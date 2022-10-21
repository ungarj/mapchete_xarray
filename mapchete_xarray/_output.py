"""
Contains all classes required to use the xarray driver as mapchete output.
"""

import logging
import math
import os

import croniter
import dateutil
import numpy as np
import xarray as xr
import zarr
from mapchete.config import snap_bounds, validate_values
from mapchete.errors import MapcheteConfigError
from mapchete.formats import base
from mapchete.formats.tools import compare_metadata_params, dump_metadata, load_metadata
from mapchete.io import fs_from_path, path_exists
from mapchete.io.raster import bounds_to_ranges, create_mosaic, extract_from_array
from rasterio.transform import from_origin
from tilematrix import Bounds
from zarr.storage import FSStore

logger = logging.getLogger(__name__)

METADATA = {
    "driver_name": "xarray",
    "data_type": "raster",
    "mode": "w",
    "file_extensions": ["zarr"],
}

DEFAULT_TIME_CHUNKSIZE = 8


class OutputDataReader(base.SingleFileOutputReader):

    METADATA = METADATA
    _ds = None

    def __init__(self, output_params, *args, **kwargs):
        super().__init__(output_params)
        if output_params.get("pixelbuffer", 0) > 0:
            raise MapcheteConfigError(
                "a pixelbuffer larger than 0 is not allowed with zarr"
            )
        self.output_params = output_params
        self.nodata = output_params.get("nodata", 0)
        self.storage = "zarr"
        self.file_extension = ".zarr"
        self.path = output_params["path"]
        if not self.path.endswith(self.file_extension):
            raise MapcheteConfigError("output path must end with .zarr")
        self.fs = fs_from_path(self.path)
        self.output_params = output_params
        self.zoom = output_params["delimiters"]["zoom"][0]

        if output_params.get("band_names"):
            self.band_names = output_params.get("band_names")
            self.count = len(self.band_names)
        elif output_params.get("bands"):
            self.count = output_params.get("bands")
            self.band_names = [f"Band{i}" for i in range(1, self.count + 1)]
        else:  # pragma: no cover
            raise ValueError("either 'count' or 'band_names' has to be provided")

        self.dtype = output_params.get("dtype", "uint8")
        self.x_axis_name = output_params.get("x_axis_name", "X")
        self.y_axis_name = output_params.get("y_axis_name", "Y")
        self.band_axis_name = output_params.get("band_axis_name", "band")
        self.time_axis_name = output_params.get("time_axis_name", "time")
        self.area_or_point = output_params.get("area_or_point", "Area")
        self.bounds = snap_bounds(
            bounds=output_params["delimiters"]["process_bounds"],
            pyramid=self.pyramid,
            zoom=self.zoom,
        )
        self.affine = from_origin(
            self.bounds.left,
            self.bounds.top,
            self.pyramid.pixel_x_size(self.zoom),
            self.pyramid.pixel_y_size(self.zoom),
        )
        self.shape = (
            math.ceil(
                (self.bounds.top - self.bounds.bottom)
                / self.pyramid.pixel_x_size(self.zoom)
            ),
            math.ceil(
                (self.bounds.right - self.bounds.left)
                / self.pyramid.pixel_x_size(self.zoom)
            ),
        )
        self.time = output_params.get("time", {})
        self.start_time = self.time.get("start")
        self.end_time = self.time.get("end")

    @property
    def ds(self):
        if self._ds is None:
            self._ds = xr.open_zarr(
                self.path,
                mask_and_scale=False,
                consolidated=True,
                chunks=None,
            )
        return self._ds

    @property
    def bands(self):
        """Return band names in correct order."""
        return self.band_names

    @property
    def axis_names(self):
        if self.time:
            return (self.time_axis_name, self.y_axis_name, self.x_axis_name)
        else:
            return (self.y_axis_name, self.x_axis_name)

    def read(self, output_tile):
        """
        Read existing process output.

        Parameters
        ----------
        output_tile : ``BufferedTile``
            must be member of output ``TilePyramid``

        Returns
        -------
        process output : array or list
        """
        return self._read(bounds=output_tile.bounds)[self.band_names]

    def empty(self, *args):
        """
        Return empty data.

        Returns
        -------
        empty data : array or list
            empty array with correct data type for raster data or empty list
            for vector data
        """
        return xr.Dataset()

    def open(self, tile, process, **kwargs):  # pragma: no cover
        """
        Open process output as input for other process.

        Parameters
        ----------
        tile : ``Tile``
        process : ``MapcheteProcess``
        kwargs : keyword arguments
        """
        return InputTile(
            tile,
            process,
            time=self.time,
            time_axis_name=self.time_axis_name,
            bands=self.bands,
            band_axis_name=self.band_axis_name,
        )

    def extract_subset(self, input_data_tiles=None, out_tile=None):
        # for mapchete.io.raster.create_mosaic() the input arrays must have
        # the bands as first dimension, the rest will get stitched together
        def _transpose_darrays(input_data_tiles):
            for tile, ds in input_data_tiles:
                # convert Dataset to DataArray
                darr = ds.to_array(dim=self.band_axis_name)
                if (
                    self.time and darr.dims[0] != self.band_axis_name
                ):  # pragma: no cover
                    yield (
                        tile,
                        darr.transpose(
                            self.band_axis_name,
                            self.time_axis_name,
                            self.y_axis_name,
                            self.x_axis_name,
                        ).data,
                    )
                else:
                    yield tile, darr.data

        mosaic = create_mosaic(_transpose_darrays(input_data_tiles))
        arr = extract_from_array(
            in_raster=mosaic,
            in_affine=mosaic.affine,
            out_tile=out_tile,
        )
        coords = {self.time_axis_name: input_data_tiles[0][1].time.values}
        return xr.Dataset(
            data_vars={
                band_name: (self.axis_names, band)
                for band_name, band in zip(self.band_names, arr)
            },
            coords=coords,
        )

    def _bounds_to_ranges(self, bounds):
        return bounds_to_ranges(
            out_bounds=bounds, in_affine=self.affine, in_shape=self.shape
        )

    def _timestamp_regions(self, timestamps):
        slice_idxs = list()
        slice_timestamps = list()

        for t in sorted(timestamps):
            try:
                idx = list(self.ds.time.values).index(t)
            except ValueError:  # pragma: no cover
                raise ValueError(
                    f"time slice {t} not available to insert: {self.ds.time.values}"
                )

            if slice_idxs and idx > slice_idxs[-1] + 1:
                yield slice_timestamps, slice(slice_idxs[0], slice_idxs[-1] + 1)
                slice_idxs = list()
                slice_timestamps = list()

            slice_idxs.append(idx)
            slice_timestamps.append(t)

        if slice_idxs:
            yield slice_timestamps, slice(slice_idxs[0], slice_idxs[-1] + 1)

    def _read(self, bounds):

        selector = {
            self.x_axis_name: slice(bounds.left, bounds.right),
            self.y_axis_name: slice(bounds.top, bounds.bottom),
        }

        if self.time:
            selector[self.time_axis_name] = slice(self.start_time, self.end_time)

        return self.ds.sel(**selector)


class OutputDataWriter(base.SingleFileOutputWriter, OutputDataReader):

    METADATA = METADATA

    def __init__(self, output_params, *args, **kwargs):
        super().__init__(output_params, *args, **kwargs)

    def prepare(self, **kwargs):
        if path_exists(self.path):
            # verify it is compatible with our output parameters / chunking
            archive = zarr.open(FSStore(f"{self.path}"))
            mapchete_params = archive.attrs.get("mapchete")
            if mapchete_params is None:  # pragma: no cover
                raise TypeError(
                    f"zarr archive at {self.path} exists but does not hold mapchete metadata"
                )
            existing = load_metadata(mapchete_params)
            current = load_metadata(dump_metadata(self.output_params))
            compare_metadata_params(existing, current)
        else:
            # if output does not exist, create an empty one
            initialize_zarr(
                path=self.path,
                bounds=self.bounds,
                shape=self.shape,
                crs=self.pyramid.crs,
                time=self.time,
                chunksize=self.pyramid.tile_size * self.pyramid.metatiling,
                fill_value=self.nodata,
                band_names=self.band_names,
                dtype=self.dtype,
                x_axis_name=self.x_axis_name,
                y_axis_name=self.y_axis_name,
                time_axis_name=self.time_axis_name,
                area_or_point=self.area_or_point,
                output_metadata=dump_metadata(self.output_params),
            )

    def _zarr_chunk_from_xy(self, x, y):

        # determine row
        pixel_y_size = _pixel_y_size(self.bounds.top, self.bounds.bottom, self.shape[0])
        tile_y_size = round(
            pixel_y_size * self.pyramid.tile_size * self.pyramid.metatiling, 20
        )
        row = abs(int((self.ds[self.y_axis_name].max() - y) / tile_y_size))

        # determine column
        pixel_x_size = _pixel_x_size(self.bounds.right, self.bounds.left, self.shape[1])
        tile_x_size = round(
            pixel_x_size * self.pyramid.tile_size * self.pyramid.metatiling, 20
        )
        col = abs(int((x - self.ds[self.x_axis_name].min()) / tile_x_size))

        return row, col

    def tiles_exist(self, process_tile=None, output_tile=None):
        """
        Check whether output tiles of a tile (either process or output) exists.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``
        output_tile : ``BufferedTile``
            must be member of output ``TilePyramid``

        Returns
        -------
        exists : bool
        """

        tile = process_tile or output_tile
        zarr_chunk_row, zarr_chunk_col = self._zarr_chunk_from_xy(
            tile.bbox.centroid.x, tile.bbox.centroid.y
        )

        for var in self.ds:

            if self.time:

                if path_exists(
                    os.path.join(
                        self.path,
                        var,
                        f"0.{zarr_chunk_row}.{zarr_chunk_col}",
                    )
                ):
                    return True
            else:
                if path_exists(
                    os.path.join(self.path, var, f"{zarr_chunk_row}.{zarr_chunk_col}")
                ):
                    return True
        return False

    def is_valid_with_config(self, config):
        """
        Check if output format is valid with other process parameters.

        Parameters
        ----------
        config : dictionary
            output configuration parameters

        Returns
        -------
        is_valid : bool
        """
        if len(config["delimiters"]["zoom"]) > 1:
            raise ValueError("zarr output can only be used with a single zoom")
        if "time" in config:
            if "pattern" not in config["time"] and "steps" not in config["time"]:
                raise ValueError(
                    "when using a time axis, please specify the time stamps either through "
                    "'pattern' or 'steps'"
                )
        return validate_values(config, [("path", str)])

    def write(self, process_tile, data):
        """
        Write data from one or more process tiles.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``
        """
        if len(data) == 0:
            logger.debug("output empty, nothing to write")
            return
        minrow, maxrow, mincol, maxcol = self._bounds_to_ranges(process_tile.bounds)
        coords = {}
        region = {
            self.x_axis_name: slice(mincol, maxcol),
            self.y_axis_name: slice(minrow, maxrow),
        }

        if self.time:
            coords[self.time_axis_name] = data.time.values

        def write_zarr(ds, region):
            ds.to_zarr(
                FSStore(self.path),
                mode="r+",
                compute=True,
                safe_chunks=True,
                region=region,
            )

        ds = self.output_cleaned(data)
        if self.time:
            for timestamps, time_region in self._timestamp_regions(ds.time.values):
                region[self.time_axis_name] = time_region
                write_zarr(ds.sel(time=timestamps), region)
        else:
            write_zarr(ds, region)

    def _dataarray_to_dataset(self, darr):
        coords = {}
        if self.time:
            coords[self.time_axis_name] = np.array(
                darr.time.values, dtype=np.datetime64
            )
            # make sure the band axis is first
            if darr.dims[0] != self.band_axis_name:
                darr = darr.transpose(
                    self.band_axis_name,
                    self.time_axis_name,
                    self.y_axis_name,
                    self.x_axis_name,
                )
        return xr.Dataset(
            data_vars={
                band_name: (self.axis_names, band.values)
                for band_name, band in zip(self.band_names, darr)
            },
            coords=coords,
        )

    def _ndarray_to_dataset(self, ndarr):
        coords = {}
        if self.time:
            coords[self.time_axis_name] = np.array(
                self.ds.time.values, dtype=np.datetime64
            )
            slices, bands = ndarr.shape[:2]
            if slices != len(self.ds.time.values):  # pragma: no cover
                raise ValueError(
                    f"NumPy array ({slices} slices) does not fit into "
                    f"Zarr on time axis ({len(self.ds.time.values)} slices)."
                )
            elif bands != len(self.ds.data_vars):  # pragma: no cover
                raise ValueError(
                    f"NumPy array ({bands} bands) does not fit into "
                    f"Zarr on band axis ({len(self.ds.data_vars)} bands)."
                )
            # make sure the band axis is first
            ndarr = np.transpose(ndarr, (1, 0, 2, 3))
        return xr.Dataset(
            data_vars={
                band_name: (self.axis_names, band)
                for band_name, band in zip(self.band_names, ndarr)
            },
            coords=coords,
        )

    def output_is_valid(self, process_data):
        """
        Check whether process output is allowed with output driver.

        Parameters
        ----------
        process_data : raw process output

        Returns
        -------
        True or False
        """
        return isinstance(process_data, (xr.Dataset, xr.DataArray, np.ndarray))

    def output_cleaned(self, process_data):
        """
        Return verified and cleaned output.

        Parameters
        ----------
        process_data : raw process output

        Returns
        -------
        xarray
        """
        if isinstance(process_data, xr.Dataset):
            # we have to clean all of the Dataset metadata, otherwise there
            # will be errors when updating an existing Dataset

            # delete mapchete metadata from root
            process_data.attrs.pop("mapchete", None)
            # delete GDAL specific metadata from DataArrays
            for darr in process_data.values():
                for attr in ["_FillValue", "AREA_OR_POINT", "_CRS"]:
                    darr.attrs.pop(attr, None)
            # delete GDAL specific metadata from coordinates
            for coord in process_data.coords:
                for attr in ["_FillValue", "AREA_OR_POINT", "_CRS"]:
                    process_data[coord].attrs.pop(attr, None)
            return process_data

        elif isinstance(process_data, xr.DataArray):
            return self._dataarray_to_dataset(process_data)

        elif isinstance(process_data, np.ndarray):
            return self._ndarray_to_dataset(process_data)

        else:  # pragma: no cover
            raise TypeError(
                f"xarray driver only accepts xarray.DataArray or xarray.Dataset as output, not {type(process_data)}"
            )

    def close(self, exc_type=None, exc_value=None, exc_traceback=None):
        """Gets called if process is closed."""
        try:
            if self._ds is not None:
                logger.debug("close dataset")
                self._ds.close()
        except Exception as e:  # pragma: no cover
            logger.debug(e)


class InputTile(base.InputTile):
    """
    Target Tile representation of input data.

    Parameters
    ----------
    tile : ``Tile``
    kwargs : keyword arguments
        driver specific parameters
    """

    def __init__(
        self,
        tile,
        process,
        time=None,
        time_axis_name=None,
        bands=None,
        band_axis_name=None,
        **kwargs,
    ):
        """Initialize."""
        self.tile = tile
        self.process = process
        self.time = time
        self.time_axis_name = time_axis_name
        self.bands = bands
        self.band_axis_name = band_axis_name

    def read(
        self, indexes=None, start_time=None, end_time=None, timestamps=None, **kwargs
    ):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        selector = {}
        if self.time:
            if start_time or end_time:
                selector[self.time_axis_name] = slice(
                    start_time or self.time.get("start"),
                    end_time or self.time.get("end"),
                )
            elif timestamps:
                selector[self.time_axis_name] = np.array(
                    timestamps, dtype=np.datetime64
                )

        ds = self.process.get_raw_output(self.tile)
        return ds[self._get_indexes(indexes)].sel(**selector)

    def is_empty(self):  # pragma: no cover
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        return not self.tile.bbox.intersects(self.process.config.area_at_zoom())

    def _get_indexes(self, indexes=None):
        if indexes is None:
            return self.bands
        indexes = indexes if isinstance(indexes, list) else [indexes]
        out = []
        for i in indexes:
            if isinstance(i, int):
                out.append(self.bands[i])
            elif isinstance(i, str):
                out.append(i)
            else:  # pragma: no cover
                raise TypeError(
                    f"band indexes must either be integers or strings, not: {i}"
                )
        return out


def _pixel_x_size(right, left, width):
    return (right - left) / width


def _pixel_y_size(top, bottom, height):
    return (top - bottom) / -height


def initialize_zarr(
    path=None,
    bounds=None,
    shape=None,
    crs=None,
    time=None,
    fill_value=None,
    chunksize=256,
    band_names=None,
    dtype="uint8",
    x_axis_name="X",
    y_axis_name="Y",
    time_axis_name="time",
    area_or_point="Area",
    output_metadata=None,
):
    if path_exists(path):  # pragma: no cover
        raise IOError(f"cannot initialize zarr storage as path already exists: {path}")

    height, width = shape
    bounds = Bounds(*bounds)
    pixel_x_size = _pixel_x_size(bounds.right, bounds.left, width)
    pixel_y_size = _pixel_y_size(bounds.top, bounds.bottom, height)

    coord_x = [bounds.left + pixel_x_size / 2 + i * pixel_x_size for i in range(width)]
    coord_y = [bounds.top + pixel_y_size / 2 + i * pixel_y_size for i in range(height)]

    axis_names = [y_axis_name, x_axis_name]
    coords = {
        x_axis_name: ([x_axis_name], coord_x),
        y_axis_name: ([y_axis_name], coord_y),
    }

    if time:
        start_time = (
            dateutil.parser.parse(time["start"])
            if isinstance(time["start"], str)
            else time["start"]
        )

        end_time = (
            dateutil.parser.parse(time["end"])
            if isinstance(time["end"], str)
            else time["end"]
        )

        if "pattern" in time:
            coord_time = [
                t
                for t in croniter.croniter_range(
                    start_time,
                    end_time,
                    time["pattern"],
                )
            ]
        elif "steps" in time:
            # convert timestamp steps into np.datetime64
            coord_time = np.array(
                [
                    dateutil.parser.parse(t) if isinstance(t, str) else t
                    for t in time["steps"]
                ],
                dtype=np.datetime64,
            )
        else:  # pragma: no cover
            raise ValueError(
                "timestamps have to be provied either as list in 'steps' or a pattern in 'pattern'"
            )
        coords[time_axis_name] = coord_time

        output_shape = (len(coord_time), *shape)
        output_chunks = (
            time.get("chunksize", DEFAULT_TIME_CHUNKSIZE),
            chunksize,
            chunksize,
        )
        axis_names = [time_axis_name] + axis_names

    else:
        output_shape = shape
        output_chunks = (chunksize, chunksize)

    try:
        # write zarr
        ds = xr.Dataset(coords=coords)
        ds.to_zarr(
            FSStore(path),
            compute=False,
            encoding={var: {"_FillValue": fill_value} for var in ds.data_vars},
            safe_chunks=True,
        )

        # add GDAL metadata for each band
        for band_name in band_names:
            store = FSStore(f"{path}/{band_name}")
            zarr.creation.create(
                shape=output_shape,
                chunks=output_chunks,
                dtype=dtype,
                store=store,
            )

            attrs = zarr.open(store).attrs
            attrs["_ARRAY_DIMENSIONS"] = axis_names
            attrs["_CRS"] = {"wkt": crs.wkt}
            attrs["AREA_OR_POINT"] = area_or_point

        # add global metadata
        if output_metadata:
            zarr.open(FSStore(f"{path}")).attrs.update(mapchete=output_metadata)
        zarr.consolidate_metadata(path)

    except Exception:  # pragma: no cover
        # remove leftovers if something failed during initialization
        try:
            fs_from_path(path).rm(path, recursive=True)
        except FileNotFoundError:
            pass
        raise
