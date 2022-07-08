"""
Contains all classes required to use the xarray driver as mapchete output.
"""

import logging
import math

import croniter
import dask.array as da
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
from mapchete.tile import BufferedTile
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


class OutputDataReader(base.SingleFileOutputReader):

    METADATA = METADATA

    def __init__(self, output_params, *args, **kwargs):
        super().__init__(output_params)
        if output_params.get("pixelbuffer", 0) > 0:
            raise ValueError("a pixelbuffer larger than 0 is not allowed with zarr")
        self.output_params = output_params
        self.nodata = output_params.get("nodata", 0)
        self.storage = "zarr"
        self.file_extension = ".zarr"
        self.path = output_params["path"]
        if not self.path.endswith(self.file_extension):
            raise ValueError("output path must end with .zarr")
        self.fs = fs_from_path(self.path)
        self.output_params = output_params
        self.zoom = output_params["delimiters"]["zoom"][0]
        self.count = output_params.get("bands", 1)
        self.dtype = output_params.get("dtype", "uint8")
        self.x_axis_name = output_params.get("x_axis_name", "X")
        self.y_axis_name = output_params.get("y_axis_name", "Y")
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

        self._ds = None

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
        return self._read(bounds=output_tile.bounds)

    def empty(self, process_tile):
        """
        Return empty data.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``

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
        return InputTile(tile, process)

    def _bounds_to_ranges(self, bounds):
        return bounds_to_ranges(
            out_bounds=bounds, in_affine=self.affine, in_shape=self.shape
        )

    def _timestamp_regions(self, timestamps):

        slice_idxs = list()
        slice_timestamps = list()

        for t in sorted(timestamps):
            idx = list(self.ds.time.values).index(t)

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
            selector["time"] = slice(self.start_time, self.end_time)

        return self.ds.sel(**selector)


class OutputDataWriter(base.SingleFileOutputWriter, OutputDataReader):

    METADATA = METADATA

    def __init__(self, output_params, *args, **kwargs):
        super().__init__(output_params, *args, **kwargs)

    def prepare(self, process_area=None, **kwargs):
        if path_exists(self.path):
            # verify it is compatible with our output parameters / chunking
            archive = zarr.open(FSStore(f"{self.path}"))
            mapchete_params = archive.attrs.get("mapchete")
            if mapchete_params is None:
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
                count=self.count,
                dtype=self.dtype,
                x_axis_name=self.x_axis_name,
                y_axis_name=self.y_axis_name,
                area_or_point=self.area_or_point,
                output_metadata=dump_metadata(self.output_params),
            )

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
        bounds = process_tile.bounds if process_tile else output_tile.bounds
        for var in self._read(bounds=bounds).values():
            if np.any(var != self.nodata):
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
            raise ValueError("single zarr output can only be used with a single zoom")
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
        axis_names = [self.y_axis_name, self.x_axis_name]

        if self.time:
            coords["time"] = data.time.values
            axis_names = ["time"] + axis_names

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
            for timestamps, time_region in self._timestamp_regions(data.time.values):
                region["time"] = time_region
                write_zarr(ds.sel(time=timestamps), region)
        else:
            write_zarr(ds, region)

    def _dataarray_to_dataset(self, darr):
        coords = {}
        axis_names = [self.y_axis_name, self.x_axis_name]
        if self.time:
            coords["time"] = darr.time.values
            axis_names = ["time"] + axis_names
        return xr.Dataset(
            data_vars={
                f"Band{i}": (axis_names, band.values)
                for i, band in zip(range(1, self.count + 1), darr)
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
            # TODO: deleting the attributes is a hackaround attempt
            process_data.attrs.pop("mapchete", None)
            for darr in process_data.values():
                darr.attrs = {}
            return process_data
        elif isinstance(process_data, xr.DataArray):
            return self._dataarray_to_dataset(process_data)
        else:
            raise TypeError(
                f"xarray driver only accepts xarray.DataArray or xarray.Dataset as output, not {type(process_data)}"
            )

    def close(self, exc_type=None, exc_value=None, exc_traceback=None):
        """Gets called if process is closed."""
        try:
            if self._ds is not None:
                logger.debug("close dataset")
                self._ds.close()
        except Exception as e:
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

    def __init__(self, tile, process, **kwargs):
        """Initialize."""
        self.tile = tile
        self.process = process

    def read(self, **kwargs):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        return self.process.get_raw_output(self.tile)

    def is_empty(self):
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        return not self.tile.bbox.intersects(self.process.config.area_at_zoom())


def initialize_zarr(
    path=None,
    bounds=None,
    shape=None,
    crs=None,
    time=None,
    fill_value=None,
    chunksize=256,
    count=None,
    dtype="uint8",
    x_axis_name="X",
    y_axis_name="Y",
    area_or_point="Area",
    output_metadata=None,
):

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

        coord_time = [
            t
            for t in croniter.croniter_range(
                start_time,
                end_time,
                time["pattern"],
            )
        ]

        output_shape = (len(coord_time), *shape)
        output_chunks = (time["chunksize"], chunksize, chunksize)
    else:
        output_shape = shape
        output_chunks = (chunksize, chunksize)

    height, width = shape
    bounds = Bounds(*bounds)
    pixel_x_size = (bounds.right - bounds.left) / width
    pixel_y_size = (bounds.top - bounds.bottom) / -height

    coord_x = [bounds.left + pixel_x_size / 2 + i * pixel_x_size for i in range(width)]
    coord_y = [bounds.top + pixel_y_size / 2 + i * pixel_y_size for i in range(height)]

    coords = {
        x_axis_name: ([x_axis_name], coord_x),
        y_axis_name: ([y_axis_name], coord_y),
    }

    axis_names = (
        ["time", y_axis_name, x_axis_name] if time else [y_axis_name, x_axis_name]
    )

    if time:
        coords["time"] = coord_time

    ds = xr.Dataset(coords=coords)

    ds.to_zarr(
        FSStore(path),
        compute=False,
        encoding={var: {"_FillValue": fill_value} for var in ds.data_vars},
        safe_chunks=True,
    )

    for i in range(1, count + 1):
        store = FSStore(f"{path}/Band{i}")
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
