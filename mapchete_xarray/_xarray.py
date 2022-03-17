import logging
from mapchete.config import validate_values
from mapchete.errors import MapcheteConfigError
from mapchete.formats import base
from mapchete.io import path_exists, fs_from_path
from mapchete.io.raster import create_mosaic, extract_from_array, bounds_to_ranges
from mapchete.tile import BufferedTile
import numpy as np
import os
from rasterio.transform import from_origin
import tempfile
import xarray as xr
import zarr

from mapchete_xarray._zarr import initialize_zarr


logger = logging.getLogger(__name__)

METADATA = {"driver_name": "xarray", "data_type": "raster", "mode": "w"}


class OutputDataWriter:
    """
    Constructor class which either returns XarraySingleFileOutputWriter or
    XarrayTileDirectoryOutputWriter.

    Parameters
    ----------
    output_params : dictionary
        output parameters from Mapchete file

    Attributes
    ----------
    path : string
        path to output directory
    file_extension : string
        file extension for output files (.tif)
    output_params : dictionary
        output parameters from Mapchete file
    nodata : integer or float
        nodata value used when writing GeoTIFFs
    pixelbuffer : integer
        buffer around output tiles
    pyramid : ``tilematrix.TilePyramid``
        output ``TilePyramid``
    crs : ``rasterio.crs.CRS``
        object describing the process coordinate reference system
    srid : string
        spatial reference ID of CRS (e.g. "{'init': 'epsg:4326'}")
    """

    def __new__(self, output_params, **kwargs):
        """Initialize."""
        self.path = output_params["path"]
        self.file_extension = ".zarr"
        if self.path.endswith(self.file_extension):
            return XarrayZarrOutputDataWriter(output_params, **kwargs)
        else:
            return XarrayTileDirectoryOutputDataWriter(output_params, **kwargs)


class OutputDataReader:
    def __new__(self, output_params, **kwargs):
        """Initialize."""
        self.path = output_params["path"]
        self.file_extension = ".zarr"
        if self.path.endswith(self.file_extension):
            return XarrayZarrOutputDataReader(output_params, **kwargs)
        else:
            return XarrayTileDirectoryOutputDataReader(output_params, **kwargs)


class XarrayZarrOutputDataReader(base.SingleFileOutputReader):

    METADATA = METADATA

    def __init__(self, output_params, *args, **kwargs):
        super(XarrayTileDirectoryOutputDataReader, self).__init__(output_params)
        self.path = output_params["path"]
        self.output_params = output_params
        self.nodata = output_params.get("nodata", 0)
        self.storage = "zarr"
        self.file_extension = ".zarr"
        self.fs = fs_from_path(self.path)

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

        raise NotImplementedError()

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
        raise NotImplementedError()


class XarrayZarrOutputDataWriter(base.SingleFileOutputWriter):

    METADATA = METADATA

    def __init__(self, output_params, *args, **kwargs):
        super(XarrayZarrOutputDataWriter, self).__init__(output_params)
        self.file_extension = ".zarr"
        self.output_params = output_params
        zoom = output_params["delimiters"]["zoom"][0]
        self.path = output_params["path"]
        self.fill_value = output_params.get("fill_value", None)
        self.count = output_params.get("bands", 1)
        self.dtype = output_params.get("dtype", "uint8")
        self.x_axis_name = output_params.get("x_axis_name", "X")
        self.y_axis_name = output_params.get("y_axis_name", "Y")
        self.area_or_point = output_params.get("area_or_point", "Area")
        self.bounds = output_params["delimiters"]["process_bounds"]
        self.affine = from_origin(
            self.bounds.left,
            self.bounds.top,
            self.pyramid.pixel_x_size(zoom),
            self.pyramid.pixel_y_size(zoom),
        )
        self.shape = (
            self.pyramid.matrix_height(zoom) * self.pyramid.tile_height(zoom),
            self.pyramid.matrix_width(zoom) * self.pyramid.tile_width(zoom),
        )
        # check if archive exists
        try:
            self.fs.ls(self.path)
            raise NotImplementedError
        except FileNotFoundError:
            # if not, create an empty one
            initialize_zarr(
                path=self.path,
                bounds=self.bounds,
                shape=self.shape,
                crs=self.pyramid.crs,
                chunksize=self.pyramid.tile_size * self.pyramid.metatiling,
                fill_value=self.fill_value,
                count=self.count,
                dtype=self.dtype,
                x_axis_name=self.x_axis_name,
                y_axis_name=self.y_axis_name,
                area_or_point=self.area_or_point,
            )
        # if yes, verify it is compatible with our output parameters

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
        # TODO: find method to check whether tile output was already written
        return False
        raise NotImplementedError()

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
        ds = xr.Dataset(
            # coords={
            #     self.x_axis_name: ([self.x_axis_name], coord_x),
            #     self.y_axis_name: ([self.y_axis_name], coord_y),
            # },
            data_vars={
                f"Band{i}": ([self.y_axis_name, self.x_axis_name], array)
                for i, array in zip(range(1, self.count + 1), data)
            }
        )
        minrow, maxrow, mincol, maxcol = bounds_to_ranges(
            out_bounds=process_tile.bounds, in_affine=self.affine, in_shape=self.shape
        )
        ds.to_zarr(
            self.path,
            compute=False,
            safe_chunks=True,
            region={
                self.x_axis_name: slice(mincol, maxcol),
                self.y_axis_name: slice(minrow, maxrow),
            },
        )

        # dataset.to_zarr(region...)
        # use "region" kwarg to determine where in the zarr archive to write the data to
        raise NotImplementedError

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
        return isinstance(process_data, xr.DataArray)

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
        return process_data

    def close(self, exc_type=None, exc_value=None, exc_traceback=None):
        """Gets called if process is closed."""
        pass


class XarrayTileDirectoryOutputDataReader(base.TileDirectoryOutputReader):

    METADATA = METADATA

    def __init__(self, output_params, **kwargs):
        """Initialize."""
        super(XarrayTileDirectoryOutputDataReader, self).__init__(output_params)
        self.path = output_params["path"]
        self.output_params = output_params
        self.nodata = output_params.get("nodata", 0)
        self.storage = output_params.get("storage", "netcdf")
        if self.storage not in ["netcdf", "zarr"]:
            raise ValueError("'storage' must either be 'netcdf' or 'zarr'")
        self.file_extension = ".nc" if self.storage == "netcdf" else ".zarr"
        self.fs = fs_from_path(self.path)

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
        # We need to use special code for zarr output to check whether tiles
        # exist on an S3 bucket.
        if process_tile and output_tile:
            raise ValueError("just one of 'process_tile' and 'output_tile' allowed")
        if process_tile:
            if self.storage == "netcdf":
                return any(
                    path_exists(self.get_path(tile))
                    for tile in self.pyramid.intersecting(process_tile)
                )
            else:
                return any(
                    path_exists(os.path.join(*[self.get_path(tile), "data", ".zarray"]))
                    for tile in self.pyramid.intersecting(process_tile)
                )
        if output_tile:
            if self.storage == "netcdf":
                return path_exists(self.get_path(output_tile))
            else:
                return path_exists(
                    os.path.join(*[self.get_path(output_tile), "data", ".zarray"])
                )


class XarrayTileDirectoryOutputDataWriter(
    base.TileDirectoryOutputWriter, XarrayTileDirectoryOutputDataReader
):

    METADATA = METADATA

    def __init__(self, output_params, **kwargs):
        """Initialize."""
        super(XarrayTileDirectoryOutputDataWriter, self).__init__(output_params)

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
        return validate_values(config, [("path", str)])

    def empty(self, process_tile):
        """
        Return empty data.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``

        Returns
        -------
        empty data : xarray
            empty xarray
        """
        return xr.DataArray([])

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
        return isinstance(process_data, xr.DataArray)

    def output_cleaned(self, process_data):
        """
        Clean up process_data if necessary.

        Parameters
        ----------
        process_data : raw process output

        Returns
        -------
        xarray.DataArray
        """
        return process_data

    def write(self, process_tile, data):
        """
        Write data from process tiles into GeoTIFF file(s).

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``
        data : ``np.ndarray``
        """
        if data.shape == (0,):
            logger.debug("process output data empty, nothing to write")
            return
        # Convert from process_tile to output_tiles and write
        for out_tile in self.pyramid.intersecting(process_tile):
            out_tile = BufferedTile(out_tile, self.pixelbuffer)
            out_path = self.get_path(out_tile)
            logger.debug("write output to %s", out_path)
            self.prepare_path(out_tile)
            out_xarr = _copy_metadata(
                base_darr=data,
                new_data=extract_from_array(
                    in_raster=data.data,
                    in_affine=process_tile.affine,
                    out_tile=out_tile,
                ),
            )
            if np.where(out_xarr.data == self.nodata, True, False).all():
                logger.debug("output tile data empty, nothing to write")
            else:
                if self.storage == "netcdf":
                    if out_path.startswith("s3://"):
                        # this below does not work as writing to a file object is only
                        # supported by the "scipy" engine which does not accept our
                        # encoding dict
                        # with self.fs.open(out_path, "wb") as dst:
                        #     dst.write(
                        #         out_xarr.to_dataset(name="data").to_netcdf(
                        #             self.fs.get_mapper(out_path),
                        #             encoding={"data": self._get_encoding()},
                        #         )
                        #     )
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmp_path = os.path.join(tmpdir, "temp.nc")
                            logger.debug("write to temporary file %s", tmp_path)
                            out_xarr.to_dataset(name="data").to_netcdf(
                                tmp_path, encoding={"data": self._get_encoding()}
                            )
                            logger.debug("upload %s to %s", tmp_path, out_path)
                            self.fs.upload(tmp_path, out_path)
                    else:
                        out_xarr.to_dataset(name="data").to_netcdf(
                            out_path, encoding={"data": self._get_encoding()}
                        )
                elif self.storage == "zarr":
                    out_path = self.fs.get_mapper(out_path)
                    logger.debug("write output to %s", out_path)
                    out_xarr.to_dataset(name="data").to_zarr(
                        out_path, mode="w", encoding={"data": self._get_encoding()}
                    )

    def read(self, output_tile, **kwargs):
        """
        Read existing process output.

        Parameters
        ----------
        output_tile : ``BufferedTile``
            must be member of output ``TilePyramid``

        Returns
        -------
        NumPy array
        """
        try:
            out_path = self.get_path(output_tile)
            if self.storage == "netcdf":
                if out_path.startswith("s3://"):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = os.path.join(tmpdir, "temp.nc")
                        logger.debug("download to temporary file %s", tmp_path)
                        self.fs.download(out_path, tmp_path)
                        return xr.open_dataset(tmp_path)["data"]
                return xr.open_dataset(self.get_path(output_tile))["data"]
            elif self.storage == "zarr":
                out_path = self.fs.get_mapper(out_path)
                return xr.open_zarr(out_path, chunks=None)["data"]
        except (FileNotFoundError, ValueError):
            return self.empty(output_tile)

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

    def extract_subset(self, input_data_tiles=None, out_tile=None):
        """
        Extract subset from multiple tiles.

        input_data_tiles : list of (``Tile``, process data) tuples
        out_tile : ``Tile``

        Returns
        -------
        NumPy array or list of features.
        """
        if input_data_tiles[0][0].tp.metatiling < out_tile.tp.metatiling:
            raise MapcheteConfigError(
                "process metatiling must be smaller than xarray output metatiling"
            )
        return _copy_metadata(
            base_darr=input_data_tiles[0][1],
            new_data=extract_from_array(
                in_raster=create_mosaic([(i[0], i[1].data) for i in input_data_tiles]),
                out_tile=out_tile,
            ),
        )

    def _read_as_tiledir(
        self,
        out_tile=None,
        td_crs=None,
        tiles_paths=None,
        profile=None,
        validity_check=False,
        indexes=None,
        resampling=None,
        dst_nodata=None,
        gdal_opts=None,
        **kwargs,
    ):
        """
        Read reprojected & resampled input data.

        Parameters
        ----------
        validity_check : bool
            vector file: also run checks if reprojected geometry is valid,
            otherwise throw RuntimeError (default: True)
        indexes : list or int
            raster file: a list of band numbers; None will read all.
        dst_nodata : int or float, optional
            raster file: if not set, the nodata value from the source dataset
            will be used
        gdal_opts : dict
            raster file: GDAL options passed on to rasterio.Env()

        Returns
        -------
        data : list for vector files or numpy array for raster files
        """
        if not tiles_paths:
            return self.empty(out_tile)
        source_tile = tiles_paths[0][0]
        if source_tile.tp.grid != out_tile.tp.grid:
            raise MapcheteConfigError(
                "xarray tile directory must have same grid as process pyramid"
            )
        return self.extract_subset(
            input_data_tiles=[(tile, self.read(tile)) for tile, _ in tiles_paths],
            out_tile=out_tile,
        )

    def _get_encoding(self):
        if self.storage == "netcdf":
            return dict(
                zlib=self.output_params.get("zlib", True),
                complevel=self.output_params.get("complevel", 4),
                shuffle=self.output_params.get("shuffle", True),
                fletcher32=self.output_params.get("fletcher32", False),
            )
        elif self.storage == "zarr":
            return dict(
                compressor=zarr.Blosc(
                    cname=self.output_params.get("compressor", "zstd"),
                    clevel=self.output_params.get("complevel", 3),
                    shuffle=self.output_params.get("shuffle", 1),
                )
            )


class InputTile(base.InputTile):  # pragma: no cover
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


def _copy_metadata(base_darr=None, new_data=None):
    return xr.DataArray(
        data=new_data,
        coords=base_darr.coords,
        dims=base_darr.dims,
        name=base_darr.name,
        attrs=base_darr.attrs,
    )
