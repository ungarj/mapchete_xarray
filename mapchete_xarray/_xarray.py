import logging
from mapchete.config import validate_values
from mapchete.errors import MapcheteConfigError
from mapchete.formats import base
from mapchete.io import path_exists
from mapchete.io.raster import create_mosaic, extract_from_array
from mapchete.tile import BufferedTile
import numpy as np
import os
import s3fs
import xarray as xr
import zarr


logger = logging.getLogger(__name__)

METADATA = {
    "driver_name": "xarray",
    "data_type": "raster",
    "mode": "w"
}

class OutputDataReader(base.TileDirectoryOutputReader):

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


class OutputDataWriter(base.TileDirectoryOutputWriter, OutputDataReader):

    METADATA = METADATA

    def __init__(self, output_params, **kwargs):
        """Initialize."""
        super(OutputDataWriter, self).__init__(output_params)
        self.path = output_params["path"]
        self.output_params = output_params
        self.nodata = output_params.get("nodata", 0)
        self.storage = output_params.get("storage", "netcdf")
        if self.storage not in ["netcdf", "zarr"]:
            raise ValueError("'storage' must either be 'netcdf' or 'zarr'")
        self.file_extension = ".nc" if self.storage == "netcdf" else ".zarr"

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
                    out_tile=out_tile
                )
            )
            if np.where(out_xarr.data == self.nodata, True, False).all():
                logger.debug("output tile data empty, nothing to write")
            else:
                if self.storage == "netcdf":
                    out_xarr.to_dataset(name="data").to_netcdf(
                        out_path, encoding={"data": self._get_encoding()}
                    )
                elif self.storage == "zarr":
                    out_path = self.get_path(out_tile)
                    if out_path.startswith("s3://"):
                        # Initilize the S3 file system
                        s3 = s3fs.S3FileSystem()
                        out_path = s3fs.S3Map(root=out_path, s3=s3, check=False)
                    logger.debug("write output to %s", out_path)
                    out_xarr.to_dataset(name="data").to_zarr(
                        out_path,
                        encoding={"data": self._get_encoding()}
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
            if self.storage == "netcdf":
                return xr.open_dataset(self.get_path(output_tile))["data"]
            elif self.storage == "zarr":
                out_path = self.get_path(output_tile)
                if out_path.startswith("s3://"):
                    # Initilize the S3 file system
                    s3 = s3fs.S3FileSystem()
                    out_path = s3fs.S3Map(root=out_path, s3=s3, check=False)
                return xr.open_zarr(
                    out_path,
                    chunks=None
                )["data"]
        except (FileNotFoundError, ValueError):
            return self.empty(output_tile)

    def open(self, tile, process, **kwargs):
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
                out_tile=out_tile
            )
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
        **kwargs
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
        source_tile = tiles_paths[0][0]
        if source_tile.tp.grid != out_tile.tp.grid:
            raise MapcheteConfigError(
                "xarray tile directory must have same grid as process pyramid"
            )
        return self.extract_subset(
            input_data_tiles=[(tile, self.read(tile)) for tile, _ in tiles_paths],
            out_tile=out_tile
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
                    shuffle=self.output_params.get("shuffle", 1)
                )
            )


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


def _copy_metadata(base_darr=None, new_data=None):
    return xr.DataArray(
        data=new_data,
        coords=base_darr.coords,
        dims=base_darr.dims,
        name=base_darr.name,
        attrs=base_darr.attrs
    )
