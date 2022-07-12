"""
Contains all classes required to use the xarray driver as mapchete input.
"""

import numpy as np
import xarray as xr
import zarr
from mapchete.config import snap_bounds
from mapchete.formats import base, load_metadata
from mapchete.io import path_exists
from mapchete.io.vector import reproject_geometry
from shapely.geometry import box
from zarr.storage import FSStore


class InputData(base.InputData):
    """In case this driver is used when being a readonly input to another process."""

    _ds = None

    def __init__(self, input_params, **kwargs):
        """Initialize."""
        super().__init__(input_params, **kwargs)
        self.path = input_params["path"]
        if not path_exists(self.path):  # pragma: no cover
            raise FileNotFoundError(f"path {self.path} does not exist")
        mapchete_params = self.ds.attrs.get("mapchete")
        if mapchete_params is None:  # pragma: no cover
            raise TypeError(
                f"zarr archive at {self.path} exists but does not hold mapchete metadata"
            )
        mapchete_metadata = load_metadata(mapchete_params)
        self.zarr_pyramid = mapchete_metadata["pyramid"]
        if self.zarr_pyramid.crs != self.pyramid.crs:  # pragma: no cover
            raise NotImplementedError(
                f"single zarr output ({self.zarr_pyramid.crs}) cannot be reprojected to "
                f"different CRS ({self.pyramid.crs})"
            )
        self._bounds = snap_bounds(
            bounds=mapchete_metadata["driver"]["delimiters"]["process_bounds"],
            pyramid=self.zarr_pyramid,
            zoom=mapchete_metadata["driver"]["delimiters"]["zoom"][0],
        )
        self.x_axis_name = mapchete_metadata["driver"].get("x_axis_name", "X")
        self.y_axis_name = mapchete_metadata["driver"].get("y_axis_name", "Y")
        self.time_axis_name = mapchete_metadata["driver"].get("time_axis_name", "time")
        self.time = mapchete_metadata["driver"].get("time", {})
        self.band_names = mapchete_metadata["driver"].get(
            "band_names", [v for v in self.ds.data_vars]
        )

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

    def open(self, tile, **kwargs):
        """
        Return InputTile object.

        Parameters
        ----------
        tile : ``Tile``

        Returns
        -------
        input tile : ``InputTile``
            tile view of input data
        """
        return InputTile(
            tile,
            path=self.path,
            x_axis_name=self.x_axis_name,
            y_axis_name=self.y_axis_name,
            time_axis_name=self.time_axis_name,
            time=self.time,
            band_names=self.band_names,
            bbox=self.bbox(),
            **kwargs,
        )

    def bbox(self, out_crs=None):
        """
        Return data bounding box.

        Parameters
        ----------
        out_crs : ``rasterio.crs.CRS``
            rasterio CRS object (default: CRS of process pyramid)

        Returns
        -------
        bounding box : geometry
            Shapely geometry object
        """
        return reproject_geometry(
            box(*self._bounds),
            src_crs=self.zarr_pyramid.crs,
            dst_crs=self.pyramid.crs if out_crs is None else out_crs,
            segmentize_on_clip=True,
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

    def __init__(
        self,
        tile,
        path=None,
        x_axis_name=None,
        y_axis_name=None,
        time_axis_name=None,
        time=None,
        band_names=None,
        bbox=None,
        **kwargs,
    ):
        """Initialize."""
        self.path = path
        self.tile = tile
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.time_axis_name = time_axis_name
        self.time = time
        self.band_names = band_names
        self.bbox = bbox
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

    @property
    def bands(self):
        """Return band names in correct order."""
        return self.band_names

    def _get_indexes(self, indexes=None):
        """Return a list of band names (i.e. Zarr data variable names)."""
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

    def read(
        self, indexes=None, start_time=None, end_time=None, timestamps=None, **kwargs
    ):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : xarray.Dataset
        """
        bounds = self.tile.bounds
        selector = {
            self.x_axis_name: slice(bounds.left, bounds.right),
            self.y_axis_name: slice(bounds.top, bounds.bottom),
        }

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

        return self.ds[self._get_indexes(indexes)].sel(**selector)

    def is_empty(self):  # pragma: no cover
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        return not self.tile.bbox.intersects(self.bbox)
