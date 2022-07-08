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

    def __init__(self, input_params, **kwargs):
        """Initialize."""
        super().__init__(input_params, **kwargs)
        self.path = input_params["path"]
        if not path_exists(self.path):
            raise FileNotFoundError(f"path {self.path} does not exist")
        archive = zarr.open(FSStore(f"{self.path}"))
        mapchete_params = archive.attrs.get("mapchete")
        if mapchete_params is None:
            raise TypeError(
                f"zarr archive at {self.path} exists but does not hold mapchete metadata"
            )
        metadata = load_metadata(mapchete_params)
        self.zarr_pyramid = metadata["pyramid"]
        if self.zarr_pyramid.crs != self.pyramid.crs:
            raise NotImplementedError(
                f"single zarr output ({self.zarr_pyramid.crs}) cannot be reprojected to different CRS ({self.pyramid.crs})"
            )
        self._bounds = snap_bounds(
            bounds=metadata["driver"]["delimiters"]["process_bounds"],
            pyramid=self.zarr_pyramid,
            zoom=metadata["driver"]["delimiters"]["zoom"][0],
        )
        self.x_axis_name = metadata["driver"].get("x_axis_name", "X")
        self.y_axis_name = metadata["driver"].get("y_axis_name", "Y")
        self.time = metadata["driver"].get("time", {})

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
            time=self.time,
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


class InputTile(base.InputTile):  # pragma: no cover
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
        time=None,
        bbox=None,
        **kwargs,
    ):
        """Initialize."""
        self.path = path
        self.tile = tile
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name
        self.time = time
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

    def read(self, indexes=None, start_time=None, end_time=None, **kwargs):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        return self.read_dataset(start_time=start_time, end_time=end_time, **kwargs)

    def read_dataarray(self, indexes=None, start_time=None, end_time=None, **kwargs):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        dataset = self.read_dataset(start_time=start_time, end_time=end_time, **kwargs)
        return xr.concat(dataset.data_vars.values(), dim="bands")

    def read_dataset(self, indexes=None, start_time=None, end_time=None, **kwargs):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        bounds = self.tile.bounds
        selector = {
            self.x_axis_name: slice(bounds.left, bounds.right),
            self.y_axis_name: slice(bounds.top, bounds.bottom),
        }

        if self.time:
            selector["time"] = slice(
                start_time or self.time.get("start"), end_time or self.time.get("end")
            )

        return self.ds.sel(**selector)

    def is_empty(self):
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        return not self.tile.bbox.intersects(self.bbox)
