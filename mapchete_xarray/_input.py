from mapchete.formats import base


class InputData(base.InputData):
    """In case this driver is used when being a readonly input to another process."""

    def __init__(self, input_params, **kwargs):
        """Initialize."""
        super().__init__(input_params, **kwargs)
        pass

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
        raise NotImplementedError()
        # return reproject_geometry(
        #     box(*self._bounds),
        #     src_crs=self.td_pyramid.crs,
        #     dst_crs=self.pyramid.crs if out_crs is None else out_crs,
        #     segmentize_on_clip=True,
        # )


class InputTile(base.InputTile):  # pragma: no cover
    """
    Target Tile representation of input data.

    Parameters
    ----------
    tile : ``Tile``
    kwargs : keyword arguments
        driver specific parameters
    """

    def __init__(self, tile, **kwargs):
        """Initialize."""
        self.tile = tile

    def read(self, **kwargs):
        """
        Read reprojected & resampled input data.

        Returns
        -------
        data : array or list
            NumPy array for raster data or feature list for vector data
        """
        raise NotImplementedError()

    def is_empty(self):
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        raise NotImplementedError()
        # return not self.tile.bbox.intersects(self.process.config.area_at_zoom())
