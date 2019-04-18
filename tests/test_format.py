import mapchete
from mapchete.errors import MapcheteConfigError
from mapchete.formats import available_output_formats
import numpy as np
import pytest
import xarray as xr


def test_format_available():
    assert "xarray" in available_output_formats()


def test_write_read_output(example_config):
    with mapchete.open(example_config.path) as mp:
        data_tile = next(mp.get_process_tiles(5))

        # basic functions
        empty_xarr = mp.config.output.empty(data_tile)
        assert isinstance(empty_xarr, xr.DataArray)
        assert mp.config.output.get_path(data_tile)

        # check if tile exists
        assert not mp.config.output.tiles_exist(data_tile)

        # write
        mp.batch_process(tile=data_tile.id)

        # check if tile exists
        assert mp.config.output.tiles_exist(data_tile)

        # read again, this time with data
        xarr = mp.config.output.read(data_tile)
        assert isinstance(xarr, xr.DataArray)
        assert xarr.data.all()
        assert not set(('time', 'bands', 'x', 'y')).difference(set(xarr.dims))

        # handle empty data
        process_tile = next(mp.get_process_tiles(6))
        mp.config.output.write(process_tile, mp.config.output.empty(process_tile))
        # check if tile exists
        assert not mp.config.output.tiles_exist(process_tile)
        xarr = mp.config.output.read(process_tile)
        assert isinstance(xarr, xr.DataArray)
        assert not xarr.data.any()

        # write nodata array
        process_tile = next(mp.get_process_tiles(7))
        mp.config.output.write(process_tile, xr.DataArray(np.zeros(process_tile.shape)))
        # check if tile exists
        assert not mp.config.output.tiles_exist(process_tile)
        xarr = mp.config.output.read(process_tile)
        assert isinstance(xarr, xr.DataArray)
        assert not xarr.data.any()


def test_read_from_tile_directory(xarray_tiledir_input_mapchete, written_output):
    # read from xarray tile directory output
    with mapchete.open(
        dict(
            xarray_tiledir_input_mapchete.dict,
            input=dict(xarray_output=written_output.dict["output"]["path"])
        )
    ) as mp:
        data_tile = mp.config.process_pyramid.tile(5, 0, 0)
        mp_tile = mapchete.MapcheteProcess(
            mp.config.process_pyramid.tile(*data_tile.id),
            config=mp.config,
            params=mp.config.params_at_zoom(5)
        )
        xarr_tile = mp_tile.open("xarray_output")
        assert not xarr_tile.is_empty()
        xarr = xarr_tile.read()
        assert isinstance(xarr, xr.DataArray)
        assert xarr.data.all()
        assert ('time', 'bands', 'x', 'y') == xarr.dims
        assert xarr.data.shape[-2:] == data_tile.shape

    # raise error if process metatiling is bigger than output metatiling
    with mapchete.open(
        dict(
            xarray_tiledir_input_mapchete.dict,
            input=dict(xarray_output=written_output.dict["output"]["path"]),
            pyramid=dict(xarray_tiledir_input_mapchete.dict["pyramid"], metatiling=4)
        )
    ) as mp:
        with pytest.raises(MapcheteConfigError):
            mapchete.MapcheteProcess(
                mp.config.process_pyramid.tile(5, 0, 0),
                config=mp.config,
                params=mp.config.params_at_zoom(5)
            ).open("xarray_output").read()


def test_tile_directory_grid_error(xarray_tiledir_input_mapchete, written_output):
    # raise error if tile pyramid grid differs
    with mapchete.open(
        dict(
            xarray_tiledir_input_mapchete.dict,
            input=dict(xarray_output=written_output.dict["output"]["path"]),
            pyramid=dict(grid="mercator")
        )
    ) as mp:
        with pytest.raises(MapcheteConfigError):
            mapchete.MapcheteProcess(
                mp.config.process_pyramid.tile(5, 0, 0),
                config=mp.config,
                params=mp.config.params_at_zoom(5)
            ).open("xarray_output").read()


def test_read_from_mapchete_output(xarray_mapchete_input_mapchete, written_output):
    # read from xarray tile directory output
    with mapchete.open(
        dict(
            xarray_mapchete_input_mapchete.dict,
            input=dict(xarray_output=written_output.path)
        )
    ) as mp:
        data_tile = mp.config.process_pyramid.tile(5, 0, 0)
        mp_tile = mapchete.MapcheteProcess(
            mp.config.process_pyramid.tile(*data_tile.id),
            config=mp.config,
            params=mp.config.params_at_zoom(5)
        )
        xarr_tile = mp_tile.open("xarray_output")
        assert not xarr_tile.is_empty()
        xarr = xarr_tile.read()
        assert isinstance(xarr, xr.DataArray)
        assert xarr.data.all()
        assert ('time', 'bands', 'x', 'y') == xarr.dims
        assert xarr.data.shape[-2:] == data_tile.shape

    # raise error if process metatiling is bigger than output metatiling
    with mapchete.open(
        dict(
            xarray_mapchete_input_mapchete.dict,
            input=dict(xarray_output=written_output.dict["output"]["path"]),
            pyramid=dict(xarray_mapchete_input_mapchete.dict["pyramid"], metatiling=4)
        )
    ) as mp:
        with pytest.raises(MapcheteConfigError):
            mapchete.MapcheteProcess(
                mp.config.process_pyramid.tile(5, 0, 0),
                config=mp.config,
                params=mp.config.params_at_zoom(5)
            ).open("xarray_output").read()
