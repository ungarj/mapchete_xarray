import mapchete
from mapchete.formats import available_output_formats
import numpy as np
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
        mp.batch_process(tile=data_tile.id)
        assert mp.config.output.tiles_exist(data_tile)
        # TODO: use mapchete process read function
        mp_tile = mapchete.MapcheteProcess(
            mp.config.process_pyramid.tile(*data_tile.id),
            config=mp.config,
            params=mp.config.params_at_zoom(5)
        )
        xarr = mp_tile.open("xarray_output").read()
        assert isinstance(xarr, xr.DataArray)
        assert xarr.data.all()
        assert not set(('time', 'bands', 'x', 'y')).difference(set(xarr.dims))


def test_read_from_mapchete_output(xarray_tiledir_input_mapchete, written_output):
    # read from xarray tile directory output
    with mapchete.open(
        dict(
            xarray_tiledir_input_mapchete.dict,
            input=dict(xarray_output=written_output.path)
        )
    ) as mp:
        data_tile = mp.config.process_pyramid.tile(5, 0, 0)
        mp.batch_process(tile=data_tile.id)
        assert mp.config.output.tiles_exist(data_tile)
        # TODO: use mapchete process read function
        mp_tile = mapchete.MapcheteProcess(
            mp.config.process_pyramid.tile(*data_tile.id),
            config=mp.config,
            params=mp.config.params_at_zoom(5)
        )
        xarr = mp_tile.open("xarray_output").read()
        assert isinstance(xarr, xr.DataArray)
        assert xarr.data.all()
        assert not set(('time', 'bands', 'x', 'y')).difference(set(xarr.dims))
