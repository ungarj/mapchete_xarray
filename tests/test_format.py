import mapchete
from mapchete.formats import available_output_formats
import numpy as np
import xarray as xr


def test_format_available():
    assert "xarray" in available_output_formats()


def test_write_read_output(
    example_config, xarray_tiledir_input_mapchete
):
    with mapchete.open(example_config) as mp:
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

        # read from xarray tile directory output
        with mapchete.open(
            dict(
                xarray_tiledir_input_mapchete,
                input=dict(xarray_output=mp.config.output.path)
            )
        ) as mp_tiledir_input:
            mp_tiledir_input.batch_process(tile=data_tile.id)
            assert mp_tiledir_input.config.output.tiles_exist(data_tile)
            # TODO: use mapchete process read function
            arr = mp_tiledir_input.config.output.read(data_tile)
            assert isinstance(arr, np.ndarray)
            assert arr.data.all()

        # TODO # read from xarray mapchete output
        # with mapchete.open(
        #     dict(
        #         xarray_mapchete_input_mapchete,
        #         input=dict(xarray_output=mp.config.output.path)
        #     )
        # ) as mp_mapchete_input:
        #     mp_mapchete_input.batch_process(tile=data_tile.id)
        #     assert mp_mapchete_input.config.output.tiles_exist(data_tile)
        #     xarr = mp_mapchete_input.config.output.read(data_tile)
        #     assert isinstance(xarr, xr.DataArray)
        #     assert not xarr.data.any()
