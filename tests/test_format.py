import mapchete
from mapchete.formats import available_output_formats
import numpy as np
import xarray as xr


def test_format_available():
    assert "xarray" in available_output_formats()


def test_write_read_output(example_config):
    with mapchete.open(example_config) as mp:
        process_tile = next(mp.get_process_tiles(5))

        # basic functions
        empty_xarr = mp.config.output.empty(process_tile)
        assert isinstance(empty_xarr, xr.DataArray)
        assert mp.config.output.get_path(process_tile)

        # check if tile exists
        assert not mp.config.output.tiles_exist(process_tile)

        # write
        mp.batch_process(tile=process_tile.id)

        # check if tile exists
        assert mp.config.output.tiles_exist(process_tile)

        # read again, this time with data
        xarr = mp.config.output.read(process_tile)
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
