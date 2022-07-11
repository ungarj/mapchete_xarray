import numpy as np
import xarray as xr
from mapchete.formats.tools import available_output_formats, driver_from_extension


def test_format_available():
    assert "xarray" in available_output_formats()


def test_format_extension_findable():
    assert driver_from_extension("zarr") == "xarray"


def test_empty_tile(output_3d_mapchete):
    mp = output_3d_mapchete.mp()
    tile = output_3d_mapchete.first_process_tile()
    empty_output = mp.config.output.empty(tile)
    assert isinstance(empty_output, xr.Dataset)


def test_3d_processed_tile_exists(output_3d_mapchete):
    mp = output_3d_mapchete.mp()
    tile = output_3d_mapchete.first_process_tile()
    list(mp.compute(tile=tile, concurrency=None))
    assert mp.config.output.tiles_exist(tile)
    assert mp.config.output.tiles_exist(output_tile=tile)


def test_3d_tile_not_exists(output_3d_mapchete):
    mp = output_3d_mapchete.mp()
    tile = output_3d_mapchete.first_process_tile()
    assert not mp.config.output.tiles_exist(tile)
    assert not mp.config.output.tiles_exist(output_tile=tile)


def test_3d_write_empty_data(output_3d_mapchete):
    mp = output_3d_mapchete.mp()
    tile = output_3d_mapchete.first_process_tile()
    mp.config.output.write(tile, mp.config.output.empty(tile))
    assert not mp.config.output.tiles_exist(tile)
    output = mp.config.output.read(tile)
    for var in output.values():
        assert np.all(var == mp.config.output.nodata)


def test_4d_processed_tile_exists(output_4d_mapchete):
    mp = output_4d_mapchete.mp()
    tile = output_4d_mapchete.first_process_tile()
    list(mp.compute(tile=tile))
    assert mp.config.output.tiles_exist(tile)
    assert mp.config.output.tiles_exist(output_tile=tile)


def test_4d_tile_not_exists(output_4d_mapchete):
    mp = output_4d_mapchete.mp()
    tile = output_4d_mapchete.first_process_tile()
    assert not mp.config.output.tiles_exist(tile)
    assert not mp.config.output.tiles_exist(output_tile=tile)


def test_4d_write_empty_data(output_4d_mapchete):
    mp = output_4d_mapchete.mp()
    tile = output_4d_mapchete.first_process_tile()
    mp.config.output.write(tile, mp.config.output.empty(tile))
    assert not mp.config.output.tiles_exist(tile)
    output = mp.config.output.read(tile)
    for var in output.values():
        assert np.all(var == mp.config.output.nodata)


def test_s3_processed_tile_exists(output_4d_s3_mapchete):
    mp = output_4d_s3_mapchete.mp()
    tile = output_4d_s3_mapchete.first_process_tile()
    list(mp.compute(tile=tile))
    assert mp.config.output.tiles_exist(tile)
    assert mp.config.output.tiles_exist(output_tile=tile)


def test_s3_tile_not_exists(output_4d_s3_mapchete):
    mp = output_4d_s3_mapchete.mp()
    tile = output_4d_s3_mapchete.first_process_tile()
    assert not mp.config.output.tiles_exist(tile)
    assert not mp.config.output.tiles_exist(output_tile=tile)


def test_s3_write_empty_data(output_4d_s3_mapchete):
    mp = output_4d_s3_mapchete.mp()
    tile = output_4d_s3_mapchete.first_process_tile()
    mp.config.output.write(tile, mp.config.output.empty(tile))
    assert not mp.config.output.tiles_exist(tile)
    output = mp.config.output.read(tile)
    for var in output.values():
        assert np.all(var == mp.config.output.nodata)


def test_zarr_as_input(zarr_as_input_mapchete):
    list(zarr_as_input_mapchete.mp().compute(concurrency=None))


def test_zarr_process_output_as_input(zarr_process_output_as_input_mapchete):
    # NOTE: this only reads an empty output, thus maybe not testing reading real data
    list(zarr_process_output_as_input_mapchete.mp().compute(concurrency=None))


# TODO: test if global grid creation on high zoom level adds performance issues
