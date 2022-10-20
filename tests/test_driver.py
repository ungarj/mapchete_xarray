import mapchete
import numpy as np
import pytest
import xarray as xr
from mapchete.errors import MapcheteConfigError
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
    list(zarr_process_output_as_input_mapchete.mp().compute(concurrency=None))


def test_zarr_process_output_as_input_tile_exists(
    zarr_process_output_as_input_mapchete,
):
    first_run = list(
        zarr_process_output_as_input_mapchete.mp().compute(concurrency=None)
    )
    assert first_run[0]._result.written is True

    second_run = list(
        zarr_process_output_as_input_mapchete.mp().compute(concurrency=None)
    )
    assert second_run[0]._result.written is False


def test_custom_band_names_read_kwargs_no_indexes(output_3d_custom_band_names_mapchete):
    mp = output_3d_custom_band_names_mapchete.mp()
    tile = output_3d_custom_band_names_mapchete.first_process_tile()
    list(mp.compute(tile=tile))
    assert mp.config.output.tiles_exist(tile)
    data_vars = [v for v in mp.config.output.read(tile).data_vars]
    assert data_vars == ["red", "green", "blue"]


def test_zarr_as_input_read_kwargs_no_indexes(zarr_as_input_mapchete):
    mp = zarr_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        data_vars = [v for v in src.read().data_vars]
        assert data_vars == ["Band1", "Band2", "Band3"]


def test_zarr_as_input_read_kwargs_indexes_by_index(zarr_as_input_mapchete):
    mp = zarr_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:

        data_vars = [v for v in src.read(indexes=[0, 2]).data_vars]
        assert data_vars == ["Band1", "Band3"]


def test_zarr_as_input_read_kwargs_indexes_by_name(zarr_as_input_mapchete):
    mp = zarr_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:

        data_vars = [v for v in src.read(indexes=["Band1", "Band3"]).data_vars]
        assert data_vars == ["Band1", "Band3"]


def test_zarr_as_input_read_kwargs_time_range(zarr_as_input_mapchete):
    mp = zarr_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        assert len(src.read(start_time="2022-06-05").time.values) == 3
        assert (
            len(src.read(start_time="2022-06-05", end_time="2022-06-09").time.values)
            == 2
        )
        assert len(src.read(end_time="2022-06-09").time.values) == 4


def test_zarr_as_input_read_kwargs_timestamps(zarr_as_input_mapchete):
    mp = zarr_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        assert len(src.read(timestamps=["2022-06-04", "2022-06-09"]).time.values) == 2


def test_zarr_process_output_as_input_read_kwargs_no_indexes(
    zarr_process_output_as_input_mapchete,
):
    mp = zarr_process_output_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        data_vars = [v for v in src.read().data_vars]
        assert data_vars == ["Band1", "Band2", "Band3"]


def test_zarr_process_output_as_input_read_kwargs_indexes_by_index(
    zarr_process_output_as_input_mapchete,
):
    mp = zarr_process_output_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:

        data_vars = [v for v in src.read(indexes=[0, 2]).data_vars]
        assert data_vars == ["Band1", "Band3"]


def test_zarr_process_output_as_input_read_kwargs_indexes_by_name(
    zarr_process_output_as_input_mapchete,
):
    mp = zarr_process_output_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:

        data_vars = [v for v in src.read(indexes=["Band1", "Band3"]).data_vars]
        assert data_vars == ["Band1", "Band3"]


def test_zarr_process_output_as_input_read_kwargs_time_range(
    zarr_process_output_as_input_mapchete,
):
    mp = zarr_process_output_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        assert len(src.read(start_time="2022-06-05").time.values) == 3
        assert (
            len(src.read(start_time="2022-06-05", end_time="2022-06-09").time.values)
            == 2
        )
        assert len(src.read(end_time="2022-06-09").time.values) == 4


def test_zarr_process_output_as_input_read_kwargs_timestamps(
    zarr_process_output_as_input_mapchete,
):
    mp = zarr_process_output_as_input_mapchete.process_mp()
    with mp.open("zarr") as src:
        assert len(src.read(timestamps=["2022-06-04", "2022-06-09"]).time.values) == 2


def test_output_pixelbuffer_error(output_3d_mapchete):
    config = output_3d_mapchete.dict
    config["output"].update(pixelbuffer=5)
    with pytest.raises(MapcheteConfigError):
        mapchete.open(config)


def test_output_file_extension_error(output_3d_mapchete):
    config = output_3d_mapchete.dict
    config["output"].update(path="foo")
    with pytest.raises(MapcheteConfigError):
        mapchete.open(config)


def test_zoom_levels_error(output_3d_mapchete):
    config = output_3d_mapchete.dict
    config.update(zoom_levels=dict(min=0, max=5))
    with pytest.raises(MapcheteConfigError):
        mapchete.open(config)


def test_timestamps_error(output_4d_mapchete):
    config = output_4d_mapchete.dict
    config["output"]["time"].pop("pattern")
    with pytest.raises(MapcheteConfigError):
        mapchete.open(config)


def test_3d_numpy_processed_tile_exists(output_3d_numpy_mapchete):
    mp = output_3d_numpy_mapchete.mp()
    tile = output_3d_numpy_mapchete.first_process_tile()
    list(mp.compute(tile=tile, concurrency=None))
    assert mp.config.output.tiles_exist(tile)
    assert mp.config.output.tiles_exist(output_tile=tile)


def test_4d_numpy_processed_tile_exists(output_4d_numpy_mapchete):
    mp = output_4d_numpy_mapchete.mp()
    tile = output_4d_numpy_mapchete.first_process_tile()
    list(mp.compute(tile=tile, concurrency=None))
    assert mp.config.output.tiles_exist(tile)
    assert mp.config.output.tiles_exist(output_tile=tile)


# TODO: test if global grid creation on high zoom level adds performance issues
