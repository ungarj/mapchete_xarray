import json
import os

import dateutil
import zarr
from mapchete.io import fs_from_path
from rasterio.crs import CRS
from tilematrix import TilePyramid

from mapchete_xarray._output import initialize_zarr


def test_initialize_zarr(tmpdir):
    out_path = os.path.join(tmpdir, "test.zarr")
    tp = TilePyramid("geodetic")
    initialize_zarr(
        path=out_path,
        bounds=(0, 10, 10, 20),
        shape=(1024, 1024),
        crs=tp.crs,
        fill_value=0,
        chunksize=256,
        count=3,
        dtype="uint8",
    )
    fs = fs_from_path(out_path)
    bands = ["Band1", "Band2", "Band3"]
    axes = ["X", "Y"]
    required_files = [".zgroup", ".zmetadata"] + bands + axes
    ls = fs.ls(out_path)
    for required_file in required_files:
        for file in ls:
            if file.endswith(required_file):
                break
        else:
            raise ValueError(f"required file {required_file} does not exist in {ls}")
    with zarr.open(out_path) as src:
        assert set(src.array_keys()) == set(bands + axes)
        for b in bands:
            attrs = json.loads(src.store[b + "/.zattrs"])
            for field in ["AREA_OR_POINT", "_ARRAY_DIMENSIONS", "_CRS"]:
                assert field in attrs
            crs = attrs["_CRS"]
            assert "wkt" in crs
            assert CRS.from_wkt(crs["wkt"])

        for array_name, array in src.arrays():
            if array_name in bands:
                assert array.shape == (1024, 1024)
                assert array.chunks == (256, 256)
            elif array_name in axes:
                assert array.shape == (1024,)
                assert array.chunks == (1024,)
                if array_name == "X":
                    for coord in array[:]:
                        assert 0 < coord < 10
                else:
                    for coord in array[:]:
                        assert 10 < coord < 20


def test_initialize_zarr_time(tmpdir):
    out_path = os.path.join(tmpdir, "test.zarr")
    tp = TilePyramid("geodetic")
    initialize_zarr(
        path=out_path,
        bounds=(0, 10, 10, 20),
        shape=(1024, 1024),
        crs=tp.crs,
        time=dict(
            start=dateutil.parser.parse("2022-03-01"),
            end=dateutil.parser.parse("2022-03-31"),
            pattern="0 0 * * *",
            chunksize=10,
        ),
        fill_value=0,
        chunksize=256,
        count=3,
        dtype="uint8",
    )
    fs = fs_from_path(out_path)
    bands = ["Band1", "Band2", "Band3"]
    axes = ["time", "X", "Y"]
    required_files = [".zgroup", ".zmetadata"] + bands + axes
    ls = fs.ls(out_path)
    for required_file in required_files:
        for file in ls:
            if file.endswith(required_file):
                break
        else:
            raise ValueError(f"required file {required_file} does not exist in {ls}")
    with zarr.open(out_path) as src:
        assert set(src.array_keys()) == set(bands + axes)
        for b in bands:
            attrs = json.loads(src.store[b + "/.zattrs"])
            for field in ["AREA_OR_POINT", "_ARRAY_DIMENSIONS", "_CRS"]:
                assert field in attrs
            crs = attrs["_CRS"]
            assert "wkt" in crs
            assert CRS.from_wkt(crs["wkt"])

        for array_name, array in src.arrays():
            if array_name in bands:
                assert array.shape == (31, 1024, 1024)
                assert array.chunks == (10, 256, 256)
            elif array_name in axes:
                if array_name == "X":
                    assert array.shape == (1024,)
                    assert array.chunks == (1024,)
                    for coord in array[:]:
                        assert 0 < coord < 10
                elif array_name == "Y":
                    assert array.shape == (1024,)
                    assert array.chunks == (1024,)
                    for coord in array[:]:
                        assert 10 < coord < 20
                elif array_name == "time":
                    assert array.shape == (31,)
                    assert array.chunks == (31,)
                    assert array.attrs["units"] == "days since 2022-03-01 00:00:00"
                    for coord in array[:]:
                        assert 0 <= coord <= 30


def test_write_mapchete_output_metadata(zarr_single_time_mapchete):
    # process zarr_single_time_mapchete
    mp = zarr_single_time_mapchete.mp()
    list(mp.compute(tile=(5, 0, 0)))

    # verify output data is somewhere in zarr metadata
    # --> . zarr.open("filtered.zarr") as a:
    # ...     a.attrs["mapchete"] = dict with process output metadata
    z = zarr.open(mp.config.output_reader.path)
    assert isinstance(z.attrs["mapchete"], dict)
    mp_metadata = z.attrs["mapchete"]
    assert "pyramid" in mp_metadata
    assert "driver" in mp_metadata


def test_single_zarr_as_input(single_zarr_input_mapchete):
    # use single zarr path as process input
    # TODO create process fixture
    list(single_zarr_input_mapchete.mp().compute(concurrency=None))


def test_single_zarr_mapchete_as_input(single_zarr_process_mapchete):
    # use mapchete process with single zarr output as process input
    # TODO create process fixture
    list(single_zarr_process_mapchete.mp().compute(concurrency=None))


def test_read_as_dataarray():
    # TODO: input.read() returns xarray.DataArray
    raise NotImplementedError()


def test_read_as_dataset():
    # TODO: input.read(as_dataset=True) returns xarray.DataSet
    # TODO: input.read_dataset() returns xarray.DataSet
    raise NotImplementedError()
