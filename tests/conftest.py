import os
import uuid
from tempfile import TemporaryDirectory

import mapchete
import pytest
import yaml
from mapchete.io import fs_from_path
from mapchete.testing import ProcessFixture

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
S3_TEMP_DIR = "s3://mapchete-test/tmp/" + uuid.uuid4().hex


@pytest.fixture
def mp_s3_tmpdir():
    """Setup and teardown temporary directory."""
    fs = fs_from_path(S3_TEMP_DIR)

    def _cleanup():
        try:
            fs.rm(S3_TEMP_DIR, recursive=True)
        except FileNotFoundError:
            pass

    _cleanup()
    yield S3_TEMP_DIR
    _cleanup()


@pytest.fixture(scope="session")
def written_output():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "example.mapchete"), output_tempdir=tempdir
        ) as example:
            data_tile = next(example.mp().get_process_tiles(5))
            example.mp().batch_process(tile=data_tile.id)
            yield example


@pytest.fixture
def example_config():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "example.mapchete"), output_tempdir=tempdir
        ) as example:
            yield example


@pytest.fixture
def zarr_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def zarr_single_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr_single.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def zarr_single_s3_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr_single.mapchete"),
        output_tempdir=os.path.join(S3_TEMP_DIR),
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def zarr_single_time_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr_single_time.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def xarray_tiledir_input_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "xarray_tiledir_input.mapchete"),
            output_tempdir=tempdir,
        ) as example:
            yield example


@pytest.fixture
def xarray_mapchete_input_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "xarray_mapchete_input.mapchete"),
            output_tempdir=tempdir,
        ) as example:
            yield example


@pytest.fixture
def convert_to_xarray_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "convert_to_xarray.mapchete"),
            output_tempdir=tempdir,
        ) as example:
            yield example


@pytest.fixture
def single_example_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "single_example.mapchete"),
            output_tempdir=tempdir,
        ) as example:
            yield example


@pytest.fixture
def single_zarr_input_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "single_zarr_input.mapchete"),
            output_tempdir=tempdir,
            output_suffix=".zarr",
        ) as example:
            yield example


@pytest.fixture
def single_zarr_process_mapchete():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            os.path.join(TESTDATA_DIR, "single_zarr_process.mapchete"),
            output_tempdir=tempdir,
            output_suffix=".zarr",
        ) as example:
            yield example


@pytest.fixture
def rgb_tif():
    return os.path.join(TESTDATA_DIR, "rgb.tif")


@pytest.fixture
def example_zarr():
    return os.path.join(TESTDATA_DIR, "example.zarr")
