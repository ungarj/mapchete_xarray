import os
import uuid
from tempfile import TemporaryDirectory

import pytest
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
            os.path.join(TESTDATA_DIR, "output_4d.mapchete"), output_tempdir=tempdir
        ) as example:
            data_tile = next(example.mp().get_process_tiles(5))
            example.mp().batch_process(tile=data_tile.id)
            yield example


@pytest.fixture
def convert_to_zarr_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "convert_to_zarr.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def output_3d_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "output_3d.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def output_3d_numpy_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "output_3d_numpy.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def output_4d_s3_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "output_4d.mapchete"),
        output_tempdir=os.path.join(S3_TEMP_DIR),
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def output_4d_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "output_4d.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def output_4d_numpy_mapchete():
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "output_4d_numpy.mapchete"),
    ) as example:
        yield example


@pytest.fixture
def zarr_as_input_mapchete(tmp_path):
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr_as_input.mapchete"),
        output_tempdir=tmp_path,
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def zarr_process_output_as_input_mapchete(tmp_path):
    with ProcessFixture(
        os.path.join(TESTDATA_DIR, "zarr_process_output_as_input.mapchete"),
        output_tempdir=tmp_path,
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def example_zarr():
    return os.path.join(TESTDATA_DIR, "example.zarr")
