import os
import uuid
from tempfile import TemporaryDirectory

import pytest
from mapchete.path import MPath
from mapchete.testing import ProcessFixture

SCRIPT_DIR = MPath(os.path.dirname(os.path.realpath(__file__)))
TESTDATA_DIR = SCRIPT_DIR / "testdata"
S3_TEMP_DIR = MPath("s3://mapchete-test/tmp/" + uuid.uuid4().hex)


@pytest.fixture
def mp_s3_tmpdir():
    """Setup and teardown temporary directory."""
    S3_TEMP_DIR.rm(recursive=True, ignore_errors=True)
    S3_TEMP_DIR.makedirs()
    yield S3_TEMP_DIR
    S3_TEMP_DIR.rm(recursive=True, ignore_errors=True)


@pytest.fixture(autouse=True)
def mp_tmpdir():
    """Setup and teardown temporary directory."""
    with TemporaryDirectory() as tempdir_path:
        tempdir = MPath(tempdir_path)
        tempdir.makedirs()
        yield tempdir


@pytest.fixture(scope="session")
def written_output():
    with TemporaryDirectory() as tempdir:
        with ProcessFixture(
            TESTDATA_DIR / "output_4d.mapchete", output_tempdir=tempdir
        ) as example:
            data_tile = next(example.mp().get_process_tiles(5))
            example.mp().batch_process(tile=data_tile.id)
            yield example


@pytest.fixture
def convert_to_zarr_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "convert_to_zarr.mapchete",
    ) as example:
        yield example


@pytest.fixture
def output_3d_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "output_3d.mapchete",
    ) as example:
        yield example


@pytest.fixture
def output_3d_custom_band_names_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "output_3d_custom_band_names.mapchete",
    ) as example:
        yield example


@pytest.fixture
def output_3d_numpy_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "output_3d_numpy.mapchete",
    ) as example:
        yield example


@pytest.fixture
def output_4d_s3_mapchete(mp_s3_tmpdir):
    with ProcessFixture(
        TESTDATA_DIR / "output_4d.mapchete",
        output_tempdir=mp_s3_tmpdir,
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def output_4d_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "output_4d.mapchete",
    ) as example:
        yield example


@pytest.fixture
def output_4d_numpy_mapchete():
    with ProcessFixture(
        TESTDATA_DIR / "output_4d_numpy.mapchete",
    ) as example:
        yield example


@pytest.fixture
def zarr_as_input_mapchete(mp_tmpdir):
    with ProcessFixture(
        TESTDATA_DIR / "zarr_as_input.mapchete",
        output_tempdir=mp_tmpdir,
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def zarr_process_output_as_input_mapchete(mp_tmpdir):
    with ProcessFixture(
        TESTDATA_DIR / "zarr_process_output_as_input.mapchete",
        output_tempdir=mp_tmpdir,
        output_suffix=".zarr",
    ) as example:
        yield example


@pytest.fixture
def example_zarr():
    return TESTDATA_DIR / "example.zarr"
