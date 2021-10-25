import boto3
from collections import namedtuple
from contextlib import contextmanager
import mapchete
import os
import pytest
from tempfile import TemporaryDirectory
import uuid
import yaml


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
S3_TEMP_DIR = "s3://mapchete-test/tmp/" + uuid.uuid4().hex

ExampleConfig = namedtuple("ExampleConfig", ("path", "dict"))


# helper functions
@contextmanager
def _tempdir_mapchete(path, update={}):
    with TemporaryDirectory() as temp_dir:
        abs_dir, filename = os.path.split(path)

        # load config to dictionary
        config = yaml.safe_load(open(path).read())

        # add config directory, point output to temp_dir and make path to process
        # file absolute
        config.update(config_dir=os.path.dirname(path))
        config["output"].update(path=temp_dir)
        if config["process"].endswith(".py"):
            config["process"] = os.path.join(abs_dir, config["process"])

        # if required apply custom changes to configuration
        config.update(**update)

        # dump temporary mapchete file to temporary direcotry
        temp_mapchete_file = os.path.join(temp_dir, filename)
        with open(temp_mapchete_file, "w") as mapchete_file:
            yaml.dump(config, mapchete_file, default_flow_style=False)
        yield ExampleConfig(path=temp_mapchete_file, dict=config)


# temporary directory for I/O tests
@pytest.fixture
def mp_s3_tmpdir():
    """Setup and teardown temporary directory."""

    def _cleanup():
        for obj in boto3.resource('s3').Bucket(S3_TEMP_DIR.split("/")[2]).objects.filter(
            Prefix="/".join(S3_TEMP_DIR.split("/")[-2:])
        ):
            obj.delete()

    _cleanup()
    yield S3_TEMP_DIR
    _cleanup()


@pytest.fixture(scope="session")
def written_output():
    with _tempdir_mapchete(os.path.join(TESTDATA_DIR, "example.mapchete")) as config:
        with mapchete.open(config.path) as mp:
            data_tile = next(mp.get_process_tiles(5))
            mp.batch_process(tile=data_tile.id)
            yield config


@pytest.fixture
def example_config():
    with _tempdir_mapchete(os.path.join(TESTDATA_DIR, "example.mapchete")) as config:
        yield config


@pytest.fixture
def zarr_config():
    with _tempdir_mapchete(os.path.join(TESTDATA_DIR, "zarr_example.mapchete")) as config:
        yield config


@pytest.fixture
def xarray_tiledir_input_mapchete():
    with _tempdir_mapchete(
        os.path.join(TESTDATA_DIR, "xarray_tiledir_input.mapchete")
    ) as config:
        yield config


@pytest.fixture
def xarray_mapchete_input_mapchete():
    with _tempdir_mapchete(
        os.path.join(TESTDATA_DIR, "xarray_mapchete_input.mapchete")
    ) as config:
        yield config
