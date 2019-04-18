from collections import namedtuple
import mapchete
import os
import pytest
from tempfile import TemporaryDirectory
import yaml


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")

ExampleConfig = namedtuple("ExampleConfig", ("path", "dict"))


# helper functions
def _tempdir_mapchete(path, update={}):
    with TemporaryDirectory() as temp_dir:
        abs_dir, filename = os.path.split(path)

        # load config to dictionary
        config = yaml.load(open(path).read())

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


@pytest.fixture(scope="session")
def written_output():
    temp_fixture = _tempdir_mapchete(os.path.join(TESTDATA_DIR, "example.mapchete"))
    temp_process = next(temp_fixture)
    with mapchete.open(temp_process.path) as mp:
        data_tile = next(mp.get_process_tiles(5))
        mp.batch_process(tile=data_tile.id)
        yield temp_process
    # triggers deletion of temporary directory
    next(temp_fixture)


@pytest.fixture
def example_config():
    """Fixture for example.mapchete."""
    yield from _tempdir_mapchete(os.path.join(TESTDATA_DIR, "example.mapchete"))


@pytest.fixture
def xarray_tiledir_input_mapchete():
    """Fixture for example.mapchete."""
    yield from _tempdir_mapchete(
        os.path.join(TESTDATA_DIR, "xarray_tiledir_input.mapchete")
    )
