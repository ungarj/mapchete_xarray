import os
import pytest
from tempfile import TemporaryDirectory
import yaml


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")


# helper functions
def _dict_from_mapchete(path):
    config = yaml.load(open(path).read())
    config.update(config_dir=os.path.dirname(path))
    return config


@pytest.fixture
def example_config():
    """Fixture for example.mapchete."""
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(TESTDATA_DIR, "example.mapchete")
        config = _dict_from_mapchete(path)
        config["output"].update(path=temp_dir)
        yield config


@pytest.fixture
def xarray_tiledir_input_mapchete():
    """Fixture for example.mapchete."""
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(TESTDATA_DIR, "xarray_tiledir_input.mapchete")
        config = _dict_from_mapchete(path)
        config["output"].update(path=temp_dir)
        yield config
