import numpy as np
import xarray as xr

from mapchete_xarray.processes import convert_to_xarray


def test_convert_to_xarray(convert_to_xarray_mapchete):
    mp = convert_to_xarray_mapchete.process_mp(tile=(1, 0, 0))

    # default settings
    output = convert_to_xarray.execute(mp)
    assert isinstance(output, xr.DataArray)
    assert output.data.all()
    assert ("bands", "x", "y") == output.dims
    assert output.data.shape[-2:] == mp.tile.shape

    # band name
    output = convert_to_xarray.execute(mp, band_names=["elevation"])
    assert isinstance(output, xr.DataArray)
    assert output.data.all()
    assert ("bands", "x", "y") == output.dims
    assert output.data.shape[-2:] == mp.tile.shape
    assert isinstance(output.loc["elevation"], xr.DataArray)

    mp = convert_to_xarray_mapchete.process_mp(tile=(5, 0, 0))
    output = convert_to_xarray.execute(mp)
    assert output == "empty"
