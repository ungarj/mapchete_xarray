import dateutil
import numpy as np
import xarray as xr


def execute(
    mp,
    bands=3,
    dtype="uint16",
    x_axis_name="X",
    y_axis_name="Y",
    add_timestamps=True,
    timestamps=None,
    return_dataarray=True,
):
    timestamps = timestamps or [
        "2022-03-01",
        "2022-03-02",
        "2022-03-04",
        "2022-03-07",
        "2022-03-09",
    ]
    if add_timestamps:
        shape = (len(timestamps), bands, *mp.tile.shape)
        dims = ["time", "band", y_axis_name, x_axis_name]
        coords = {"time": [dateutil.parser.parse(t) for t in timestamps]}
    else:
        shape = (bands, *mp.tile.shape)
        dims = ["bands", y_axis_name, x_axis_name]
        coords = {}

    arr = np.full(shape=shape, fill_value=500, dtype=dtype)

    if return_dataarray:
        return xr.DataArray(
            data=arr,
            dims=dims,
            coords=coords,
        )
    else:
        return arr
