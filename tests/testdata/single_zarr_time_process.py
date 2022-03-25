import numpy as np
import xarray as xr
import dateutil


def execute(mp, bands=3, dtype="uint16", x_axis_name="X", y_axis_name="Y"):
    timestamps = [
        dateutil.parser.parse(t)
        for t in ["2022-03-01", "2022-03-04", "2022-03-07", "2022-03-09"]
    ]
    data = np.full(
        shape=(bands, len(timestamps), *mp.tile.shape), fill_value=500, dtype=dtype
    )

    return xr.DataArray(
        data=data, dims=["band", "time", "Y", "X"], coords={"time": timestamps}
    )
