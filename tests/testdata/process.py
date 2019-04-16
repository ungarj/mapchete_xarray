from dateutil import parser
import numpy as np
import xarray as xr


def execute(mp, stack_height=10):
    # create 4D arrays with current tile shape and dtype
    arrs = [
        np.ones(
            (mp.config.output.output_params["bands"], ) + mp.tile.shape,
            dtype=mp.config.output.output_params["dtype"]
        )
        for _ in range(1, stack_height)
    ]

    # create timestamps for each array
    timestamps = [parser.parse("2018-04-0%s" % i) for i in range(1, stack_height)]

    # build xarray with time axis
    timeseries = xr.DataArray(
        np.stack(arrs), coords={'time': timestamps}, dims=('time', 'bands', 'x', 'y')
    )

    # return to write
    return timeseries
