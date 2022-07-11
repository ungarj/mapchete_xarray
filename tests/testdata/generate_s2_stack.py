import dateutil
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def execute(mp, timestamps):
    products = np.stack([product.read() for _, product in mp.open("s2_products")])
    return xr.DataArray(
        data=products,
        dims=["time", "band", "Y", "X"],
        coords={"time": np.array(timestamps, dtype=np.datetime64)},
    )
