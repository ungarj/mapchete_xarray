import numpy as np


def execute(mp, bands=3, dtype="uint16", x_axis_name="X", y_axis_name="Y"):
    return np.full(shape=(bands, *mp.tile.shape), fill_value=500, dtype=dtype)
