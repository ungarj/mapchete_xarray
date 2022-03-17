import numpy as np


def execute(mp, bands=3, dtype="uint16", x_axis_name="X", y_axis_name="Y"):
    return np.ones(shape=(bands, *mp.tile.shape), dtype=dtype)
