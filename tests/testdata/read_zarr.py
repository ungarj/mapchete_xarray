def execute(mp, indexes=None):
    with mp.open("zarr") as zarr:
        return zarr.read(indexes=indexes)
