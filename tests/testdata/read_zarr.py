def execute(mp):
    with mp.open("zarr") as zarr:
        return zarr.read_dataarray()
