def execute(mp):
    with mp.open("zarr") as zarr:
        zarr.read()
        # TODO: define which class should be returned
