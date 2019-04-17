def execute(mp):
    with mp.open("xarray_output") as xarr_output:
        return xarr_output.read().data[0]
