import json
import xarray as xr


def execute(
    mp,
    band_names=None,
    index_band=None,
    index_field="index",
    slice_id_field="slice_id",
):
    """
    Convert raster input array to xarray with optionally named axes.

    """
    coords = {}
    attrs = {}
    with mp.open("raster") as raster:
        if raster.is_empty():
            return "empty"
        data = raster.read()

    if "indexes" in mp.input:  # pragma: no cover
        if index_band is None:
            raise ValueError("index_band has to be specified if indexes are provided")
        s2_indexes = {
            i["properties"][slice_id_field]: i["properties"][index_field]
            for i in mp.open("indexes").read()
        }
        attrs.update(slice_ids=s2_indexes)

    if band_names:
        if len(band_names) != data.shape[0]:  # pragma: no cover
            raise ValueError("band_names has to be the same length than input array")
        coords.update(bands=band_names)

    return xr.DataArray(
        # nd array
        data,
        # named dimension indexes
        coords=coords,
        # named dimensions
        dims=("bands", "x", "y"),
        # additional attributes
        attrs=dict(json=json.dumps(attrs)),
    )
