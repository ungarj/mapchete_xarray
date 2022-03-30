import dask.array as da
import json
from mapchete.io import fs_from_path
import os
from tilematrix._funcs import Bounds
import xarray as xr
from zarr.storage import FSStore
import croniter
import zarr


def initialize_zarr(
    path=None,
    bounds=None,
    shape=None,
    crs=None,
    time=None,
    fill_value=None,
    chunksize=256,
    count=None,
    dtype="uint8",
    x_axis_name="X",
    y_axis_name="Y",
    area_or_point="Area",
):
    fs = fs_from_path(path)

    if time:
        coord_time = [
            t
            for t in croniter.croniter_range(
                time["start"],
                time["end"],
                time["pattern"],
            )
        ]

        output_shape = (len(coord_time), *shape)
        output_chunks = (time["chunksize"], chunksize, chunksize)
    else:
        output_shape = shape
        output_chunks = (chunksize, chunksize)

    height, width = shape
    bounds = Bounds(*bounds)
    pixel_x_size = (bounds.right - bounds.left) / width
    pixel_y_size = (bounds.top - bounds.bottom) / -height

    coord_x = [bounds.left + pixel_x_size / 2 + i * pixel_x_size for i in range(width)]
    coord_y = [bounds.top + pixel_y_size / 2 + i * pixel_y_size for i in range(height)]
    array = da.full(
        shape=output_shape,
        fill_value=fill_value,
        chunks=output_chunks,
        dtype=dtype,
    )
    coords = {
        x_axis_name: ([x_axis_name], coord_x),
        y_axis_name: ([y_axis_name], coord_y),
    }

    axis_names = (
        ["time", y_axis_name, x_axis_name] if time else [y_axis_name, x_axis_name]
    )

    data_vars = {f"Band{i}": ([*axis_names], array) for i in range(1, count + 1)}

    if time:
        coords["time"] = coord_time

    ds = xr.Dataset(coords=coords, data_vars=data_vars)

    attrs = {
        "_ARRAY_DIMENSIONS": axis_names,
        # xarray cannot write attributes values as dictionaries!
        # "_CRS": {"wkt": crs.wkt},
        "AREA_OR_POINT": area_or_point,
    }
    for data_var in ds.data_vars:
        ds[data_var].attrs = attrs
    ds.to_zarr(
        FSStore(path),
        compute=False,
        encoding={var: {"_FillValue": fill_value} for var in ds.data_vars},
        safe_chunks=True,
    )

    # TODO: find a better way!
    for data_var in ds.data_vars:
        attrs_path = os.path.join(path, data_var, ".zattrs")
        with fs.open(attrs_path, mode="r") as src:
            metadata = json.loads(src.read())
        metadata["_CRS"] = {"wkt": crs.wkt}
        with fs.open(attrs_path, mode="w") as dst:
            dst.write(json.dumps(metadata))

    zarr.consolidate_metadata(path)

    return ds
