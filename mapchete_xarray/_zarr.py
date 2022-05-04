from tilematrix._funcs import Bounds
import xarray as xr
from zarr.storage import FSStore
import croniter
import zarr
import dateutil


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

    if time:
        start_time = (
            dateutil.parser.parse(time["start"])
            if isinstance(time["start"], str)
            else time["start"]
        )

        end_time = (
            dateutil.parser.parse(time["end"])
            if isinstance(time["end"], str)
            else time["end"]
        )

        coord_time = [
            t
            for t in croniter.croniter_range(
                start_time,
                end_time,
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

    coords = {
        x_axis_name: ([x_axis_name], coord_x),
        y_axis_name: ([y_axis_name], coord_y),
    }

    axis_names = (
        ["time", y_axis_name, x_axis_name] if time else [y_axis_name, x_axis_name]
    )

    if time:
        coords["time"] = coord_time

    ds = xr.Dataset(coords=coords)

    ds.to_zarr(
        FSStore(path),
        compute=False,
        encoding={var: {"_FillValue": fill_value} for var in ds.data_vars},
        safe_chunks=True,
    )

    for i in range(1, count + 1):
        store = FSStore(f"{path}/Band{i}")
        zarr.creation.create(
            shape=output_shape,
            chunks=output_chunks,
            dtype=dtype,
            store=store,
        )

        attrs = zarr.open(store).attrs
        attrs["_ARRAY_DIMENSIONS"] = axis_names
        attrs["_CRS"] = {"wkt": crs.wkt}
        attrs["AREA_OR_POINT"] = area_or_point

    zarr.consolidate_metadata(path)
