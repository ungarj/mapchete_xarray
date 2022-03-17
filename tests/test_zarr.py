import json
from mapchete.io import fs_from_path
import os
from rasterio.crs import CRS
from tilematrix import TilePyramid
import zarr

from mapchete_xarray._zarr import initialize_zarr


def test_initialize_zarr(tmpdir):
    out_path = os.path.join(tmpdir, "test.zarr")
    tp = TilePyramid("geodetic")
    initialize_zarr(
        path=out_path,
        bounds=(0, 10, 10, 20),
        shape=(1024, 1024),
        crs=tp.crs,
        fill_value=0,
        chunksize=256,
        count=3,
        dtype="uint8",
    )
    fs = fs_from_path(out_path)
    bands = ["Band1", "Band2", "Band3"]
    axes = ["X", "Y"]
    required_files = [".zgroup", ".zmetadata"] + bands + axes
    ls = fs.ls(out_path)
    for required_file in required_files:
        for file in ls:
            if file.endswith(required_file):
                break
        else:
            raise ValueError(f"required file {required_file} does not exist in {ls}")
    with zarr.open(out_path) as src:
        assert set(src.array_keys()) == set(bands + axes)
        for b in bands:
            attrs = json.loads(src.store[b + "/.zattrs"])
            for field in ["AREA_OR_POINT", "_ARRAY_DIMENSIONS", "_CRS"]:
                assert field in attrs
            crs = attrs["_CRS"]
            assert "wkt" in crs
            assert CRS.from_wkt(crs["wkt"])

        for array_name, array in src.arrays():
            if array_name in bands:
                assert array.shape == (1024, 1024)
                assert array.chunks == (256, 256)
            elif array_name in axes:
                assert array.shape == (1024,)
                assert array.chunks == (1024,)
                if array_name == "X":
                    for coord in array[:]:
                        assert 0 < coord < 10
                else:
                    for coord in array[:]:
                        assert 10 < coord < 20
