#!/usr/bin/env python3

import os
import time
from collections import namedtuple
from itertools import product
from tempfile import TemporaryDirectory

import xarray as xr
import zarr

example_stack = xr.open_dataset("testdata/example.nc")["data"]

CompressorArgs = namedtuple("CompressorArgs", ["cname", "clevel", "shuffle"])


def _folder_size(path="."):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += _folder_size(entry.path)
    return total


blosc_compressors = [
    CompressorArgs(cname=cname, clevel=clevel, shuffle=shuffle)
    for cname, clevel, shuffle in product(
        # ["zstd", "snappy"],
        # range(1, 5),
        # range(-1, 3)
        ["zstd", "blosclz", "lz4", "lz4hc", "zlib", "snappy"],
        range(1, 10),
        range(-1, 3),
    )
]

results = dict()
for args in blosc_compressors:
    with TemporaryDirectory() as tempdir:
        start = time.time()
        example_stack.to_dataset(name="data").to_zarr(
            tempdir,
            encoding={
                "data": {
                    "compressor": zarr.Blosc(
                        cname=args.cname, clevel=args.clevel, shuffle=args.shuffle
                    )
                }
            },
        )
        elapsed = time.time() - start
        size = _folder_size(tempdir)
        print("size: {}, time: {}, args: {}".format(size, round(elapsed, 3), args))
        results[args] = {"size": size, "time": elapsed}


def _sort_by(results, key=None):
    return sorted(results.items(), key=lambda x: x[1][key])


def _add_ranks(results):
    # add size rank
    for rank, i in enumerate(_sort_by(results, key="size")):
        results[i[0]].update(size_rank=rank + 1)
    # add speed rank
    for rank, i in enumerate(_sort_by(results, key="time")):
        results[i[0]].update(time_rank=rank + 1)
    # add combined rank
    for v in results.values():
        v.update(combined_rank=v["size_rank"] + v["time_rank"])
    return results


for k, v in _add_ranks(results).items():
    print("{}: {}".format(k, v))

print("top 10 by combined ranks")
for i in _sort_by(results, key="combined_rank")[:10]:
    print(i)
