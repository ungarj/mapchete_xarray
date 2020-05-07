===============
mapchete xarray
===============

This driver enables mapchete to write multidimensional arrays into a tile directory structure.

.. image:: https://badge.fury.io/py/mapchete-xarray.svg
    :target: https://badge.fury.io/py/mapchete-xarray

.. image:: https://travis-ci.org/ungarj/mapchete_xarray.svg?branch=master
    :target: https://travis-ci.org/ungarj/mapchete_xarray

.. image:: https://coveralls.io/repos/github/ungarj/mapchete_xarray/badge.svg?branch=master
    :target: https://coveralls.io/github/ungarj/mapchete_xarray?branch=master

.. image:: https://img.shields.io/pypi/pyversions/mapchete_xarray.svg
    :target: https://pypi.python.org/pypi/mapchete_xarray



-----
Usage
-----

Example ``.mapchete`` file:

.. code-block:: yaml

    process: process.py
    zoom_levels:
        min: 0
        max: 12
    input:
    output:
        format: xarray
        path: /some/output/path
        dtype: uint16
        bands: 3
        storage: zarr  # or netcdf
    pyramid:
        grid: geodetic
        metatiling: 2



Example process file:

.. code-block:: python

    from dateutil import parser
    import numpy as np
    import xarray as xr


    def execute(mp, stack_height=10):
        # create 4D arrays with current tile shape and dtype
        arrs = [
            np.ones((3, ) + mp.tile.shape, dtype="uint16")
            for _ in range(1, stack_height)
        ]

        # create timestamps for each array
        timestamps = [parser.parse("2018-04-0%s" % i) for i in range(1, stack_height)]

        # build xarray with time axis
        timeseries = xr.DataArray(
            np.stack(arrs), coords={'time': timestamps}, dims=('time', 'bands', 'x', 'y')
        )

        # return to write
        return timeseries


------------
Installation
------------

.. code-block:: shell

    # install using pip:
    pip install mapchete_xarray
    # verify driver is vailable ('xarray' should be listed as output format):
    mapchete formats


-------------------
Current Limitations
-------------------

- no reprojection allowed
- when reading from existing output, process metatiling must be smaller than xarray output metatiling


-------
License
-------

MIT License

Copyright (c) 2019-2020 `EOX IT Services`_

.. _`EOX IT Services`: https://eox.at/
