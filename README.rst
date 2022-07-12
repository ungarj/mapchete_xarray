===============
mapchete xarray
===============

This driver enables mapchete to write multidimensional arrays into a `Zarr`_ archive.

.. image:: https://badge.fury.io/py/mapchete-xarray.svg
    :target: https://badge.fury.io/py/mapchete-xarray

.. image:: https://github.com/ungarj/mapchete_xarray/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ungarj/mapchete_xarray/actions

.. image:: https://coveralls.io/repos/github/ungarj/mapchete_xarray/badge.svg?branch=master
    :target: https://coveralls.io/github/ungarj/mapchete_xarray?branch=master

.. image:: https://img.shields.io/pypi/pyversions/mapchete_xarray.svg
    :target: https://pypi.python.org/pypi/mapchete_xarray


This driver creates a Zarr according to the structure used by the `GDAL Zarr driver`_. Bands are stored in separate 2D arrays (y, x). If a time dimension is configured, the time axis is applied to the individual bands (time, y, x).

When using a time axis, please note that it has to be initialized with the full time range, i.e. it is not possible to extend the time axis after it was initialized.

If you plan extending your Zarr over multiple process runs you can achieve this by always specifying the full time range in the output configuration and then simply write a couple of slices per process run. Please note that for every process run after initialization you will have to use the `--overwrite` flag, otherwise the process tile will be skipped.

-----
Usage
-----

A process file can either return a `xarray.Dataset`, `xarray.DataArray` or a `numpy.ndarray` object. Please make sure though that when using a time axis, the timestamps of the slices have to be provided. In case of returning a `numpy.ndarray` this information is not available so this array has to match exactly to the output Zarr shape on the time and bands axes.

3D output array
---------------

For a simple 3D (bands, x, y) output:

.. code-block:: yaml

    # example.mapchete
    process: process.py
    zoom_levels:
        min: 0
        max: 12
    input:
    output:
        format: xarray
        path: output.zarr
        dtype: uint16
        bands: 3
    pyramid:
        grid: geodetic
        metatiling: 2


.. code-block:: python

    # process.py
    import numpy as np
    import xarray as xr


    def execute(
        mp,
        bands=3,
        dtype="uint16",
    ):
        shape = (bands, *mp.tile.shape)
        dims = ["bands", "Y", "X"]
        coords = {}

        return xr.DataArray(
            data=np.full(shape=shape, fill_value=500, dtype=dtype),
            dims=dims,
            coords=coords,
        )


4D output array
---------------

For a simple 4 (time, bands, x, y) output:

.. code-block:: yaml

    # example.mapchete
    process: process.py
    zoom_levels:
        min: 0
        max: 12
    input:
    output:
        format: xarray
        path: output.zarr
        dtype: uint16
        bands: 3
        time:
            start: 2022-03-01
            end: 2022-03-31
            pattern: 0 0 * * *
            chunksize: 10
            # alternatively you can use steps:
            # steps:
            #     - 2022-06-01
            #     - 2022-06-04
            #     - 2022-06-06
            #     - 2022-06-09
            #     - 2022-06-11
    pyramid:
        grid: geodetic
        metatiling: 2


.. code-block:: python

    # process.py
    import dateutil
    import numpy as np
    import xarray as xr


    def execute(
        mp,
        bands=3,
        dtype="uint16",
        timestamps=None,
    ):
        timestamps = [
            "2022-03-01",
            "2022-03-02",
            "2022-03-04",
            "2022-03-07",
            "2022-03-09",
        ]
        shape = (bands, len(timestamps), *mp.tile.shape)
        dims = ["band", "time", "Y", "X"]
        coords = {"time": [dateutil.parser.parse(t) for t in timestamps]}

        return xr.DataArray(
            data=np.full(shape=shape, fill_value=500, dtype=dtype),
            dims=dims,
            coords=coords,
        )


------------
Installation
------------

.. code-block:: shell

    # install using pip:
    $ pip install mapchete_xarray
    # verify driver is vailable ('xarray' should be listed as output format):
    $ mapchete formats


-------------------
Current Limitations
-------------------

- No reprojection allowed when reading from a Zarr archive.
- No output pixelbuffer possible.


-------
License
-------

MIT License

Copyright (c) 2019-2022 `EOX IT Services`_

.. _`EOX IT Services`: https://eox.at/
.. _`Zarr`: https://zarr.readthedocs.io/en/stable/index.html
.. _`GDAL Zarr driver`: https://gdal.org/drivers/raster/zarr.html
