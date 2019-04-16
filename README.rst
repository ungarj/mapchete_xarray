mapchete xarray
===============

This driver enables mapchete to write multidimensional arrays into a tile directory structure.

.. image:: https://badge.fury.io/py/mapchete_xarray.svg
    :target: https://badge.fury.io/py/mapchete_xarray

.. image:: https://travis-ci.org/ungarj/mapchete_xarray.svg?branch=master
    :target: https://travis-ci.org/ungarj/mapchete_xarray

.. image:: https://coveralls.io/repos/github/ungarj/mapchete_xarray/badge.svg?branch=master
    :target: https://coveralls.io/github/ungarj/mapchete_xarray?branch=master

.. image:: https://img.shields.io/pypi/pyversions/mapchete_xarray.svg
    :target: https://pypi.python.org/pypi/mapchete_xarray



Usage
-----

Example ``.mapchete`` file:

.. include:: tests/testdata/example.mapchete
    :code: yaml

Example process file:

.. include:: tests/testdata/process.py
    :code: python


Installation
------------

.. code-block:: shell

    # install using pip:
    pip install mapchete_xarray
    # verify driver is vailable ('xarray' should be listed as output format):
    mapchete formats


License
-------

MIT License

Copyright (c) 2019 `EOX IT Services`_

.. _`EOX IT Services`: https://eox.at/
