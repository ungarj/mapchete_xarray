#########
Changelog
#########

----------------------
2022.10.0 - 2022.10.21
----------------------

* core

    * faster tiles exist check


---------------------
2022.7.0 - 2022.05.04
---------------------

Note: major overhaul of package.

* core

    * removed support for all storage functionalities except single Zarr output

* tests

    * rewrote whole test suite; added Sentinel-2 test fixtures


---------------------
2022.5.0 - 2022.05.04
---------------------

* core
    * add support for single ZARR archives


----------------------
2021.11.0 - 2021.11.03
----------------------

* core

    * add S3 read/write support for NetCDF
    * add ``mapchete_xarray.processes.convert_to_xarray`` process

* tests

    * use ``mapchete.testing``

----------------------
2021.10.0 - 2021.10.25
----------------------

* packaging

    * change version numbering scheme to ``YYYY.MM.x``
    * use GitHub actions instead of travis

* tests

    * fix fixtures


---
0.4
---
* enable writing to zarr (#1)

---
0.3
---
* fixed PyPi packaging

---
0.2
---
* enable netCDF compression
* adapted to updated mapchete driver interface

---
0.1
---

* writing to and reading from xarray dumped as NetCDF
