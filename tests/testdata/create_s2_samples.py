#!/usr/bin/env python3

"""
Create product list with:

$ mapchete satellite cat-results -b 16 48 17 49 2022-06-01 2022-06-30 vienna_2022-06.gpkg

"""
import math
import os

import click
import fiona
import rasterio
from affine import Affine
from mapchete import Timer
from rasterio.vrt import WarpedVRT
from tqdm import tqdm


def _parse_bands(_, __, value):
    if value:
        return value.split(",")


@click.command()
@click.argument(
    "PRODUCTS",
    type=click.Path(exists=True),
)
@click.option(
    "--bands",
    "-b",
    type=click.STRING,
    default="B04,B03,B02",
    callback=_parse_bands,
    help="""Bands used to create samples.\n"""
    """Available bands:\n"""
    """AOT, B01, B02, B03, B04, B05, B06, B07, B08,\n"""
    """B09, B11, B12, B8A, L2A_PVI, SCL, TCI, WVP""",
)
@click.option(
    "--resolution",
    "-r",
    type=click.INT,
    default=10,
    help="Output resolution in meters.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory to write the GeoTIFF files to.",
)
@click.option(
    "--granule",
    "-g",
    type=click.STRING,
    help="Filter out products by granule (e.g. '33UWP')",
)
def gen_samples(products, bands=None, resolution=None, out_dir=None, granule=None):
    with fiona.open(products) as src:
        if granule:
            filtered_products = [
                p
                for p in src
                if "".join(p["properties"]["baseurl"].split("/")[4:7]) == granule
            ]
        else:
            filtered_products = [p for p in src]
        for product in tqdm(filtered_products):
            product_id = product["properties"]["product_id"]
            baseurl = product["properties"]["baseurl"].replace(
                "/sentinel-s2-l2a/", "/sentinel-cogs/"
            )
            granule = "".join(baseurl.split("/")[4:7])
            generate_sample(
                product_id, baseurl, out_dir, bands, resolution, granule=granule
            )


def generate_sample(product_id, baseurl, out_dir, bands, resolution, granule=None):
    out_path = os.path.join(out_dir, f"{product_id}_{''.join(bands)}_{resolution}m.tif")
    if os.path.isfile(out_path):
        return

    with rasterio.open(os.path.join(baseurl, f"{bands[0]}.tif")) as src:
        meta = src.meta
        src_transform = src.transform
        src_res = src.transform[0]
    dst_transform = Affine.from_gdal(
        *(src_transform[2], resolution, 0.0, src_transform[5], 0.0, -resolution)
    )
    dst_width = int(math.ceil(src.width * (src_res / resolution)))
    dst_height = int(math.ceil(src.height * (src_res / resolution)))
    meta.update(
        count=len(bands),
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        compress="deflate",
    )
    with Timer() as product_timer:
        with rasterio.open(out_path, "w", **meta) as dst:
            tqdm.write(f"write {out_path} ...")
            for i, band in enumerate(bands, 1):
                with Timer() as band_timer:
                    src_path = os.path.join(baseurl, f"{band}.tif")
                    tqdm.write(f"read {src_path} and resample to {resolution}m ...")
                    with rasterio.open(src_path) as src:
                        with WarpedVRT(
                            src,
                            width=dst_width,
                            height=dst_height,
                            transform=dst_transform,
                        ) as warped:
                            dst.write(warped.read(1), i)
                tqdm.write(f"band read in {band_timer}")
    tqdm.write(f"{out_path} written successfully in {product_timer}")


if __name__ == "__main__":
    gen_samples()
