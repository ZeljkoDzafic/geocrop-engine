"""SYNTHETIC study with an irregular AOI for the AOI-aware tests (not real imagery).

A 640 x 576 px grid (10 m, EPSG:32634: 2 x 2 blocks of 512 px, the second row/column partial) holds 40 x 36 parcels of
six crop classes. The AOI is an L-shaped polygon with a diagonal edge. Its right edge ends exactly at column 512, so
the block to the right (columns 512 ...) holds no AOI pixel but lies within 3 px of one: it is computed only when
the halo is at least 3 px (the smoothing radius of a 7 px window). The bottom-right block is far from the AOI.
Training polygons and parcels lie inside the AOI.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geocrop import demo
from geocrop.aoi import AoiMask, default_mask_path, rasterize

PARCELS_PER_SIDE, PARCEL_PX = 36, 16
WIDTH = HEIGHT = PARCELS_PER_SIDE * PARCEL_PX           # 576 px: blocks 512 + 64
X0, Y0 = demo.ORIGIN


def pixel_polygon(points) -> Polygon:
    """Polygon from (column, row) pixel-corner coordinates."""
    return Polygon([(X0 + c * 10, Y0 - r * 10) for c, r in points])


def aoi_geometry():
    top = pixel_polygon([(0, 0), (512, 0), (512, 150), (260, 230), (0, 230)])
    left = pixel_polygon([(0, 200), (201, 200), (201, HEIGHT), (0, HEIGHT)])
    return unary_union([top, left])


def build(root: Path, mask: bool = True) -> dict:
    """Scenes, reference grid (+ AOI mask) and reference layers restricted to the AOI; returns the paths."""
    root = Path(root)
    data = demo.generate_scenes(root / "data", s2_every=30, s1_every=60, parcels_per_side=PARCELS_PER_SIDE,
                                parcel_px=PARCEL_PX)
    geom = aoi_geometry()
    grid = root / "grid.tif"
    with rasterio.open(grid, "w", driver="GTiff", width=WIDTH, height=HEIGHT, count=1, dtype="uint8", nodata=0,
                       crs=demo.CRS, transform=from_origin(X0, Y0, 10, 10), compress="deflate") as dst:
        import numpy as np
        dst.write(np.ones((HEIGHT, WIDTH), "uint8"), 1)
    mask_path = rasterize(geom, grid, default_mask_path(grid)) if mask else None
    layers = {}
    for key in ("training", "parcels", "observations"):
        frame = gpd.read_file(data[key])
        frame = frame[frame.geometry.within(geom)] if key != "observations" else frame[frame.geometry.within(geom)]
        path = root / f"{key}.gpkg"
        frame.to_file(path, driver="GPKG")
        layers[key] = str(path)
    (root / "aoi.geojson").write_text(gpd.GeoDataFrame(geometry=[geom], crs=demo.CRS).to_json())
    return {"root": root, "grid": grid, "mask": mask_path, "aoi": root / "aoi.geojson", "geometry": geom,
            "s2": Path(data["scene_root"]) / "s2", "s1": Path(data["scene_root"]) / "s1", **layers}


def config(study: dict, name: str, aoi: bool, **extra) -> dict:
    """A pipeline configuration for the study; `aoi=False` turns the (auto-discovered) mask off."""
    root = Path(study["root"])
    cfg = {"windows": {"P1": {"start": "2023-03-01", "end": "2023-06-30"},
                       "P2": {"start": "2023-07-01", "end": "2023-10-31"}},
           "grid": {"reference": str(study["grid"]), **({} if aoi else {"aoi_mask": False}),
                    **extra.pop("grid", {})},
           "paths": {"s2_scenes": str(study["s2"]), "s1_scenes": str(study["s1"]),
                     "composites": str(root / name / "composites"), "features": str(root / name / "stack.tif"),
                     "training": study["training"], "parcels": study["parcels"],
                     "observations": study["observations"], "output": str(root / name / "run")},
           "compositing": {"mask_dilation_px": 1},
           "features": {"s2_bands": ["B03", "B04", "B08", "B11"], "s2_indices": ["NDVI"], "s1": ["VH_dB"],
                        "deltas": ["NDVI"], "seasonal": ["NDVI"]},
           "model": {"n_estimators": 15, "n_jobs": 1, "cv_folds": 2, "spatial_block_m": 500,
                     "max_samples_per_polygon": 20},
           "predict": {"smoothing": {"enabled": True, "window": 7, "smoothness": 10.0}}}
    for key, value in extra.items():
        cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
    return cfg


def load_mask(study: dict, halo: int = 0) -> AoiMask:
    return AoiMask(study["mask"], halo)
