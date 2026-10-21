"""SYNTHETIC demonstration data and end-to-end run.

Generates a small 10 m UTM scene with six classes whose NDVI follows plausible phenological
trajectories over four windows, matching Sentinel-1 backscatter, reference polygons, parcels with
~10 % deliberately wrong declarations and independent field points. It exists to exercise the
software; its accuracy numbers say nothing about real-world performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from .raster import NODATA

PERIODS = ("P1", "P2", "P3", "P4")
# NDVI per window (P1 spring ... P4 late season)
PROFILES = {
    "winter_crops": (0.75, 0.55, 0.15, 0.20),
    "maize": (0.20, 0.60, 0.82, 0.50),
    "soybean": (0.15, 0.50, 0.85, 0.35),
    "sunflower": (0.20, 0.72, 0.55, 0.22),
    "grassland": (0.55, 0.62, 0.55, 0.50),
    "forest": (0.80, 0.86, 0.85, 0.75),
}
SWIR_OFFSET = {"winter_crops": 0.0, "maize": 0.02, "soybean": -0.03, "sunflower": 0.04, "grassland": 0.01, "forest": -0.05}
VH_OFFSET = {"winter_crops": -1.0, "maize": 1.5, "soybean": 0.0, "sunflower": 2.0, "grassland": -1.5, "forest": 3.0}
ORIGIN = (600000.0, 4980000.0)   # EPSG:32634, northern Bosnia latitude band
CRS = "EPSG:32634"


def _bands(ndvi, cls_swir, rng):
    red = 0.03 + (1 - ndvi) * 0.08
    nir = red * (1 + ndvi) / (1 - ndvi)
    out = {
        "B02": red * 0.8, "B03": red * 1.1 + 0.01, "B04": red,
        "B05": red + 0.30 * (nir - red), "B06": red + 0.60 * (nir - red), "B07": red + 0.85 * (nir - red),
        "B08": nir, "B8A": nir * 1.02, "B11": 0.30 - 0.15 * ndvi + cls_swir,
    }
    out["B12"] = out["B11"] * 0.7
    return {k: np.clip(v + rng.normal(0, 0.006, v.shape), 0.001, 1).astype("float32") for k, v in out.items()}


def generate(root: str | Path, parcels_per_side: int = 12, parcel_px: int = 16, seed: int = 2023) -> dict:
    root = Path(root)
    rng = np.random.default_rng(seed)
    n = parcels_per_side * parcel_px
    transform = from_origin(*ORIGIN, 10, 10)
    names = list(PROFILES)
    label = np.zeros((n, n), dtype=int)
    jitter = np.zeros((n, n), dtype="float32")
    parcels, training, points = [], [], []
    for iy in range(parcels_per_side):
        for ix in range(parcels_per_side):
            k = int(rng.integers(len(names)))
            y0, x0 = iy * parcel_px, ix * parcel_px
            label[y0:y0 + parcel_px, x0:x0 + parcel_px] = k
            jitter[y0:y0 + parcel_px, x0:x0 + parcel_px] = rng.normal(0, 0.06)
            geom = box(ORIGIN[0] + x0 * 10, ORIGIN[1] - (y0 + parcel_px) * 10,
                       ORIGIN[0] + (x0 + parcel_px) * 10, ORIGIN[1] - y0 * 10)
            pid = f"P{iy:02d}{ix:02d}"
            declared = names[k] if rng.random() > 0.1 else names[int(rng.integers(len(names)))]
            parcels.append({"parcel_id": pid, "declared_crop": declared, "true_class": names[k], "geometry": geom})
            if (ix + iy) % 3 == 0:          # one third of fields interpreted as training polygons
                training.append({"polygon_id": f"T{pid}", "class_name": names[k], "geometry": geom.buffer(-30)})
            elif (ix + 2 * iy) % 4 == 1:    # disjoint fields visited in the field season
                c = geom.centroid
                points.append({"obs_id": f"O{pid}", "class_name": names[k], "observed_at": "2023-07-20T10:00:00Z",
                               "geometry": Point(c.x + rng.normal(0, 8), c.y + rng.normal(0, 8))})
    comp = root / "composites"
    profile = {"driver": "GTiff", "height": n, "width": n, "count": 1, "dtype": "float32", "crs": CRS,
               "transform": transform, "nodata": NODATA, "compress": "deflate"}
    swir = np.array([SWIR_OFFSET[c] for c in names], dtype="float32")[label]
    vh_off = np.array([VH_OFFSET[c] for c in names], dtype="float32")[label]
    for p_i, period in enumerate(PERIODS):
        folder = comp / period
        folder.mkdir(parents=True, exist_ok=True)
        ndvi = np.array([PROFILES[c][p_i] for c in names], dtype="float32")[label]
        ndvi = np.clip(ndvi + jitter + rng.normal(0, 0.12, ndvi.shape), 0.02, 0.95)
        for band, arr in _bands(ndvi, swir, rng).items():
            with rasterio.open(folder / f"{band}.tif", "w", **profile) as dst:
                dst.write(arr, 1)
        vh_db = -23 + 9 * ndvi + vh_off + rng.normal(0, 2.0, ndvi.shape)
        vv_db = -13 + 3 * ndvi + 0.4 * vh_off + rng.normal(0, 2.0, ndvi.shape)
        for pol, db in (("VV", vv_db), ("VH", vh_db)):
            with rasterio.open(folder / f"{pol}.tif", "w", **profile) as dst:
                dst.write((10 ** (db / 10)).astype("float32"), 1)
    ref = root / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    for name, rows in (("parcels", parcels), ("training", training), ("observations", points)):
        path = ref / f"{name}.gpkg"
        path.unlink(missing_ok=True)
        gpd.GeoDataFrame(rows, crs=CRS).to_file(path, driver="GPKG")
    (root / "SYNTHETIC_DATA.txt").write_text("Synthetic test data generated by geocrop demo. Not real imagery.\n")
    return {"root": str(root), "composites": str(comp), "parcels": str(ref / "parcels.gpkg"),
            "training": str(ref / "training.gpkg"), "observations": str(ref / "observations.gpkg"),
            "classes": names}


def run(root: str | Path, quick: bool = False) -> dict:
    """Generate synthetic data and run every stage of the pipeline on it."""
    from . import experiments, features, model, parcels, validation
    from .samples import extract

    root = Path(root)
    data = generate(root)
    stack = features.build_stack(data["composites"], root / "features" / "stack.tif")
    samples = extract(stack, data["training"], max_per_polygon=60)
    samples.save(root / "features" / "samples.npz")
    params = {"n_estimators": 60 if quick else 200, "max_features": "sqrt",
              "class_weight": "balanced_subsample", "n_jobs": -1}
    report = model.train(samples, root / "model", params=params, n_splits=3, spatial_block_m=500,
                         bootstrap=100 if quick else 500)
    model.predict(stack, root / "model" / "model.joblib", root / "prediction")
    parcel_summary = parcels.aggregate(root / "prediction" / "probabilities.tif", data["parcels"],
                                       root / "parcels" / "parcels.gpkg", strategy="combined",
                                       min_confidence=0.5)
    val = validation.evaluate_points(root / "prediction", data["observations"], root / "validation",
                                     year=2023, training=data["training"], training_exclusion_m=20, window_px=1,
                                     bootstrap=100 if quick else 500)
    table = experiments.run(samples, features.load_manifest(stack), root / "experiments", params=params,
                            n_splits=3, spatial_block_m=500, targets=["soybean", "sunflower"])
    result = {
        "mode": "SYNTHETIC ONLY - software verification, not scientific accuracy",
        "cv_macro_f1": round(report["cross_validation"]["macro_f1"], 3),
        "field_points_oa": round(val["accuracy"]["overall_accuracy"], 3),
        "declaration_status": parcel_summary["declaration_status_counts"],
        "ablation": table[["experiment", "macro_f1"]].round(3).to_dict(orient="records"),
        "outputs": str(root),
    }
    (root / "demo_summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result
