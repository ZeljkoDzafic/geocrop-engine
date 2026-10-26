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


WINDOW_CENTRES = (121, 174, 220, 266)   # day of year of P1..P4 centres


def class_curve(name: str, doy) -> np.ndarray:
    """Smooth NDVI trajectory of a class, through the window values (bare soil outside the season)."""
    knots = (60, *WINDOW_CENTRES, 330)
    prof = PROFILES[name]
    edge = 0.75 if name == "forest" else (0.6 if name == "winter_crops" else 0.15)
    return np.interp(doy, knots, (edge, *prof, max(0.15, prof[-1] - 0.25) if name != "forest" else 0.7))


def generate_scenes(root: str | Path, year: int = 2023, s2_every: int = 5, s1_every: int = 12,
                    cloud_fraction: float = 0.35, seed: int = 7, **grid) -> dict:
    """Per-acquisition scene folders (S2 with SCL clouds, S1 linear VV/VH) following the scene contract."""
    from datetime import date, timedelta

    from . import scenes as scene_meta

    root = Path(root)
    base = generate(root, **grid)
    rng = np.random.default_rng(seed)
    with rasterio.open(Path(base["composites"]) / "P1" / "B02.tif") as ref:
        profile, shape = ref.profile, ref.shape
    lab = _labels(base["parcels"], shape)
    names = list(PROFILES)
    swir = np.array([SWIR_OFFSET[c] for c in names], dtype="float32")[lab]
    vh_off = np.array([VH_OFFSET[c] for c in names], dtype="float32")[lab]
    start = date(year, 3, 15)
    s2_dirs, s1_dirs = [], []
    for k in range(0, 230, s2_every):
        d = start + timedelta(days=k)
        doy = d.timetuple().tm_yday
        ndvi = np.array([class_curve(c, doy) for c in names], dtype="float32")[lab]
        ndvi = np.clip(ndvi + rng.normal(0, 0.03, shape), 0.02, 0.95)
        folder = root / "scenes" / "s2" / d.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        for band, arr in _bands(ndvi, swir, rng).items():
            with rasterio.open(folder / f"{band}.tif", "w", **profile) as dst:
                dst.write(arr, 1)
        scl = np.full(shape, 4, dtype="uint8")
        if rng.random() < cloud_fraction * 2:
            cy, cx, r = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(10, shape[0] // 2)
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            scl[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 9
        prof8 = {**profile, "dtype": "uint8", "nodata": None}
        with rasterio.open(folder / "SCL.tif", "w", **prof8) as dst:
            dst.write(scl, 1)
        scene_meta.write(folder, sensor="S2", datetime=f"{d.isoformat()}T09:40:00", synthetic=True)
        s2_dirs.append(folder)
    for k in range(0, 230, s1_every):
        d = start + timedelta(days=k)
        doy = d.timetuple().tm_yday
        ndvi = np.array([class_curve(c, doy) for c in names], dtype="float32")[lab]
        folder = root / "scenes" / "s1" / d.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        vh = -23 + 9 * ndvi + vh_off + rng.normal(0, 1.5, shape)
        vv = -13 + 3 * ndvi + 0.4 * vh_off + rng.normal(0, 1.5, shape)
        for pol, db in (("VV", vv), ("VH", vh)):
            with rasterio.open(folder / f"{pol}.tif", "w", **profile) as dst:
                dst.write((10 ** (db / 10)).astype("float32"), 1)
        scene_meta.write(folder, sensor="S1", datetime=f"{d.isoformat()}T16:50:00", orbit_direction="ASCENDING",
                         synthetic=True)
        s1_dirs.append(folder)
    return {**base, "s2_scenes": [str(p) for p in s2_dirs], "s1_scenes": [str(p) for p in s1_dirs],
            "scene_root": str(root / "scenes"), "reference": str(Path(base["composites"]) / "P1" / "B02.tif")}


def _labels(parcels_path, shape) -> np.ndarray:
    gdf = gpd.read_file(parcels_path)
    names = list(PROFILES)
    lab = np.zeros(shape, dtype=int)
    for row in gdf.itertuples():
        x0, y0, x1, y1 = row.geometry.bounds
        c0, c1 = int((x0 - ORIGIN[0]) / 10), int((x1 - ORIGIN[0]) / 10)
        r0, r1 = int((ORIGIN[1] - y1) / 10), int((ORIGIN[1] - y0) / 10)
        lab[r0:r1, c0:c1] = names.index(row.true_class)
    return lab


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
                                       conformal=root / "model" / "conformal.json", alpha=0.1,
                                       aoa=root / "prediction" / "aoa.tif")
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
        "traffic_light": parcel_summary["traffic_light_counts"],
        "ablation": table[["experiment", "macro_f1"]].round(3).to_dict(orient="records"),
        "outputs": str(root),
    }
    (root / "demo_summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def synthetic_parcel_series(n_side: int = 12, cell_px: int = 10, start=None, end=None, revisit_days: int = 5,
                            cloud_fraction: float = 0.3, pixel_noise: float = 0.08, shift_days: float = 12.0,
                            amplitude_sd: float = 0.08, mixed_fraction: float = 0.15, step_days: int = 5,
                            output_step_days: int = 15, lam: float = 20.0, seed: int = 11) -> dict:
    """SYNTHETIC parcel signatures and matching pixel time series (no rasters; for parcel-level tests).

    Every parcel follows its class's NDVI curve (`class_curve`) shifted by a parcel-specific sowing offset
    (sd `shift_days`) and scaled by a parcel amplitude (sd `amplitude_sd`). `mixed_fraction` of each
    parcel's pixels is contaminated 50/50 with another class (boundary/weed mixing); each observation has
    iid pixel noise `pixel_noise`; on each acquisition a parcel is fully clouded with `cloud_fraction`.

    Returns ``parcels`` (GeoDataFrame: parcel_id, true_class, declared_crop with ~10 % wrong declarations,
    rectangles of random size inside a grid of `cell_px` 10 m cells), ``signatures`` (long table in the
    `cbm.parcel_signatures` layout, index NDVI), ``pixels`` (`Samples` of Whittaker-smoothed per-pixel
    ``ts_NDVI_d<doy>`` features, polygon = parcel id), ``start`` and ``end``. Software tests only.
    """
    from datetime import date, timedelta

    import pandas as pd

    from . import timeseries as ts
    from .samples import Samples

    start = start or date(2023, 3, 15)
    end = end or date(2023, 10, 31)
    rng = np.random.default_rng(seed)
    names = list(PROFILES)
    acq = [start + timedelta(days=k) for k in range(0, (end - start).days + 1, revisit_days)]
    doy = np.array([d.timetuple().tm_yday for d in acq], dtype=float)
    grid = ts.make_grid(start, end, step_days)
    gdoy = np.array([d.timetuple().tm_yday for d in grid])
    keep = np.arange(0, len(grid), output_step_days // step_days)
    parcels, sig_rows, X, y, pid, xs, ys = [], [], [], [], [], [], []
    for iy in range(n_side):
        for ix in range(n_side):
            k = int(rng.integers(len(names)))
            cls = names[k]
            w, h = (int(v) for v in rng.integers(cell_px // 2, cell_px + 1, size=2))
            x0, y0 = ORIGIN[0] + ix * cell_px * 10, ORIGIN[1] - iy * cell_px * 10
            ident = f"S{iy:02d}{ix:02d}"
            declared = cls if rng.random() > 0.1 else names[int(rng.integers(len(names)))]
            parcels.append({"parcel_id": ident, "true_class": cls, "declared_crop": declared,
                            "geometry": box(x0, y0 - h * 10, x0 + w * 10, y0)})
            n = w * h
            curve = class_curve(cls, doy - rng.normal(0, shift_days)) * rng.normal(1, amplitude_sd)
            values = np.repeat(curve[None, :], n, axis=0)
            mixed = rng.random(n) < mixed_fraction
            if mixed.any():
                other = names[int(rng.choice([j for j in range(len(names)) if j != k]))]
                values[mixed] = 0.5 * values[mixed] + 0.5 * class_curve(other, doy)[None, :]
            values = np.clip(values + rng.normal(0, pixel_noise, values.shape), -0.2, 1.0)
            clear = rng.random(len(acq)) >= cloud_fraction
            for j, d in enumerate(acq):
                v = values[:, j]
                sig_rows.append({"parcel_id": ident, "date": d.isoformat(), "index": "NDVI",
                                 "mean": float(v.mean()) if clear[j] else np.nan,
                                 "std": float(v.std()) if clear[j] else np.nan,
                                 "n_valid": n if clear[j] else 0, "n_pixels": n})
            yb, wb = ts.bin_to_grid(acq, values, np.broadcast_to(clear, values.shape), grid)
            X.append(ts.whittaker(yb, wb, lam=lam, min_valid=4)[:, keep])
            y += [cls] * n
            pid += [ident] * n
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            xs.append(x0 + (cols.ravel() + 0.5) * 10)
            ys.append(y0 - (rows.ravel() + 0.5) * 10)
    feats = [f"ts_NDVI_d{int(gdoy[k])}" for k in keep]
    Xa = np.concatenate(X).astype("float32")
    ok = np.isfinite(Xa).all(axis=1)
    pixels = Samples(Xa[ok], np.array(y)[ok], np.array(pid)[ok], np.concatenate(xs)[ok], np.concatenate(ys)[ok],
                     feats, CRS)
    return {"parcels": gpd.GeoDataFrame(parcels, crs=CRS), "signatures": pd.DataFrame(sig_rows),
            "pixels": pixels, "start": start, "end": end, "classes": names}


def write_parcel_series(root: str | Path, **kwargs) -> dict:
    """Write `synthetic_parcel_series` to ``parcels.gpkg``, ``signatures.csv`` and ``pixel_samples.npz``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = synthetic_parcel_series(**kwargs)
    paths = {"parcels": root / "parcels.gpkg", "signatures": root / "signatures.csv",
             "pixels": root / "pixel_samples.npz"}
    paths["parcels"].unlink(missing_ok=True)
    data["parcels"].to_file(paths["parcels"], driver="GPKG")
    data["signatures"].to_csv(paths["signatures"], index=False)
    data["pixels"].save(paths["pixels"])
    (root / "SYNTHETIC_DATA.txt").write_text("Synthetic parcel series generated by geocrop. Not real imagery.\n")
    return {**{k: str(v) for k, v in paths.items()}, "start": data["start"].isoformat(),
            "end": data["end"].isoformat(), "parcels_count": int(len(data["parcels"])),
            "pixels_count": int(len(data["pixels"].y))}
