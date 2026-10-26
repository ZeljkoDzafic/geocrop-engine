"""Parcel-level classification directly on parcel signatures (CbM crop-type marker style).

The pixel workflow classifies pixels and aggregates their probabilities per parcel (`geocrop.parcels`).
Operational Checks-by-Monitoring systems (e.g. the Sen4CAP L4A crop-type product or crop-type markers in
area-monitoring services; ideas only, no code reused) instead classify one feature vector per parcel built
from the parcel's time series. This module does the latter on the long signature table written by
`geocrop.cbm.parcel_signatures` (``parcel_id, date, index, mean, std, n_valid, n_pixels``):

``parcel_features``   per parcel and index: each signature statistic (``mean``, ``std``, or any further
                      numeric column such as within-parcel percentiles) is screened for cloud coverage,
                      binned to a regular grid and Whittaker-smoothed, then summarised as
                        ts_<INDEX>_<stat>_d<doy>     smoothed series every `output_step_days` (channel-major,
                                                     the layout TempCNN / L-TAE reshape to (n, T, C))
                        stat_<INDEX>_<stat>_p<q>     temporal percentiles (10/25/50/75/90) of the series
                        stat_<INDEX>_mean_peakdoy    day of year of the series maximum
                        win_<INDEX>_<stat>_d<doy>    mean per `window_days` window (start day of year)
                        shape_*                      area, perimeter, shape index, elongation, solidity,
                                                     interior pixel count
                      plus ``qa_clear_observations`` / ``qa_n_pixels`` (quality, never model inputs).
``train``             one sample per parcel through `geocrop.model.train`: the same parcel-grouped spatial
                      CV, locked holdout, tuning, temperature calibration, conformal thresholds and AOA as
                      the pixel model (one calibration object = one parcel). Any registered estimator works,
                      including the temporal models `tempcnn` and `ltae` (they use the `ts_` block only).
``predict``           parcel table with the same decision columns as `geocrop.parcels.aggregate`
                      (predicted_class, confidence, margin, entropy, p_<class>, prediction_set, cp_status,
                      decision, decision_reasons, declaration_status, traffic_light ...), so the CbM
                      traffic light, field-work ranking and exports work unchanged.
``compare_pixel_aggregation``   parcel classifier vs pixel classifier + parcel aggregation (mean
                      probability and majority) on the same parcels, the same locked holdout and the same
                      parcel-grouped spatial folds; per-fold scores, pooled metrics and McNemar test.

Labels must come from independently verified parcels, never from the declarations being checked.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import metrics
from . import timeseries as ts
from .samples import Samples

FEATURE_GROUPS = ("ts", "stats", "windows", "shape")
GROUP_PREFIX = {"ts": "ts_", "stats": "stat_", "windows": "win_", "shape": "shape_"}
PERCENTILES = (10, 25, 50, 75, 90)
DEFAULT_PARAMS = {"algorithm": "rf", "n_estimators": 500, "max_features": "sqrt",
                  "class_weight": "balanced_subsample", "n_jobs": -1}


# --------------------------------------------------------------------------- features

def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def _projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    if gdf.crs.is_projected and math.isclose(gdf.crs.axis_info[0].unit_conversion_factor, 1.0):
        return gdf
    return gdf.to_crs(gdf.estimate_utm_crs())


def shape_descriptors(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area (m2), perimeter (m), shape index P/(2 sqrt(pi A)), elongation (long/short side of the minimum
    rotated rectangle) and solidity (area / convex-hull area) per parcel, in a metric CRS."""
    rows = []
    for geom in _projected(parcels).geometry:
        if geom is None or geom.is_empty or geom.area <= 0:
            rows.append({k: np.nan for k in ("area_m2", "perimeter_m", "index", "elongation", "solidity")})
            continue
        rect = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)
        sides = np.hypot(*np.diff(rect[:3], axis=0).T)
        rows.append({"area_m2": geom.area, "perimeter_m": geom.length,
                     "index": geom.length / (2 * math.sqrt(math.pi * geom.area)),
                     "elongation": float(sides.max() / sides.min()) if sides.min() > 0 else np.nan,
                     "solidity": geom.area / geom.convex_hull.area})
    return pd.DataFrame(rows).add_prefix("shape_")


def signature_series(signatures: pd.DataFrame, start: date, end: date, index: str, stats=("mean", "std"),
                     step_days: int = 5, lam: float = 20.0, min_coverage: float = 0.5, min_valid: int = 4,
                     parcel_ids=None):
    """Whittaker-smoothed regular series of one index: (ids, grid days, {stat: (n_parcels, n_grid)}, clear counts).

    Observations covering less than `min_coverage` of the parcel's interior pixels are discarded (as in
    `cbm.markers`); parcels with fewer than `min_valid` clear grid bins get NaN series.
    """
    missing = [c for c in ("parcel_id", "date", "index", "n_valid", "n_pixels", *stats) if c not in signatures]
    if missing:
        raise ValueError(f"Signatures lack columns {missing}")
    ids = (sorted(signatures.parcel_id.astype(str).unique()) if parcel_ids is None
           else [str(p) for p in parcel_ids])
    pos = {p: k for k, p in enumerate(ids)}
    sig = signatures[signatures["index"] == index]
    if sig.empty:
        raise ValueError(f"No signatures for index {index!r}")
    sig = sig[sig.parcel_id.astype(str).isin(pos)]
    grid = ts.make_grid(start, end, step_days)
    days = np.array([_doy(d) for d in grid])
    obs = sorted({date.fromisoformat(str(d)[:10]) for d in sig.date})
    dpos = {d.isoformat(): k for k, d in enumerate(obs)}
    rows = sig.parcel_id.astype(str).map(pos).to_numpy()
    cols = np.array([dpos[str(d)[:10]] for d in sig.date], dtype=int)
    clear = ((sig.n_pixels > 0) & (sig.n_valid >= min_coverage * sig.n_pixels)).to_numpy()
    out, counts = {}, np.zeros(len(ids), dtype=int)
    for stat in stats:
        values = np.full((len(ids), len(obs)), np.nan)
        valid = np.zeros_like(values, dtype=bool)
        v = pd.to_numeric(sig[stat], errors="coerce").to_numpy(dtype=float)
        values[rows, cols] = v
        valid[rows, cols] = clear & np.isfinite(v)
        y, w = ts.bin_to_grid(obs, values, valid, grid)
        out[stat] = ts.whittaker(y, w, lam=lam, min_valid=min_valid)
        if stat == stats[0]:
            counts = (w > 0).sum(axis=1)
    return ids, days, out, counts


def parcel_features(signatures: pd.DataFrame, parcels: gpd.GeoDataFrame | str | Path | None, start: date, end: date,
                    id_field: str = "parcel_id", indices=None, stats=("mean", "std"), groups=FEATURE_GROUPS,
                    step_days: int = 5, output_step_days: int = 15, window_days: int = 30, lam: float = 20.0,
                    min_coverage: float = 0.5, min_valid: int = 4) -> pd.DataFrame:
    """One row per parcel (index = parcel id as str) with the feature groups described in the module doc.

    Parcels without enough clear observations keep NaN features; `train` excludes them and `predict`
    reports them as ``insufficient_resolution``.
    """
    if output_step_days < step_days or output_step_days % step_days:
        raise ValueError("output_step_days must be a positive multiple of step_days")
    if window_days < step_days:
        raise ValueError("window_days must be at least step_days")
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups {sorted(unknown)}; choose from {FEATURE_GROUPS}")
    stats = tuple(stats)
    gdf = None
    if parcels is not None:
        gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels
        if id_field not in gdf or gdf[id_field].astype(str).duplicated().any():
            raise ValueError(f"Parcels need a unique {id_field!r} column")
        ids = gdf[id_field].astype(str).tolist()
    else:
        ids = sorted(signatures.parcel_id.astype(str).unique())
    indices = list(indices) if indices else sorted(signatures["index"].astype(str).unique())
    columns: dict[str, np.ndarray] = {}
    clear = []
    for index in indices:
        _, days, series, counts = signature_series(signatures, start, end, index, stats, step_days, lam,
                                                   min_coverage, min_valid, parcel_ids=ids)
        clear.append(counts)
        keep = np.arange(0, len(days), output_step_days // step_days)
        windows = [np.arange(s, min(s + window_days // step_days, len(days)))
                   for s in range(0, len(days), window_days // step_days)]
        for stat in stats:
            z = series[stat]
            if "ts" in groups:
                for k in keep:
                    columns[f"ts_{index}_{stat}_d{int(days[k])}"] = z[:, k]
            if "stats" in groups:
                with np.errstate(all="ignore"):
                    pct = np.nanpercentile(z, PERCENTILES, axis=1) if np.isfinite(z).any() else \
                        np.full((len(PERCENTILES), len(ids)), np.nan)
                for q, values in zip(PERCENTILES, pct):
                    columns[f"stat_{index}_{stat}_p{q}"] = values
                if stat == "mean":
                    ok = np.isfinite(z).all(axis=1)
                    peak = np.full(len(ids), np.nan)
                    peak[ok] = days[np.argmax(z[ok], axis=1)]
                    columns[f"stat_{index}_mean_peakdoy"] = peak
            if "windows" in groups:
                for w in windows:
                    columns[f"win_{index}_{stat}_d{int(days[w[0]])}"] = z[:, w].mean(axis=1)
    frame = pd.DataFrame(columns, index=pd.Index(ids, name="parcel_id"))
    if "shape" in groups and gdf is not None:
        shape = shape_descriptors(gdf).set_index(frame.index)
        frame = pd.concat([frame, shape], axis=1)
    n_pixels = (signatures.assign(parcel_id=signatures.parcel_id.astype(str))
                .groupby("parcel_id")["n_pixels"].max().reindex(frame.index))
    if "shape" in groups:
        frame["shape_n_pixels"] = n_pixels.to_numpy(dtype=float)
    frame["qa_clear_observations"] = np.min(clear, axis=0) if clear else 0
    frame["qa_n_pixels"] = n_pixels.fillna(0).astype(int).to_numpy()
    return frame


def feature_columns(frame: pd.DataFrame, groups=FEATURE_GROUPS) -> list[str]:
    """Model input columns of `frame` for the requested groups (in frame order)."""
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    return [c for c in frame.columns if c.startswith(prefixes)]


def to_samples(frame: pd.DataFrame, parcels: gpd.GeoDataFrame, label_field: str, columns: list[str],
               id_field: str = "parcel_id") -> tuple[Samples, list[str]]:
    """Labelled parcels with finite features as `Samples` (one sample per parcel, polygon = parcel id,
    coordinates = centroid in a metric CRS); also returns the labelled ids excluded for missing features."""
    gdf = _projected(parcels)
    if label_field not in gdf:
        raise ValueError(f"Parcels need the reference label column {label_field!r}")
    labels = gdf[label_field]
    labelled = gdf[labels.notna() & labels.astype(str).str.strip().ne("")]
    ids = labelled[id_field].astype(str).to_numpy()
    X = frame.reindex(ids)[columns].to_numpy(dtype="float32")
    finite = np.isfinite(X).all(axis=1)
    if not finite.any():
        raise ValueError("No labelled parcel has complete features")
    c = labelled.geometry.centroid
    s = Samples(X[finite], labelled[label_field].astype(str).to_numpy()[finite], ids[finite],
                c.x.to_numpy()[finite], c.y.to_numpy()[finite], list(columns), gdf.crs.to_string())
    return s, sorted(ids[~finite].tolist())


def _resolve_columns(frame: pd.DataFrame, params: dict, groups) -> list[str]:
    from .estimators import SEQUENCE_ALGORITHMS

    sequence = params.get("algorithm") in SEQUENCE_ALGORITHMS
    columns = feature_columns(frame, ("ts",) if sequence else tuple(groups))
    if not columns:
        raise ValueError(f"No feature columns for groups {list(groups)}" +
                         (" (temporal models need the ts group)" if sequence else ""))
    return columns


# --------------------------------------------------------------------------- training and prediction

def train(frame: pd.DataFrame, parcels: gpd.GeoDataFrame | str | Path, label_field: str, output_dir: str | Path,
          params: dict | None = None, groups=FEATURE_GROUPS, id_field: str = "parcel_id", n_splits: int = 5,
          spatial_block_m: float | None = 5000, seed: int = 42, holdout_fraction: float = 0.0,
          calibration_mode: str = "oof", calibration_fraction: float = 0.2, spatial_buffer_m: float = 0,
          bootstrap: int = 200, conformal_alphas=(0.05, 0.1, 0.2), tune_grid: dict | None = None) -> dict:
    """Parcel classifier through `geocrop.model.train` (one sample per parcel; same CV/holdout/calibration).

    Writes the usual training artifacts (model.joblib, conformal.json, aoa.json, train_report.json,
    oof_polygons.csv with one row per parcel) plus ``parcel_features.csv``.
    """
    from . import model

    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels
    params = dict(params or DEFAULT_PARAMS)
    columns = _resolve_columns(frame, params, groups)
    samples, excluded = to_samples(frame, gdf, label_field, columns, id_field)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = model.train(samples, out, features=columns, params=params, n_splits=n_splits,
                         spatial_block_m=spatial_block_m, seed=seed, tune_grid=tune_grid, bootstrap=bootstrap,
                         conformal_alphas=conformal_alphas, spatial_buffer_m=spatial_buffer_m,
                         calibration_mode=calibration_mode, calibration_fraction=calibration_fraction,
                         holdout_fraction=holdout_fraction)
    report.update({"level": "parcel", "sample_unit": "one feature vector per parcel (polygon = parcel)",
                   "feature_groups": sorted({g for g, p in GROUP_PREFIX.items() if any(c.startswith(p) for c in columns)}),
                   "excluded_incomplete_parcels": excluded, "label_field": label_field,
                   "training_parcel_ids": sorted(samples.polygon.tolist())})
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    frame.to_csv(out / "parcel_features.csv")
    return report


def _entropy(p: np.ndarray) -> float:
    k = len(p)
    if k < 2:
        return 0.0
    return float(-np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0)) / math.log(k))


def predict(model_dir: str | Path, frame: pd.DataFrame, parcels: gpd.GeoDataFrame | str | Path,
            output: str | Path, id_field: str = "parcel_id", declared_field: str | None = "declared_crop",
            alpha: float | None = 0.1, min_valid_pixels: int = 4, min_confidence: float | None = None,
            min_margin: float | None = None, min_aoa_fraction: float = 0.5, crop_mapping: dict | None = None,
            markers: pd.DataFrame | None = None, calendar: dict | None = None) -> dict:
    """Classify every parcel and write GeoPackage/GeoJSON + CSV + summary JSON (`parcels.aggregate` schema).

    ``n_valid`` is the parcel's interior pixel count from the signatures; ``aoa_fraction`` is 1 when the
    parcel's feature vector lies inside the model's area of applicability, else 0; ``in_training`` marks
    parcels whose labels trained the model (their predictions are in-sample).
    """
    from joblib import load

    from . import applicability, cbm, contracts, uncertainty
    from .hashing import sha256
    from .parcels import _decision_evidence, decision_details, declaration_status

    if not np.isfinite(min_aoa_fraction) or not 0 <= min_aoa_fraction <= 1:
        raise ValueError("min_aoa_fraction must be finite and in [0, 1]")
    mdir = Path(model_dir)
    payload = contracts.check("model", load(mdir / "model.joblib"), mdir / "model.joblib")
    est, columns = payload["model"], payload["features"]
    temperature = float(payload.get("temperature", 1.0))
    classes = [str(c) for c in est.classes_]
    model_sha = sha256(mdir / "model.joblib")
    thresholds = None
    if alpha is not None and (mdir / "conformal.json").exists():
        conf = json.loads((mdir / "conformal.json").read_text(encoding="utf8"))
        if conf.get("model_sha256") != model_sha:
            raise ValueError("conformal.json belongs to a different model.joblib; retrain or restore both")
        if str(alpha) not in conf["thresholds"]:
            raise ValueError(f"alpha {alpha} not calibrated; available {list(conf['thresholds'])}")
        thresholds = conf["thresholds"][str(alpha)]
    aoa = applicability.Applicability.from_json(mdir / "aoa.json") if (mdir / "aoa.json").exists() else None
    report_path = mdir / "train_report.json"
    trained = set(json.loads(report_path.read_text(encoding="utf8")).get("training_parcel_ids", [])) \
        if report_path.exists() else set()
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf or gdf[id_field].astype(str).duplicated().any():
        raise ValueError(f"Parcels need a unique {id_field!r} column")
    ids = gdf[id_field].astype(str).to_numpy()
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"Parcel features lack model inputs {missing[:5]}; rebuild them with the training settings")
    X = frame.reindex(ids)[columns].to_numpy(dtype="float32")
    finite = np.isfinite(X).all(axis=1)
    proba = np.full((len(ids), len(classes)), np.nan)
    if finite.any():
        proba[finite] = uncertainty.apply_temperature(est.predict_proba(X[finite]), temperature)
    di = np.full(len(ids), np.nan)
    if aoa is not None and finite.any():
        di[finite] = aoa.dissimilarity(X[finite])
    quality = frame.reindex(ids)
    rows = []
    for j, pid in enumerate(ids):
        n_px = quality["qa_n_pixels"].iloc[j] if "qa_n_pixels" in quality else np.nan
        row = {"n_valid": int(n_px) if np.isfinite(n_px) else 0,
               "clear_observations": (int(quality["qa_clear_observations"].iloc[j])
                                      if "qa_clear_observations" in quality and
                                      np.isfinite(quality["qa_clear_observations"].iloc[j]) else 0),
               "in_training": pid in trained}
        reasons, warnings = [], []
        if finite[j]:
            p = proba[j]
            order = np.sort(p)[::-1]
            best = int(p.argmax())
            row.update({"mean_class": classes[best], "majority_class": classes[best], "confidence": float(p[best]),
                        "majority_share": None, "strategies_agree": True,
                        "margin": float(order[0] - order[1]) if len(p) > 1 else 1.0, "entropy": _entropy(p)})
            row.update({f"p_{c}": float(p[i]) for i, c in enumerate(classes)})
            if thresholds is not None:
                members = [c for c in classes if 1 - row[f"p_{c}"] <= thresholds.get(c, 1.0)]
                row["prediction_set"], row["set_size"] = ";".join(members), len(members)
            if aoa is not None:
                row["aoa_dissimilarity"] = float(di[j])
                row["aoa_fraction"] = float(di[j] <= aoa.threshold)
            detail = decision_details(row, "mean_probability", min_valid_pixels, min_confidence, None, min_margin)
            cls, decision, reasons = detail["predicted_class"], detail["decision"], detail["reasons"]
            if aoa is not None and decision != "insufficient_resolution" and row["aoa_fraction"] < min_aoa_fraction:
                reasons.append("aoa_below_threshold")
                decision = "review"
        else:
            cls, decision, reasons = None, "insufficient_resolution", ["insufficient_observations"]
        row["decision_confidence"], row["decision_margin"] = _decision_evidence(row, cls)
        row["decision_reasons"] = ";".join(reasons)
        row["evidence_warnings"] = ";".join(warnings)
        declared = gdf[declared_field].iloc[j] if declared_field and declared_field in gdf else None
        if declared is not None and crop_mapping:
            declared = crop_mapping.get(str(declared), declared)
        if declared is not None and isinstance(declared, float) and np.isnan(declared):
            declared = None
        row.update({"predicted_class": cls, "decision": decision, "strategy": "parcel_classifier",
                    "declared_class": None if declared is None else str(declared),
                    "declaration_status": declaration_status(declared, cls, decision)})
        if thresholds is not None and row["declaration_status"] not in ("not_declared", "insufficient_resolution"):
            members = row.get("prediction_set", "").split(";") if row.get("prediction_set") else []
            row["cp_status"] = ("uncertain" if not members else
                                "discrepancy" if str(declared) not in members else
                                "agreement" if members == [str(declared)] else "uncertain")
        rows.append(row)
    table = pd.DataFrame(rows)
    if markers is not None:
        m = markers.assign(parcel_id=markers.parcel_id.astype(str)).set_index("parcel_id").reindex(ids)
        m = m.reset_index(drop=True)
        for col in ("peak_doy", "harvest_doy", "bare_soil_days", "clear_observations"):
            if col in m:
                table[f"marker_{col}" if col == "clear_observations" else col] = m[col].to_numpy()
        if calendar:
            table["calendar_consistent"] = cbm.calendar_consistency(m, table["declared_class"], calendar).to_numpy()
    table["traffic_light"] = cbm.traffic_light(table)
    table = table.drop(columns=["majority_class", "majority_share", "strategies_agree"], errors="ignore")
    result = gdf.drop(columns=[c for c in table.columns if c in gdf.columns]).reset_index(drop=True)
    result = gpd.GeoDataFrame(pd.concat([result, table], axis=1), geometry=gdf.geometry.name, crs=gdf.crs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    result.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
    result.drop(columns=result.geometry.name).to_csv(out.with_suffix(".csv"), index=False)
    summary = {
        "parcels": int(len(result)), "strategy": "parcel_classifier", "model_sha256": model_sha,
        "calibration_model_sha256": model_sha if thresholds is not None else None,
        "thresholds": {"min_valid_pixels": min_valid_pixels, "min_confidence": min_confidence,
                       "min_margin": min_margin},
        "conformal_alpha": alpha if thresholds is not None else None, "min_aoa_fraction": min_aoa_fraction,
        "decision_counts": result["decision"].value_counts().to_dict(),
        "declaration_status_counts": result["declaration_status"].value_counts().to_dict(),
        "traffic_light_counts": result["traffic_light"].value_counts().to_dict(),
        "decision_reason_counts": dict(sorted(Counter(code for codes in table["decision_reasons"]
                                                       for code in codes.split(";") if code).items())),
        "in_training": int(table["in_training"].sum()),
        "note": "Parcel-level classifier on signatures. Review flags support inspection prioritisation; "
                "they are not findings of non-compliance. Parcels marked in_training were used to fit the model.",
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    return summary


# --------------------------------------------------------------------------- pixel vs parcel experiment

def _take(s: Samples, mask) -> Samples:
    return Samples(s.X[mask], s.y[mask], s.polygon[mask], s.x[mask], s.yy[mask], s.features, s.crs)


def _aggregate_pixels(proba: np.ndarray, pixel_ids: np.ndarray, classes: list[str], ids) -> tuple[list, list]:
    """Mean-probability and majority class per parcel id (order of `ids`)."""
    frame = pd.DataFrame(proba, columns=classes)
    frame["pid"] = pixel_ids
    frame["vote"] = np.asarray(classes)[proba.argmax(axis=1)]
    mean = frame.groupby("pid")[classes].mean()
    votes = frame.groupby("pid")["vote"].agg(lambda v: sorted(Counter(v).items(), key=lambda t: (-t[1], t[0]))[0][0])
    return ([classes[int(np.argmax(mean.loc[i].to_numpy()))] for i in ids], [votes.loc[i] for i in ids])


def _fit_predict(params: dict, names: list[str], seed: int, X, y, X_test) -> tuple[np.ndarray, list[str]]:
    from .estimators import make_estimator, sequence_params

    est = make_estimator(sequence_params(params, names), seed).fit(X, y)
    return est.predict_proba(X_test), [str(c) for c in est.classes_]


def compare_pixel_aggregation(parcel_samples: Samples, pixel_samples: Samples, output_dir: str | Path,
                              parcel_params: dict | None = None, pixel_params: dict | None = None,
                              n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
                              holdout_fraction: float = 0.0, bootstrap: int = 500) -> dict:
    """Parcel classifier vs pixel classifier + aggregation on identical parcels and parcel-grouped folds.

    `parcel_samples`: one sample per parcel (`to_samples`); `pixel_samples`: pixels whose `polygon` is the
    parcel id (e.g. `samples.extract(stack, parcels, label_field=..., id_field="parcel_id")`). Only parcels
    present in both are used. The locked holdout and the folds are computed once on the parcels
    (`model.split_final_holdout`, `model._fold_plan`) and mapped to pixels by parcel id, so a parcel's pixels
    never appear in training and test of the same fold. Each fold k fits both models with seed + k.
    """
    import time

    from .model import _fold_plan, spatial_blocks, split_final_holdout

    parcel_params = dict(parcel_params or DEFAULT_PARAMS)
    pixel_params = dict(pixel_params or DEFAULT_PARAMS)
    common = np.intersect1d(parcel_samples.polygon.astype(str), pixel_samples.polygon.astype(str))
    if len(common) < 4:
        raise ValueError("Parcel and pixel samples share fewer than four parcels")
    parcels = _take(parcel_samples, np.isin(parcel_samples.polygon.astype(str), common))
    pixels = _take(pixel_samples, np.isin(pixel_samples.polygon.astype(str), common))
    if len(np.unique(parcels.polygon)) != len(parcels.polygon):
        raise ValueError("parcel_samples must hold exactly one sample per parcel")
    label = dict(zip(parcels.polygon.astype(str), parcels.y.astype(str)))
    if any(label[p] != str(c) for p, c in zip(pixels.polygon.astype(str), pixels.y)):
        raise ValueError("Pixel labels disagree with parcel labels")
    dev, test = split_final_holdout(parcels, spatial_block_m, holdout_fraction, seed)
    groups = spatial_blocks(dev, spatial_block_m)
    plan = _fold_plan(dev.y, groups, n_splits, seed, polygon=dev.polygon)
    classes = sorted(set(dev.y.astype(str)))
    methods = ("parcel_classifier", "pixel_mean_probability", "pixel_majority")
    oof = {m: np.empty(len(dev.y), dtype=object) for m in methods}
    seconds = {"parcel_classifier": 0.0, "pixel_model": 0.0}
    per_fold, fold_records = [], []
    pix_ids = pixels.polygon.astype(str)
    for k, (tr, te, _) in enumerate(plan):
        train_ids, test_ids = set(dev.polygon[tr].astype(str)), set(dev.polygon[te].astype(str))
        if train_ids & test_ids or set(groups[tr]) & set(groups[te]):
            raise AssertionError("Parcel leakage between training and test folds")
        ptr, pte = np.isin(pix_ids, list(train_ids)), np.isin(pix_ids, list(test_ids))
        if set(pix_ids[ptr]) & set(pix_ids[pte]):
            raise AssertionError("Pixel leakage between training and test parcels")
        t0 = time.perf_counter()
        proba, cls = _fit_predict(parcel_params, dev.features, seed + k, dev.X[tr], dev.y[tr], dev.X[te])
        oof["parcel_classifier"][te] = np.asarray(cls)[proba.argmax(axis=1)]
        seconds["parcel_classifier"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        proba, cls = _fit_predict(pixel_params, pixels.features, seed + k, pixels.X[ptr], pixels.y[ptr],
                                  pixels.X[pte])
        mean, major = _aggregate_pixels(proba, pix_ids[pte], cls, dev.polygon[te].astype(str))
        oof["pixel_mean_probability"][te], oof["pixel_majority"][te] = mean, major
        seconds["pixel_model"] += time.perf_counter() - t0
        row = {"fold": k, "test_parcels": int(len(te)), "train_parcels": int(len(tr)),
               "train_pixels": int(ptr.sum()), "test_pixels": int(pte.sum())}
        for m in methods:
            s = metrics.summary(dev.y[te], oof[m][te].astype(str), labels=classes)
            row[f"{m}_overall_accuracy"], row[f"{m}_macro_f1"] = s["overall_accuracy"], s["macro_f1"]
        per_fold.append(row)
        fold_records.append(sorted(test_ids))
    results = {}
    for m in methods:
        s = metrics.summary(dev.y, oof[m].astype(str), labels=classes)
        if bootstrap:
            s["bootstrap"] = metrics.bootstrap(dev.y, oof[m].astype(str), n_boot=bootstrap, seed=seed)
        s["fold_macro_f1_mean"] = float(np.mean([f[f"{m}_macro_f1"] for f in per_fold]))
        s["fold_macro_f1_std"] = float(np.std([f[f"{m}_macro_f1"] for f in per_fold]))
        results[m] = s
    report = {
        "schema": "geocrop-parcel-vs-pixel/1", "internal_only": True,
        "parcels": int(len(parcels.y)), "development_parcels": int(len(dev.y)), "pixels": int(len(pixels.y)),
        "classes": classes, "n_folds": len(plan), "seed": seed, "spatial_block_m": spatial_block_m,
        "holdout_fraction": holdout_fraction,
        "fold_fingerprint": hashlib.sha256(json.dumps(fold_records).encode()).hexdigest()[:16],
        "parcel_params": parcel_params, "pixel_params": pixel_params,
        "parcel_features": len(parcels.features), "pixel_features": len(pixels.features),
        "seconds": seconds, "folds": per_fold, "cross_validation": results,
        "mcnemar_parcel_vs_pixel_mean": metrics.mcnemar(dev.y, oof["parcel_classifier"],
                                                        oof["pixel_mean_probability"]),
        "mcnemar_parcel_vs_pixel_majority": metrics.mcnemar(dev.y, oof["parcel_classifier"], oof["pixel_majority"]),
        "note": "Same parcels, same locked holdout and same parcel-grouped spatial folds for both approaches. "
                "Internal comparison on reference parcels, not independent field validation.",
    }
    if test is not None:
        test_ids = test.polygon.astype(str)
        proba, cls = _fit_predict(parcel_params, dev.features, seed, dev.X, dev.y, test.X)
        hold = {"parcel_classifier": np.asarray(cls)[proba.argmax(axis=1)]}
        ptr, pte = np.isin(pix_ids, dev.polygon.astype(str)), np.isin(pix_ids, test_ids)
        proba, cls = _fit_predict(pixel_params, pixels.features, seed, pixels.X[ptr], pixels.y[ptr], pixels.X[pte])
        hold["pixel_mean_probability"], hold["pixel_majority"] = _aggregate_pixels(proba, pix_ids[pte], cls, test_ids)
        labels = sorted(set(classes) | set(test.y.astype(str)))
        report["holdout"] = {m: {k: v for k, v in metrics.summary(test.y, np.asarray(p).astype(str), labels=labels).items()
                                 if k in ("overall_accuracy", "macro_f1", "kappa", "per_class")}
                             for m, p in hold.items()}
        report["holdout"]["parcels"] = int(len(test.y))
        report["holdout"]["mcnemar_parcel_vs_pixel_mean"] = metrics.mcnemar(
            test.y, hold["parcel_classifier"], hold["pixel_mean_probability"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    pd.DataFrame({"parcel_id": dev.polygon, "label": dev.y, **{m: oof[m] for m in methods}}).to_csv(
        out / "comparison_oof.csv", index=False)
    pd.DataFrame([{"method": m, "overall_accuracy": r["overall_accuracy"], "macro_f1": r["macro_f1"],
                   "kappa": r["kappa"], "fold_macro_f1_std": r["fold_macro_f1_std"],
                   **({"holdout_macro_f1": report["holdout"][m]["macro_f1"]} if test is not None else {})}
                  for m, r in results.items()]).to_csv(out / "comparison.csv", index=False)
    return report
