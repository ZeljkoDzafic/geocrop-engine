"""External validation against independent, time-stamped field observations (points).

Independence safeguards built in:
* only observations from the classification season (`year`) are used (crop rotation);
* observations inside, or closer than `training_exclusion_m` to, any training polygon are
  excluded and counted, so training and validation cannot share fields;
* an optional (2r+1)^2 majority window absorbs GNSS position error at field edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from . import metrics
from .raster import check_grid


def _window_vote(ds, row, col, radius, nodata=255, return_share=False):
    win = Window(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1)
    block = ds.read(1, window=win, boundless=True, fill_value=nodata)
    vals = block[block != nodata]
    if vals.size == 0:
        return (None, None) if return_share else None
    counts = np.bincount(vals.astype(int))
    winner = int(counts.argmax())
    return (winner, float(counts[winner] / vals.size)) if return_share else winner


def evaluate_points(prediction_dir: str | Path, observations: str | Path | gpd.GeoDataFrame, output_dir: str | Path,
                    label_field: str = "class_name", date_field: str | None = "observed_at", year: int | None = None,
                    training: str | Path | None = None, training_exclusion_m: float = 0.0, window_px: int = 0,
                    label_mapping: dict | None = None, bootstrap: int = 1000, seed: int = 42,
                    group_field: str | None = None) -> dict:
    """Compare `classes.tif` (and probabilities, if present) in `prediction_dir` with field points."""
    pred_dir, out = Path(prediction_dir), Path(output_dir)
    if isinstance(window_px, bool) or not isinstance(window_px, int) or window_px < 0:
        raise ValueError("window_px must be a non-negative integer")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    out.mkdir(parents=True, exist_ok=True)
    legend = json.loads((pred_dir / "class_legend.json").read_text(encoding="utf8"))
    obs = gpd.read_file(observations) if not isinstance(observations, gpd.GeoDataFrame) else observations.copy()
    if label_field not in obs:
        raise ValueError(f"Observations need {label_field!r}")
    if obs.crs is None:
        raise ValueError("Observations need a CRS")
    log = {"input": int(len(obs))}
    obs = obs[obs.geometry.notna() & (obs.geometry.geom_type == "Point")]
    if year is not None:
        if not date_field or date_field not in obs:
            raise ValueError(f"Filtering by year needs a {date_field!r} column")
        dates = pd.to_datetime(obs[date_field], errors="coerce", utc=True)
        obs = obs[dates.dt.year == year]
        log["after_year_filter"] = int(len(obs))
    if obs[label_field].isna().any():
        raise ValueError(f"Observation label field {label_field!r} has missing labels")
    if label_mapping:
        obs = obs.assign(**{label_field: obs[label_field].astype(str).map(lambda v: label_mapping.get(v, v))})
    if (obs[label_field].isna() | obs[label_field].astype(str).str.strip().eq("")).any():
        raise ValueError(f"Observation label field {label_field!r} has missing or blank labels")
    if group_field is not None:
        if group_field not in obs:
            raise ValueError(f"Observations need grouping field {group_field!r}")
        if (obs[group_field].isna() | obs[group_field].astype(str).str.strip().eq("")).any():
            raise ValueError(f"Grouping field {group_field!r} has missing or blank identifiers")
    with rasterio.open(pred_dir / "classes.tif") as ds:
        obs = obs.to_crs(ds.crs)
        if training is not None:
            train = gpd.read_file(training).to_crs(ds.crs)
            zone = train.geometry.buffer(training_exclusion_m) if training_exclusion_m else train.geometry
            near = obs.geometry.apply(lambda p: bool(zone.sindex.query(p, predicate="intersects").size))
            log["excluded_near_training"] = int(near.sum())
            obs = obs[~near]
        prob_path = pred_dir / "probabilities.tif"
        prob = rasterio.open(prob_path) if prob_path.exists() else None
        records = []
        try:
            if prob is not None:
                check_grid(prob, ds, "validation probabilities")
                classes = list(prob.descriptions)
                if len(set(classes)) != len(classes) or set(classes) != set(legend.values()):
                    raise ValueError("Probability band descriptions must uniquely match the class legend")
            for _, rec in obs.iterrows():
                r, c = ds.index(rec.geometry.x, rec.geometry.y)
                if not (0 <= r < ds.height and 0 <= c < ds.width):
                    continue
                if window_px:
                    k, confidence = _window_vote(ds, r, c, window_px, return_share=True)
                else:
                    k, confidence = int(ds.read(1, window=Window(c, r, 1, 1))[0, 0]), None
                if k is None or k == 255 or str(k) not in legend:
                    continue
                item = {"observed": str(rec[label_field]), "predicted": legend[str(k)],
                        "x": rec.geometry.x, "y": rec.geometry.y, "decision_confidence": confidence}
                if group_field is not None:
                    item["bootstrap_group"] = str(rec[group_field])
                if prob is not None:
                    p = prob.read(window=Window(c, r, 1, 1))[:, 0, 0]
                    item["probabilities"] = p.tolist()
                records.append(item)
        finally:
            if prob is not None:
                prob.close()
    if not records:
        raise ValueError("No valid observations overlap the prediction")
    df = pd.DataFrame(records)
    log["evaluated"] = int(len(df))
    unseen = sorted(set(df.observed) - set(legend.values()))
    log["observed_classes_not_in_model"] = unseen
    result = {"selection": log, "accuracy": metrics.summary(df.observed, df.predicted),
              "validation_protocol": {
                  "decision": "window_majority" if window_px else "centre_pixel", "window_radius_px": window_px,
                  "confidence": "window_vote_share" if window_px else "predicted_class_probability",
                  "calibration": "centre_pixel_probability_argmax (separate from window majority)",
                  "accuracy_weighting": "per observation; grouping changes resampling only",
                  "group_field": group_field}}
    if bootstrap:
        groups = df.bootstrap_group.to_numpy() if group_field is not None else None
        if groups is not None and len(np.unique(groups)) < 2:
            raise ValueError("Cluster bootstrap needs at least two evaluated groups; use bootstrap=0 for descriptive accuracy only")
        result["bootstrap"] = metrics.bootstrap(df.observed, df.predicted, groups=groups, n_boot=bootstrap, seed=seed)
        result["bootstrap"].update({"resampling": "group" if group_field is not None else "point",
                                    "group_field": group_field})
    if "probabilities" in df:
        P = np.vstack(df.probabilities.to_numpy())
        known = df.observed.isin(classes).to_numpy()
        valid = np.isfinite(P).all(axis=1) & (P >= 0).all(axis=1) & (P <= 1).all(axis=1) & np.isclose(P.sum(axis=1), 1, atol=1e-4)
        log["invalid_centre_probabilities"] = int((~valid).sum())
        if (known & valid).any():
            result["calibration"] = metrics.calibration(P[known & valid], df.observed[known & valid], classes)
        df["max_probability"] = np.where(valid, P.max(axis=1), np.nan)
        if not window_px:
            chosen = np.array([classes.index(label) for label in df.predicted])
            df["decision_confidence"] = np.where(valid, P[np.arange(len(P)), chosen], np.nan)
        df = df.drop(columns="probabilities")
    usable = df.decision_confidence.notna()
    log["with_decision_confidence"] = int(usable.sum())
    if usable.any() and (window_px or "calibration" in result):
        result["reject_curve"] = metrics.reject_curve(df.loc[usable, "decision_confidence"],
                                                      (df.loc[usable, "observed"] == df.loc[usable, "predicted"]).to_numpy())
    result["note"] = ("Independent only if observations did not inform training-polygon delineation, "
                      "feature selection or hyperparameter tuning. Bootstrap assumes independent resampling units; "
                      "use group_field for repeated observations within parcels or spatial blocks. Grouping does "
                      "not correct a non-probability sampling design.")
    df.to_csv(out / "observations_evaluated.csv", index=False)
    acc = result["accuracy"]
    pd.DataFrame(acc["confusion_matrix"], index=acc["labels"], columns=acc["labels"]).to_csv(out / "confusion_matrix.csv")
    (out / "validation_report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf8")
    return result


def mapped_area(classes_tif: str | Path) -> dict:
    """Mapped area per class (map units², e.g. m²) for area-adjusted estimation."""
    pred_dir = Path(classes_tif).parent
    legend = json.loads((pred_dir / "class_legend.json").read_text(encoding="utf8"))
    counts = np.zeros(256, dtype=np.int64)
    with rasterio.open(classes_tif) as ds:
        pixel_area = abs(ds.transform.a * ds.transform.e)
        for _, w in ds.block_windows(1):
            counts += np.bincount(ds.read(1, window=w).ravel(), minlength=256)
    return {name: float(counts[int(k)] * pixel_area) for k, name in legend.items()}
