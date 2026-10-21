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


def _window_vote(ds, row, col, radius, nodata=255):
    win = Window(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1)
    block = ds.read(1, window=win, boundless=True, fill_value=nodata)
    vals = block[block != nodata]
    if vals.size == 0:
        return None
    counts = np.bincount(vals.astype(int))
    return int(counts.argmax())


def evaluate_points(prediction_dir: str | Path, observations: str | Path | gpd.GeoDataFrame, output_dir: str | Path,
                    label_field: str = "class_name", date_field: str | None = "observed_at", year: int | None = None,
                    training: str | Path | None = None, training_exclusion_m: float = 0.0, window_px: int = 0,
                    label_mapping: dict | None = None, bootstrap: int = 1000, seed: int = 42) -> dict:
    """Compare `classes.tif` (and probabilities, if present) in `prediction_dir` with field points."""
    pred_dir, out = Path(prediction_dir), Path(output_dir)
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
    if label_mapping:
        obs = obs.assign(**{label_field: obs[label_field].astype(str).map(lambda v: label_mapping.get(v, v))})
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
            for rec in obs.itertuples():
                r, c = ds.index(rec.geometry.x, rec.geometry.y)
                if not (0 <= r < ds.height and 0 <= c < ds.width):
                    continue
                k = _window_vote(ds, r, c, window_px) if window_px else int(ds.read(1, window=Window(c, r, 1, 1))[0, 0])
                if k is None or k == 255 or str(k) not in legend:
                    continue
                item = {"observed": str(getattr(rec, label_field)), "predicted": legend[str(k)],
                        "x": rec.geometry.x, "y": rec.geometry.y}
                if prob is not None:
                    p = prob.read(window=Window(c, r, 1, 1))[:, 0, 0]
                    item["probabilities"] = p.tolist()
                records.append(item)
        finally:
            if prob is not None:
                classes = list(prob.descriptions)
                prob.close()
    if not records:
        raise ValueError("No valid observations overlap the prediction")
    df = pd.DataFrame(records)
    log["evaluated"] = int(len(df))
    unseen = sorted(set(df.observed) - set(legend.values()))
    log["observed_classes_not_in_model"] = unseen
    result = {"selection": log, "accuracy": metrics.summary(df.observed, df.predicted)}
    if bootstrap:
        result["bootstrap"] = metrics.bootstrap(df.observed, df.predicted, n_boot=bootstrap, seed=seed)
    if "probabilities" in df:
        P = np.vstack(df.probabilities.to_numpy())
        known = df.observed.isin(classes).to_numpy()
        if known.any():
            result["calibration"] = metrics.calibration(P[known], df.observed[known], classes)
            conf = P.max(axis=1)
            result["reject_curve"] = metrics.reject_curve(conf, (df.observed == df.predicted).to_numpy())
        df["max_probability"] = P.max(axis=1)
        df = df.drop(columns="probabilities")
    result["note"] = ("Independent only if observations did not inform training-polygon delineation, "
                      "feature selection or hyperparameter tuning.")
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
