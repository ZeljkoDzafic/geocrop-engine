"""Parcel-level aggregation, decision strategies and declaration comparison.

For each parcel j, pixels whose centres fall inside the parcel after an inward buffer of
`interior_buffer_px` pixels (to exclude mixed boundary pixels) are summarised:

    n_valid                valid pixels used
    majority_class/share   argmax_k  (1/n) sum_i 1[argmax p_i = k]      (strategy A, "majority")
    mean_class/confidence  argmax_k  (1/n) sum_i p_ik                   (strategy B, "mean_probability")
    margin, entropy        top-2 margin and normalised entropy of the mean probability vector
    prob_std               within-parcel std. dev. of the chosen class probability (homogeneity)
    p_<class>, f_<class>   mean probability and pixel fraction per class (parcel-model descriptors)
    area_m2, shape_index   geometry descriptors (perimeter / (2 sqrt(pi A)), 1 = circle)

Strategy C ("combined") accepts a class only when A and B agree and all configured thresholds
pass; otherwise the parcel is sent to review. Thresholds are *not* built in: they must be chosen
on validation data (see `metrics.reject_curve`).

Declaration status (never a legal finding):
    agreement | discrepancy | uncertain | insufficient_resolution | not_declared
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from .raster import valid_mask
from .samples import polygon_window

STRATEGIES = ("majority", "mean_probability", "combined")


def _summarise(z: np.ndarray, classes: list[str]) -> dict:
    """z: (k, n) per-pixel class probabilities of one parcel."""
    k, n = z.shape
    mean = z.mean(axis=1)
    votes = np.bincount(z.argmax(axis=0), minlength=k) / n
    order = np.sort(mean)[::-1]
    best = int(mean.argmax())
    ent = -np.sum(np.where(mean > 0, mean * np.log(np.maximum(mean, 1e-12)), 0))
    row = {
        "n_valid": int(n),
        "majority_class": classes[int(votes.argmax())], "majority_share": float(votes.max()),
        "mean_class": classes[best], "confidence": float(mean[best]),
        "median_probability": float(np.median(z[best])), "prob_std": float(z[best].std()),
        "margin": float(order[0] - order[1]) if k > 1 else 1.0,
        "entropy": float(ent / math.log(k)) if k > 1 else 0.0,
    }
    row["strategies_agree"] = row["majority_class"] == row["mean_class"]
    for i, c in enumerate(classes):
        row[f"p_{c}"] = float(mean[i])
        row[f"f_{c}"] = float(votes[i])
    return row


def decide(row: dict, strategy: str, min_valid_pixels: int, min_confidence=None, min_majority_share=None,
           min_margin=None) -> tuple[str | None, str]:
    """Return (predicted_class, decision) where decision is accepted | review | insufficient_resolution."""
    if row.get("n_valid", 0) < min_valid_pixels:
        return None, "insufficient_resolution"
    if strategy == "majority":
        cls = row["majority_class"]
    elif strategy == "mean_probability":
        cls = row["mean_class"]
    elif strategy == "combined":
        cls = row["mean_class"] if row["strategies_agree"] else None
    else:
        raise ValueError(f"Unknown strategy {strategy}; use one of {STRATEGIES}")
    checks = [(min_confidence, row["confidence"]), (min_majority_share, row["majority_share"]), (min_margin, row["margin"])]
    if cls is None or any(t is not None and v < t for t, v in checks):
        return cls, "review"
    return cls, "accepted"


def declaration_status(declared, predicted, decision) -> str:
    if declared is None or (isinstance(declared, float) and np.isnan(declared)) or str(declared) == "":
        return "not_declared"
    if decision == "insufficient_resolution":
        return "insufficient_resolution"
    if decision == "review" or predicted is None:
        return "uncertain"
    return "agreement" if str(declared) == str(predicted) else "discrepancy"


def aggregate(probabilities: str | Path, parcels: str | Path | gpd.GeoDataFrame, output: str | Path,
              id_field: str = "parcel_id", declared_field: str | None = "declared_crop",
              strategy: str = "mean_probability", interior_buffer_px: float = 1, min_valid_pixels: int = 4,
              min_confidence: float | None = None, min_majority_share: float | None = None,
              min_margin: float | None = None, crop_mapping: dict | None = None,
              conformal: str | Path | dict | None = None, alpha: float = 0.1, aoa: str | Path | None = None,
              min_aoa_fraction: float = 0.5, markers: pd.DataFrame | None = None, calendar: dict | None = None) -> dict:
    """Summarise the probability raster per parcel and write GeoPackage + CSV + summary JSON.

    crop_mapping   maps declaration codes to model class names (e.g. {"1101": "soybean"}).
    conformal      `conformal.json` from training (or its dict): adds the prediction set at level 1-alpha and
                   `cp_status` (agreement: set == {declared}; discrepancy: declared not in a non-empty set;
                   otherwise uncertain).
    aoa            `aoa.tif` from prediction: parcels with < min_aoa_fraction of pixels inside the area of
                   applicability are sent to review (the model extrapolates there).
    markers        table from `cbm.markers`; with `calendar` adds `calendar_consistent`.
    The `traffic_light` column combines all available evidence (see `geocrop.cbm.traffic_light`).
    """
    from . import cbm

    cp = None
    if conformal is not None:
        cp = conformal if isinstance(conformal, dict) else json.loads(Path(conformal).read_text(encoding="utf8"))
        if str(alpha) not in cp["thresholds"]:
            raise ValueError(f"alpha {alpha} not calibrated; available {list(cp['thresholds'])}")
        cp = cp["thresholds"][str(alpha)]
    aoa_ds = rasterio.open(aoa) if aoa else None
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf:
        raise ValueError(f"Parcels need {id_field!r}")
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    rows = []
    with rasterio.open(probabilities) as ds:
        classes = list(ds.descriptions)
        if not all(classes):
            raise ValueError("Probability bands need class-name descriptions")
        gdf = gdf.to_crs(ds.crs)
        pixel = abs(ds.transform.a)
        for parcel in gdf.itertuples():
            geom = parcel.geometry
            core = geom.buffer(-interior_buffer_px * pixel) if interior_buffer_px else geom
            buffered = bool(interior_buffer_px) and not core.is_empty
            if core.is_empty:
                core = geom   # narrow parcel: fall back to full geometry, flagged below
            row = {"area_m2": float(geom.area),
                   "shape_index": float(geom.length / (2 * math.sqrt(math.pi * geom.area))) if geom.area > 0 else None,
                   "interior_buffer_applied": buffered, "n_valid": 0}
            win = polygon_window(core, ds)
            if win is not None:
                cube = ds.read(window=win)
                inside = geometry_mask([core], out_shape=cube.shape[1:], transform=ds.window_transform(win), invert=True)
                z = cube[:, inside & valid_mask(cube, ds.nodata)]
                if z.shape[1] > 0:
                    row.update(_summarise(z, classes))
                    if cp is not None:
                        members = [c for c in classes if 1 - row[f"p_{c}"] <= cp.get(c, 1.0)]
                        row["prediction_set"] = ";".join(members)
                        row["set_size"] = len(members)
                if aoa_ds is not None:                      # aoa.tif shares the probability grid
                    a = aoa_ds.read(2, window=win)
                    ins = inside & (a != aoa_ds.nodata)
                    row["aoa_fraction"] = float(a[ins].mean()) if ins.any() else None
            cls, decision = decide(row, strategy, min_valid_pixels, min_confidence, min_majority_share, min_margin)
            if decision == "accepted" and row.get("aoa_fraction") is not None and row["aoa_fraction"] < min_aoa_fraction:
                decision = "review"
            declared = getattr(parcel, declared_field) if declared_field and declared_field in gdf else None
            if declared is not None and crop_mapping:
                declared = crop_mapping.get(str(declared), declared)
            row.update({"predicted_class": cls, "decision": decision, "strategy": strategy,
                        "declared_class": None if declared is None else str(declared),
                        "declaration_status": declaration_status(declared, cls, decision)})
            if cp is not None and row["declaration_status"] not in ("not_declared", "insufficient_resolution"):
                members = row.get("prediction_set", "").split(";") if row.get("prediction_set") else []
                # empty set = atypical object (no class conforms) -> uncertain, never a discrepancy
                row["cp_status"] = ("uncertain" if not members else
                                    "discrepancy" if str(declared) not in members else
                                    "agreement" if members == [str(declared)] else "uncertain")
            rows.append(row)
    if aoa_ds is not None:
        aoa_ds.close()
    table = pd.DataFrame(rows)
    if markers is not None:
        ids = gdf[id_field].astype(str).reset_index(drop=True)
        m = markers.assign(parcel_id=markers.parcel_id.astype(str)).set_index("parcel_id").reindex(ids).reset_index(drop=True)
        for col in ("peak_doy", "harvest_doy", "bare_soil_days", "clear_observations"):
            table[col] = m[col].to_numpy()
        if calendar:
            table["calendar_consistent"] = cbm.calendar_consistency(m, table["declared_class"], calendar).to_numpy()
    table["traffic_light"] = cbm.traffic_light(table)
    result = gdf.drop(columns=[c for c in table.columns if c in gdf.columns]).reset_index(drop=True)
    result = gpd.GeoDataFrame(pd.concat([result, table], axis=1), geometry=gdf.geometry.name, crs=gdf.crs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    result.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
    result.drop(columns=result.geometry.name).to_csv(out.with_suffix(".csv"), index=False)
    summary = {
        "parcels": int(len(result)), "strategy": strategy, "interior_buffer_px": interior_buffer_px,
        "thresholds": {"min_valid_pixels": min_valid_pixels, "min_confidence": min_confidence,
                       "min_majority_share": min_majority_share, "min_margin": min_margin},
        "decision_counts": result["decision"].value_counts().to_dict(),
        "declaration_status_counts": result["declaration_status"].value_counts().to_dict(),
        "traffic_light_counts": result["traffic_light"].value_counts().to_dict(),
        "conformal_alpha": alpha if cp is not None else None,
        "status_by_declared_class": (result.groupby("declared_class")["declaration_status"].value_counts()
                                     .unstack(fill_value=0).to_dict(orient="index")
                                     if result["declared_class"].notna().any() else {}),
        "note": "Review flags support inspection prioritisation; they are not findings of non-compliance.",
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    return summary


# --------------------------------------------------------------------------- parcel-level model

def descriptor_columns(frame: pd.DataFrame) -> list[str]:
    base = ["confidence", "majority_share", "margin", "entropy", "prob_std", "median_probability", "area_m2",
            "shape_index", "n_valid"]
    return [c for c in frame.columns if c.startswith(("p_", "f_"))] + [c for c in base if c in frame.columns]


def train_parcel_model(parcels: str | Path, reference_field: str, output_dir: str | Path, n_splits: int = 5,
                       spatial_block_m: float | None = 5000, seed: int = 42, params: dict | None = None) -> dict:
    """Second-stage classifier on aggregated parcel descriptors (strategy 2 in the study design).

    Labels must come from independently verified parcels, never from the declarations being checked.
    Uses the same spatially blocked CV as the pixel model and reports it next to the
    direct-aggregation baselines (majority, mean probability) on the same folds.
    """
    from joblib import dump

    from . import metrics
    from .model import cross_validate, make_rf

    gdf = gpd.read_file(parcels)
    labelled = gdf[gdf[reference_field].notna() & gdf["n_valid"].fillna(0).gt(0)].copy()
    if labelled.empty:
        raise ValueError("No labelled parcels with valid pixels")
    cols = descriptor_columns(labelled)
    X = labelled[cols].fillna(0).to_numpy(dtype="float32")
    y = labelled[reference_field].astype(str).to_numpy()
    ids = labelled.index.astype(str).to_numpy()
    if spatial_block_m:
        c = labelled.geometry.centroid
        groups = (np.floor(c.x / spatial_block_m).astype(int).astype(str) + ":" +
                  np.floor(c.y / spatial_block_m).astype(int).astype(str)).to_numpy()
    else:
        groups = ids
    params = params or {"n_estimators": 300, "class_weight": "balanced_subsample", "n_jobs": -1}
    cv = cross_validate(X, y, ids, groups, params, n_splits, seed)
    for k in ("_oof", "_oof_proba", "_classes"):
        cv.pop(k)
    # Direct aggregation needs no fitting, so it is evaluated on all labelled parcels.
    baselines = {name: metrics.summary(y, labelled[col].astype(str).to_numpy())
                 for name, col in (("majority", "majority_class"), ("mean_probability", "mean_class"))}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = make_rf(params, seed).fit(X, y)
    dump({"model": model, "features": cols}, out / "parcel_model.joblib", compress=3)
    report = {"n_parcels": int(len(y)), "features": cols, "cross_validation": cv, "baselines": baselines}
    (out / "parcel_model_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    return report
