"""Parcel-level aggregation, decision strategies and declaration comparison.

For each parcel j, pixels whose centres fall inside the parcel after an inward buffer of
`interior_buffer_px` pixels (to exclude mixed boundary pixels) are summarised:

    n_valid                valid pixels used
    majority_class/share   argmax_k  (1/n) sum_i 1[argmax p_i = k]      (strategy A, "majority")
    mean_class/confidence  argmax_k  (1/n) sum_i p_ik                   (strategy B, "mean_probability")
    margin, entropy        top-2 margin and normalised entropy of the mean probability vector
    prob_std               within-parcel std. dev. of the chosen class probability (homogeneity)
    p_<class>, f_<class>   mean probability and pixel fraction per class (parcel-model descriptors)
    decision_confidence, decision_margin   probability and signed margin of the strategy's selected class
    area_m2, shape_index   geometry descriptors (perimeter / (2 sqrt(pi A)), 1 = circle)

Strategy C ("combined") accepts a class only when A and B agree and all configured thresholds
pass; otherwise the parcel is sent to review. Thresholds are *not* built in: choose them on
development/calibration data and freeze them before independent validation.

Declaration status (never a legal finding):
    agreement | discrepancy | uncertain | insufficient_resolution | not_declared
"""
from __future__ import annotations

import json
import math
from collections import Counter
from contextlib import ExitStack
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


def _decision_evidence(row: dict, cls: str | None) -> tuple[float | None, float | None]:
    """Probability and signed margin for the selected class, leaving mean-winner descriptors intact.

    A majority winner can differ from the mean-probability winner. In that case its margin is
    measured against the strongest competing mean probability and is negative. Missing candidate
    probabilities cannot justify a confidence or margin threshold.
    """
    if cls is None:
        return None, None
    if cls == row.get("mean_class"):
        return row.get("confidence"), row.get("margin")
    confidence = row.get(f"p_{cls}")
    others = [v for k, v in row.items() if k.startswith("p_") and k != f"p_{cls}"]
    margin = confidence - max(others) if confidence is not None and others else None
    return confidence, margin


def decision_details(row: dict, strategy: str, min_valid_pixels: int, min_confidence=None, min_majority_share=None,
                     min_margin=None) -> dict:
    """Return the selected class, decision and every failed decision condition.

    `min_majority_share` always constrains the largest pixel-vote share; `min_confidence` and
    `min_margin` constrain the selected class's mean probability and signed probability margin.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}; use one of {STRATEGIES}")
    if isinstance(min_valid_pixels, bool) or not isinstance(min_valid_pixels, (int, np.integer)) or min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be a positive integer")
    for name, threshold, lower in (("min_confidence", min_confidence, 0),
                                    ("min_majority_share", min_majority_share, 0), ("min_margin", min_margin, -1)):
        if threshold is not None and (not np.isfinite(threshold) or not lower <= threshold <= 1):
            raise ValueError(f"{name} must be finite and in [{lower}, 1]")
    if row.get("n_valid", 0) < min_valid_pixels:
        return {"predicted_class": None, "decision": "insufficient_resolution", "reasons": ["insufficient_valid_pixels"]}
    if strategy == "majority":
        cls = row["majority_class"]
    elif strategy == "mean_probability":
        cls = row["mean_class"]
    elif strategy == "combined":
        cls = row["mean_class"] if row["strategies_agree"] else None
    confidence, margin = _decision_evidence(row, cls)
    reasons = ["strategies_disagree"] if cls is None else []
    checks = [("confidence", min_confidence, confidence), ("majority_share", min_majority_share, row["majority_share"]),
              ("margin", min_margin, margin)]
    for name, threshold, value in checks:
        if threshold is not None:
            if value is None or not np.isfinite(value):
                reasons.append(f"missing_{name}")
            elif value < threshold:
                reasons.append(f"low_{name}")
    return {"predicted_class": cls, "decision": "review" if reasons else "accepted", "reasons": reasons}


def decide(row: dict, strategy: str, min_valid_pixels: int, min_confidence=None, min_majority_share=None,
           min_margin=None) -> tuple[str | None, str]:
    """Backward-compatible class/decision pair; use decision_details for reason codes."""
    detail = decision_details(row, strategy, min_valid_pixels, min_confidence, min_majority_share, min_margin)
    return detail["predicted_class"], detail["decision"]


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
              min_aoa_fraction: float = 0.5, markers: pd.DataFrame | None = None, calendar: dict | None = None,
              marker_rules: dict | None = None, temporal_support: pd.DataFrame | None = None,
              temporal_rules: dict | None = None) -> dict:
    """Summarise the probability raster per parcel and write GeoPackage + CSV + summary JSON.

    crop_mapping   maps declaration codes to model class names (e.g. {"1101": "soybean"}).
    conformal      `conformal.json` from training (or its dict): adds the prediction set at level 1-alpha and
                   `cp_status` (agreement: set == {declared}; discrepancy: declared not in a non-empty set;
                   otherwise uncertain).
    aoa            `aoa.tif` from prediction: parcels with < min_aoa_fraction of pixels inside the area of
                   applicability are sent to review (the model extrapolates there).
    markers        table from `cbm.markers`; with `calendar` adds `calendar_consistent`.
    The `traffic_light` column combines all available evidence (see `geocrop.cbm.traffic_light`).
    marker_rules   enabled `cbm.rules` (see `geocrop.cbm_markers.RULE_DEFAULTS`): copies the event-marker
                   columns of `markers` (from `cbm.event_markers`), lets them escalate the traffic light and
                   adds `traffic_light_reasons`. Without it the output is unchanged.
    temporal_support  per-parcel table from `temporal_quality.parcel_support`: its ``tq_*`` columns are
                   copied into the output.
    temporal_rules enabled `cbm.temporal_support` (`geocrop.temporal_quality.RULE_DEFAULTS`): weakly
                   supported parcels are routed to review with reason codes (needs `temporal_support`).
    """
    from . import cbm

    if not np.isfinite(min_aoa_fraction) or not 0 <= min_aoa_fraction <= 1:
        raise ValueError("min_aoa_fraction must be finite and in [0, 1]")
    cp = None
    calibration_model = None
    if conformal is not None:
        cp = conformal if isinstance(conformal, dict) else json.loads(Path(conformal).read_text(encoding="utf8"))
        calibration_model = cp.get("model_sha256")
        if str(alpha) not in cp["thresholds"]:
            raise ValueError(f"alpha {alpha} not calibrated; available {list(cp['thresholds'])}")
        cp = cp["thresholds"][str(alpha)]
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf:
        raise ValueError(f"Parcels need {id_field!r}")
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    rows = []
    with ExitStack() as resources:
        ds = resources.enter_context(rasterio.open(probabilities))
        if ds.crs is None or not ds.crs.is_projected or not math.isclose(ds.crs.linear_units_factor[1], 1):
            raise ValueError("Parcel aggregation requires a projected CRS in metres")
        aoa_ds = resources.enter_context(rasterio.open(aoa)) if aoa else None
        if aoa_ds is not None and (aoa_ds.count < 2 or aoa_ds.crs != ds.crs or aoa_ds.shape != ds.shape
                                   or not aoa_ds.transform.almost_equals(ds.transform)):
            raise ValueError("AOA raster needs two bands and the same CRS, shape and transform as probabilities")
        classes = list(ds.descriptions)
        if not all(classes) or len(set(classes)) != len(classes):
            raise ValueError("Probability bands need unique class-name descriptions")
        if cp is not None:
            if set(cp) != set(classes):
                raise ValueError("Calibration threshold classes must exactly match probability-band classes")
            if calibration_model and ds.tags().get("model_sha256") != calibration_model:
                raise ValueError("Calibration model_sha256 does not match the probability raster; "
                                 "regenerate probabilities and thresholds from the same saved model")
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
            warnings = ["interior_buffer_fallback"] if interior_buffer_px and not buffered else []
            win = polygon_window(core, ds)
            if win is not None:
                cube = ds.read(window=win)
                inside = geometry_mask([core], out_shape=cube.shape[1:], transform=ds.window_transform(win), invert=True)
                prob_valid = inside & valid_mask(cube, ds.nodata)
                z = cube[:, prob_valid]
                if z.shape[1] > 0:
                    if (np.any(z < -1e-6) or np.any(z > 1 + 1e-6)
                            or not np.allclose(z.sum(axis=0), 1, atol=1e-4, rtol=1e-5)):
                        raise ValueError("Valid probability pixels must lie in [0, 1] and sum to one")
                    row.update(_summarise(z, classes))
                    if cp is not None:
                        members = [c for c in classes if 1 - row[f"p_{c}"] <= cp.get(c, 1.0)]
                        row["prediction_set"] = ";".join(members)
                        row["set_size"] = len(members)
                if aoa_ds is not None:                      # aoa.tif shares the probability grid
                    a = aoa_ds.read(2, window=win)
                    known = prob_valid & np.isfinite(a) & np.isin(a, [0, 1])
                    if aoa_ds.nodata is not None:
                        known &= a != aoa_ds.nodata
                    n_valid = int(prob_valid.sum())
                    row["n_aoa_valid"] = int(known.sum())
                    row["aoa_coverage"] = float(known.sum() / n_valid) if n_valid else None
                    # Unknown AOA pixels cannot contribute evidence that a parcel is inside.
                    row["aoa_fraction"] = float((a[known] == 1).sum() / n_valid) if known.any() and n_valid else None
                    if n_valid and known.sum() < n_valid:
                        warnings.append("aoa_partial_support")
            detail = decision_details(row, strategy, min_valid_pixels, min_confidence, min_majority_share, min_margin)
            cls, decision, reasons = detail["predicted_class"], detail["decision"], detail["reasons"]
            row["decision_confidence"], row["decision_margin"] = _decision_evidence(row, cls)
            if aoa_ds is not None and decision != "insufficient_resolution":
                if row.get("aoa_fraction") is None:
                    reasons.append("aoa_missing_support")
                elif row["aoa_fraction"] < min_aoa_fraction:
                    reasons.append("aoa_below_threshold")
                if reasons:
                    decision = "review"
            row["decision_reasons"] = ";".join(reasons)
            row["evidence_warnings"] = ";".join(warnings)
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
    table = pd.DataFrame(rows)
    if markers is not None:
        ids = gdf[id_field].astype(str).reset_index(drop=True)
        m = markers.assign(parcel_id=markers.parcel_id.astype(str)).set_index("parcel_id").reindex(ids).reset_index(drop=True)
        for col in ("peak_doy", "harvest_doy", "bare_soil_days", "clear_observations"):
            if col in m or marker_rules is None:
                table[col] = m[col].to_numpy()
        if calendar:
            table["calendar_consistent"] = cbm.calendar_consistency(m, table["declared_class"], calendar).to_numpy()
    rules_on = marker_rules is not None and cbm.cbm_markers.resolve_rules(marker_rules)["enabled"]
    tq_on = temporal_rules is not None and cbm.temporal_quality.resolve(temporal_rules)["enabled"]
    if temporal_support is not None:
        ids = gdf[id_field].astype(str).reset_index(drop=True)
        table = cbm.temporal_quality.attach(table.assign(_pid=ids.to_numpy()), temporal_support, "_pid")
        table = table.drop(columns="_pid")
    if rules_on or tq_on:
        if rules_on and markers is not None:
            for col in cbm.cbm_markers.EVENT_COLUMNS:
                if col in m:
                    table[col] = m[col].to_numpy()
        details = cbm.traffic_light_details(table, marker_rules, temporal_rules)
        table["traffic_light"] = details["traffic_light"].to_numpy()
        table["traffic_light_reasons"] = details["traffic_light_reasons"].to_numpy()
    else:
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
        "calibration_model_sha256": calibration_model,
        "min_aoa_fraction": min_aoa_fraction,
        "decision_reason_counts": dict(sorted(Counter(code for codes in table["decision_reasons"]
                                                       for code in codes.split(";") if code).items())),
        "evidence_warning_counts": dict(sorted(Counter(code for codes in table["evidence_warnings"]
                                                        for code in codes.split(";") if code).items())),
        "status_by_declared_class": (result.groupby("declared_class")["declaration_status"].value_counts()
                                     .unstack(fill_value=0).to_dict(orient="index")
                                     if result["declared_class"].notna().any() else {}),
        "note": "Review flags support inspection prioritisation; they are not findings of non-compliance.",
    }
    if rules_on or tq_on:
        summary["traffic_light_reason_counts"] = dict(sorted(Counter(
            code for codes in table["traffic_light_reasons"] for code in codes.split(";") if code).items()))
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
