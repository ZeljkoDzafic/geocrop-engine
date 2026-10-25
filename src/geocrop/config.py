"""Experiment configuration: one YAML file describes a study; the engine hard-codes none of it.

Unknown keys are rejected so that typos never silently fall back to defaults.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import indices

DEFAULTS: dict[str, Any] = {
    "project": "geocrop-experiment",
    "seed": 42,
    "grid": {"crs": None, "resolution_m": 10, "reference": None},
    "cache": {"hash": "content"},
    "windows": {},
    "paths": {
        "s2_scenes": "data/interim/s2",
        "s1_scenes": "data/interim/s1",
        "composites": "data/processed/composites",
        "features": "data/features/stack.tif",
        "training": "data/reference/training.gpkg",
        "parcels": "data/reference/parcels.gpkg",
        "observations": "data/reference/observations.gpkg",
        "output": "outputs/run",
    },
    "compositing": {
        "statistic": "median",
        "scl_invalid": [0, 1, 3, 8, 9, 10, 11],
        "cloud_probability_max": None,
        "mask_dilation_px": 0,
        "min_valid_observations": 1,
        "temporal_screening": False,
    },
    "s1": {"separate_orbits": False, "statistic": "mean"},
    "features": {
        "s2_bands": list(indices.S2_BANDS),
        "s2_indices": list(indices.DEFAULT_S2_INDICES),
        "s1": list(indices.DEFAULT_S1_FEATURES),
        "deltas": ["NDVI", "NDRE", "NDMI"],
        "seasonal": ["NDVI"],
    },
    "timeseries": {
        "enabled": False,
        "start": None,
        "end": None,
        "s2_indices": ["NDVI", "NDRE", "NDMI"],
        "s1_indices": ["VHVV_dB"],
        "step_days": 5,
        "output_step_days": 15,
        "lambda": 50.0,
        "envelope_iterations": 0,
        "harmonic_order": 2,
        "min_observed_fraction": 0,
        "max_gap_days": None,
    },
    "cbm": {
        "enabled": False,
        "index": "NDVI",
        "calendar": None,
        "min_coverage": 0.5,
    },
    "classes": {"label_field": "class_name", "id_field": "polygon_id", "targets": []},
    "model": {
        "algorithm": "rf",
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "max_samples_per_polygon": 250,
        "cv_folds": 5,
        "spatial_block_m": 5000,
        "spatial_buffer_m": 0,
        "n_jobs": -1,
        "max_depth": None,
        "learning_rate": None,
        "tune_grid": None,
        "calibrate": True,
        "calibration_mode": "oof",
        "calibration_fraction": 0.2,
        "holdout_fraction": 0.0,
        "inner_splits": 3,
    },
    "experiments": {},
    "parcels": {
        "id_field": "parcel_id",
        "declared_field": "declared_crop",
        "strategy": "mean_probability",
        "interior_buffer_px": 1,
        "min_valid_pixels": 4,
        "min_confidence": None,
        "min_majority_share": None,
        "min_margin": None,
        "crop_mapping": None,
        "conformal_alpha": 0.1,
        "min_aoa_fraction": 0.5,
    },
    "validation": {
        "label_field": "class_name",
        "date_field": "observed_at",
        "year": None,
        "training_exclusion_m": 0,
        "window_px": 1,
        "bootstrap": 1000,
    },
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            # Free-form sections (windows, experiments) accept any key.
            if path in ("windows", "experiments"):
                out[key] = value
                continue
            raise ConfigError(f"Unknown configuration key: {where}")
        if isinstance(base[key], dict) and base[key] and isinstance(value, dict):
            out[key] = _merge(base[key], value, where)
        else:
            out[key] = value
    return out


def parse_windows(raw: dict) -> dict[str, tuple[date, date]]:
    windows = {}
    for name, spec in (raw or {}).items():
        try:
            start, end = date.fromisoformat(str(spec["start"])), date.fromisoformat(str(spec["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Window {name} needs ISO start/end dates") from exc
        if end < start:
            raise ConfigError(f"Window {name} ends before it starts")
        windows[str(name)] = (start, end)
    ordered = sorted(windows.items(), key=lambda kv: kv[1][0])
    for (a, (_, a_end)), (b, (b_start, _)) in zip(ordered, ordered[1:]):
        if b_start <= a_end:
            raise ConfigError(f"Windows {a} and {b} overlap")
    return dict(ordered)


def validate(cfg: dict) -> dict:
    parse_windows(cfg["windows"])
    t = cfg["timeseries"]
    for key in ("step_days", "output_step_days"):
        if isinstance(t[key], bool) or not isinstance(t[key], int) or t[key] < 1:
            raise ConfigError(f"timeseries.{key} must be a positive integer")
    if t["output_step_days"] % t["step_days"]:
        raise ConfigError("timeseries.output_step_days must be a multiple of step_days")
    for key, upper in (("min_observed_fraction", 1), ("max_gap_days", math.inf)):
        value = t[key]
        if value is None and key == "max_gap_days":
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not 0 <= value <= upper):
            raise ConfigError(f"Invalid timeseries.{key}")
    buffer = cfg["model"]["spatial_buffer_m"]
    if isinstance(buffer, bool) or not isinstance(buffer, (int, float)) or not math.isfinite(buffer) or buffer < 0:
        raise ConfigError("model.spatial_buffer_m must be a finite nonnegative distance in metres")
    h = cfg["model"]["holdout_fraction"]
    if isinstance(h, bool) or not isinstance(h, (int, float)) or not 0 <= h < 0.5:
        raise ConfigError("model.holdout_fraction must be in [0, 0.5)")
    if cfg["model"]["calibration_mode"] not in ("oof", "holdout"):
        raise ConfigError("model.calibration_mode must be oof or holdout")
    fraction = cfg["model"]["calibration_fraction"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ConfigError("model.calibration_fraction must be a finite fraction between 0 and 1")
    for name in cfg["features"]["s2_indices"] + cfg["features"]["s1"]:
        indices.get(name)
    if cfg["compositing"]["statistic"] not in ("median", "mean") and not str(
            cfg["compositing"]["statistic"]).startswith("p"):
        raise ConfigError("compositing.statistic must be median, mean or pNN (percentile)")
    if cfg["parcels"]["strategy"] not in ("majority", "mean_probability", "combined"):
        raise ConfigError("parcels.strategy must be majority, mean_probability or combined")
    for name, spec in cfg["experiments"].items():
        if not isinstance(spec, dict) or not ({"groups", "features", "periods"} & set(spec)):
            raise ConfigError(f"Experiment {name} needs groups, features and/or periods")
    return cfg


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    raw = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
    cfg = _merge(DEFAULTS, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    return validate(cfg)


def fingerprint(cfg: dict) -> str:
    """Stable hash of a resolved configuration, recorded in every run manifest."""
    body = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(body).hexdigest()[:16]
