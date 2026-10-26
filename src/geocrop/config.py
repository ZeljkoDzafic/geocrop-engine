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

from . import cbm_markers, indices

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
        "embeddings": None,     # optional foundation-model feature group, see EMBEDDING_DEFAULTS
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
        # Event/homogeneity markers (geocrop.cbm_markers.EVENT_DEFAULTS) and traffic-light marker rules
        # (RULE_DEFAULTS); rules are off by default so existing outputs are unchanged.
        "events": copy.deepcopy(cbm_markers.EVENT_DEFAULTS),
        "rules": copy.deepcopy(cbm_markers.RULE_DEFAULTS),
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
    "cv": {
        "strategy": "spatial_blocks",   # spatial_blocks | region | year | region_year (transfer validation)
        "region_field": None,           # polygon attribute copied into sample region ids
        "year_field": None,             # polygon attribute copied into sample year ids
        "year": None,                   # constant year id for a single-season sample set
    },
    "predict": {
        "smoothing": {"enabled": False, "window": 7, "smoothness": 10.0},
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
    # Optional openEO back-end (geocrop openeo-extract / openeo-graph); local processing stays the default.
    "openeo": {
        "url": "https://openeo.dataspace.copernicus.eu",
        "products": ["s2_composites"],
        "s2_collection": "SENTINEL2_L2A",
        "s1_collection": "SENTINEL1_GRD",
        "s1_coefficient": "sigma0-ellipsoid",
        "s1_card4l": False,
        "max_cloud_cover": None,
        "signature_indices": ["NDVI", "BSI"],
        "start": None,
        "end": None,
        "job_options": None,
        "work_dir": "data/openeo",
    },
    # Where Sentinel-2 scenes come from (docs/datacube.md). `geocrop cube load --config` reads this section;
    # the pipeline itself always reads scene folders from paths.s2_scenes, whatever filled them.
    "data": {
        "source": "local",          # local (SAFE / stac-import scene folders) | odc-stac
        "endpoint": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "max_cloud": 80,
        "min_clear_fraction": 0.2,
    },
    # Machine protection: free-space guard before heavy stages (see docs/preflight.md).
    "resources": {
        "disk_guard": True,
        "min_free_gb": 5.0,
        "safety_factor": 1.25,
        "compression_ratio": 0.8,
    },
    # Thresholds for `geocrop preflight` (warnings only; hard blockers are fixed by the pipeline's own rules).
    "preflight": {
        "min_s2_scenes_per_window": 3,
        "min_s1_scenes_per_window": 2,
        "max_unobserved_fraction": 0.05,
        "coverage_max_px": 512,
        "inventories": [],
    },
}


class ConfigError(ValueError):
    pass


#: Keys of the optional ``features.embeddings`` mapping (docs/embeddings.md).
EMBEDDING_DEFAULTS: dict[str, Any] = {
    "source": None,            # a feature provider (alphaearth | tessera | presto | galileo | anysat | plugin) or a list
    "year": None,
    "dir": None,               # default: <paths.features parent>/embeddings
    "resampling": "nearest",   # pre-computed sources: nearest | bilinear
    "mirror": "source.coop",   # AlphaEarth: source.coop | gcs
    "index": None,             # AlphaEarth: local/remote aef_index.parquet override
    "monthly": None,           # Presto: monthly composites root (<YYYY-MM>/B02.tif ..., VV.tif, VH.tif)
    "start": None,             # Presto: first month YYYY-MM (default <year>-01)
    "months": 12,              # Presto: number of monthly timesteps (1..24)
    "checkpoint": None,        # Presto: local checkpoint (default: pinned official download)
    "parcels": False,          # Presto: also write parcel-level embeddings from paths.parcels
    "attach": True,            # append the embeddings raster to the feature stack
    "namespace": None,         # feature names emb_<source>_NN (true) / legacy emb_NN (false); default: new providers
                               # and lists of sources are namespaced, alphaearth/tessera/presto alone are legacy
    "allow_license": [],       # licence ids acknowledged for providers that are not permissive
}


def embedding_sources(opts: dict) -> list[str]:
    """``features.embeddings.source`` as a list (one or several feature providers)."""
    src = opts.get("source")
    return [] if not src else [src] if isinstance(src, str) else list(src)


def embedding_namespace(opts: dict) -> bool | None:
    """Namespace option: several sources always need namespaced feature names."""
    return True if len(embedding_sources(opts)) > 1 and opts.get("namespace") is None else opts.get("namespace")


def embedding_options(cfg: dict, always: bool = False) -> dict | None:
    """Resolved ``features.embeddings`` options (None when the group is not configured, unless `always`)."""
    raw = cfg["features"].get("embeddings")
    if not raw and not always:
        return None
    opts = {**EMBEDDING_DEFAULTS, **(raw or {})}
    if not opts["dir"]:
        opts["dir"] = str(Path(cfg["paths"]["features"]).parent / "embeddings")
    return opts


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
    _validate_embeddings(cfg["features"].get("embeddings"))
    if cfg["compositing"]["statistic"] not in ("median", "mean") and not str(
            cfg["compositing"]["statistic"]).startswith("p"):
        raise ConfigError("compositing.statistic must be median, mean or pNN (percentile)")
    if cfg["parcels"]["strategy"] not in ("majority", "mean_probability", "combined"):
        raise ConfigError("parcels.strategy must be majority, mean_probability or combined")
    try:
        cbm_markers.resolve_event_params(cfg["cbm"]["events"])
        cbm_markers.resolve_rules(cfg["cbm"]["rules"])
    except ValueError as exc:
        raise ConfigError(f"cbm: {exc}") from exc
    from .splits import STRATEGIES
    cv = cfg["cv"]
    if cv["strategy"] not in STRATEGIES:
        raise ConfigError(f"cv.strategy must be one of {', '.join(STRATEGIES)}")
    if cv["year_field"] and cv["year"] is not None:
        raise ConfigError("Set cv.year_field or cv.year, not both")
    sm = cfg["predict"]["smoothing"]
    if not isinstance(sm["enabled"], bool):
        raise ConfigError("predict.smoothing.enabled must be true or false")
    w = sm["window"]
    if isinstance(w, bool) or not isinstance(w, int) or w < 3 or w % 2 == 0:
        raise ConfigError("predict.smoothing.window must be an odd integer >= 3")
    k = sm["smoothness"]
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
        raise ConfigError("predict.smoothing.smoothness must be a positive number")
    o = cfg["openeo"]
    unknown = sorted(set(o["products"]) - {"s2_composites", "s1_composites", "s2_scenes", "signatures"})
    if unknown:
        raise ConfigError(f"Unknown openeo.products {unknown}")
    if o["s1_coefficient"] not in ("sigma0-ellipsoid", "gamma0-terrain"):
        raise ConfigError("openeo.s1_coefficient must be sigma0-ellipsoid or gamma0-terrain")
    _validate_resources(cfg["resources"], cfg["preflight"])
    _validate_data(cfg["data"])
    for name, spec in cfg["experiments"].items():
        if not isinstance(spec, dict) or not ({"groups", "features", "periods"} & set(spec)):
            raise ConfigError(f"Experiment {name} needs groups, features and/or periods")
    return cfg


def _validate_data(d: dict) -> None:
    if d["source"] not in ("local", "odc-stac"):
        raise ConfigError("data.source must be local or odc-stac")
    for key in ("endpoint", "collection"):
        if not isinstance(d[key], str) or not d[key]:
            raise ConfigError(f"data.{key} must be a non-empty string")
    mc = d["max_cloud"]
    if mc is not None and (isinstance(mc, bool) or not isinstance(mc, (int, float)) or not 0 <= mc <= 100):
        raise ConfigError("data.max_cloud must be a percentage in [0, 100] or null")
    f = d["min_clear_fraction"]
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 <= f <= 1:
        raise ConfigError("data.min_clear_fraction must be in [0, 1]")


def _validate_embeddings(raw) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ConfigError("features.embeddings must be a mapping {source: ..., year: ...}")
    unknown = set(raw) - set(EMBEDDING_DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown configuration key: features.embeddings.{sorted(unknown)[0]}")
    from . import plugins
    sources = embedding_sources(raw)
    known = plugins.feature_providers.names()
    if not sources or not all(isinstance(x, str) and x in plugins.feature_providers for x in sources):
        raise ConfigError(f"features.embeddings.source must be one or a list of: {', '.join(known)}")
    if len(set(sources)) != len(sources):
        raise ConfigError("features.embeddings.source lists a provider twice")
    if raw.get("namespace") not in (None, True, False):
        raise ConfigError("features.embeddings.namespace must be true, false or unset")
    if len(sources) > 1 and raw.get("namespace") is False:
        raise ConfigError("Several embedding sources need namespaced feature names (features.embeddings.namespace)")
    allow = raw.get("allow_license", [])
    if not isinstance(allow, list) or not all(isinstance(x, str) for x in allow):
        raise ConfigError("features.embeddings.allow_license must be a list of licence ids")
    year = raw.get("year")
    if isinstance(year, bool) or not isinstance(year, int) or not 2015 <= year <= 2100:
        raise ConfigError("features.embeddings.year must be a year, e.g. 2023")
    if raw.get("resampling", "nearest") not in ("nearest", "bilinear"):
        raise ConfigError("features.embeddings.resampling must be nearest or bilinear")
    if raw.get("mirror", "source.coop") not in ("source.coop", "gcs"):
        raise ConfigError("features.embeddings.mirror must be source.coop or gcs")
    months = raw.get("months", 12)
    if isinstance(months, bool) or not isinstance(months, int) or not 1 <= months <= 24:
        raise ConfigError("features.embeddings.months must be an integer in 1..24")
    start = raw.get("start")
    if start is not None:
        try:
            y, m = (int(p) for p in str(start).split("-"))
            date(y, m, 1)
        except (TypeError, ValueError) as exc:
            raise ConfigError("features.embeddings.start must be YYYY-MM") from exc


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_resources(r: dict, p: dict) -> None:
    if not isinstance(r["disk_guard"], bool):
        raise ConfigError("resources.disk_guard must be true or false")
    if not _number(r["min_free_gb"]) or r["min_free_gb"] < 0:
        raise ConfigError("resources.min_free_gb must be a finite nonnegative number of gigabytes")
    if not _number(r["safety_factor"]) or r["safety_factor"] < 1:
        raise ConfigError("resources.safety_factor must be a finite number >= 1")
    if not _number(r["compression_ratio"]) or not 0 < r["compression_ratio"] <= 1:
        raise ConfigError("resources.compression_ratio must be in (0, 1]")
    for key in ("min_s2_scenes_per_window", "min_s1_scenes_per_window", "coverage_max_px"):
        if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < (16 if key == "coverage_max_px" else 0):
            raise ConfigError(f"preflight.{key} must be a nonnegative integer" +
                              (" of at least 16" if key == "coverage_max_px" else ""))
    if not _number(p["max_unobserved_fraction"]) or not 0 <= p["max_unobserved_fraction"] <= 1:
        raise ConfigError("preflight.max_unobserved_fraction must be in [0, 1]")
    if not isinstance(p["inventories"], list) or not all(isinstance(x, str) for x in p["inventories"]):
        raise ConfigError("preflight.inventories must be a list of CSV paths")


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
