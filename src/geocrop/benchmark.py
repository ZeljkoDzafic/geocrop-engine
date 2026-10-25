"""Compare classifiers on identical sampled support and spatial folds.

This is an internal model-selection experiment, not evidence of external accuracy or
superiority over other software. Timings describe this run and include CV fit/prediction.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .estimators import ALGORITHMS, channels_from_names
from .model import _fold_plan, cross_validate, selection_samples, spatial_blocks
from .samples import Samples


def _digest_array(values) -> str:
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(str(values.shape).encode())
    if values.dtype.hasobject:
        h.update(json.dumps(values.tolist(), ensure_ascii=False).encode())
    else:
        h.update(memoryview(values).cast("B"))
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def run(samples: Samples, output_dir: str | Path,
        candidates: dict[str, dict] | None = None, n_splits: int = 5,
        spatial_block_m: float | None = 5000, seed: int = 42,
        spatial_buffer_m: float = 0, calibration_mode: str = "oof", calibration_fraction: float = 0.2) -> dict:
    """Run named parameter sets on the same folds, persisting successful and failed outcomes.

    Optional-model dependency failures remain visible as failed candidates. A partial run
    has status ``incomplete``; callers must inspect it instead of selecting from only the
    survivors. No winner is chosen automatically and no mapping model is fitted afterward.
    """
    if candidates is None:
        candidates = {a: {"algorithm": a, "n_estimators": 100, "n_jobs": 1,
                           "class_weight": "balanced_subsample"}
                      for a in ("rf", "extra_trees", "hist_gb")}
    if not candidates:
        raise ValueError("Supply at least one benchmark candidate")
    for name, params in candidates.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
            raise ValueError("Candidates must map nonempty names to estimator parameter dictionaries")
        if params.get("algorithm", "rf") not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm in candidate {name!r}")
    input_count = len(samples.y)
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"polygon": samples.polygon, "coordinates": np.column_stack([samples.x, samples.yy]),
               "crs": samples.crs, "spatial_buffer_m": spatial_buffer_m}
    # Validate the shared design before doing costly candidate fits or creating a report.
    plan = _fold_plan(samples.y, groups, n_splits, seed, **spatial)
    identity = {key: _digest_array(getattr(samples, key)) for key in ("X", "y", "polygon", "x", "yy")}
    identity.update({"features": samples.features, "crs": samples.crs})
    sample_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fold_records = [{"fold": k, **diagnostics, "n_test": len(te),
                     "train_index_sha256": _digest_array(tr), "test_index_sha256": _digest_array(te)}
                    for k, (tr, te, diagnostics) in enumerate(plan)]
    fold_fingerprint = hashlib.sha256(json.dumps(fold_records, sort_keys=True).encode()).hexdigest()
    rows, results = [], {}
    for name, raw in candidates.items():
        params = dict(raw)
        started = time.perf_counter()
        try:
            if params.get("algorithm") == "tempcnn" and "n_channels" not in params:
                params["n_channels"] = channels_from_names(samples.features)
            cv = cross_validate(samples.X, samples.y, samples.polygon, groups, params, n_splits, seed,
                                coordinates=spatial["coordinates"], crs=samples.crs,
                                spatial_buffer_m=spatial_buffer_m)
            for key in ("_oof", "_oof_proba", "_classes"):
                cv.pop(key)
            row = {"candidate": name, "algorithm": params.get("algorithm", "rf"), "status": "ok",
                   **{k: cv[k] for k in ("overall_accuracy", "macro_f1", "kappa", "fold_macro_f1_std")},
                   "ece": cv["calibration"]["ece"]}
            results[name] = {"status": "ok", "params": params, "cv": cv}
        except Exception as exc:  # Candidate/backend failures are report outcomes; interrupts still propagate.
            row = {"candidate": name, "algorithm": params.get("algorithm", "rf"), "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
            results[name] = {"status": "failed", "params": params, "error": row["error"]}
        row["seconds"] = round(time.perf_counter() - started, 6)
        rows.append(row)
    report = {"schema": "geocrop-benchmark/1", "status": "complete" if all(
        r["status"] == "ok" for r in rows) else "incomplete", "internal_only": True,
        "note": "Internal model-selection diagnostics on supplied reference samples. No external accuracy or "
                "cross-engine superiority is established. Timings include CV fit/prediction on this machine.",
        "samples": len(samples.y), "input_samples": input_count, "features": samples.features,
        "calibration_mode": calibration_mode, "calibration_split": calibration_split,
        "sample_fingerprint": sample_fingerprint,
        "fold_fingerprint": fold_fingerprint, "folds": fold_records,
        "seed": seed, "requested_folds": n_splits, "actual_folds": len(plan),
        "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "sklearn": sklearn.__version__, "numpy": np.__version__},
        "results": results, "summary": rows}
    report = _json_safe(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    return report
