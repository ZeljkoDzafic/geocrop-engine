"""SYNTHETIC experiment: parcel classifier on signatures vs pixel classifier + parcel aggregation.

Every arm uses the same parcels, the same locked holdout and the same parcel-grouped spatial folds
(`geocrop.parcel_classify.compare_pixel_aggregation`). Results describe the synthetic generator only
(`geocrop.demo.synthetic_parcel_series`), not real-world accuracy.

    PYTHONPATH=src python tools/parcel_vs_pixel_experiment.py --out outputs/parcel_vs_pixel
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from geocrop import parcel_classify as pc
from geocrop.demo import synthetic_parcel_series

SCENARIOS = {
    "moderate": {"pixel_noise": 0.08, "cloud_fraction": 0.3, "mixed_fraction": 0.15, "shift_days": 12.0},
    "hard": {"pixel_noise": 0.15, "cloud_fraction": 0.5, "mixed_fraction": 0.3, "shift_days": 18.0},
}
RF = {"algorithm": "rf", "n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced_subsample",
      "n_jobs": -1}
PARCEL_MODELS = {
    "rf": RF,
    "ltae": {"algorithm": "ltae", "epochs": 100, "patience": 10},
    "tempcnn": {"algorithm": "tempcnn", "epochs": 60, "patience": 10},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-side", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--block-m", type=float, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--models", nargs="+", default=list(PARCEL_MODELS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = []
    for scenario, settings in SCENARIOS.items():
        for seed in a.seeds:
            data = synthetic_parcel_series(n_side=a.n_side, seed=seed, **settings)
            frame = pc.parcel_features(data["signatures"], data["parcels"], data["start"], data["end"])
            for name in a.models:
                params = PARCEL_MODELS[name]
                cols = pc._resolve_columns(frame, params, pc.FEATURE_GROUPS)
                parcels, _ = pc.to_samples(frame, data["parcels"], "true_class", cols)
                t0 = time.perf_counter()
                rep = pc.compare_pixel_aggregation(parcels, data["pixels"], out / scenario / f"seed{seed}" / name,
                                                   params, RF, a.folds, a.block_m, seed, a.holdout, bootstrap=0)
                for method, r in rep["cross_validation"].items():
                    if method != "parcel_classifier" and name != a.models[0]:
                        continue            # pixel arms are identical across parcel models; report once
                    rows.append({"scenario": scenario, "seed": seed,
                                 "method": f"parcel_{name}" if method == "parcel_classifier" else f"{method} (rf)",
                                 "cv_overall_accuracy": r["overall_accuracy"], "cv_macro_f1": r["macro_f1"],
                                 "holdout_macro_f1": rep.get("holdout", {}).get(method, {}).get("macro_f1"),
                                 "parcels": rep["parcels"], "pixels": rep["pixels"],
                                 "fold_fingerprint": rep["fold_fingerprint"],
                                 "mcnemar_p_vs_pixel_mean": (rep["mcnemar_parcel_vs_pixel_mean"]["p_value"]
                                                             if method == "parcel_classifier" else None),
                                 "seconds": time.perf_counter() - t0})
                print(json.dumps(rows[-1]))
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "runs.csv", index=False)
    summary = (table.groupby(["scenario", "method"])[["cv_overall_accuracy", "cv_macro_f1", "holdout_macro_f1"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
