"""Release gate, check (b): combined integration of the engine on ONE study, driven through the CLI.

Runs the main `geocrop` sub-commands together on the same SYNTHETIC study (plus, optionally, a small real
area produced by `geocrop acceptance`) and asserts that their file contracts fit together:

  1. demo                 `geocrop demo --quick` and a season of synthetic S2/S1 scene folders
  2. embeddings           Presto from monthly composites -> attached to the stack -> samples
                          (+ AlphaEarth / Presto on the real area with --network / --real-composites)
  3. benchmark_models     LightTAE + RF + CatBoost on identical spatial folds (time-series features)
  4. active_learning      train -> predict -> parcels -> select -> simulated labels -> ingest -> retrain
  5. transfer             leave-one-region-out on regional samples merged across two synthetic seasons
  6. ftw_benchmark        FTW segmenter inside `benchmark-segmentation` (+ FTW on the real area)
  7. cbm_markers          s1-signatures, signatures, event-markers, traffic-light
  8. preflight_clean      preflight, then clean --dry-run and clean on a config for this study

Every step records status (ok / fail / skip), duration, evidence and the tail of the error output. Skips are
used only when an optional dependency or the network is unavailable. Exit code 1 when any non-skipped step
failed. All numbers produced on the synthetic study are SYNTHETIC and say nothing about real accuracy.

Usage:
    python tools/gate_integration.py --work DIR [--network] [--real-composites DIR] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PRESTO_PINNED_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"
FTW_MODEL = "FTW_PRUE_EFNET_B3_CCBY"
WINDOWS = {"P1": {"start": "2023-04-01", "end": "2023-05-31"}, "P2": {"start": "2023-06-01", "end": "2023-07-15"},
           "P3": {"start": "2023-07-16", "end": "2023-08-31"}, "P4": {"start": "2023-09-01", "end": "2023-10-15"}}
SEASON = ("2023-03-15", "2023-10-31")       # span of the synthetic scene folders (generate_scenes)
BLOCK_M = "500"
FOLDS = "3"


class Skip(Exception):
    """Optional dependency or network not available."""


class StepError(Exception):
    def __init__(self, message: str, tail: str = ""):
        super().__init__(message)
        self.tail = tail


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def first_json(text: str):
    """First JSON object/array printed by a command (some commands print several or add text)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise StepError("no JSON in command output", _tail(text))


def has_module(*names: str) -> list[str]:
    return [n for n in names if importlib.util.find_spec(n) is None]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: Path) -> dict[str, tuple[int, float]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime) for p in root.rglob("*") if p.is_file()}


def polygon_coverage(samples_path: Path, training: str | Path) -> dict:
    """Which reference polygons/classes contributed samples (`samples` skips polygons without a valid pixel)."""
    from collections import Counter

    import geopandas as gpd

    from geocrop.samples import Samples
    s = Samples.load(samples_path)
    tr = gpd.read_file(training)
    present = set(map(str, s.polygon))
    dropped = tr[~tr["polygon_id"].astype(str).isin(present)]
    return {"samples_per_class": dict(sorted(Counter(map(str, s.y)).items())),
            "polygons": [int(len(tr) - len(dropped)), int(len(tr))],
            "dropped_polygons_per_class": dict(sorted(Counter(dropped["class_name"].astype(str)).items())),
            "missing_classes": sorted(set(tr["class_name"].astype(str)) - set(map(str, s.y)))}


class Gate:
    def __init__(self, work: Path, network: bool, real: Path | None, cache: Path):
        self.work, self.network, self.real, self.cache = work, network, real, cache
        self.steps: list[dict] = []
        self.ctx: dict = {}
        self.env = dict(os.environ, GEOCROP_CACHE=str(cache), MPLBACKEND="Agg")
        self.commands: list[str] = []

    # ------------------------------------------------------------------ helpers
    def cli(self, *args, timeout: int = 1200, expect: tuple[int, ...] = (0,)) -> str:
        """Run `python -m geocrop.cli ...` in the work directory; raise StepError with the output tail."""
        cmd = [sys.executable, "-m", "geocrop.cli", *[str(a) for a in args]]
        self.commands.append(" ".join(cmd[2:]))
        try:
            r = subprocess.run(cmd, cwd=self.work, env=self.env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise StepError(f"timeout after {timeout}s: geocrop {args[0]}", _tail(str(exc.stderr or ""))) from None
        if r.returncode not in expect:
            raise StepError(f"geocrop {' '.join(str(a) for a in args[:2])} exited {r.returncode}",
                            _tail(r.stderr) or _tail(r.stdout))
        self._last_stderr = r.stderr
        return r.stdout

    def py(self, code: str, timeout: int = 900) -> str:
        """Run a small Python snippet (library entry points without a CLI, e.g. synthetic scene generation)."""
        self.commands.append(f"python -c <{code.strip().splitlines()[0][:60]}...>")
        r = subprocess.run([sys.executable, "-c", code], cwd=self.work, env=self.env, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode:
            raise StepError("python snippet failed", _tail(r.stderr))
        return r.stdout

    def need(self, *keys: str):
        missing = [k for k in keys if k not in self.ctx]
        if missing:
            raise StepError(f"not run: prerequisite output(s) {missing} missing because an earlier step failed")

    def step(self, name: str, fn):
        t0 = time.perf_counter()
        rec = {"name": name, "status": "ok", "evidence": {}}
        print(f"[gate] {name} ...", flush=True)
        try:
            fn(rec["evidence"])
        except Skip as exc:
            rec["status"], rec["reason"] = "skip", str(exc)
        except StepError as exc:
            rec["status"], rec["error"], rec["error_tail"] = "fail", str(exc), exc.tail
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, the gate continues
            rec["status"], rec["error"] = "fail", f"{type(exc).__name__}: {exc}"
            rec["error_tail"] = _tail(traceback.format_exc())
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        self.steps.append(rec)
        extra = rec.get("reason") or rec.get("error") or ""
        print(f"[gate] {name}: {rec['status']} ({rec['seconds']} s) {extra}", flush=True)

    # ------------------------------------------------------------------ shared inputs
    def base_stack(self) -> Path:
        """Handcrafted multi-window stack of the study (no embeddings), built once."""
        if "base_stack" not in self.ctx:
            self.need("study")
            out = self.work / "features" / "stack_base.tif"
            self.cli("features", "--composites", self.ctx["study"]["composites"], "--out", out)
            self.ctx["base_stack"] = out
        return self.ctx["base_stack"]

    def write_config(self, path: Path, embeddings: dict | None = None) -> Path:
        import yaml
        st = self.ctx["study"]
        root = Path(st["root"])
        cfg = {"grid": {"crs": "EPSG:32634", "reference": st["reference"]},
               "windows": WINDOWS,
               "paths": {"s2_scenes": str(root / "scenes" / "s2"), "s1_scenes": str(root / "scenes" / "s1"),
                         "composites": str(self.work / "run" / "composites"),
                         "features": str(self.work / "run" / "features" / "stack.tif"),
                         "training": st["training"], "parcels": st["parcels"], "observations": st["observations"],
                         "output": str(self.work / "run" / "out")},
               "model": {"spatial_block_m": int(BLOCK_M), "cv_folds": int(FOLDS), "n_estimators": 30},
               "resources": {"disk_guard": False}}
        if embeddings:
            cfg["features"] = {"embeddings": embeddings}
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
        return path

    # ------------------------------------------------------------------ 1. demo
    def s1_demo(self, ev):
        quick = self.work / "demo_quick"
        out = first_json(self.cli("demo", "--quick", "--root", quick))
        if "SYNTHETIC" not in out.get("mode", ""):
            raise StepError("demo summary is not marked SYNTHETIC")
        ev["demo_quick"] = {"cv_macro_f1": out["cv_macro_f1"], "field_points_oa": out["field_points_oa"],
                            "summary": str(quick / "demo_summary.json")}
        root = self.work / "study"
        txt = self.py(f"""
import json
from geocrop.demo import generate_scenes
d = generate_scenes({str(root)!r}, year=2023, s2_every=6, s1_every=12)
print(json.dumps({{k: v for k, v in d.items() if k not in ('s2_scenes', 's1_scenes')}} |
                 {{'n_s2': len(d['s2_scenes']), 'n_s1': len(d['s1_scenes']),
                   'first': d['s2_scenes'][0], 'last': d['s2_scenes'][-1]}}))
""")
        st = json.loads(txt.strip().splitlines()[-1])
        s2 = sorted((root / "scenes" / "s2").iterdir())
        s1 = sorted((root / "scenes" / "s1").iterdir())
        for folder, req in ((s2[0], ("B02.tif", "B12.tif", "SCL.tif", "scene.json")),
                            (s1[0], ("VV.tif", "VH.tif", "scene.json"))):
            miss = [f for f in req if not (folder / f).exists()]
            if miss:
                raise StepError(f"scene folder {folder} misses {miss}")
        if not (root / "SYNTHETIC_DATA.txt").exists():
            raise StepError("SYNTHETIC_DATA.txt marker missing")
        self.ctx["study"] = st
        ev.update({"synthetic": True, "study": str(root), "s2_scene_folders": len(s2), "s1_scene_folders": len(s1),
                   "s2_dates": [s2[0].name, s2[-1].name], "s1_dates": [s1[0].name, s1[-1].name],
                   "grid": st["reference"], "classes": st["classes"]})

    # ------------------------------------------------------------------ 2. embeddings (Presto)
    def _presto_checkpoint(self, ev) -> list[str]:
        """--checkpoint argument: [] lets the CLI download the pinned file into $GEOCROP_CACHE/presto."""
        cached = [p for p in (self.cache / "presto").glob("presto_*_default_model.pt")
                  if sha256(p) == PRESTO_PINNED_SHA256]
        if cached:
            ev["checkpoint"] = f"pinned, cached ({cached[0]})"
            return []
        if self.network:
            ev["checkpoint"] = f"pinned, downloaded by the CLI into {self.cache / 'presto'}"
            return []
        stand_in = self.work / "presto_stand_in.pt"
        self.py(f"""
import torch
from geocrop import embeddings
torch.manual_seed(0)
m = embeddings.load_presto(None)
torch.save({{f"encoder.{{k}}": v for k, v in m.state_dict().items()}}, {str(stand_in)!r})
""")
        ev["checkpoint"] = ("STAND-IN random weights (no --network and no cached pinned checkpoint); "
                            "tests the contract, not the pretrained model")
        return ["--checkpoint", stand_in]

    def s2_embeddings(self, ev):
        self.need("study")
        if has_module("torch"):
            raise Skip("torch not installed (pip install -e '.[deep]')")
        st = self.ctx["study"]
        emb_dir = self.work / "embeddings"
        cfg = self.write_config(self.work / "study_presto.yaml",
                                {"source": "presto", "year": 2023, "start": "2023-03", "months": 8,
                                 "dir": str(emb_dir), "attach": True})
        ck = self._presto_checkpoint(ev)
        out = first_json(self.cli("embeddings", "--config", cfg, "--make-monthly", "--parcels", st["parcels"], *ck))
        raster = emb_dir / "presto_2023.tif"
        parcels_csv = emb_dir / "presto_2023_parcels.csv"
        months = sorted(p.name for p in (emb_dir / "monthly").iterdir() if p.is_dir())
        for p in (raster, parcels_csv, raster.with_suffix(".features.json")):
            if not p.exists():
                raise StepError(f"missing embeddings output {p}")
        import numpy as np
        import pandas as pd
        import rasterio
        with rasterio.open(raster) as ds, rasterio.open(st["reference"]) as ref:
            if ds.shape != ref.shape or ds.crs != ref.crs or ds.transform != ref.transform:
                raise StepError("Presto raster is not on the study grid")
            data = ds.read()
            valid = float((data != ds.nodata).all(axis=0).mean())
            tags = ds.tags()
            if ds.count != 128 or ds.descriptions[0] != "emb_00":
                raise StepError(f"unexpected Presto bands: {ds.count} {ds.descriptions[:2]}")
        if ck == [] and tags.get("checkpoint_sha256") != PRESTO_PINNED_SHA256:
            raise StepError(f"Presto raster records checkpoint {tags.get('checkpoint_sha256')}")
        if valid < 0.9:
            raise StepError(f"only {valid:.2f} of pixels have an embedding")
        table = pd.read_csv(parcels_csv)
        n_parcels = len(__import__("geopandas").read_file(st["parcels"]))
        if len(table) != n_parcels or not np.isfinite(table.filter(like="emb_").to_numpy()).all():
            raise StepError("parcel embeddings incomplete or non-finite")
        ev["presto"] = {"raster": str(raster), "bands": 128, "valid_fraction": round(valid, 4),
                        "monthly_folders": months, "parcels_csv_rows": len(table),
                        "tags": {k: tags.get(k) for k in ("embedding_source", "start", "checkpoint_sha256")},
                        "cli_summary_keys": sorted(out)[:12]}
        # attach to the handcrafted stack through features --config (features.embeddings.attach)
        stack = self.work / "features" / "stack_emb.tif"
        self.cli("features", "--composites", st["composites"], "--out", stack, "--config", cfg)
        from geocrop.features import load_manifest
        man = load_manifest(stack)
        groups = sorted({f.group for f in man})
        n_emb = sum(f.group == "emb" for f in man)
        if n_emb != 128 or len(groups) < 2:
            raise StepError(f"embeddings not attached: groups {groups}, emb features {n_emb}")
        samples = self.work / "features" / "samples_emb.npz"
        self.cli("samples", "--features", stack, "--polygons", st["training"], "--out", samples,
                 "--max-per-polygon", "40")
        from geocrop.samples import Samples
        s = Samples.load(samples)
        emb_cols = [i for i, n in enumerate(s.features) if n.startswith("emb_")]
        if len(emb_cols) != 128 or not np.isfinite(s.X[:, emb_cols]).all():
            raise StepError("samples lack finite emb_* features")
        cov = polygon_coverage(samples, st["training"])
        if cov["dropped_polygons_per_class"]:
            raise StepError(f"samples on the embedding stack dropped reference polygons: {cov}")
        ev["stack"] = {"path": str(stack), "features": len(man), "groups": groups, "emb_features": n_emb}
        ev["samples"] = {"path": str(samples), "n": int(len(s.y)), "features": len(s.features),
                         "polygons": int(len(set(s.polygon)))}

    # ------------------------------------------------------------------ 2b/2c real area
    def _real_layout(self) -> dict:
        if self.real is None:
            raise Skip("no --real-composites given (output directory of `geocrop acceptance --out`)")
        r = self.real
        grid = r / "grid.tif"
        if not grid.exists():
            raise StepError(f"{grid} not found: --real-composites must be an acceptance output directory")
        comp = r / "composites"
        windows = sorted(p for p in comp.iterdir() if p.is_dir() and (p / "B04.tif").exists()) if comp.exists() else []
        s1_root = r / "scenes" / "s1_coregistered"
        if not s1_root.exists():
            s1_root = r / "scenes" / "s1"
        return {"grid": grid, "windows": windows, "s2_scenes": r / "scenes" / "s2", "s1_scenes": s1_root,
                "parcels": r / "parcels" / "parcels.gpkg", "segments": r / "segmentation" / "segments.gpkg",
                "report": r / "acceptance_report.json"}

    def s2b_alphaearth_real(self, ev):
        if not self.network:
            raise Skip("AlphaEarth reads public COGs over the network; pass --network")
        lay = self._real_layout()
        if has_module("pyarrow"):
            raise Skip("pyarrow not installed (needed for the AlphaEarth tile index; pip install -e '.[parquet]')")
        out_dir = self.work / "real" / "embeddings"
        out = first_json(self.cli("embeddings", "--source", "alphaearth", "--year", "2023", "--reference",
                                  lay["grid"], "--out", out_dir, timeout=1800))
        raster = out_dir / "alphaearth_2023.tif"
        from geocrop import embeddings
        s = embeddings.summary(raster)
        if s["bands"] != 64 or s["valid_fraction"] < 0.99 or abs(s["vector_norm_mean"] - 1) > 0.02:
            raise StepError(f"AlphaEarth summary out of contract: {s}")
        ev.update({"grid": str(lay["grid"]), "raster": str(raster), "cli": {k: out[k] for k in list(out)[:6]},
                   "summary": {k: s[k] for k in ("bands", "valid_fraction", "min", "max", "vector_norm_mean")}})

    def s2c_presto_real(self, ev):
        lay = self._real_layout()
        if has_module("torch"):
            raise Skip("torch not installed (pip install -e '.[deep]')")
        n_s2 = len([p for p in lay["s2_scenes"].iterdir() if p.is_dir()]) if lay["s2_scenes"].exists() else 0
        if not n_s2:
            raise StepError(f"no S2 scene folders under {lay['s2_scenes']}")
        dates = []
        for d in lay["s2_scenes"].iterdir():
            meta = d / "scene.json"
            if meta.exists():
                dates.append(json.loads(meta.read_text())["datetime"][:7])
        start, last = min(dates), max(dates)
        months = (int(last[:4]) - int(start[:4])) * 12 + int(last[5:7]) - int(start[5:7]) + 1
        import yaml
        cfg = {"grid": {"reference": str(lay["grid"])},
               "paths": {"s2_scenes": str(lay["s2_scenes"]), "s1_scenes": str(lay["s1_scenes"]),
                         "features": str(self.work / "real" / "features" / "stack.tif")},
               "features": {"embeddings": {"source": "presto", "year": int(start[:4]), "start": start,
                                           "months": months, "dir": str(self.work / "real" / "embeddings")}}}
        path = self.work / "real_presto.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg), encoding="utf8")
        ck = self._presto_checkpoint(ev)
        self.cli("embeddings", "--config", path, "--make-monthly", *ck, timeout=1800)
        from geocrop import embeddings
        raster = self.work / "real" / "embeddings" / f"presto_{start[:4]}.tif"
        s = embeddings.summary(raster)
        if s["bands"] != 128 or s["valid_fraction"] < 0.5:
            raise StepError(f"Presto on the real area out of contract: {s}")
        ev.update({"layout": "acceptance scene folders scenes/s2 (+ S1) -> --make-monthly",
                   "s2_scene_folders": n_s2, "s1_root": str(lay["s1_scenes"]), "start": start, "months": months,
                   "raster": str(raster),
                   "summary": {k: s[k] for k in ("bands", "valid_fraction", "vector_norm_mean")}})

    # ------------------------------------------------------------------ 3. benchmark-models
    def s3_benchmark(self, ev):
        self.need("study")
        st = self.ctx["study"]
        root = Path(st["root"])
        ts = self.work / "ts" / "stack_ts.tif"
        self.cli("ts-features", "--scenes", root / "scenes" / "s2", root / "scenes" / "s1", "--reference",
                 st["reference"], "--start", SEASON[0], "--end", SEASON[1], "--out", ts, "--output-step-days", "20")
        from geocrop.features import load_manifest
        man = load_manifest(ts)
        n_ts = sum(f.group == "ts" for f in man)
        if not n_ts:
            raise StepError("ts-features wrote no `ts` group")
        samples = self.work / "ts" / "samples_ts.npz"
        # extract only the `ts` features the temporal models use: phenology metrics are NaN where no season is
        # detected (winter crops whose season starts before the series), which used to drop the whole class
        self.cli("samples", "--features", ts, "--polygons", st["training"], "--out", samples, "--max-per-polygon", "30",
                 "--groups", "ts")
        coverage = polygon_coverage(samples, st["training"])
        # the guard: the same path on every feature must refuse to benchmark a reduced class set (exit != 0)
        all_feats = self.work / "ts" / "samples_all.npz"
        self.cli("samples", "--features", ts, "--polygons", st["training"], "--out", all_feats, "--max-per-polygon",
                 "30")
        guard_cov = polygon_coverage(all_feats, st["training"])
        guard = None
        if guard_cov["missing_classes"]:
            self.cli("benchmark-models", "--samples", all_feats, "--out", self.work / "bench" / "guard", "--models", "rf",
                     "--block-m", BLOCK_M, "--folds", FOLDS, "--repeats", "1", expect=(1,))
            guard = {"missing_classes": guard_cov["missing_classes"], "exit_code": 1,
                     "message": (self._last_stderr.strip().splitlines() or [""])[-1][:300]}
            if "MissingClassError" not in self._last_stderr:
                raise StepError(f"benchmark-models accepted samples that lost classes {guard_cov['missing_classes']}")
        common = ["--samples", samples, "--features", ts, "--groups", "ts", "--block-m", BLOCK_M, "--folds", FOLDS,
                  "--repeats", "1"]
        models = ["ltae", "rf", "catboost"]
        missing = has_module("torch", "catboost")
        if missing:
            raise Skip(f"optional model dependencies missing: {missing} (pip install -e '.[deep,boost]')")
        joint = self.work / "bench" / "joint"
        self.cli("benchmark-models", "--out", joint, "--models", *models, *common, timeout=1800)
        import pandas as pd
        table = pd.read_csv(joint / "benchmark_models.csv")
        status = dict(zip(table["model"], table["status"]))
        bad = {m: s for m, s in status.items() if s != "ok"}
        if bad or set(status) != set(models):
            raise StepError(f"benchmark-models statuses: {status}")
        report = json.loads((joint / "benchmark_models.json").read_text())
        sig = report["fold_signature"]
        per_model = report.get("fold_signatures_per_model") or {}
        if set(per_model) != set(models) or len({tuple(v) for v in per_model.values()}) != 1 \
                or any(v[0] != sig for v in per_model.values()):
            raise StepError(f"per-model fold signatures differ: {per_model} (run signature {sig})")
        # fold identity: (a) separate single-model runs must reproduce the same fold signature,
        # (b) the signature must match an independent recomputation from the samples.
        sigs = {"joint": sig}
        for m in models:
            d = self.work / "bench" / m
            self.cli("benchmark-models", "--out", d, "--models", m, *common, timeout=1800)
            sigs[m] = json.loads((d / "benchmark_models.json").read_text())["fold_signature"]
        import warnings

        from geocrop.benchmark import fold_signature
        from geocrop.model import selection_samples
        from geocrop.samples import Samples
        s = Samples.load(samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dev, _ = selection_samples(s, float(BLOCK_M), 42, 0, "oof", 0.2)
            recomputed = fold_signature(dev, float(BLOCK_M), int(FOLDS), 42)
        if len(set(sigs.values())) != 1 or recomputed != sig:
            raise StepError(f"fold signatures differ: {sigs}, recomputed {recomputed}")
        mc = {m: report["details"][m]["runs"][0].get("mcnemar_p_vs_reference") for m in models}
        ev.update({"ts_stack": str(ts), "ts_features": n_ts, "samples": int(len(s.y)),
                   "models": {r.model: {"status": r.status, "macro_f1": round(float(r.macro_f1), 3),
                                        "overall_accuracy": round(float(r.overall_accuracy), 3)}
                              for r in table.itertuples()},
                   "metrics_note": "SYNTHETIC data", "reference_model": report["reference_model"],
                   "fold_signature": sig, "fold_signatures_per_run": sigs, "fold_signature_recomputed": recomputed,
                   "fold_signatures_per_model": per_model, "lost_class_guard": guard,
                   "mcnemar_p_vs_reference": mc, "outputs": [str(joint / "benchmark_models.csv"),
                                                             str(joint / "benchmark_models.json")],
                   "reference_coverage": coverage})
        if coverage["missing_classes"]:
            raise StepError(f"samples --groups ts lost classes {coverage['missing_classes']} "
                            f"(dropped polygons per class: {coverage['dropped_polygons_per_class']})")

    # ------------------------------------------------------------------ 4. active learning
    def s4_active_learning(self, ev):
        self.need("study")
        import geopandas as gpd
        import pandas as pd
        st = self.ctx["study"]
        stack = self.base_stack()
        al = self.work / "al"
        params = '{"n_estimators": 60, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}'
        s0 = al / "r0" / "samples.npz"
        self.cli("samples", "--features", stack, "--polygons", st["training"], "--out", s0, "--max-per-polygon", "40")
        self.cli("train", "--samples", s0, "--out", al / "r0" / "model", "--block-m", BLOCK_M, "--folds", FOLDS,
                 "--params", params)
        self.cli("predict", "--features", stack, "--model", al / "r0" / "model" / "model.joblib",
                 "--out", al / "r0" / "prediction")
        parcels_out = al / "r0" / "parcels" / "parcels.gpkg"
        psum = first_json(self.cli("parcels", "--probabilities", al / "r0" / "prediction" / "probabilities.tif",
                                   "--parcels", st["parcels"], "--out", parcels_out,
                                   "--conformal", al / "r0" / "model" / "conformal.json",
                                   "--aoa", al / "r0" / "prediction" / "aoa.tif"))
        cov0 = polygon_coverage(s0, st["training"])
        if cov0["dropped_polygons_per_class"]:
            raise StepError(f"samples dropped reference polygons: {cov0}")
        self.ctx["parcels_out"] = parcels_out
        sel = al / "r1" / "select.gpkg"
        summary = first_json(self.cli("active-learning", "select", "--parcels", parcels_out, "--out", sel,
                                      "--budget", "8", "--min-spacing-m", "200", "--training", st["training"],
                                      "--exclude", st["observations"], "--trips", "2", "--round", "1"))
        picked = gpd.read_file(sel)
        form = pd.read_csv(al / "r1" / "select_label_form.csv", dtype=str)
        truth = gpd.read_file(st["parcels"]).set_index("parcel_id").true_class
        tr0 = gpd.read_file(st["training"])
        from geocrop.active_learning import training_parcel_ids
        par = gpd.read_file(st["parcels"])
        already = set(training_parcel_ids(tr0, par, "parcel_id").dropna())
        if not len(picked) or set(picked.parcel_id) & already:
            raise StepError(f"selection empty or proposes training parcels: {sorted(set(picked.parcel_id) & already)}")
        form["label"] = [truth[p] for p in form.parcel_id]           # SYNTHETIC truth stands in for field visits
        form["source"], form["date"], form["confidence"] = "field", "2023-07-20", "1"
        labels = al / "r1" / "labels.csv"
        form.to_csv(labels, index=False)
        tr1 = al / "r1" / "training_r1.gpkg"
        ing = first_json(self.cli("active-learning", "ingest", "--labels", labels, "--parcels", st["parcels"],
                                  "--training", st["training"], "--out", tr1, "--round", "1",
                                  "--validation", st["observations"]))
        new = gpd.read_file(tr1)
        added = new[new["al_round"] == 1] if "al_round" in new else new.iloc[0:0]
        prov = pd.read_csv(al / "r1" / "al_provenance.csv", dtype=str)
        if not len(added) or len(new) != len(tr0) + len(added):
            raise StepError(f"ingest added {len(added)} polygons; training {len(tr0)} -> {len(new)}")
        s1 = al / "r1" / "samples.npz"
        self.cli("samples", "--features", stack, "--polygons", tr1, "--out", s1, "--max-per-polygon", "40")
        self.cli("train", "--samples", s1, "--out", al / "r1" / "model", "--block-m", BLOCK_M, "--folds", FOLDS,
                 "--params", params)
        from geocrop.samples import Samples
        smp = Samples.load(s1)
        ingested_ids = set(added["polygon_id"].astype(str))
        used = ingested_ids & set(map(str, smp.polygon))
        rep0 = json.loads((al / "r0" / "model" / "train_report.json").read_text())
        rep1 = json.loads((al / "r1" / "model" / "train_report.json").read_text())
        if used != ingested_ids or rep1["polygons"] != rep0["polygons"] + len(added):
            raise StepError(f"retrained model does not use the ingested labels: used {len(used)}/{len(ingested_ids)}, "
                            f"polygons {rep0['polygons']} -> {rep1['polygons']}")
        if rep1["model_sha256"] == rep0["model_sha256"]:
            raise StepError("retrained model is byte-identical to round 0")
        ev.update({"parcels": {"path": str(parcels_out), "traffic_light": psum.get("traffic_light_counts")},
                   "selected": sorted(picked.parcel_id), "select_summary": {k: summary.get(k) for k in
                                                                            ("selected", "shortfall", "quota_relaxed")},
                   "ingest": {k: ing.get(k) for k in list(ing)[:8]},
                   "provenance_actions": prov["action"].value_counts().to_dict(),
                   "training_polygons": [len(tr0), len(new)], "added_polygon_ids": sorted(ingested_ids),
                   "train_polygons_round0_round1": [rep0["polygons"], rep1["polygons"]],
                   "train_samples_round0_round1": [rep0["samples"], rep1["samples"]],
                   "cv_macro_f1_round0_round1 (SYNTHETIC)": [round(rep0["cross_validation"]["macro_f1"], 3),
                                                             round(rep1["cross_validation"]["macro_f1"], 3)]})

    # ------------------------------------------------------------------ 5. transfer (region)
    def s5_transfer(self, ev):
        self.need("study")
        import geopandas as gpd
        st = self.ctx["study"]
        tr = self.work / "transfer"
        tr.mkdir(parents=True, exist_ok=True)
        # second SYNTHETIC season on the same grid (different fields and labels, same feature definitions)
        root24 = self.work / "study_2024"
        self.py(f"from geocrop.demo import generate; generate({str(root24)!r}, seed=2024)")
        stack24 = self.work / "features" / "stack_base_2024.tif"
        self.cli("features", "--composites", root24 / "composites", "--out", stack24)
        stacks = {"2023": self.base_stack(), "2024": stack24}
        sources = {"2023": st["training"], "2024": str(root24 / "reference" / "training.gpkg")}
        parts = []
        for year, src in sources.items():
            g = gpd.read_file(src)
            xs = g.geometry.centroid.x
            edges = xs.quantile([1 / 3, 2 / 3]).to_numpy()
            g["region"] = ["west" if x <= edges[0] else "centre" if x <= edges[1] else "east" for x in xs]
            poly = tr / f"training_regions_{year}.gpkg"
            poly.unlink(missing_ok=True)
            g.to_file(poly, driver="GPKG")
            out = tr / f"samples_{year}.npz"
            self.cli("samples", "--features", stacks[year], "--polygons", poly, "--out", out, "--max-per-polygon",
                     "30", "--region-field", "region", "--year", year)
            parts.append(out)
        merged = tr / "samples_merged.npz"
        m = first_json(self.cli("samples-merge", "--inputs", *parts, "--stacks", stacks["2023"], stacks["2024"],
                                "--out", merged))
        if m.get("years") != ["2023", "2024"] or sorted(m.get("regions") or []) != ["centre", "east", "west"]:
            raise StepError(f"samples-merge metadata unexpected: {m}")
        out = tr / "result"
        self.cli("transfer", "--samples", merged, "--out", out, "--strategies", "region", "--folds", FOLDS,
                 "--block-m", BLOCK_M, "--params", '{"n_estimators": 50, "n_jobs": -1}')
        import pandas as pd
        table = pd.read_csv(out / "transfer.csv")
        report = json.loads((out / "transfer.json").read_text())
        text = json.dumps(report)
        if "region" not in table.to_string() or "region" not in text:
            raise StepError(f"transfer outputs carry no region strategy: {table.to_string()}")
        ev.update({"samples_merge": m, "transfer_csv": str(out / "transfer.csv"),
                   "transfer_json": str(out / "transfer.json"), "table (SYNTHETIC)": json.loads(
                       table.round(3).to_json(orient="records"))})

    # ------------------------------------------------------------------ 6. FTW in benchmark-segmentation
    def _ftw_checkpoint(self, ev) -> Path:
        missing = has_module("torch", "segmentation_models_pytorch")
        if missing:
            raise Skip(f"FTW needs {missing} (pip install -e '.[deep]' segmentation-models-pytorch)")
        cache = self.work / "ftw_cache"
        if "ftw_ckpt" in self.ctx:
            return self.ctx["ftw_ckpt"]
        try:
            out = first_json(self.cli("ftw-boundaries", "--download-only", "--no-download", "--cache-dir", cache,
                                      "--model", FTW_MODEL))
            ev["checkpoint"] = f"cached {out['checkpoint']}"
        except StepError:
            if not self.network:
                raise Skip("FTW checkpoint not cached and no --network for the pinned download") from None
            out = first_json(self.cli("ftw-boundaries", "--download-only", "--cache-dir", cache, "--model", FTW_MODEL,
                                      timeout=1800))
            ev["checkpoint"] = f"downloaded {out['checkpoint']}"
        self.ctx["ftw_ckpt"] = Path(out["checkpoint"])
        return self.ctx["ftw_ckpt"]

    def s6_ftw(self, ev):
        self.need("study")
        self._ftw_checkpoint(ev)
        st = self.ctx["study"]
        comp = Path(st["composites"])
        out = self.work / "segbench"
        methods = {"ftw": {"segmenter": "ftw", "model": FTW_MODEL, "cache_dir": str(self.work / "ftw_cache"),
                           "window_a": str(comp / "P1"), "window_b": str(comp / "P3"), "patch": 128, "overlap": 32,
                           # the pretrained model finds no field interior on SYNTHETIC reflectance at 0.5
                           # ("No field interiors below threshold"); 1.01 keeps the watershed running so the
                           # registry/benchmark contract is tested. Real FTW behaviour: step 6b_ftw_real_area.
                           "threshold": 1.01,
                           "work_dir": str(out / "ftw_work")}}
        self.cli("benchmark-segmentation", "--raster", comp / "P1" / "B04.tif", "--reference", st["parcels"],
                 "--out", out, "--tile-size", "96", "--segments-per-km2", "40", "--methods", json.dumps(methods),
                 timeout=1800)
        import pandas as pd
        table = pd.read_csv(out / "benchmark_segmentation.csv")
        rows = table.set_index("method")
        if "ftw" not in rows.index or rows.loc["ftw", "status"] != "ok":
            raise StepError(f"ftw row not ok: {table[['method', 'status']].to_dict(orient='records')}")
        summ = json.loads((out / "ftw_work" / "ftw_summary.json").read_text())
        cols = ["method", "status", "segments", "object_f1", "boundary_f1", "seconds"]
        ev.update({"table (SYNTHETIC)": json.loads(table[[c for c in cols if c in table]].round(3)
                                                   .to_json(orient="records")),
                   "ftw_model": summ.get("model"), "ftw_licence": summ.get("licence"),
                   "outputs": [str(out / "benchmark_segmentation.csv"), str(out / "ftw.gpkg")],
                   "note": "FTW applied to SYNTHETIC reflectance with threshold 1.01: scores test the plumbing, "
                           "not FTW quality"})

    def s6b_ftw_real(self, ev):
        lay = self._real_layout()
        if not self.network and not (self.work / "ftw_cache").exists():
            raise Skip("FTW checkpoint needs --network for the pinned download")
        self._ftw_checkpoint(ev)
        wins = lay["windows"]
        if len(wins) < 2:
            raise Skip(f"FTW needs two composite windows with B02..B08; found {[w.name for w in wins]}")
        miss = [b for w in (wins[0], wins[-1]) for b in ("B02", "B03", "B04", "B08") if not (w / f"{b}.tif").exists()]
        if miss:
            raise StepError(f"real composites lack FTW bands: {miss}")
        out = self.work / "real" / "ftw"
        ref = lay["segments"] if lay["segments"].exists() else None
        args = ["ftw-boundaries", "--window-a", wins[0], "--window-b", wins[-1], "--grid", lay["grid"], "--out", out,
                "--model", FTW_MODEL, "--cache-dir", self.work / "ftw_cache", "--no-download", "--patch", "256"]
        if ref:
            args += ["--reference", ref]
        info = first_json(self.cli(*args, timeout=1800))
        for name in ("ftw_stack.tif", "ftw_not_field.tif", "ftw_parcels.gpkg", "ftw_summary.json"):
            if not (out / name).exists():
                raise StepError(f"missing FTW output {name}")
        ev.update({"layout": f"acceptance composites windows A={wins[0].name}, B={wins[-1].name} on grid.tif",
                   "segments": info.get("segments"), "licence": info.get("licence"),
                   "inference_seconds": info.get("inference_seconds"),
                   "reference_for_metrics": str(ref) + " (OBIA segments of the acceptance run, NOT ground truth)"
                   if ref else None,
                   "segment_metrics": {k: info.get("segment_metrics", {}).get(k)
                                       for k in ("object_f1", "matched_iou_mean")} if ref else None,
                   "outputs": str(out)})

    # ------------------------------------------------------------------ 7. CbM markers and rules
    def s7_cbm(self, ev):
        self.need("study", "parcels_out")
        st = self.ctx["study"]
        root = Path(st["root"])
        cbm = self.work / "cbm"
        s1sig = first_json(self.cli("s1-signatures", "--scenes", root / "scenes" / "s1", "--parcels", st["parcels"],
                                    "--reference", st["reference"], "--start", SEASON[0], "--end", SEASON[1],
                                    "--out", cbm / "s1_signatures.csv"))
        s2sig = first_json(self.cli("signatures", "--scenes", root / "scenes" / "s2", "--parcels", st["parcels"],
                                    "--reference", st["reference"], "--start", SEASON[0], "--end", SEASON[1],
                                    "--out", cbm / "s2_signatures.csv", "--indices", "NDVI", "BSI", "--moments"))
        cfg = self.write_config(cbm / "study_cbm.yaml")
        ev_m = first_json(self.cli("event-markers", "--signatures", cbm / "s2_signatures.csv", "--s1-signatures",
                                   cbm / "s1_signatures.csv", "--start", SEASON[0], "--end", SEASON[1], "--out",
                                   cbm / "events.csv", "--config", cfg))
        n_parcels = s2sig["parcels"]
        if ev_m["parcels"] != n_parcels or s1sig["parcels"] != n_parcels:
            raise StepError(f"parcel counts differ: s2 {n_parcels}, s1 {s1sig['parcels']}, events {ev_m['parcels']}")
        tl = first_json(self.cli("traffic-light", "--parcels", self.ctx["parcels_out"], "--markers", cbm / "events.csv",
                                 "--config", cfg, "--rules", '{"harvest_expected_classes": ["maize", "winter_crops"]}',
                                 "--out", cbm / "traffic_light.gpkg"))
        import geopandas as gpd
        out = gpd.read_file(cbm / "traffic_light.gpkg")
        if len(out) != n_parcels or not {"traffic_light", "traffic_light_reasons", "harvest_detected"} <= set(out):
            raise StepError(f"traffic-light output incomplete: {len(out)} rows, columns {sorted(out.columns)[:20]}")
        ev.update({"s1_signatures": s1sig, "s2_signatures": s2sig,
                   "event_markers (SYNTHETIC)": {k: ev_m[k] for k in ev_m if k != "out"},
                   "traffic_light (SYNTHETIC)": {k: tl[k] for k in ("traffic_light_counts", "reason_counts")},
                   "outputs": [str(cbm / n) for n in ("events.csv", "traffic_light.gpkg", "traffic_light.csv")]})

    # ------------------------------------------------------------------ 8. preflight + clean
    def s8_preflight_clean(self, ev):
        self.need("study")
        st = self.ctx["study"]
        cfg = self.write_config(self.work / "study_run.yaml")
        pf = self.work / "preflight"
        self.cli("preflight", "--config", cfg, "--out", pf / "preflight.json", "--markdown", pf / "preflight.md")
        report = json.loads((pf / "preflight.json").read_text())
        errors = [c["code"] for c in report["checks"] if c["status"] == "error"]
        if report["verdict"] == "blocked" or errors:
            raise StepError(f"preflight blocked: {errors}")
        run = first_json(self.cli("run", "--config", cfg, "--stages", "composite"))
        out_dir, comp = self.work / "run" / "out", self.work / "run" / "composites"
        planted = {"tmp": out_dir / ".stages-gate.tmp", "part": comp / "P1" / "B02.tif.part",
                   "quarantine": out_dir / ".quarantine" / "features-gate" / "stack.tif"}
        for p in planted.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("partial")
        keep_user = comp / "P1" / "notes.txt"
        keep_user.write_text("user notes")
        scene_tmp = Path(st["root"]) / "scenes" / "s2" / "leftover.tmp"      # inside a protected input
        scene_tmp.write_text("x")
        study_root = Path(st["root"])
        before_inputs = {p: sha256(Path(p)) for p in snapshot(study_root)}
        before_run = snapshot(self.work / "run")
        dry = first_json(self.cli("clean", "--config", cfg, "--dry-run", "--all-intermediates"))
        if snapshot(self.work / "run") != before_run:
            raise StepError("clean --dry-run changed files")
        listed = {i["path"] for i in dry["would_remove"]}
        real = first_json(self.cli("clean", "--config", cfg, "--all-intermediates"))
        removed = {i["path"] for i in real["removed"]}
        after_run = snapshot(self.work / "run")
        gone = set(before_run) - set(after_run)

        def covered(path: str, roots: set[str]) -> bool:
            return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)
        unlisted_gone = sorted(p for p in gone if not covered(p, removed))
        still_there = sorted(p for p in removed if Path(p).exists())
        problems = []
        if removed != listed:
            problems.append(f"removed != dry-run list: only removed {sorted(removed - listed)[:5]}, "
                            f"only listed {sorted(listed - removed)[:5]}")
        if unlisted_gone:
            problems.append(f"files removed without being reported: {unlisted_gone[:5]}")
        if still_there:
            problems.append(f"reported as removed but still present: {still_there[:5]}")
        if any(p.exists() for p in planted.values()):
            problems.append("planted temporary/quarantine files survived")
        if not keep_user.exists():
            problems.append("unknown user file in a composite folder was removed")
        after_inputs = {p: sha256(Path(p)) for p in snapshot(study_root)}
        if after_inputs != before_inputs:
            problems.append("protected study inputs changed")
        if real.get("errors"):
            problems.append(f"clean errors: {real['errors']}")
        if problems:
            raise StepError("; ".join(problems))
        ev.update({"config": str(cfg), "verdict": report["verdict"],
                   "warnings": sorted({c["code"] for c in report["checks"] if c["status"] == "warn"}),
                   "pipeline_composite": run if isinstance(run, dict) and len(json.dumps(run)) < 2000 else "ok",
                   "dry_run_would_remove": len(listed), "dry_run_bytes": dry.get("bytes"),
                   "removed": len(removed), "files_removed": len(gone),
                   "skipped_protected": [i["path"] for i in real.get("skipped_protected", [])][:5],
                   "kept_user_file": str(keep_user), "protected_inputs_unchanged": len(after_inputs)})

    # ------------------------------------------------------------------ driver
    def run(self) -> dict:
        t0 = time.perf_counter()
        for name, fn in (("1_demo_synthetic_study", self.s1_demo),
                         ("2_embeddings_presto_stack_samples", self.s2_embeddings),
                         ("2b_alphaearth_real_area", self.s2b_alphaearth_real),
                         ("2c_presto_real_area", self.s2c_presto_real),
                         ("3_benchmark_models_ltae_rf_catboost", self.s3_benchmark),
                         ("4_active_learning_round", self.s4_active_learning),
                         ("5_transfer_region", self.s5_transfer),
                         ("6_ftw_benchmark_segmentation", self.s6_ftw),
                         ("6b_ftw_real_area", self.s6b_ftw_real),
                         ("7_cbm_s1_markers_rules", self.s7_cbm),
                         ("8_preflight_clean", self.s8_preflight_clean)):
            self.step(name, fn)
        failed = [s["name"] for s in self.steps if s["status"] == "fail"]
        return {"check": "b_integration_one_study", "python": sys.executable,
                "geocrop": _geocrop_version(), "work": str(self.work), "network": self.network,
                "real_composites": str(self.real) if self.real else None, "cache": str(self.cache),
                "data_note": "steps 1-8 run on SYNTHETIC data unless named *_real_area",
                "seconds": round(time.perf_counter() - t0, 1),
                "summary": {k: sum(s["status"] == k for s in self.steps) for k in ("ok", "fail", "skip")},
                "failed": failed, "steps": self.steps,
                "evidence": {s["name"]: s["evidence"] for s in self.steps}, "commands": self.commands}


def _geocrop_version() -> str:
    try:
        import geocrop
        return f"{geocrop.__version__} ({Path(geocrop.__file__).parent})"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", required=True, help="working directory (created; previous content is replaced)")
    p.add_argument("--network", action="store_true", help="allow downloads (Presto/FTW checkpoints, AlphaEarth)")
    p.add_argument("--real-composites", help="output directory of `geocrop acceptance --out` (real small area)")
    p.add_argument("--cache", help="download cache (default $GEOCROP_CACHE, else <work>/cache)")
    p.add_argument("--json", help="write the gate report here")
    p.add_argument("--keep", action="store_true", help="keep an existing work directory instead of replacing it")
    a = p.parse_args(argv)
    work = Path(a.work).resolve()
    cache = Path(a.cache or os.environ.get("GEOCROP_CACHE") or work / "cache").resolve()
    if work.exists() and not a.keep:
        keep_cache = cache.is_relative_to(work)
        for child in work.iterdir():
            if keep_cache and child == work / "cache" or child.name == "ftw_cache":
                continue                                   # keep downloaded checkpoints between runs
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    work.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    real = Path(a.real_composites).resolve() if a.real_composites else None
    report = Gate(work, a.network, real, cache).run()
    text = json.dumps(report, indent=2, default=str)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(text, encoding="utf8")
    print(json.dumps({"summary": report["summary"], "failed": report["failed"], "seconds": report["seconds"]}))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
