"""Config-driven, resumable pipeline.

    composite -> features (+ time series) -> samples -> train -> predict -> cbm -> parcels -> validate -> experiments

Each stage records in `<output>/stages.json` its status and a fingerprint made of its configuration,
the SHA-256 content digests of its inputs (cached per path/size/mtime, see `geocrop.cache`), the engine
source code and the fingerprints of the stages it depends on, plus the identity of every output. A stage
is skipped only if it succeeded, its fingerprint is unchanged and all outputs are present and intact, so
long regional runs resume after interruption while any input, code or upstream change re-runs dependent
stages. `force=True` reruns everything. A lock prevents concurrent runs on one output directory and
`run.log` records timing and peak memory per stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from . import __version__, scenes
from . import config as cfgmod
from .cache import HashCache, RunLock

STAGES = ("composite", "features", "samples", "train", "predict", "cbm", "parcels", "validate", "experiments")
CACHE_SCHEMA = 3
_HASHED_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".txt", ".py"}
DEPENDS = {"composite": (), "features": ("composite",), "samples": ("features",), "train": ("samples",),
           "predict": ("train", "features"), "cbm": (), "parcels": ("predict", "train", "cbm"),
           "validate": ("predict",), "experiments": ("samples", "features")}
_HASHER: HashCache | None = None   # content digests (SHA-256, cached by path/size/mtime) when a run is active


def _peak_rss_mb() -> float | None:
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, AttributeError):
        return None


def _file_identity(path: Path) -> dict:
    """Content identity: SHA-256 (cached) during runs, so touching or copying a file does not invalidate a stage;
    without an active hash cache, size + mtime plus a hash of small metadata files."""
    st = path.stat()
    if _HASHER is not None:
        return {"bytes": st.st_size, "sha256": _HASHER.file(path)}
    identity = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if path.suffix.lower() in _HASHED_SUFFIXES and st.st_size <= 1024 * 1024:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _part_identity(part):
    if isinstance(part, (str, Path)):
        path = Path(part)
        if path.is_file():
            return {"path": str(path.resolve()), **_file_identity(path)}
        if path.is_dir():
            return {"directory": str(path.resolve()), "files": {
                str(p.relative_to(path)): _file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}}
    if isinstance(part, dict):
        return {str(k): _part_identity(v) for k, v in part.items()}
    if isinstance(part, (list, tuple)):
        return [_part_identity(v) for v in part]
    return part


def _fp(*parts) -> str:
    body = json.dumps([_part_identity(p) for p in parts], sort_keys=True, default=str).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def _output_identity(outputs) -> dict | None:
    result = {}
    for raw in outputs:
        path = Path(raw)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        result[str(path.resolve())] = _file_identity(path)
    return result or None


class Runner:
    def __init__(self, cfg: dict, force: bool = False, log=print):
        self.cfg, self.force, self.log = cfg, force, log
        self.paths = {k: Path(v) for k, v in cfg["paths"].items()}
        self.out = self.paths["output"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "stages.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if not isinstance(self.state, dict):
            raise ValueError(f"Invalid stage state in {self.state_path}: expected a JSON object")
        global _HASHER
        _HASHER = HashCache(self.out / ".hashcache.json", cfg["cache"]["hash"])
        self.hashes = _HASHER
        self.code_fp = _fp(sorted(Path(__file__).parent.glob("*.py")))

    def _note(self, message: str, echo: bool = True):
        """Console message via `self.log` (replaceable) and, always, a timestamped line in run.log."""
        from datetime import datetime, timezone
        if echo:
            self.log(message)
        with (self.out / "run.log").open("a", encoding="utf8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")

    def _done(self, stage, fp, outputs) -> bool:
        entry = self.state.get(stage, {})
        identity = _output_identity(outputs)
        return (not self.force and entry.get("cache_schema") == CACHE_SCHEMA and entry.get("status") == "success"
                and entry.get("fingerprint") == fp and identity is not None and entry.get("outputs") == identity)

    def _save_state(self):
        # The previous JSON remains readable if the process stops before replacement.
        with tempfile.NamedTemporaryFile("w", dir=self.out, prefix=".stages-", suffix=".tmp", delete=False,
                                         encoding="utf8") as fh:
            tmp = Path(fh.name)
            try:
                json.dump(self.state, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp, self.state_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _record(self, stage, fp, started, status, info=None, outputs=None, error=None):
        self.state[stage] = {"cache_schema": CACHE_SCHEMA, "status": status, "fingerprint": fp,
                             "seconds": round(time.time() - started, 1), "peak_rss_mb": _peak_rss_mb(),
                             "engine": __version__, "info": info or {}}
        if outputs is not None:
            self.state[stage]["outputs"] = outputs
        if error is not None:
            self.state[stage]["error"] = {"type": type(error).__name__, "message": str(error)}
        self._save_state()

    def stage(self, name, fp, outputs, fn):
        upstream = {d: self.state.get(d, {}).get("fingerprint") for d in DEPENDS.get(name, ())}
        fp = _fp(CACHE_SCHEMA, self.code_fp, upstream, fp)
        if self._done(name, fp, outputs):
            self._note(f"[skip] {name} (unchanged)")
            return self.state[name].get("info")
        self._disk_guard(name, outputs)
        self._note(f"[run ] {name}")
        started, started_ns = time.time(), time.time_ns() - 2_000_000_000   # 2 s: coarse filesystem clocks
        self._record(name, fp, started, "running")
        try:
            info = fn()
            identity = _output_identity(outputs)
            if identity is None:
                missing = [str(p) for p in outputs if not Path(p).is_file() or Path(p).stat().st_size == 0]
                raise RuntimeError(f"Stage {name} did not produce its complete nonempty output set: {missing}")
            self._record(name, fp, started, "success", info, identity)
        except BaseException as exc:
            from .cleanup import quarantine_partial
            moved = quarantine_partial(outputs, started_ns, self.out, name)   # partial outputs never look final
            self._record(name, fp, started, "failed", {"quarantined": moved} if moved else None, error=exc)
            self._note(f"[fail] {name}: {type(exc).__name__}: {exc}", echo=False)
            if moved:
                self._note(f"[clean] {name}: {len(moved)} partial output(s) moved to {self.out / '.quarantine'}")
            raise
        finally:
            self.hashes.save()
        self._note(f"[done] {name} in {self.state[name]['seconds']} s, peak RSS {self.state[name]['peak_rss_mb']} MB",
                   echo=False)
        return info

    def _disk_guard(self, name, outputs):
        """Fail before a heavy stage starts when its output volume lacks the estimated space (see preflight)."""
        from .preflight import HEAVY_STAGES, DiskSpaceError, check_disk
        if name not in HEAVY_STAGES or not self.cfg.get("resources", {}).get("disk_guard", True):
            return
        existing = sum(Path(p).stat().st_size for p in outputs if Path(p).is_file())
        try:
            info = check_disk(self.cfg, name, existing)
        except DiskSpaceError as exc:
            self._note(f"[fail] {name}: {exc}")
            raise
        if info:
            self._note(f"[disk] {name}: {info['free_gb']} GB free, {info['need_gb']} GB required", echo=False)

    # ------------------------------------------------------------------ stages
    def composite(self):
        from . import s1, s2
        windows = cfgmod.parse_windows(self.cfg["windows"])
        if not windows:
            raise cfgmod.ConfigError("Define windows: in the configuration")
        c = self.cfg["compositing"]
        comp = self.paths["composites"]
        s2_scenes = scenes.assign(scenes.discover(self.paths["s2_scenes"]), windows) if self.paths["s2_scenes"].exists() else {}
        s1_scenes = scenes.assign(scenes.discover(self.paths["s1_scenes"]), windows) if self.paths["s1_scenes"].exists() else {}
        grid = self.cfg["grid"].get("reference")
        missing = [w for w in windows if not s2_scenes.get(w) and not s1_scenes.get(w)]
        if missing:
            raise RuntimeError(f"No scenes for windows {missing}; inspect the inventory and adjust windows")
        outputs = []
        for w in windows:
            if s2_scenes.get(w):
                outputs.extend(comp / w / name for name in
                               [*[f"{band}.tif" for band in s2.S2_BANDS], "valid_count.tif", "composite.json"])
            if s1_scenes.get(w):
                prefixes = {s1.ORBITS.get(str(m.get("orbit_direction", "")).upper(), "")
                            for m in s1_scenes[w]} if self.cfg["s1"]["separate_orbits"] else {""}
                outputs.extend(comp / w / f"{prefix + '_' if prefix else ''}{pol}.tif"
                               for prefix in sorted(prefixes) for pol in ("VV", "VH"))
                outputs.append(comp / w / "composite_s1.json")
        known_products = {*[f"{band}.tif" for band in s2.S2_BANDS], "valid_count.tif", "composite.json",
                          "composite_s1.json", *[f"{prefix}{pol}.tif" for prefix in ("", "ASC_", "DESC_")
                                                  for pol in ("VV", "VH")]}
        expected = set(outputs)
        conflicting = [comp / w / name for w in windows for name in sorted(known_products)
                       if (comp / w / name).is_file() and comp / w / name not in expected]
        if conflicting:
            raise RuntimeError("Composite products conflict with the current sensor/orbit selection: "
                               f"{[str(p) for p in conflicting]}. Choose a new paths.composites directory "
                               "or deliberately clean the conflicting generated products before retrying.")
        fp = _fp(c, self.cfg["s1"], self.cfg["windows"], grid,
                 [s["path"] for v in list(s2_scenes.values()) + list(s1_scenes.values()) for s in v])

        def run():
            info = {"s2": scenes.inventory(s2_scenes) if s2_scenes else [],
                    "s1": scenes.inventory(s1_scenes) if s1_scenes else []}
            for w in windows:
                if s2_scenes.get(w):
                    s2.composite([s["path"] for s in s2_scenes[w]], comp / w, reference=grid,
                                 statistic=c["statistic"], scl_invalid=c["scl_invalid"],
                                 cloud_probability_max=c["cloud_probability_max"],
                                 mask_dilation_px=c["mask_dilation_px"],
                                 min_valid_observations=c["min_valid_observations"],
                                 temporal_screening=c["temporal_screening"])
                if s1_scenes.get(w):
                    ref = grid or (comp / w / "B02.tif")
                    s1.composite([s["path"] for s in s1_scenes[w]], comp / w, reference=ref,
                                 separate_orbits=self.cfg["s1"]["separate_orbits"], statistic=self.cfg["s1"]["statistic"])
            return info
        return self.stage("composite", fp, outputs, run)

    def features(self):
        from . import features
        f = self.cfg["features"]
        periods = list(cfgmod.parse_windows(self.cfg["windows"])) or None
        comp = self.paths["composites"]
        inputs = sorted(p for p in comp.rglob("*.tif")) if comp.exists() else []
        t = self.cfg["timeseries"]
        target = self.paths["features"]
        outputs = [target, features.manifest_path(target)]
        ts_inputs = []
        if t["enabled"]:
            ts_target = target.with_name(target.stem + "_ts.tif")
            outputs.extend([ts_target, features.manifest_path(ts_target)])
            from .tsfeatures import quality_path
            outputs.extend([quality_path(ts_target), quality_path(ts_target).with_suffix(".json")])
            if periods and comp.exists():
                win = target.with_name(target.stem + "_windows.tif")
                outputs.extend([win, features.manifest_path(win)])
            ts_inputs = [self.paths["s2_scenes"], self.paths["s1_scenes"], self.cfg["grid"].get("reference"),
                         self.cfg["compositing"]["scl_invalid"], self.cfg["compositing"]["cloud_probability_max"]]
        fp = _fp(f, t, periods, inputs, ts_inputs)

        def run():
            parts = []
            if periods and comp.exists():
                win = target.with_name(target.stem + "_windows.tif") if t["enabled"] else target
                parts.append(features.build_stack(comp, win, periods=periods, s2_bands=f["s2_bands"],
                                                  s2_indices=f["s2_indices"], s1=f["s1"], deltas=f["deltas"],
                                                  seasonal=f["seasonal"]))
            if t["enabled"]:
                from datetime import date

                from .tsfeatures import build_timeseries_stack, merge_stacks
                if not (t["start"] and t["end"]):
                    raise cfgmod.ConfigError("timeseries.start and timeseries.end are required")
                ref = self.cfg["grid"].get("reference") or (comp / periods[0] / "B02.tif")
                tsp = build_timeseries_stack(
                    [self.paths["s2_scenes"], self.paths["s1_scenes"]], ref, target.with_name(target.stem + "_ts.tif"),
                    date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"])), t["s2_indices"],
                    t["s1_indices"], t["step_days"], t["output_step_days"], t["lambda"], t["envelope_iterations"],
                    t["harmonic_order"], self.cfg["compositing"]["scl_invalid"],
                    self.cfg["compositing"]["cloud_probability_max"],
                    min_observed_fraction=t["min_observed_fraction"], max_gap_days=t["max_gap_days"])
                parts.append(tsp)
                if len(parts) > 1:
                    merge_stacks(parts, target)
                else:
                    import shutil
                    shutil.copy(tsp, target)
                    shutil.copy(features.manifest_path(tsp), features.manifest_path(target))
            if not parts:
                raise cfgmod.ConfigError("Features require window composites or enabled time-series inputs")
            return {"stack": str(target), "parts": [str(p) for p in parts]}
        return self.stage("features", fp, outputs, run)

    def samples(self):
        from .samples import extract
        target = self.out / "samples.npz"
        m, cl = self.cfg["model"], self.cfg["classes"]
        from .features import manifest_path
        fp = _fp(m["max_samples_per_polygon"], cl, self.cfg["seed"], self.paths["features"],
                 manifest_path(self.paths["features"]), self.paths["training"])

        def run():
            s = extract(self.paths["features"], self.paths["training"], label_field=cl["label_field"],
                        id_field=cl["id_field"], max_per_polygon=m["max_samples_per_polygon"], seed=self.cfg["seed"])
            s.save(target)
            return {"samples": int(len(s.y)), "polygons": int(len(set(s.polygon)))}
        return self.stage("samples", fp, [target], run)

    def train(self):
        from . import model
        from .samples import Samples
        m = self.cfg["model"]
        target = self.out / "model" / "model.joblib"
        outputs = [target.with_name(name) for name in (
            "model.joblib", "train_report.json", "feature_importance.csv", "cv_confusion.csv",
            "oof_polygons.csv", "conformal.json", "suspect_labels.csv", "aoa.json")]
        if m.get("holdout_fraction"):
            outputs.append(target.with_name("holdout_predictions.csv"))
        if m.get("calibration_mode") == "holdout":
            outputs.extend(target.with_name(name) for name in ("calibration_polygons.csv", "calibration_split.json"))
        fp = _fp(m, self.cfg["seed"], self.out / "samples.npz")

        def run():
            params = {k: m[k] for k in model.RF_KEYS if k in m}
            r = model.train(Samples.load(self.out / "samples.npz"), self.out / "model", params=params,
                            n_splits=m["cv_folds"], spatial_block_m=m["spatial_block_m"], seed=self.cfg["seed"],
                            tune_grid=m.get("tune_grid"), provenance={"config_fingerprint": cfgmod.fingerprint(self.cfg)},
                            calibrate=m["calibrate"], spatial_buffer_m=m["spatial_buffer_m"],
                            calibration_mode=m.get("calibration_mode", "oof"),
                            calibration_fraction=m.get("calibration_fraction", 0.2),
                            holdout_fraction=m["holdout_fraction"], inner_splits=m["inner_splits"])
            return {"cv_macro_f1": r["cross_validation"]["macro_f1"], "cv_oa": r["cross_validation"]["overall_accuracy"]}
        return self.stage("train", fp, outputs, run)

    def predict(self):
        from . import model
        pred = self.out / "prediction"
        outputs = [pred / name for name in ("classes.tif", "probabilities.tif", "uncertainty.tif", "class_legend.json",
                                           "prediction_manifest.json")]
        aoa = self.out / "model" / "aoa.json"
        if aoa.exists():
            outputs.append(pred / "aoa.tif")
        fp = _fp(self.paths["features"], self.out / "model" / "model.joblib", aoa)
        return self.stage("predict", fp, outputs, lambda: model.predict(
            self.paths["features"], self.out / "model" / "model.joblib", self.out / "prediction"))

    def cbm(self):
        from datetime import date

        from . import cbm
        c, t = self.cfg["cbm"], self.cfg["timeseries"]
        if not c["enabled"]:
            return None
        if not self.paths["parcels"].exists():
            self.log("[skip] cbm (no parcel layer)")
            return None
        if not (t["start"] and t["end"]):
            raise cfgmod.ConfigError("cbm needs timeseries.start/end for the season grid")
        start, end = date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"]))
        target = self.out / "cbm" / "markers.csv"
        ref = self.cfg["grid"].get("reference") or next(self.paths["composites"].glob("*/B02.tif"))
        fp = _fp(c, t, self.paths["parcels"], self.paths["s2_scenes"], ref,
                 self.cfg["parcels"]["id_field"], self.cfg["parcels"]["interior_buffer_px"])

        def run():
            idx = [c["index"], "BSI"] if c["index"] != "BSI" else ["BSI"]
            sig = cbm.parcel_signatures(self.paths["s2_scenes"], self.paths["parcels"], ref,
                                        self.out / "cbm" / "signatures.csv", start, end, idx,
                                        id_field=self.cfg["parcels"]["id_field"],
                                        interior_buffer_px=self.cfg["parcels"]["interior_buffer_px"])
            m = cbm.markers(sig, start, end, index=c["index"], min_coverage=c["min_coverage"])
            m.to_csv(target, index=False)
            return {"parcels": int(len(m))}
        return self.stage("cbm", fp, [target, self.out / "cbm" / "signatures.csv"], run)

    def parcels(self):
        from . import parcels
        p = self.cfg["parcels"]
        if not self.paths["parcels"].exists():
            self.log("[skip] parcels (no parcel layer)")
            return None
        target = self.out / "parcels" / "parcels.gpkg"
        fp = _fp(p, self.paths["parcels"], self.out / "prediction" / "probabilities.tif")
        mapping = None
        if p.get("crop_mapping"):
            from .adapters import read_mapping
            mapping = read_mapping(p["crop_mapping"])
        conformal = self.out / "model" / "conformal.json"
        aoa = self.out / "prediction" / "aoa.tif"
        markers_csv = self.out / "cbm" / "markers.csv"
        calendar = None
        if self.cfg["cbm"]["calendar"]:
            import yaml
            calendar = yaml.safe_load(Path(self.cfg["cbm"]["calendar"]).read_text(encoding="utf8"))

        def run():
            import pandas as pd
            return parcels.aggregate(
                self.out / "prediction" / "probabilities.tif", self.paths["parcels"], target,
                id_field=p["id_field"], declared_field=p["declared_field"], strategy=p["strategy"],
                interior_buffer_px=p["interior_buffer_px"], min_valid_pixels=p["min_valid_pixels"],
                min_confidence=p["min_confidence"], min_majority_share=p["min_majority_share"],
                min_margin=p["min_margin"], crop_mapping=mapping,
                conformal=conformal if conformal.exists() else None, alpha=p["conformal_alpha"],
                aoa=aoa if aoa.exists() else None, min_aoa_fraction=p["min_aoa_fraction"],
                markers=pd.read_csv(markers_csv) if self.cfg["cbm"]["enabled"] and markers_csv.exists() else None,
                calendar=calendar)
        fp = _fp(fp, mapping, conformal, aoa, markers_csv, calendar, self.cfg["cbm"]["enabled"],
                 self.out / "prediction" / "prediction_manifest.json")
        return self.stage("parcels", fp, [target, target.with_suffix(".csv"), target.with_suffix(".summary.json")], run)

    def validate(self):
        from . import validation
        v = self.cfg["validation"]
        if not self.paths["observations"].exists():
            self.log("[skip] validate (no independent observations)")
            return None
        target = self.out / "validation" / "validation_report.json"
        fp = _fp(v, self.cfg["seed"], self.paths["observations"], self.paths["training"],
                 [self.out / "prediction" / name for name in
                  ("classes.tif", "probabilities.tif", "class_legend.json", "prediction_manifest.json")])

        def run():
            r = validation.evaluate_points(
                self.out / "prediction", self.paths["observations"], self.out / "validation",
                label_field=v["label_field"], date_field=v["date_field"], year=v["year"],
                training=self.paths["training"] if self.paths["training"].exists() else None,
                training_exclusion_m=v["training_exclusion_m"], window_px=v["window_px"],
                bootstrap=v["bootstrap"], seed=self.cfg["seed"])
            return {"oa": r["accuracy"]["overall_accuracy"], "macro_f1": r["accuracy"]["macro_f1"],
                    "selection": r["selection"]}
        return self.stage("validate", fp, [target, target.with_name("observations_evaluated.csv"),
                                           target.with_name("confusion_matrix.csv")], run)

    def experiments(self):
        from . import experiments, features
        from .samples import Samples
        m = self.cfg["model"]
        exps = self.cfg["experiments"] or experiments.DEFAULT_EXPERIMENTS
        target = self.out / "experiments" / "ablation.csv"
        fp = _fp(exps, m, self.cfg["seed"], self.cfg["classes"]["targets"], self.out / "samples.npz",
                 features.manifest_path(self.paths["features"]))

        def run():
            from .model import RF_KEYS
            params = {k: m[k] for k in RF_KEYS if k in m and m[k] is not None}
            t = experiments.run(Samples.load(self.out / "samples.npz"), features.load_manifest(self.paths["features"]),
                                self.out / "experiments", exps, params, m["cv_folds"], m["spatial_block_m"],
                                self.cfg["seed"], self.cfg["classes"]["targets"],
                                spatial_buffer_m=m["spatial_buffer_m"], calibration_mode=m["calibration_mode"],
                                calibration_fraction=m["calibration_fraction"],
                                holdout_fraction=m["holdout_fraction"])
            return {"rows": int(len(t))}
        return self.stage("experiments", fp, [target, target.with_suffix(".json")], run)


def run(config_path: str | Path, stages=None, force: bool = False, log=print) -> dict:
    cfg = cfgmod.load(config_path)
    runner = Runner(cfg, force=force, log=log)
    chosen = list(stages) if stages else list(STAGES)
    unknown = set(chosen) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}; choose from {STAGES}")
    results = {}
    global _HASHER
    try:
        with RunLock(runner.out):
            runner._note(f"geocrop {__version__}: stages {chosen}, config {config_path}", echo=False)
            for name in STAGES:
                if name in chosen:
                    results[name] = getattr(runner, name)()
            from .provenance import run_manifest
            run_manifest(runner.out, config=cfg,
                         inputs=[runner.paths[k] for k in ("training", "parcels", "observations")])
    finally:
        runner.hashes.save()
        _HASHER = None
    return results
