"""Config-driven, resumable pipeline.

    composite -> features -> samples -> train -> predict -> parcels -> validate -> experiments

Each stage records a fingerprint of its configuration section and input files in
`<output>/stages.json`; a stage whose fingerprint is unchanged and whose outputs exist is
skipped, so long regional runs can be resumed after interruption. `force=True` reruns everything.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import config as cfgmod
from . import scenes

STAGES = ("composite", "features", "samples", "train", "predict", "parcels", "validate", "experiments")


def _fp(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (str, Path)) and Path(p).is_file():
            st = Path(p).stat()
            h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class Runner:
    def __init__(self, cfg: dict, force: bool = False, log=print):
        self.cfg, self.force, self.log = cfg, force, log
        self.paths = {k: Path(v) for k, v in cfg["paths"].items()}
        self.out = self.paths["output"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "stages.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _done(self, stage, fp, outputs) -> bool:
        return (not self.force and self.state.get(stage, {}).get("fingerprint") == fp
                and all(Path(o).exists() for o in outputs))

    def _record(self, stage, fp, started, info=None):
        self.state[stage] = {"fingerprint": fp, "seconds": round(time.time() - started, 1), "info": info or {}}
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf8")

    def stage(self, name, fp, outputs, fn):
        if self._done(name, fp, outputs):
            self.log(f"[skip] {name} (unchanged)")
            return self.state[name].get("info")
        self.log(f"[run ] {name}")
        started = time.time()
        info = fn()
        self._record(name, fp, started, info)
        return info

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
        outputs = [comp / w / "B02.tif" for w in windows if s2_scenes.get(w)]
        fp = _fp(c, self.cfg["s1"], self.cfg["windows"], grid,
                 [str(s["path"]) for v in list(s2_scenes.values()) + list(s1_scenes.values()) for s in v])

        def run():
            info = {"s2": scenes.inventory(s2_scenes) if s2_scenes else [],
                    "s1": scenes.inventory(s1_scenes) if s1_scenes else []}
            for w in windows:
                if s2_scenes.get(w):
                    s2.composite([s["path"] for s in s2_scenes[w]], comp / w, reference=grid,
                                 statistic=c["statistic"], scl_invalid=c["scl_invalid"],
                                 cloud_probability_max=c["cloud_probability_max"],
                                 mask_dilation_px=c["mask_dilation_px"],
                                 min_valid_observations=c["min_valid_observations"])
                if s1_scenes.get(w):
                    ref = grid or (comp / w / "B02.tif")
                    s1.composite([s["path"] for s in s1_scenes[w]], comp / w, reference=ref,
                                 separate_orbits=self.cfg["s1"]["separate_orbits"], statistic=self.cfg["s1"]["statistic"])
            missing = [w for w in windows if not s2_scenes.get(w) and not s1_scenes.get(w)]
            if missing:
                raise RuntimeError(f"No scenes for windows {missing}; inspect the inventory and adjust windows")
            return info
        return self.stage("composite", fp, outputs, run)

    def features(self):
        from . import features
        f = self.cfg["features"]
        periods = list(cfgmod.parse_windows(self.cfg["windows"])) or None
        comp = self.paths["composites"]
        inputs = sorted(str(p) for p in comp.rglob("*.tif")) if comp.exists() else []
        fp = _fp(f, periods, *inputs)
        return self.stage("features", fp, [self.paths["features"]], lambda: {"stack": str(features.build_stack(
            comp, self.paths["features"], periods=periods, s2_bands=f["s2_bands"], s2_indices=f["s2_indices"],
            s1=f["s1"], deltas=f["deltas"], seasonal=f["seasonal"]))})

    def samples(self):
        from .samples import extract
        target = self.out / "samples.npz"
        m, cl = self.cfg["model"], self.cfg["classes"]
        fp = _fp(m["max_samples_per_polygon"], cl, self.cfg["seed"], self.paths["features"], self.paths["training"])

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
        fp = _fp(m, self.cfg["seed"], self.out / "samples.npz")

        def run():
            params = {k: m[k] for k in model.RF_KEYS if k in m}
            r = model.train(Samples.load(self.out / "samples.npz"), self.out / "model", params=params,
                            n_splits=m["cv_folds"], spatial_block_m=m["spatial_block_m"], seed=self.cfg["seed"],
                            tune_grid=m.get("tune_grid"), provenance={"config_fingerprint": cfgmod.fingerprint(self.cfg)})
            return {"cv_macro_f1": r["cross_validation"]["macro_f1"], "cv_oa": r["cross_validation"]["overall_accuracy"]}
        return self.stage("train", fp, [target], run)

    def predict(self):
        from . import model
        target = self.out / "prediction" / "classes.tif"
        fp = _fp(self.paths["features"], self.out / "model" / "model.joblib")
        return self.stage("predict", fp, [target], lambda: model.predict(
            self.paths["features"], self.out / "model" / "model.joblib", self.out / "prediction"))

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
        return self.stage("parcels", fp, [target], lambda: parcels.aggregate(
            self.out / "prediction" / "probabilities.tif", self.paths["parcels"], target,
            id_field=p["id_field"], declared_field=p["declared_field"], strategy=p["strategy"],
            interior_buffer_px=p["interior_buffer_px"], min_valid_pixels=p["min_valid_pixels"],
            min_confidence=p["min_confidence"], min_majority_share=p["min_majority_share"],
            min_margin=p["min_margin"], crop_mapping=mapping))

    def validate(self):
        from . import validation
        v = self.cfg["validation"]
        if not self.paths["observations"].exists():
            self.log("[skip] validate (no independent observations)")
            return None
        target = self.out / "validation" / "validation_report.json"
        fp = _fp(v, self.paths["observations"], self.paths["training"], self.out / "prediction" / "classes.tif")

        def run():
            r = validation.evaluate_points(
                self.out / "prediction", self.paths["observations"], self.out / "validation",
                label_field=v["label_field"], date_field=v["date_field"], year=v["year"],
                training=self.paths["training"] if self.paths["training"].exists() else None,
                training_exclusion_m=v["training_exclusion_m"], window_px=v["window_px"],
                bootstrap=v["bootstrap"], seed=self.cfg["seed"])
            return {"oa": r["accuracy"]["overall_accuracy"], "macro_f1": r["accuracy"]["macro_f1"],
                    "selection": r["selection"]}
        return self.stage("validate", fp, [target], run)

    def experiments(self):
        from . import experiments, features
        from .samples import Samples
        m = self.cfg["model"]
        exps = self.cfg["experiments"] or experiments.DEFAULT_EXPERIMENTS
        target = self.out / "experiments" / "ablation.csv"
        fp = _fp(exps, m, self.out / "samples.npz")

        def run():
            params = {k: m[k] for k in ("n_estimators", "max_features", "min_samples_leaf", "class_weight", "n_jobs") if k in m}
            t = experiments.run(Samples.load(self.out / "samples.npz"), features.load_manifest(self.paths["features"]),
                                self.out / "experiments", exps, params, m["cv_folds"], m["spatial_block_m"],
                                self.cfg["seed"], self.cfg["classes"]["targets"])
            return {"rows": int(len(t))}
        return self.stage("experiments", fp, [target], run)


def run(config_path: str | Path, stages=None, force: bool = False, log=print) -> dict:
    cfg = cfgmod.load(config_path)
    runner = Runner(cfg, force=force, log=log)
    chosen = list(stages) if stages else list(STAGES)
    unknown = set(chosen) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}; choose from {STAGES}")
    results = {}
    for name in STAGES:
        if name in chosen:
            results[name] = getattr(runner, name)()
    from .provenance import run_manifest
    run_manifest(runner.out, config=cfg, inputs=[runner.paths[k] for k in ("training", "parcels", "observations")])
    return results
