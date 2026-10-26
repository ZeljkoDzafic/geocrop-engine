"""Local, file-based model and benchmark registry (``geocrop models ...``, ``geocrop benchmarks ...``).

Layout of a registry directory (default ``<workspace>/registry``; ``$GEOCROP_REGISTRY`` or ``--registry``)::

    models/<name>/<version>/model.joblib        copied (or hard-linked) artefact, read-only
    models/<name>/<version>/<companions>        aoa.json, conformal.json, train_report.json, ... when present
    models/<name>/<version>/entry.json          immutable metadata; `entry_sha256` covers the whole body
    models/<name>/stages.jsonl                  append-only stage log (staging / production / archived)
    benchmarks/<run id>.json                    one immutable record per benchmark/ablation/transfer run

Entries are never rewritten. Stage changes are appended to a log, so the history of promotions survives.
Comparisons are only made between results that were scored on identical splits: every validation result carries
a fingerprint of its sample support and of the exact fold (or holdout, or transfer-group) assignment, and
`compare` / `leaderboard` refuse or separate results whose fingerprints differ.

Pretrained external checkpoints (Fields of The World, Presto) appear as read-only entries of kind
``pretrained``; their URL, size, SHA-256 and licence come from the pins in `geocrop.ftw` and
`geocrop.embedding_sources`. They are downloaded at runtime by those modules, never stored here.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .hashing import sha256

SCHEMA_MODEL = "geocrop-registry-model/1"
SCHEMA_BENCHMARK = "geocrop-registry-benchmark/1"
SCHEMA_VERSION = 1
ENV = "GEOCROP_REGISTRY"
#: Values of $GEOCROP_REGISTRY that switch automatic benchmark recording off.
DISABLED = frozenset({"off", "0", "none", "false", "no"})
STAGES = ("staging", "production", "archived")
COMPANIONS = ("aoa.json", "conformal.json", "train_report.json", "calibration_split.json",
              "feature_importance.csv", "cv_confusion.csv", "transfer_per_group.csv")
CLASSIFICATION_TASK = "crop-classification"
SEGMENTATION_TASK = "field-segmentation"
#: Metrics where a smaller value ranks higher in leaderboards.
LOWER_IS_BETTER = frozenset({
    "ece", "seconds", "cv_seconds", "fit_seconds", "fit_s", "predict_s", "wall_s", "runtime_seconds",
    "fold_macro_f1_std", "macro_f1_fold_std", "macro_f1_sd", "overall_accuracy_sd", "holdout_macro_f1_sd",
    "over_segmentation_mean", "under_segmentation_mean", "D_mean", "review_share", "transfer_gap_macro_f1",
    "gap_macro_f1", "gap_overall_accuracy", "gap_kappa",
})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RegistryError(ValueError):
    """A registry operation cannot be carried out (unknown entry, refused comparison, tampered file, ...)."""


class ContractError(RegistryError):
    """A feature stack does not satisfy the input contract of a registered model."""


# --------------------------------------------------------------------------- basics

def default_root(root: str | Path | None = None) -> Path:
    """Registry directory: explicit `root`, else ``$GEOCROP_REGISTRY``, else ``./registry``."""
    if root:
        return Path(root)
    env = os.environ.get(ENV, "").strip()
    if env and env.lower() not in DISABLED:
        return Path(env)
    return Path.cwd() / "registry"


def recording_enabled() -> bool:
    """Automatic benchmark recording is on unless ``$GEOCROP_REGISTRY`` is ``off`` (or 0/none/false/no)."""
    return os.environ.get(ENV, "").strip().lower() not in DISABLED


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(value):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None, tuples to lists."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(body) -> str:
    """SHA-256 of the canonical JSON form of `body`."""
    text = json.dumps(_plain(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def _digest(values) -> str:
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(str(values.shape).encode())
    if values.dtype.hasobject or values.dtype.kind in "US":
        h.update(json.dumps([str(v) for v in values.ravel().tolist()], ensure_ascii=False).encode())
    else:
        h.update(memoryview(values).cast("B"))
    return h.hexdigest()


def _engine() -> dict:
    from .provenance import _run
    here = Path(__file__).parent
    return {"version": __version__, "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=here),
            "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here))}


def _seal(body: dict) -> dict:
    body = _plain({k: v for k, v in body.items() if k != "entry_sha256"})
    body["entry_sha256"] = _canonical(body)
    return body


def _write_once(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf8") as fh:      # exclusive create: entries are never overwritten
        json.dump(body, fh, indent=2, ensure_ascii=False)
    _read_only(path)


def _read_only(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


# --------------------------------------------------------------------------- fingerprints

def support_fingerprint(samples) -> str:
    """Identity of the sampled reference support (labels, polygon ids, pixel centres, CRS); not of feature values,
    so feature-set ablations on the same samples share it."""
    return _canonical({"y": _digest(np.asarray(samples.y).astype(str)),
                       "polygon": _digest(np.asarray(samples.polygon).astype(str)),
                       "x": _digest(np.asarray(samples.x, dtype="float64")),
                       "yy": _digest(np.asarray(samples.yy, dtype="float64")), "crs": str(samples.crs)})


def features_fingerprint(samples, names=None) -> str:
    """Identity of the feature values (all features, or the `names` subset)."""
    names = list(names or samples.features)
    X = samples.X if names == list(samples.features) else samples.subset(names)
    return _canonical({"X": _digest(np.asarray(X)), "features": names})


def classification_split(samples, *, n_splits: int, seeds, spatial_block_m: float | None,
                         spatial_buffer_m: float = 0.0, holdout=None, strategies=()) -> dict:
    """Fingerprints of a classification evaluation design on `samples` (the objects actually cross-validated).

    ``cv`` hashes the exact train/test indices of every spatial fold for every seed (so it also changes when the
    fold algorithm changes), ``holdout`` the development/locked-holdout support, ``transfer`` the region/year group
    assignment of each strategy, and ``split`` all of them together.
    """
    from .model import _fold_plan, cv_groups, spatial_blocks

    support = support_fingerprint(samples)
    groups = spatial_blocks(samples, spatial_block_m)
    coords = np.column_stack([samples.x, samples.yy])
    folds, sizes = [], []
    for seed in seeds:
        plan = _fold_plan(samples.y, groups, n_splits, int(seed), polygon=samples.polygon, coordinates=coords,
                          crs=samples.crs, spatial_buffer_m=spatial_buffer_m)
        folds.append([[int(seed), k, _digest(tr), _digest(te)] for k, (tr, te, _) in enumerate(plan)])
        sizes.append([int(len(te)) for _, te, _ in plan])
    out = {"support": support,
           "cv": _canonical({"support": support, "folds": folds, "spatial_block_m": spatial_block_m,
                             "spatial_buffer_m": spatial_buffer_m}),
           "seeds": [int(s) for s in seeds], "n_folds": len(sizes[0]) if sizes else 0, "fold_test_sizes": sizes,
           "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m,
           "holdout": None, "transfer": None}
    if holdout is not None:
        out["holdout"] = _canonical({"development": support, "holdout": support_fingerprint(holdout),
                                     "spatial_block_m": spatial_block_m})
    if strategies:
        per = {}
        for strategy in strategies:
            try:
                per[strategy] = _digest(np.asarray(cv_groups(samples, strategy, spatial_block_m)).astype(str))
            except ValueError:
                per[strategy] = None
        out["transfer"] = _canonical({"support": support, "groups": per})
        out["transfer_strategies"] = list(strategies)
    out["split"] = _canonical({k: out[k] for k in ("cv", "holdout", "transfer")})
    return out


# --------------------------------------------------------------------------- model entries

def _models_dir(root) -> Path:
    return default_root(root) / "models"


def _versions(root, name: str) -> list[int]:
    d = _models_dir(root) / name
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit() and (p / "entry.json").exists())


def _entry_path(root, name: str, version: int) -> Path:
    return _models_dir(root) / name / str(version) / "entry.json"


def _stage_log(root, name: str) -> list[dict]:
    path = _models_dir(root) / name / "stages.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def stages(name: str, root=None) -> dict[int, str]:
    """Current stage of every version of `name` (``none`` when never promoted)."""
    current = {v: "none" for v in _versions(root, name)}
    for event in _stage_log(root, name):
        current[int(event["version"])] = event["stage"]
    return current


def _read_samples_summary(samples) -> dict:
    out = {}
    for key in ("year", "region"):
        values = getattr(samples, key, None)
        if values is not None:
            keys, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
            out[f"{key}s"] = {str(k): int(c) for k, c in zip(keys, counts)}
    return out


def _model_fingerprints(samples, payload: dict, report: dict | None) -> dict:
    """Recompute the train-time split fingerprints from the samples and check them against train_report.json."""
    from .model import selection_samples, split_final_holdout

    if not report:
        return {"verified": False, "reason": "no train_report.json for this model"}
    cv = report.get("cross_validation") or {}
    block = report.get("spatial_block_m")
    buffer = float(report.get("spatial_buffer_m") or 0)
    seed = int(payload.get("seed", 42))
    holdout_fraction = float((report.get("holdout_design") or {}).get("fraction") or 0)
    mode = report.get("calibration_mode", "oof")
    fraction = float((report.get("calibration_split") or {}).get("fraction_of_groups") or 0.2)
    dev, test = split_final_holdout(samples, block, holdout_fraction, seed)
    dev, _ = selection_samples(dev, block, seed, buffer, mode, fraction)
    strategy = report.get("cv_strategy", "spatial_blocks")
    n_folds = len(cv.get("folds") or []) or 5
    fp = classification_split(dev, n_splits=n_folds, seeds=[seed], spatial_block_m=block, spatial_buffer_m=buffer,
                              holdout=test, strategies=() if strategy == "spatial_blocks" else (strategy,))
    expected = [int(f["n_test"]) for f in cv.get("folds") or []]
    problems = []
    if int(report.get("samples", len(dev.y))) != len(dev.y):
        problems.append(f"train_report has {report.get('samples')} samples, reconstruction {len(dev.y)}")
    if expected and expected != fp["fold_test_sizes"][0]:
        problems.append(f"fold sizes {expected} differ from reconstruction {fp['fold_test_sizes'][0]}")
    if test is not None and (report.get("holdout_design") or {}).get("samples") not in (None, len(test.y)):
        problems.append("holdout size differs from reconstruction")
    if problems:
        return {"verified": False, "reason": "; ".join(problems)}
    return {"verified": True, "support": fp["support"], "cv": fp["cv"], "holdout": fp["holdout"],
            "transfer": fp["transfer"], "seed": seed, "n_folds": fp["n_folds"]}


def _metrics(section: dict | None, keys=("overall_accuracy", "macro_f1", "kappa")) -> dict | None:
    if not section:
        return None
    out = {k: section.get(k) for k in keys if k in section}
    boot = section.get("bootstrap") or {}
    for k in ("overall_accuracy_ci", "macro_f1_ci"):
        if k in boot:
            out[k] = boot[k]
    return out


def _validation(report: dict | None, conformal: dict | None) -> dict:
    report = report or {}
    cv = report.get("cross_validation")
    out = {"cv": _metrics(cv), "nested_cv": _metrics(report.get("nested_cv")),
           "holdout": _metrics(report.get("holdout")), "transfer": None, "conformal": None}
    if cv:
        out["cv"].update({"fold_macro_f1_std": cv.get("fold_macro_f1_std"),
                          "ece": (cv.get("calibration") or {}).get("ece"), "strategy": cv.get("strategy")})
    tv = report.get("transfer_validation")
    if tv:
        out["transfer"] = {"strategy": tv.get("strategy"),
                           **{f"out_of_domain_{k}": tv["out_of_domain"].get(k)
                              for k in ("overall_accuracy", "macro_f1", "kappa")},
                           **{f"gap_{k}": v for k, v in (tv.get("transfer_gap") or {}).items()},
                           "worst_group": tv.get("worst_group")}
    if conformal:
        out["conformal"] = {k: conformal.get(k) for k in ("level", "method", "coverage_guarantee", "calibration_mode")
                            if k in conformal}
        out["conformal"]["coverage_check"] = conformal.get("coverage_check")
    return out


def _licences(manifest: list[dict] | None, extra: dict | None) -> list[dict]:
    """Licences of pretrained components: embedding sources found in the feature manifest plus declared ones."""
    from .embeddings import SOURCE_INFO

    out = []
    sensors = {f.get("sensor") for f in manifest or [] if f.get("group") == "emb"}
    for source, info in SOURCE_INFO.items():
        if info["sensor"] in sensors:
            out.append({"component": f"{source} embeddings", "licence": info["licence"],
                        "attribution": info.get("attribution")})
    for component, licence in (extra or {}).items():
        out.append({"component": component, "licence": licence, "attribution": None})
    return out


def _stack_info(stack: str | Path) -> dict:
    import rasterio

    from .features import manifest_path
    from .raster import feature_names
    with rasterio.open(stack) as ds:
        info = {"features": feature_names(ds), "crs": ds.crs.to_string() if ds.crs else None,
                "resolution": [float(ds.res[0]), float(ds.res[1])], "nodata": ds.nodata}
    path = manifest_path(stack)
    body = _load(path) if path.is_file() else None
    info["manifest"] = body
    return info


def _hcat_mapping(path: str | Path | None, classes: list[str]) -> dict | None:
    if not path:
        return None
    from .hcat import export_mapping, read_rules
    mapping = export_mapping(read_rules(path))
    return {"source": Path(path).name, "mapping": {c: mapping[c] for c in classes if c in mapping},
            "unmapped_classes": [c for c in classes if c not in mapping]}


def register(model_path: str | Path, name: str, *, root=None, tags=(), samples: str | Path | None = None,
             features: str | Path | None = None, hcat: str | Path | None = None, taxonomy: str | None = None,
             licences: dict | None = None, description: str | None = None, link: bool = False) -> dict:
    """Copy (or hard-link) a trained ``model.joblib`` and its companions into the registry as a new version.

    `samples` (the ``samples.npz`` it was trained on; ``samples.npz`` next to or one level above the model folder
    is used when present) lets the registry recompute and verify the CV/holdout/transfer split fingerprints that
    `compare` needs. `features` (the feature stack) records the input contract (feature order, windows, CRS,
    grid resolution) and the feature manifest. `hcat` is a ``class_name,hcat_code,hcat_name`` CSV.
    """
    from joblib import load as joblib_load

    from . import contracts
    if not _NAME.match(name):
        raise RegistryError(f"Invalid model name {name!r}: use 1-64 letters, digits, '.', '_' or '-'")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise RegistryError(f"Model file {model_path} not found")
    payload = contracts.check("model", joblib_load(model_path), model_path)
    digest = sha256(model_path)
    for v in _versions(root, name):
        if _load(_entry_path(root, name, v))["artifact"]["sha256"] == digest:
            raise RegistryError(f"{name}@{v} already holds this model (sha256 {digest[:12]}...)")
    src = model_path.parent
    notes = []

    def companion(file):
        path = src / file
        if not path.is_file():
            return None
        body = _load(path)
        if body.get("model_sha256") not in (None, digest):
            notes.append(f"{file} ignored: it belongs to another model (model_sha256 mismatch)")
            return None
        return body

    report, conformal = companion("train_report.json"), companion("conformal.json")
    names = [str(f) for f in payload["features"]]
    classes = [str(c) for c in payload["classes"]]

    sample_path = Path(samples) if samples else next(
        (p for p in (src / "samples.npz", src.parent / "samples.npz") if p.is_file()), None)
    fingerprints = {"verified": False, "reason": "no samples given (register --samples samples.npz)"}
    training = {}
    if report:
        training = {k: report.get(k) for k in ("samples", "polygons", "spatial_groups", "spatial_block_m",
                                               "spatial_buffer_m", "class_counts", "polygon_counts",
                                               "final_model_samples", "holdout_design", "cv_strategy")
                    if k in report}
    if sample_path is not None:
        from .samples import Samples
        s = Samples.load(sample_path)
        if not set(names) <= set(s.features):
            fingerprints = {"verified": False, "reason": f"{sample_path.name} lacks model features"}
        else:
            fingerprints = _model_fingerprints(s, payload, report)
            if fingerprints["verified"]:
                training.update(_read_samples_summary(s))
                training["samples_sha256"] = sha256(sample_path)
        fingerprints["samples_file"] = sample_path.name
    contract = {"features": names, "n_features": len(names), "dtype": "float32", "windows": None, "crs": None,
                "resolution": None, "nodata": None}
    manifest = None
    manifest_schema = None
    if features:
        info = _stack_info(features)
        missing = [f for f in names if f not in info["features"]]
        if missing:
            raise RegistryError(f"Feature stack {features} lacks {len(missing)} model features: {missing[:5]}")
        contract.update({k: info[k] for k in ("crs", "resolution", "nodata")})
        if info["manifest"]:
            by_name = {f["name"]: f for f in info["manifest"]["features"]}
            manifest = [{k: by_name[n].get(k) for k in ("name", "group", "sensor", "period")}
                        for n in names if n in by_name]
            manifest_schema = info["manifest"].get("schema_version", 0)
            contract["windows"] = sorted({f["period"] for f in manifest if f.get("period")})
    elif sample_path is not None and fingerprints.get("verified"):
        from .samples import Samples
        contract["crs"] = Samples.load(sample_path).crs
    groups = {}
    for f in manifest or []:
        groups[f["group"]] = groups.get(f["group"], 0) + 1
    params = _plain(payload.get("params") or {})
    created = _now()
    version = (_versions(root, name) or [0])[-1] + 1
    target = _models_dir(root) / name / str(version)
    target.mkdir(parents=True, exist_ok=False)
    storage = "copy"
    if link:
        try:
            os.link(model_path, target / "model.joblib")
            storage = "hardlink"
        except OSError:
            notes.append("hard link impossible (different file system?); copied instead")
    if storage == "copy":
        shutil.copyfile(model_path, target / "model.joblib")
        _read_only(target / "model.joblib")
    if sha256(target / "model.joblib") != digest:
        raise RegistryError("Model changed while it was being registered")
    files = {}
    foreign = {f for f, body in (("train_report.json", report), ("conformal.json", conformal)) if body is None}
    for file in COMPANIONS:
        if (src / file).is_file() and file not in foreign:
            shutil.copyfile(src / file, target / file)
            _read_only(target / file)
            files[file] = sha256(target / file)
    entry = _seal({
        "schema": SCHEMA_MODEL, "schema_version": SCHEMA_VERSION, "kind": "model", "name": name,
        "version": version, "tags": sorted(set(tags or ())), "description": description, "created_utc": created,
        "engine": _engine(),
        "artifact": {"file": "model.joblib", "sha256": digest, "bytes": model_path.stat().st_size,
                     "source": str(model_path.resolve()), "storage": storage},
        "files": files,
        "algorithm": params.get("algorithm", "rf"), "params": params, "seed": payload.get("seed"),
        "sklearn": payload.get("sklearn"), "temperature": payload.get("temperature"),
        "calibration_mode": payload.get("calibration_mode"), "model_schema_version": payload.get("schema_version"),
        "feature_manifest": {"names": names, "groups": groups or None, "schema_version": manifest_schema,
                             "features": manifest, "stack": Path(features).name if features else None},
        "input_contract": contract,
        "classes": classes,
        "taxonomy": {"name": taxonomy, "hcat": _hcat_mapping(hcat, classes)},
        "training_data": training,
        "validation": _validation(report, conformal),
        "fingerprints": fingerprints,
        "licences": _licences(manifest, licences),
        "provenance": payload.get("provenance") or {},
        "notes": notes,
    })
    _write_once(target / "entry.json", entry)
    return entry


def parse_ref(ref: str) -> tuple[str, str | None]:
    """``[registry:]name[@version|stage|latest]`` -> (name, selector)."""
    ref = ref[len("registry:"):] if ref.startswith("registry:") else ref
    name, _, selector = ref.partition("@")
    if not name:
        raise RegistryError(f"Invalid registry reference {ref!r}; use name@version, name@production or name@latest")
    return name, selector or None


def get(ref: str, root=None) -> dict:
    """Entry for ``name@version``, ``name@<stage>``, ``name@latest``; plain ``name`` means its production version.
    Pretrained checkpoints are addressed as ``ftw/<model>`` or ``presto/default``."""
    name, selector = parse_ref(ref)
    if "/" in name:
        for entry in pretrained_entries():
            if entry["name"] == name:
                return entry
        raise RegistryError(f"Unknown pretrained checkpoint {name!r}")
    versions = _versions(root, name)
    if not versions:
        raise RegistryError(f"No model {name!r} in registry {default_root(root)}")
    current = stages(name, root)
    if selector is None or selector in STAGES:
        stage = selector or "production"
        chosen = [v for v in versions if current[v] == stage]
        if not chosen:
            raise RegistryError(f"{name} has no version in stage {stage!r} (stages: "
                                f"{ {v: current[v] for v in versions} }); use {name}@latest or {name}@<version>")
        version = chosen[-1]
    elif selector == "latest":
        version = versions[-1]
    elif selector.isdigit() and int(selector) in versions:
        version = int(selector)
    else:
        raise RegistryError(f"{name}@{selector}: unknown version or stage; versions {versions}, stages {list(STAGES)}")
    entry = _load(_entry_path(root, name, version))
    entry["stage"] = current[version]
    entry["path"] = str(_entry_path(root, name, version).parent / entry["artifact"]["file"])
    return entry


def list_models(root=None, kind: str = "all") -> list[dict]:
    """One summary row per model version (and per pinned pretrained checkpoint with ``kind`` all/pretrained)."""
    rows = []
    if kind in ("all", "model"):
        base = _models_dir(root)
        for d in sorted(base.iterdir()) if base.is_dir() else []:
            current = stages(d.name, root)
            for v in _versions(root, d.name):
                e = _load(_entry_path(root, d.name, v))
                cv = (e.get("validation") or {}).get("cv") or {}
                rows.append({"kind": "model", "name": e["name"], "version": v, "stage": current[v],
                             "algorithm": e.get("algorithm"), "classes": len(e.get("classes") or []),
                             "features": len(e["feature_manifest"]["names"]), "cv_macro_f1": cv.get("macro_f1"),
                             "sha256": e["artifact"]["sha256"], "tags": e.get("tags"), "created_utc": e["created_utc"],
                             "engine_version": e["engine"]["version"]})
    if kind in ("all", "pretrained"):
        for e in pretrained_entries():
            rows.append({"kind": "pretrained", "name": e["name"], "version": e["version"], "stage": "external",
                         "algorithm": e["algorithm"], "licence": e["licence"], "sha256": e["artifact"]["sha256"],
                         "url": e["artifact"]["url"], "commercial_use": e.get("commercial_use")})
    return rows


def promote(ref: str, stage: str, root=None, note: str | None = None) -> dict:
    """Append a stage change. Promoting to production archives the previous production version; the artefact is
    verified first and a tampered version cannot be promoted."""
    if stage not in STAGES:
        raise RegistryError(f"Stage must be one of {STAGES}")
    entry = get(ref, root)
    if entry["kind"] != "model":
        raise RegistryError("Pretrained checkpoints have no registry stages")
    check = verify_entry(entry)
    if stage != "archived" and not check["ok"]:
        raise RegistryError(f"{entry['name']}@{entry['version']} failed verification: {check['problems']}")
    name, version = entry["name"], entry["version"]
    events = []
    if stage == "production":
        events += [{"version": v, "stage": "archived", "reason": f"superseded by version {version}"}
                   for v, s in stages(name, root).items() if s == "production" and v != version]
    events.append({"version": version, "stage": stage, "previous": entry["stage"], "note": note})
    path = _models_dir(root) / name / "stages.jsonl"
    with open(path, "a", encoding="utf8") as fh:
        for e in events:
            fh.write(json.dumps({**e, "time_utc": _now(), "engine_version": __version__}) + "\n")
    return {"name": name, "version": version, "stage": stage, "events": events}


def verify_entry(entry: dict) -> dict:
    """Recompute artefact, companion and entry hashes of one entry."""
    problems = []
    if entry["kind"] == "pretrained":
        path = Path(entry["cache_path"])
        if not path.exists():
            return {"name": entry["name"], "version": entry["version"], "ok": True, "status": "not cached",
                    "problems": []}
        ok = sha256(path) == entry["artifact"]["sha256"]
        return {"name": entry["name"], "version": entry["version"], "ok": ok, "status": "cached",
                "problems": [] if ok else [f"cached checkpoint {path} does not match the pinned sha256"]}
    folder = Path(entry["path"]).parent
    body = _load(folder / "entry.json")
    if _canonical({k: v for k, v in body.items() if k != "entry_sha256"}) != body.get("entry_sha256"):
        problems.append("entry.json was modified after registration")
    model = folder / body["artifact"]["file"]
    if not model.is_file():
        problems.append("model artefact missing")
    elif sha256(model) != body["artifact"]["sha256"]:
        problems.append("model artefact sha256 mismatch")
    for file, digest in body.get("files", {}).items():
        if not (folder / file).is_file():
            problems.append(f"{file} missing")
        elif sha256(folder / file) != digest:
            problems.append(f"{file} sha256 mismatch")
    return {"name": body["name"], "version": body["version"], "ok": not problems, "problems": problems}


def verify(ref: str | None = None, root=None, include_pretrained: bool = True) -> list[dict]:
    """Verify one entry, or every model version (and cached pretrained checkpoint) in the registry."""
    if ref:
        return [verify_entry(get(ref, root))]
    out = [verify_entry(get(f"{r['name']}@{r['version']}", root)) for r in list_models(root, "model")]
    if include_pretrained:
        out += [verify_entry(e) for e in pretrained_entries()]
    return out


COMPARE_SECTIONS = {
    "cv": ("cv", ("overall_accuracy", "macro_f1", "kappa", "ece")),
    "nested_cv": ("cv", ("overall_accuracy", "macro_f1", "kappa")),
    "holdout": ("holdout", ("overall_accuracy", "macro_f1", "kappa")),
    "transfer": ("transfer", ("out_of_domain_overall_accuracy", "out_of_domain_macro_f1", "out_of_domain_kappa",
                              "gap_macro_f1")),
}


def compare(ref_a: str, ref_b: str, root=None) -> dict:
    """Metric deltas (B - A) for every validation section whose split fingerprints are identical.

    Sections on different folds, holdouts or transfer groups are listed under ``refused`` with the reason; when no
    section is comparable a `RegistryError` is raised instead of reporting apples-to-oranges differences.
    """
    a, b = get(ref_a, root), get(ref_b, root)
    if "pretrained" in (a["kind"], b["kind"]):
        raise RegistryError("Pretrained checkpoints carry no validation metrics to compare")
    fa, fb = a.get("fingerprints") or {}, b.get("fingerprints") or {}
    va, vb = a.get("validation") or {}, b.get("validation") or {}
    out = {"a": f"{a['name']}@{a['version']}", "b": f"{b['name']}@{b['version']}", "delta": "b - a",
           "sections": {}, "refused": {}}
    for section, (fp_key, keys) in COMPARE_SECTIONS.items():
        ma, mb = va.get(section), vb.get(section)
        if not ma or not mb:
            if ma or mb:
                out["refused"][section] = "present in only one model"
            continue
        if not (fa.get("verified") and fb.get("verified")):
            out["refused"][section] = ("split fingerprints unverified "
                                       f"(a: {fa.get('reason', 'ok')}; b: {fb.get('reason', 'ok')})")
            continue
        if not fa.get(fp_key) or fa.get(fp_key) != fb.get(fp_key):
            out["refused"][section] = (f"different {fp_key} split fingerprints "
                                       f"({str(fa.get(fp_key))[:12]} vs {str(fb.get(fp_key))[:12]})")
            continue
        rows = {}
        for k in keys:
            x, y = ma.get(k), mb.get(k)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                rows[k] = {"a": x, "b": y, "delta": y - x}
        out["sections"][section] = {"fingerprint": fa[fp_key], "metrics": rows}
    if not out["sections"]:
        raise RegistryError(f"Refusing to compare {out['a']} and {out['b']}: no validation section was scored on "
                            f"identical splits ({out['refused']})")
    return out


def compare_markdown(result: dict) -> str:
    lines = [f"# {result['b']} vs {result['a']} (delta = b - a)", ""]
    for section, body in result["sections"].items():
        lines += [f"## {section} (split `{body['fingerprint'][:12]}`)", "", "| Metric | A | B | Delta |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {k} | {v['a']:.4f} | {v['b']:.4f} | {v['delta']:+.4f} |" for k, v in body["metrics"].items()]
        lines.append("")
    for section, reason in result["refused"].items():
        lines.append(f"* {section}: not compared, {reason}")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- input contract and predict

def check_input_contract(entry: dict, stack: str | Path) -> dict:
    """Check a feature stack against a registered model's input contract; raise `ContractError` on mismatch."""
    contract = entry["input_contract"]
    info = _stack_info(stack)
    problems = []
    missing = [f for f in contract["features"] if f not in info["features"]]
    if missing:
        problems.append(f"missing {len(missing)} model features, e.g. {missing[:5]}")
    if contract.get("crs") and info["crs"]:
        from pyproj import CRS
        if not CRS.from_user_input(contract["crs"]).equals(CRS.from_user_input(info["crs"])):
            problems.append(f"CRS {info['crs']} differs from the contract CRS {contract['crs']}")
    if contract.get("resolution") and not np.allclose(contract["resolution"], info["resolution"], rtol=1e-6):
        problems.append(f"grid resolution {info['resolution']} differs from the contract {contract['resolution']}")
    registered = entry["feature_manifest"].get("features")
    if registered and info["manifest"]:
        here = {f["name"]: f for f in info["manifest"]["features"]}
        changed = [f["name"] for f in registered if f["name"] in here and any(
            here[f["name"]].get(k) != f.get(k) for k in ("group", "sensor", "period"))]
        if changed:
            problems.append(f"{len(changed)} features have a different group/sensor/window than at training, e.g. "
                            f"{changed[:5]}")
    if problems:
        raise ContractError(f"Feature stack {stack} does not satisfy the input contract of "
                            f"{entry['name']}@{entry['version']}:\n  - " + "\n  - ".join(problems))
    return {"ok": True, "checked": [k for k in ("features", "crs", "resolution", "feature_manifest")
                                    if k == "features" or contract.get(k) or (k == "feature_manifest" and registered)]}


def resolve_for_predict(ref: str, stack: str | Path, root=None) -> dict:
    """Model path for ``registry:<name>@<version|stage>`` after hash verification and the input-contract check."""
    entry = get(ref, root)
    if entry["kind"] != "model":
        raise RegistryError(f"{entry['name']} is a pretrained checkpoint, not a classification model")
    check = verify_entry(entry)
    if not check["ok"]:
        raise RegistryError(f"{entry['name']}@{entry['version']} failed verification: {check['problems']}")
    contract = check_input_contract(entry, stack)
    return {"path": entry["path"], "name": entry["name"], "version": entry["version"], "stage": entry["stage"],
            "sha256": entry["artifact"]["sha256"], "contract": contract}


def annotate_prediction(manifest: str | Path, resolved: dict) -> None:
    """Add the registry reference to a prediction_manifest.json written by `geocrop.model.predict`."""
    path = Path(manifest)
    body = _load(path)
    body["registry"] = {k: resolved[k] for k in ("name", "version", "stage", "sha256")}
    path.write_text(json.dumps(body, indent=2), encoding="utf8")


# --------------------------------------------------------------------------- pretrained checkpoints

def pretrained_entries() -> list[dict]:
    """Pinned external checkpoints (FTW, Presto) as registry entries of kind ``pretrained``."""
    from .ftw import MODELS, checkpoint_path
    out = []
    for m in MODELS.values():
        out.append({"kind": "pretrained", "name": f"ftw/{m.name}", "version": m.version, "algorithm": "unet",
                    "task": SEGMENTATION_TASK, "licence": m.licence, "commercial_use": m.commercial_use,
                    "description": m.description, "classes": m.classes,
                    "artifact": {"url": m.url, "sha256": m.sha256, "bytes": m.size},
                    "cache_path": str(checkpoint_path(m.name)), "pinned_in": "geocrop.ftw.MODELS",
                    "stage": "external"})
    try:
        from .embedding_sources import PRESTO_CHECKPOINT, PRESTO_COMMIT, cache_dir
    except ImportError:          # requests missing: the Presto adapter cannot be loaded
        return out
    out.append({"kind": "pretrained", "name": "presto/default", "version": PRESTO_COMMIT[:12], "algorithm": "presto",
                "task": "embedding", "licence": PRESTO_CHECKPOINT["licence"], "commercial_use": True,
                "description": "Presto encoder (nasaharvest/presto default_model.pt)", "classes": None,
                "artifact": {"url": PRESTO_CHECKPOINT["url"], "sha256": PRESTO_CHECKPOINT["sha256"], "bytes": None},
                # same location as geocrop.embedding_sources.presto_checkpoint()
                "cache_path": str(cache_dir() / "presto" / f"presto_{PRESTO_COMMIT[:12]}_default_model.pt"),
                "pinned_in": "geocrop.embedding_sources.PRESTO_CHECKPOINT", "stage": "external"})
    return out


# --------------------------------------------------------------------------- benchmark records

def _rows(rows, key: str) -> list[dict]:
    out = []
    for r in rows:
        r = _plain(dict(r))
        row = {"candidate": str(r.pop(key, r.get("candidate")))}
        for k, v in r.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
            elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                row[k] = v                  # confidence intervals
        out.append(row)
    return out


def record_run(kind: str, output_dir: str | Path, rows: list[dict], *, task: str, split: dict, dataset: str,
               primary_metric: str = "macro_f1", protocol: dict | None = None, runtime_seconds: float | None = None,
               root=None) -> Path:
    """Write one immutable benchmark record; returns its path."""
    created = _now()
    body = {"schema": SCHEMA_BENCHMARK, "schema_version": SCHEMA_VERSION, "kind": kind, "task": task,
            "created_utc": created, "engine": _engine(), "output_dir": str(Path(output_dir).resolve()),
            "split_fingerprint": split["split"], "split": split, "dataset_fingerprint": dataset,
            "protocol": protocol or {}, "primary_metric": primary_metric,
            "runtime_seconds": None if runtime_seconds is None else round(float(runtime_seconds), 3),
            "rows": rows}
    short = _canonical(body)[:10]
    body["id"] = f"{created[:19].replace('-', '').replace(':', '')}-{kind}-{short}"
    body = _seal(body)
    path = default_root(root) / "benchmarks" / f"{body['id']}.json"
    _write_once(path, body)
    return path


def _safely(fn):
    """Benchmark hooks never break the benchmark: failures become a warning."""
    def wrapper(*args, **kwargs):
        if not recording_enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # recording is best effort; the benchmark outputs are already written
            warnings.warn(f"Benchmark run not recorded in the registry: {type(exc).__name__}: {exc}", stacklevel=2)
            return None
    wrapper.__doc__ = fn.__doc__
    wrapper.__name__ = fn.__name__
    return wrapper


@_safely
def record_classification(kind: str, output_dir, rows, samples, *, n_splits: int, seeds, spatial_block_m,
                          spatial_buffer_m: float = 0.0, holdout=None, strategies=(), candidate_key: str = "candidate",
                          protocol: dict | None = None, started: float | None = None, task: str = CLASSIFICATION_TASK,
                          primary_metric: str = "macro_f1", root=None) -> Path | None:
    """Record a benchmark/benchmark-models/ablation run on `samples` (the objects that were cross-validated)."""
    split = classification_split(samples, n_splits=n_splits, seeds=seeds, spatial_block_m=spatial_block_m,
                                 spatial_buffer_m=spatial_buffer_m, holdout=holdout, strategies=strategies)
    protocol = {"n_splits": n_splits, "seeds": list(seeds), "spatial_block_m": spatial_block_m,
                "spatial_buffer_m": spatial_buffer_m, "holdout_samples": 0 if holdout is None else len(holdout.y),
                "features_fingerprint": features_fingerprint(samples), "n_samples": int(len(samples.y)),
                **(protocol or {})}
    return record_run(kind, output_dir, _rows(rows, candidate_key), task=task, split=split, dataset=split["support"],
                      primary_metric=primary_metric, protocol=protocol, root=root,
                      runtime_seconds=None if started is None else time.perf_counter() - started)


@_safely
def record_transfer(output_dir, rows, samples, *, strategies, features=None, n_splits: int, seed: int, spatial_block_m,
                    spatial_buffer_m: float = 0.0, params: dict | None = None, started: float | None = None,
                    root=None) -> list[Path]:
    """One record per transfer strategy (task ``crop-transfer/<strategy>``), pooled ``ALL`` rows only."""
    features = list(features or samples.features)
    candidate = f"{(params or {}).get('algorithm', 'rf')} ({len(features)} features)"
    paths = []
    for strategy in strategies:
        pooled = [r for r in rows if r.get("strategy") == strategy and r.get("held_out_group") == "ALL"]
        if not pooled:
            continue
        split = classification_split(samples, n_splits=n_splits, seeds=[seed], spatial_block_m=spatial_block_m,
                                     spatial_buffer_m=spatial_buffer_m, strategies=(strategy,))
        split["split"] = split["transfer"]
        protocol = {"strategy": strategy, "n_splits": n_splits, "seed": seed, "spatial_block_m": spatial_block_m,
                    "params": params or {}, "features_fingerprint": features_fingerprint(samples, features)}
        paths.append(record_run("transfer", output_dir, _rows([{**pooled[0], "candidate": candidate}], "candidate"),
                                task=f"crop-transfer/{strategy}", split=split, dataset=split["support"],
                                primary_metric="out_of_domain_macro_f1", protocol=protocol, root=root,
                                runtime_seconds=None if started is None else time.perf_counter() - started))
    return paths


@_safely
def record_segmentation(output_dir, rows, raster, reference, *, tolerance_px: int, iou_threshold: float,
                        started: float | None = None, root=None) -> Path | None:
    """Record a benchmark-segmentation run; the split is the (raster, reference parcels, tolerances) triple."""
    data = {"raster": sha256(raster), "reference": sha256(reference)}
    split = {**data, "tolerance_px": tolerance_px, "iou_threshold": iou_threshold}
    split["split"] = _canonical(split)
    return record_run("benchmark-segmentation", output_dir, _rows(rows, "method"), task=SEGMENTATION_TASK,
                      split=split, dataset=_canonical(data), primary_metric="object_f1",
                      protocol={"raster": Path(raster).name, "reference": Path(reference).name}, root=root,
                      runtime_seconds=None if started is None else time.perf_counter() - started)


@_safely
def record_public(report: dict, work_dir, *, roles: str, protocol_split: dict, evaluation: dict,
                  root=None) -> Path | None:
    """Record a benchmark-public run (task ``public/<protocol id>``)."""
    p = report["protocol"]
    split = {"protocol": p["id"], "protocol_version": p["version"], "roles": roles, "samples":
             report["samples"]["fingerprint"], "split_settings": protocol_split, "evaluation": evaluation}
    split["split"] = _canonical(split)
    rows = []
    for name, r in report["models"].items():
        row = {"candidate": name, "status": r.get("status"), "wall_s": r.get("wall_s")}
        if r.get("status") == "ok":
            row.update({k: r["parcel"].get(k) for k in ("overall_accuracy", "macro_f1", "kappa")})
            row.update({"overall_accuracy_ci": r["bootstrap"].get("overall_accuracy_ci"),
                        "macro_f1_ci": r["bootstrap"].get("macro_f1_ci"),
                        "pixel_macro_f1": r["pixel"].get("macro_f1"),
                        "conformal_coverage": r["conformal"].get("coverage"),
                        "review_share": r["review"].get("review_share"), "fit_s": r["time"].get("fit_s")})
        rows.append(row)
    runtime = None
    try:
        runtime = (datetime.fromisoformat(report["finished_utc"]) -
                   datetime.fromisoformat(report["started_utc"])).total_seconds()
    except (KeyError, ValueError):
        pass
    return record_run("benchmark-public", work_dir, _rows(rows, "candidate"), task=f"public/{p['id']}", split=split,
                      dataset=report["samples"]["fingerprint"], protocol={"id": p["id"], "version": p["version"],
                                                                           "sha256": p["sha256"]},
                      runtime_seconds=runtime, root=root)


def list_benchmarks(root=None, task: str | None = None, kind: str | None = None) -> list[dict]:
    base = default_root(root) / "benchmarks"
    out = []
    for path in sorted(base.glob("*.json")) if base.is_dir() else []:
        e = _load(path)
        if (task and e["task"] != task) or (kind and e["kind"] != kind):
            continue
        out.append(e)
    return out


def get_benchmark(run_id: str, root=None) -> dict:
    matches = [e for e in list_benchmarks(root) if e["id"] == run_id or e["id"].startswith(run_id)
               or e["id"].endswith(run_id)]
    if len(matches) != 1:
        raise RegistryError(f"{'No' if not matches else 'Ambiguous'} benchmark run {run_id!r}"
                            + (f": {[m['id'] for m in matches]}" if matches else ""))
    return matches[0]


def benchmark_summary(entry: dict) -> dict:
    return {"id": entry["id"], "kind": entry["kind"], "task": entry["task"], "created_utc": entry["created_utc"],
            "split_fingerprint": entry["split_fingerprint"][:12], "dataset_fingerprint": entry["dataset_fingerprint"][:12],
            "rows": len(entry["rows"]), "runtime_seconds": entry.get("runtime_seconds"),
            "engine_version": entry["engine"]["version"]}


def leaderboard(task: str, root=None, metric: str | None = None, split: str | None = None) -> list[dict]:
    """Ranked tables, one per split fingerprint: rows from different splits are never ranked together."""
    entries = list_benchmarks(root, task)
    if not entries:
        raise RegistryError(f"No benchmark runs recorded for task {task!r} in {default_root(root)}")
    groups: dict[str, list[dict]] = {}
    for e in entries:
        if split and not e["split_fingerprint"].startswith(split):
            continue
        groups.setdefault(e["split_fingerprint"], []).append(e)
    if not groups:
        raise RegistryError(f"No runs of task {task!r} with split fingerprint {split!r}")
    out = []
    for fp, runs in groups.items():
        key = metric or runs[0]["primary_metric"]
        lower = key in LOWER_IS_BETTER
        rows = []
        for e in runs:
            for r in e["rows"]:
                value = r.get(key)
                if r.get("status", "ok") != "ok" or not isinstance(value, (int, float)):
                    continue
                ci = r.get(f"{key}_ci")
                rows.append({"candidate": r["candidate"], key: value, "ci": ci, "kind": e["kind"], "run": e["id"],
                             "engine_version": e["engine"]["version"], "created_utc": e["created_utc"],
                             **{k: r[k] for k in ("overall_accuracy", "macro_f1", "kappa") if k in r and k != key}})
        rows.sort(key=lambda r: r[key], reverse=not lower)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        out.append({"task": task, "split_fingerprint": fp, "metric": key, "higher_is_better": not lower,
                    "runs": len(runs), "dataset_fingerprint": runs[0]["dataset_fingerprint"],
                    "engine_versions": sorted({e["engine"]["version"] for e in runs}), "rows": rows})
    out.sort(key=lambda g: -g["runs"])
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return "" if v is None else str(v)


def leaderboard_markdown(groups: list[dict]) -> str:
    lines = []
    for g in groups:
        key = g["metric"]
        lines += [f"## {g['task']} - split `{g['split_fingerprint'][:12]}`", "",
                  f"Ranked by `{key}` ({'higher' if g['higher_is_better'] else 'lower'} is better); {g['runs']} run(s) "
                  f"on identical splits, dataset `{g['dataset_fingerprint'][:12]}`, engine "
                  f"{', '.join(g['engine_versions'])}.", "",
                  f"| Rank | Candidate | {key} | CI | Kind | Run |", "|---:|---|---:|---|---|---|"]
        lines += [f"| {r['rank']} | {r['candidate']} | {_fmt(r[key])} | {_fmt(r['ci'])} | {r['kind']} | `{r['run']}` |"
                  for r in g["rows"]]
        lines.append("")
    return "\n".join(lines)


def leaderboard_csv(groups: list[dict]) -> str:
    buf = io.StringIO()
    fields = ["split_fingerprint", "metric", "rank", "candidate", "value", "ci_low", "ci_high", "kind", "run",
              "engine_version", "created_utc"]
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for g in groups:
        for r in g["rows"]:
            ci = r["ci"] if isinstance(r["ci"], list) else [None, None]
            w.writerow({"split_fingerprint": g["split_fingerprint"], "metric": g["metric"], "rank": r["rank"],
                        "candidate": r["candidate"], "value": r[g["metric"]], "ci_low": ci[0], "ci_high": ci[1],
                        "kind": r["kind"], "run": r["run"], "engine_version": r["engine_version"],
                        "created_utc": r["created_utc"]})
    return buf.getvalue()


# --------------------------------------------------------------------------- read-only JSON for the web layer

def overview(root=None) -> dict:
    """Read-only, JSON-serialisable snapshot of the registry (models with stages, pretrained pins, benchmark runs)
    for the web workspace. Performs no writes and no hashing."""
    return _plain({"root": str(default_root(root)), "models": list_models(root, "model"),
                   "pretrained": list_models(root, "pretrained"),
                   "benchmarks": [benchmark_summary(e) for e in list_benchmarks(root)],
                   "tasks": sorted({e["task"] for e in list_benchmarks(root)})})
