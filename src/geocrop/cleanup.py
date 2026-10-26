"""Housekeeping: quarantine partial stage outputs and remove regenerable intermediates safely.

Categories (``geocrop clean``):

* ``temporary``  – leftovers of interrupted atomic writes (``.stages-*.tmp``, ``.hashcache.tmp``, ``*.tmp``,
  ``*.part``) in the run output, composite and feature directories; always removed.
* ``quarantine`` – partial outputs of failed stages moved to ``<output>/.quarantine`` by the pipeline; always removed.
* ``intermediate`` – regenerable products (window composites, feature stacks and sidecars, ``samples.npz``,
  the hash cache); only with ``--all-intermediates``. Stages rerun automatically when they are missing.
* ``result`` – model, prediction, parcel, CbM, validation and experiment products; only with ``--include-results``.

Scene folders (raw/imported downloads), the grid, reference layers, mapping/calendar files and the config
are protected and never removed: any candidate inside, or containing, a protected path is skipped. Only
files with known names are removed; unknown files stay and their directories are left in place.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .cache import RunLock

QUARANTINE = ".quarantine"
TEMP_PATTERNS = (".stages-*.tmp", ".hashcache.tmp", "*.tmp", "*.part")
RESULT_DIRS = ("model", "prediction", "parcels", "cbm", "validation", "experiments")


def quarantine_partial(paths, since_ns: int, output: str | Path, stage: str) -> list[dict]:
    """Move outputs written or modified since `since_ns` into ``<output>/.quarantine/<stage>-<time>/``.

    A rename keeps the evidence without consuming space; across volumes the partial file is deleted
    instead (it is incomplete by definition). Never raises: failures are reported in the result."""
    moved, target = [], None
    for raw in paths:
        path = Path(raw)
        try:
            if not path.is_file() or path.stat().st_mtime_ns < since_ns:
                continue
            if target is None:
                target = Path(output) / QUARANTINE / f"{stage}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
                target.mkdir(parents=True, exist_ok=True)
            dest = target / f"{len(moved):03d}_{path.name}"
            try:
                os.replace(path, dest)
                moved.append({"path": str(path), "quarantined": str(dest)})
            except OSError:
                path.unlink(missing_ok=True)
                moved.append({"path": str(path), "quarantined": None, "deleted": True})
        except OSError as exc:
            moved.append({"path": str(path), "error": str(exc)})
    if target is not None:
        (target / "quarantine.json").write_text(json.dumps({"stage": stage, "files": moved}, indent=2),
                                                encoding="utf8")
    return moved


def _protected(cfg, config_path=None) -> list[Path]:
    p = cfg["paths"]
    items = [p["s2_scenes"], p["s1_scenes"], p["training"], p["parcels"], p["observations"],
             cfg["grid"].get("reference"), cfg["parcels"].get("crop_mapping"), cfg["cbm"].get("calendar"),
             config_path, *cfg.get("preflight", {}).get("inventories", [])]
    return [Path(x).resolve() for x in items if x]


def _blocked(path: Path, protected: list[Path]) -> bool:
    path = path.resolve()
    return any(path == q or q in path.parents or path in q.parents for q in protected)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def plan(cfg, all_intermediates: bool = False, include_results: bool = False) -> list[dict]:
    """Candidate files/directories for removal with category and size; nothing is changed."""
    from . import features, s2
    out, comp = Path(cfg["paths"]["output"]), Path(cfg["paths"]["composites"])
    stack = Path(cfg["paths"]["features"])
    found: dict[Path, str] = {}
    for root, recursive in ((out, True), (comp, True), (stack.parent, False)):
        if not root.is_dir():
            continue
        for pattern in TEMP_PATTERNS:
            for f in (root.rglob(pattern) if recursive else root.glob(pattern)):
                if f.is_file() and QUARANTINE not in f.relative_to(root).parts:
                    found.setdefault(f, "temporary")
    if (out / QUARANTINE).is_dir():
        found[out / QUARANTINE] = "quarantine"
    if all_intermediates:
        products = {*[f"{b}.tif" for b in s2.S2_BANDS], "valid_count.tif", "composite.json", "composite_s1.json",
                    *[f"{pre}{pol}.tif" for pre in ("", "ASC_", "DESC_") for pol in ("VV", "VH")]}
        if comp.is_dir():
            for window in sorted(d for d in comp.iterdir() if d.is_dir()):
                for name in sorted(products):
                    if (window / name).is_file():
                        found.setdefault(window / name, "intermediate")
        from .tsfeatures import quality_path
        stacks = [stack, stack.with_name(stack.stem + "_ts.tif"), stack.with_name(stack.stem + "_windows.tif")]
        for s in stacks:
            for f in (s, features.manifest_path(s), quality_path(s), quality_path(s).with_suffix(".json")):
                if f.is_file():
                    found.setdefault(f, "intermediate")
        for f in (out / "samples.npz", out / ".hashcache.json"):
            if f.is_file():
                found.setdefault(f, "intermediate")
    if include_results:
        for name in RESULT_DIRS:
            if (out / name).is_dir():
                found.setdefault(out / name, "result")
    return [{"path": str(p), "category": c, "bytes": _size(p)} for p, c in sorted(found.items())]


def clean(cfg, dry_run: bool = True, all_intermediates: bool = False, include_results: bool = False,
          config_path=None) -> dict:
    """Remove (or with `dry_run` only list) planned candidates, skipping protected paths."""
    out = Path(cfg["paths"]["output"])
    protected = _protected(cfg, config_path)
    candidates = plan(cfg, all_intermediates, include_results)
    removed, skipped, errors = [], [], []
    lock = RunLock(out) if out.is_dir() and not dry_run else None
    if lock is not None:
        lock.__enter__()   # refuses while a live run holds the output directory
    try:
        for item in candidates:
            path = Path(item["path"])
            if _blocked(path, protected):
                skipped.append({**item, "reason": "protected input"})
                continue
            if dry_run:
                removed.append(item)
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                removed.append(item)
            except OSError as exc:
                errors.append({**item, "error": str(exc)})
        if not dry_run and all_intermediates:
            comp = Path(cfg["paths"]["composites"])
            for window in (d for d in comp.iterdir() if d.is_dir()) if comp.is_dir() else ():
                if not any(window.iterdir()) and not _blocked(window, protected):
                    window.rmdir()
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    return {"dry_run": dry_run, "all_intermediates": all_intermediates, "include_results": include_results,
            "removed" if not dry_run else "would_remove": removed, "skipped_protected": skipped, "errors": errors,
            "bytes": sum(i["bytes"] for i in removed),
            "gb": round(sum(i["bytes"] for i in removed) / 1e9, 3),
            "protected_inputs": [str(p) for p in protected]}
