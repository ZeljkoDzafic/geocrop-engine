"""Input quality gates and reproducible run manifests."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import geopandas as gpd
import rasterio

PACKAGES = ("geocrop-engine", "numpy", "pandas", "scikit-learn", "rasterio", "geopandas", "shapely", "pyproj",
            "scipy", "scikit-image", "torch", "pystac-client")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_raster(path: str | Path) -> dict:
    with rasterio.open(path) as ds:
        problems = []
        if ds.crs is None:
            problems.append("no CRS")
        elif not ds.crs.is_projected:
            problems.append("geographic CRS: use a projected metric CRS for area and distance")
        if ds.nodata is None:
            problems.append("no explicit nodata")
        if any(not d for d in ds.descriptions):
            problems.append("missing band descriptions")
        return {"path": str(path), "width": ds.width, "height": ds.height, "bands": ds.count,
                "crs": str(ds.crs), "resolution": ds.res, "nodata": ds.nodata, "problems": problems}


def check_vector(path: str | Path, required=(), unique=(), label: str | None = None) -> dict:
    df = gpd.read_file(path)
    problems = [f"missing column {c}" for c in required if c not in df.columns]
    if df.crs is None:
        problems.append("no CRS")
    empty = int((df.geometry.isna() | df.geometry.is_empty).sum())
    invalid = int((~df.geometry.is_valid).sum())
    if empty:
        problems.append(f"{empty} null/empty geometries")
    if invalid:
        problems.append(f"{invalid} invalid geometries (fix with make_valid)")
    for c in unique:
        if c in df and df[c].duplicated().any():
            problems.append(f"{c} not unique")
    out = {"path": str(path), "features": int(len(df)), "crs": str(df.crs), "problems": problems}
    if label and label in df:
        out["class_counts"] = df[label].value_counts().to_dict()
        if df.crs is not None and df.crs.is_projected:
            out["class_area_ha"] = (df.geometry.area.groupby(df[label]).sum() / 1e4).round(2).to_dict()
    return out


def _versions() -> dict:
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    try:
        from osgeo import gdal  # noqa: F401
        out["gdal"] = gdal.__version__
    except ImportError:
        out["gdal (rasterio)"] = rasterio.__gdal_version__
    return out


def _run(cmd, cwd=None) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd, timeout=30, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_manifest(destination: str | Path, config: dict | None = None, inputs=(), snap_gpt: str | None = None,
                 notes: dict | None = None) -> Path:
    """Record everything needed to reproduce a run. Inputs are hashed, never copied or uploaded."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    body = {
        "schema": "geocrop-run/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": _versions(),
        "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent),
        "git_dirty": bool(_run(["git", "status", "--porcelain"], cwd=Path(__file__).parent)),
        "conda_explicit": _run(["conda", "list", "--explicit"]) if shutil.which("conda") else None,
        "snap_version": _run([snap_gpt, "--diag"]) if snap_gpt else None,
        "config": config, "inputs": files, "notes": notes or {},
        "privacy": "Inputs are referenced by hash only; do not publish private parcel or observation data.",
    }
    if config is not None:
        from .config import fingerprint
        body["config_fingerprint"] = fingerprint(config)
    path = root / "run_manifest.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf8")
    return path
