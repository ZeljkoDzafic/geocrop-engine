"""Scene metadata (`scene.json`) and assignment of scenes to phenological windows.

Every imported scene folder carries a small `scene.json` sidecar:
    {"schema_version": 1, "sensor": "S2"|"S1", "datetime": ISO-8601, ...optional tile/orbit fields}
so that compositing never has to guess acquisition dates from file names.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

from . import contracts

SIDE_CAR = "scene.json"
_S2_NAME = re.compile(r"MSIL2A_(\d{8}T\d{6})_N\d{4}_R(\d{3})_T(\w{5})")
_S1_NAME = re.compile(r"S1[ABC]_IW_GRD\w*?_(\d{8}T\d{6})_\d{8}T\d{6}_(\d{6})_")


def parse_product_name(name: str) -> dict:
    """Extract acquisition time and geometry identifiers from Sentinel product names."""
    if m := _S2_NAME.search(name):
        return {"sensor": "S2", "datetime": datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").isoformat(),
                "relative_orbit": int(m.group(2)), "tile": m.group(3)}
    if m := _S1_NAME.search(name):
        absolute = int(m.group(2))
        platform = name[:3]
        # Relative orbit from absolute orbit (ESA formula; S1A offset 73, S1B offset 27).
        offset = {"S1A": 73, "S1B": 27}.get(platform)
        rel = ((absolute - offset) % 175) + 1 if offset is not None else None
        return {"sensor": "S1", "datetime": datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").isoformat(),
                "absolute_orbit": absolute, "relative_orbit": rel, "platform": platform}
    raise ValueError(f"Unrecognised Sentinel product name: {name}")


def write(folder: str | Path, **meta) -> Path:
    if "datetime" not in meta or "sensor" not in meta:
        raise ValueError("scene metadata needs sensor and datetime")
    path = Path(folder) / SIDE_CAR
    path.write_text(json.dumps(contracts.stamp("scene", meta), indent=2, default=str), encoding="utf8")
    return path


def read(folder: str | Path) -> dict:
    path = Path(folder) / SIDE_CAR
    if not path.is_file():
        raise FileNotFoundError(f"{folder} has no {SIDE_CAR}; import scenes with geocrop s2-import / s1-import")
    try:
        meta = json.loads(path.read_text(encoding="utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupt scene metadata {path}: not valid JSON ({exc}); re-import the scene") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"Corrupt scene metadata {path}: expected a JSON object, got {type(meta).__name__}")
    meta = contracts.check("scene", meta, path)
    missing = [k for k in ("sensor", "datetime") if not meta.get(k)]
    if missing:
        raise ValueError(f"Scene metadata {path} lacks required keys {missing}")
    if meta["sensor"] not in ("S1", "S2"):
        raise ValueError(f"Scene metadata {path}: unknown sensor {meta['sensor']!r} (expected S1 or S2)")
    try:
        meta["date"] = datetime.fromisoformat(str(meta["datetime"]).replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"Scene metadata {path}: invalid ISO-8601 datetime {meta['datetime']!r}") from exc
    meta["path"] = Path(folder)
    return meta


def discover(root: str | Path) -> list[dict]:
    root = Path(root)
    return sorted((read(p.parent) for p in root.rglob(SIDE_CAR)), key=lambda m: m["date"])


def assign(scenes: list[dict], windows: dict[str, tuple[date, date]]) -> dict[str, list[dict]]:
    """Group scenes by window (inclusive bounds). Scenes outside every window are dropped."""
    out = {name: [] for name in windows}
    for scene in scenes:
        for name, (start, end) in windows.items():
            if start <= scene["date"] <= end:
                out[name].append(scene)
                break
    return out


def inventory(scenes_by_window: dict[str, list[dict]]) -> list[dict]:
    """Per-window availability summary, the first table to inspect before fixing window dates."""
    rows = []
    for name, scenes in scenes_by_window.items():
        dates = sorted(s["date"] for s in scenes)
        orbits = sorted({str(s.get("orbit_direction") or "") for s in scenes} - {""})
        rows.append({"window": name, "scenes": len(scenes),
                     "first": dates[0].isoformat() if dates else None,
                     "last": dates[-1].isoformat() if dates else None,
                     "orbit_directions": orbits})
    return rows
