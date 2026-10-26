"""Parcel evidence packages: a self-contained, versioned audit trail per parcel decision.

`build` writes, for every parcel of a decision table (or a selection of ids)::

    <out>/<parcel>/evidence.json   schema-versioned record (contract ``evidence``): declared and predicted
                                   class, probabilities, conformal set, AOA, markers, temporal support,
                                   every evaluated rule with inputs, thresholds, outcome and reason code,
                                   data sources (acquisitions actually observed vs grid dates that are
                                   interpolated), model sha256 and run-manifest reference
    <out>/<parcel>/evidence.html   human-readable page, no scripts and no external assets; it declares
                                   its own Content-Security-Policy (default-src 'none', style by hash)
    <out>/<parcel>/timeseries.png  parcel series: observed points vs the interpolated Whittaker curve,
                                   interpolation-dependent stretches, phenology/harvest dates, outcome
    <out>/index.csv, index.gpkg    one row per package (traffic light, reasons, support, paths, sha256)
    <out>/evidence_manifest.json   inputs with sha256, rule settings and package list

Packages are deterministic: the same inputs give byte-identical JSON, HTML and PNG (no timestamps; the
run manifest referenced carries the run time). Review flags support inspection prioritisation; they are
not findings of non-compliance.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, cbm, cbm_markers, contracts
from . import temporal_quality as tq
from .hashing import sha256

SCHEMA = "geocrop-parcel-evidence"

#: Top-level keys every package must carry, with their JSON types.
REQUIRED = {
    "schema": str, "schema_version": int, "generator": dict, "parcel_id": str, "declared_class": (str, type(None)),
    "classification": dict, "applicability": dict, "markers": dict, "temporal_support": dict, "rules": list,
    "outcome": dict, "data_sources": dict, "model": dict, "run_manifest": (dict, type(None)), "inputs": dict,
    "notes": list,
}
RULE_KEYS = ("rule", "stage", "inputs", "threshold", "triggered", "outcome", "reason_code")

_MARKER_FIELDS = ("sos_doy", "peak_doy", "eos_doy", "harvest_doy", "bare_soil_days", "clear_observations",
                  "peak_value", "calendar_consistent", *cbm_markers.EVENT_COLUMNS)


# --------------------------------------------------------------------------- helpers

def _clean(v):
    """JSON-safe, deterministic value: numpy scalars to Python, NaN/inf to None, floats rounded."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else round(f, 6)
    if v is None or (isinstance(v, str)):
        return v
    if pd.isna(v):
        return None
    return str(v)


def _dump(body: dict) -> str:
    return json.dumps(_clean(body), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _file_ref(path) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    return {"path": str(p), "sha256": sha256(p) if p.is_file() else None}


def _read_table(path, id_field: str) -> tuple[pd.DataFrame, object]:
    p = Path(path)
    if p.suffix.lower() in (".gpkg", ".geojson", ".json", ".shp", ".fgb"):
        import geopandas as gpd
        gdf = gpd.read_file(p)
        return pd.DataFrame(gdf.drop(columns=gdf.geometry.name)), gdf
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p), None
    return _numeric(pd.read_csv(p, dtype=str, keep_default_na=False).replace({"": None}), id_field), None


def _numeric(frame: pd.DataFrame, keep: str) -> pd.DataFrame:
    """Numeric columns of a string-read CSV back to numbers; ids and text columns stay strings."""
    out = frame.copy()
    for c in out.columns:
        if c == keep:
            continue
        present = out[c].dropna()
        if len(present) and present.isin(["True", "False"]).all():
            out[c] = out[c].map({"True": True, "False": False})
            continue
        converted = pd.to_numeric(out[c], errors="coerce")
        if converted.notna().sum() == out[c].notna().sum() and out[c].notna().any():
            out[c] = converted
    return out


def _signatures(path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, dtype={"parcel_id": str})


def safe_name(pid: str, used: set) -> str:
    """Filesystem-safe, unique folder name for a parcel id."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", str(pid)).strip("._") or "parcel"
    name = base[:80]
    if name in used:
        name = f"{name[:70]}-{hashlib.sha256(str(pid).encode()).hexdigest()[:8]}"
    used.add(name)
    return name


def _get(row: dict, key, default=None):
    v = row.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


def _doy_date(year: int, doy) -> str | None:
    if doy is None or not np.isfinite(doy):
        return None
    return (date(year, 1, 1) + timedelta(days=int(round(float(doy))) - 1)).isoformat()


# --------------------------------------------------------------------------- rules

def decision_rules(row: dict, summary: dict) -> list[dict]:
    """Classification decision checks reconstructed from the table row and its summary thresholds."""
    th = summary.get("thresholds", {}) or {}
    reasons = [c for c in str(_get(row, "decision_reasons", "")).split(";") if c]
    rules = []

    def add(rule, inputs, threshold, triggered, code, outcome="review"):
        rules.append({"rule": rule, "stage": "decision", "inputs": inputs, "threshold": threshold,
                      "triggered": bool(triggered), "outcome": outcome if triggered else "pass", "reason_code": code})

    n_valid = _get(row, "n_valid")
    if "min_valid_pixels" in th:
        add("decision.min_valid_pixels", {"n_valid": n_valid}, th["min_valid_pixels"],
            "insufficient_valid_pixels" in reasons or "insufficient_observations" in reasons,
            "insufficient_valid_pixels", "insufficient_resolution")
    if summary.get("strategy") == "combined" or "strategies_disagree" in reasons:
        add("decision.strategies_agree", {"majority_class": _get(row, "majority_class"),
                                          "mean_class": _get(row, "mean_class")}, None,
            "strategies_disagree" in reasons, "strategies_disagree")
    for name, key in (("confidence", "decision_confidence"), ("majority_share", "majority_share"),
                      ("margin", "decision_margin")):
        t = th.get(f"min_{name}")
        if t is not None or f"low_{name}" in reasons or f"missing_{name}" in reasons:
            add(f"decision.min_{name}", {key: _get(row, key)}, t,
                f"low_{name}" in reasons or f"missing_{name}" in reasons,
                f"missing_{name}" if f"missing_{name}" in reasons else f"low_{name}")
    if "aoa_fraction" in row or "min_aoa_fraction" in summary:
        code = "aoa_missing_support" if "aoa_missing_support" in reasons else "aoa_below_threshold"
        add("applicability.min_aoa_fraction", {"aoa_fraction": _get(row, "aoa_fraction"),
                                               "aoa_coverage": _get(row, "aoa_coverage"),
                                               "aoa_dissimilarity": _get(row, "aoa_dissimilarity")},
            summary.get("min_aoa_fraction"), code in reasons, code)
    declared, predicted = _get(row, "declared_class"), _get(row, "predicted_class")
    status = _get(row, "declaration_status")
    rules.append({"rule": "declaration.compare", "stage": "declaration",
                  "inputs": {"declared_class": declared, "predicted_class": predicted,
                             "decision": _get(row, "decision")},
                  "threshold": None, "triggered": status == "discrepancy", "outcome": status,
                  "reason_code": "declaration_discrepancy" if status == "discrepancy" else None})
    if _get(row, "cp_status") is not None or _get(row, "prediction_set") is not None:
        cp = _get(row, "cp_status")
        rules.append({"rule": "conformal.declared_in_set", "stage": "conformal",
                      "inputs": {"declared_class": declared, "prediction_set": _get(row, "prediction_set"),
                                 "set_size": _get(row, "set_size")},
                      "threshold": {"alpha": summary.get("conformal_alpha")}, "triggered": cp == "discrepancy",
                      "outcome": cp, "reason_code": {"discrepancy": "conformal_discrepancy",
                                                     "uncertain": "ambiguous_prediction_set"}.get(cp)})
    if "calendar_consistent" in row:
        cal = _get(row, "calendar_consistent")
        rules.append({"rule": "calendar.consistent", "stage": "markers",
                      "inputs": {"peak_doy": _get(row, "peak_doy"), "harvest_doy": _get(row, "harvest_doy")},
                      "threshold": None, "triggered": cal is False or str(cal) == "False",
                      "outcome": "inconsistent" if cal is False or str(cal) == "False" else
                      "consistent" if cal is True or str(cal) == "True" else "not_applicable",
                      "reason_code": "calendar_inconsistent"})
    return rules


_MARKER_FAMILIES = (
    ("harvest", ("no_harvest_evidence", "insufficient_marker_data_harvest"), "harvest_expected_classes",
     ("harvest_evidence", "harvest_detected", "harvest_confidence", "data_sufficiency"),
     ("min_marker_confidence", "no_harvest_outcome")),
    ("mowing", ("insufficient_mowing_events", "insufficient_marker_data_mowing"), "mowing_expected_classes",
     ("s2_sufficient", "mowing_events", "mowing_confidence"),
     ("min_marker_confidence", "min_mowing_events", "no_mowing_outcome")),
    ("tillage", ("tillage_detected",), "tillage_forbidden_classes", ("tillage_detected", "tillage_confidence"),
     ("min_marker_confidence", "tillage_outcome")),
    ("heterogeneity", ("heterogeneous_parcel",), None, ("homogeneity", "homogeneity_confidence"),
     ("min_marker_confidence", "heterogeneity_outcome")),
    ("mixed_class", ("mixed_class_pixels",), None, (), ("mixed_class_share", "mixed_class_outcome")),
)


def marker_rules(row: dict, frame_row: pd.DataFrame, rules: dict) -> list[dict]:
    """Every marker rule family with inputs, thresholds and whether it fired (rules from `cbm.rules`)."""
    probe = {**rules, **{k: "yellow" for k in rules if k.endswith("_outcome")}}
    mixture = cbm_markers.class_mixture(frame_row).iloc[0]
    fired = {code for code, _ in cbm_markers.marker_rule_reasons(row, probe, mixture)}
    declared = _get(row, "declared_class")
    out = []
    for family, codes, classes_key, inputs, thresholds in _MARKER_FAMILIES:
        applies = (classes_key is None or (declared is not None and str(declared) in rules[classes_key]))
        if family == "mixed_class" and rules["mixed_class_share"] is None:
            applies = False
        hit = next((c for c in codes if c in fired), None) if applies else None
        outcome_key = next(k for k in thresholds if k.endswith("_outcome"))
        configured = rules["insufficient_data_outcome"] if hit and hit.startswith("insufficient_marker_data") \
            else rules[outcome_key]
        values = {k: _get(row, k) for k in inputs}
        if family == "mixed_class":
            values["second_class_share"] = None if not np.isfinite(mixture) else float(mixture)
        if classes_key:
            values["declared_class"] = declared
        out.append({"rule": f"markers.{family}", "stage": "markers", "inputs": values,
                    "threshold": {k: rules[k] for k in thresholds if not k.endswith("_outcome")} |
                    ({classes_key: rules[classes_key]} if classes_key else {}),
                    "triggered": hit is not None,
                    "outcome": ("not_applicable" if not applies else configured if hit else "pass"),
                    "reason_code": hit or codes[0]})
    return out


def temporal_rules(row: dict, rules: dict) -> list[dict]:
    out = []
    for c in tq.evaluate(row, rules):
        out.append({"rule": c["rule"], "stage": "temporal_support", "inputs": c["inputs"],
                    "threshold": c["threshold"], "triggered": c["triggered"],
                    "outcome": (c["outcome"] if c["triggered"] else "pass"), "reason_code": c["code"]})
    return out


# --------------------------------------------------------------------------- series and sources

def _series_entry(series: dict | None, pid: str, signatures: pd.DataFrame | None, scenes: dict,
                  tol: float) -> dict:
    """Observed acquisitions, rejected acquisitions and the interpolated grid of one parcel."""
    observed, rejected, grid = [], [], []
    if signatures is not None:
        part = signatures[signatures.parcel_id.astype(str) == pid]
        idx = series["params"]["index"] if series else None
        cov = series["params"]["min_coverage"] if series else 0.5
        for r in part.sort_values(["date", "index"]).itertuples():
            d = str(r.date)[:10]
            n_valid, n_px = int(r.n_valid), int(r.n_pixels)
            clear = n_px > 0 and n_valid >= cov * n_px and pd.notna(r.mean)
            entry = {"date": d, "index": r.index, "mean": r.mean, "std": getattr(r, "std", None),
                     "n_valid": n_valid, "n_pixels": n_px, "valid_fraction": n_valid / n_px if n_px else None,
                     "scenes": scenes.get(d, [])}
            if clear:
                observed.append({**entry, "used_for_rule_index": r.index == idx})
            else:
                rejected.append({**entry, "reason": "no_pixels" if not n_px else "cloud_or_coverage_below_min"})
    if series is not None and pid in series["ids"]:
        k = series["ids"].index(pid)
        obs = series["obs"][k:k + 1]
        for j, d in enumerate(series["grid"]):
            dist = float(tq.distance_to_observation(obs, series["days"][j])[0])
            grid.append({"date": d.isoformat(), "value": series["z"][k, j],
                         "nearest_observation_days": dist if np.isfinite(dist) else None,
                         "interpolated": not np.isfinite(dist) or dist > tol})
    return {"observed": observed, "rejected": rejected, "grid": grid}


def _scene_index(root) -> dict:
    if root is None:
        return {}
    from . import scenes as scene_meta
    out: dict[str, list[dict]] = {}
    for m in scene_meta.discover(root):
        ident = m.get("source") or m.get("product_id") or m.get("id") or m["path"].name
        out.setdefault(m["date"].isoformat(), []).append(
            {k: v for k, v in (("id", str(ident)), ("sensor", m["sensor"]), ("tile", m.get("tile")),
                               ("relative_orbit", m.get("relative_orbit")),
                               ("orbit_direction", m.get("orbit_direction"))) if v is not None})
    return {d: sorted(v, key=lambda e: e["id"]) for d, v in out.items()}


# --------------------------------------------------------------------------- chart

_COL = {"obs": "#1f4e79", "curve": "#1f4e79", "interp": "#d55e00", "reject": "#8a8a8a", "event": "#444444",
        "grid": "#d9d9d9", "text": "#222222"}
_LIGHT = {"green": "#1a7f37", "yellow": "#b8860b", "red": "#c62828", "grey": "#6b6b6b"}


def _chart_data(body: dict) -> dict:
    ds = body["data_sources"]
    idx = body["temporal_support"].get("index")
    grid = ds["interpolated_grid"]
    obs = [o for o in ds["observed_acquisitions"] if o["index"] == idx and o["mean"] is not None]
    rej = sorted({o["date"] for o in ds["rejected_acquisitions"] if o["index"] == idx})
    events = []
    ts_ = body["temporal_support"]
    for name, key in (("SOS", "sos"), ("Peak", "peak"), ("EOS", "eos")):
        d = ts_.get(f"{key}_date")
        if d:
            dist = ts_.get(f"{key}_obs_distance_days")
            events.append((name, d, dist is not None and dist > ts_.get("tolerance_days", 10)))
    h = body["markers"].get("harvest_event_doy") or body["markers"].get("harvest_doy")
    if h is not None and grid:
        year = int(grid[0]["date"][:4])
        dist = ts_.get("harvest_obs_distance_days")
        events.append(("Harvest", _doy_date(year, h), dist is not None and dist > ts_.get("tolerance_days", 10)))
    return {"index": idx, "grid": grid, "obs": obs, "rejected": rej, "events": events}


def _ord(d: str) -> int:
    return date.fromisoformat(d).toordinal()


def chart_svg(body: dict, width: int = 760, height: int = 300) -> str:
    """Inline SVG (presentation attributes only, no style/script) of the parcel series."""
    c = _chart_data(body)
    if not c["grid"] and not c["obs"]:
        return "<p>No time series available for this parcel.</p>"
    left, right, top, bottom = 48, 12, 14, 40
    dates = [g["date"] for g in c["grid"]] + [o["date"] for o in c["obs"]] + c["rejected"]
    x0, x1 = min(map(_ord, dates)), max(map(_ord, dates))
    vals = [g["value"] for g in c["grid"] if g["value"] is not None] + [o["mean"] for o in c["obs"]]
    y0, y1 = (min(vals), max(vals)) if vals else (0.0, 1.0)
    pad = 0.05 * (y1 - y0 or 1)
    y0, y1 = y0 - pad, y1 + pad

    def X(d):
        return left + (width - left - right) * ((_ord(d) - x0) / ((x1 - x0) or 1))

    def Y(v):
        return top + (height - top - bottom) * (1 - (v - y0) / ((y1 - y0) or 1))

    e = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="{e(str(c["index"]))} time series: observed points and interpolated curve">']
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                     f'stroke="{_COL["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" '
                     f'fill="{_COL["text"]}">{v:.2f}</text>')
    year = int(dates[0][:4])
    for month in range(1, 13):
        d = date(year, month, 1).isoformat()
        if x0 <= _ord(d) <= x1:
            parts.append(f'<text x="{X(d):.1f}" y="{height - 22}" font-size="11" text-anchor="middle" '
                         f'fill="{_COL["text"]}">{date(year, month, 1).strftime("%b")}</text>')
    pts = [(g["date"], g["value"], g["interpolated"]) for g in c["grid"] if g["value"] is not None]
    for (d0, v0, i0), (d1, v1, i1) in zip(pts, pts[1:]):
        interp = i0 or i1
        dash = ' stroke-dasharray="5 4"' if interp else ""
        parts.append(f'<line x1="{X(d0):.1f}" y1="{Y(v0):.1f}" x2="{X(d1):.1f}" y2="{Y(v1):.1f}" '
                     f'stroke="{_COL["interp"] if interp else _COL["curve"]}" stroke-width="2"{dash}/>')
    for name, d, interp in c["events"]:
        if d and x0 <= _ord(d) <= x1:
            parts.append(f'<line x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{top}" y2="{height - bottom}" '
                         f'stroke="{_COL["interp"] if interp else _COL["event"]}" stroke-width="1" '
                         f'stroke-dasharray="2 3"/>')
            parts.append(f'<text x="{X(d) + 3:.1f}" y="{top + 10}" font-size="11" '
                         f'fill="{_COL["interp"] if interp else _COL["event"]}">{e(name)}'
                         f'{" (interp.)" if interp else ""}</text>')
    for d in c["rejected"]:
        parts.append(f'<line x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{height - bottom}" y2="{height - bottom - 7}" '
                     f'stroke="{_COL["reject"]}" stroke-width="2"><title>{e(d)}: rejected (cloud/coverage)</title>'
                     '</line>')
    for o in c["obs"]:
        parts.append(f'<circle cx="{X(o["date"]):.1f}" cy="{Y(o["mean"]):.1f}" r="3.5" fill="{_COL["obs"]}">'
                     f'<title>{e(o["date"])}: {o["mean"]:.3f} observed</title></circle>')
    ly = height - 6
    legend = [("circle", _COL["obs"], "observed (clear)"), ("solid", _COL["curve"], "smoothed, supported"),
              ("dash", _COL["interp"], "interpolated beyond tolerance"), ("tick", _COL["reject"], "rejected acquisition")]
    x = left
    for kind, col, label in legend:
        if kind == "circle":
            parts.append(f'<circle cx="{x + 5}" cy="{ly - 4}" r="3.5" fill="{col}"/>')
        elif kind == "tick":
            parts.append(f'<line x1="{x + 5}" x2="{x + 5}" y1="{ly}" y2="{ly - 8}" stroke="{col}" stroke-width="2"/>')
        else:
            dash = ' stroke-dasharray="5 4"' if kind == "dash" else ""
            parts.append(f'<line x1="{x}" x2="{x + 14}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" '
                         f'stroke-width="2"{dash}/>')
        parts.append(f'<text x="{x + 18}" y="{ly}" font-size="11" fill="{_COL["text"]}">{e(label)}</text>')
        x += 30 + 6.2 * len(label)
    parts.append("</svg>")
    return "".join(parts)


def chart_png(body: dict, path: Path) -> Path | None:
    """PNG of the same chart (matplotlib, deterministic metadata). None when matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    c = _chart_data(body)
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=100)
    try:
        pts = [(date.fromisoformat(g["date"]), g["value"], g["interpolated"]) for g in c["grid"]
               if g["value"] is not None]
        for (d0, v0, i0), (d1, v1, i1) in zip(pts, pts[1:]):
            interp = i0 or i1
            ax.plot([d0, d1], [v0, v1], color=_COL["interp"] if interp else _COL["curve"],
                    linestyle="--" if interp else "-", linewidth=1.8)
        if c["obs"]:
            ax.scatter([date.fromisoformat(o["date"]) for o in c["obs"]], [o["mean"] for o in c["obs"]],
                       s=18, color=_COL["obs"], zorder=3)
        ymin, ymax = ax.get_ylim()
        for d in c["rejected"]:
            ax.plot([date.fromisoformat(d)] * 2, [ymin, ymin + 0.04 * (ymax - ymin)], color=_COL["reject"],
                    linewidth=2)
        for name, d, interp in c["events"]:
            if d:
                ax.axvline(date.fromisoformat(d), color=_COL["interp"] if interp else _COL["event"],
                           linestyle=":", linewidth=1)
                ax.text(date.fromisoformat(d), ymax, f" {name}{' (interp.)' if interp else ''}", fontsize=8,
                        va="top", color=_COL["interp"] if interp else _COL["event"])
        ax.set_ylim(ymin, ymax)
        out = body["outcome"]
        ax.set_title(f"Parcel {body['parcel_id']} - {out['traffic_light']} "
                     f"({out['traffic_light_reasons'] or 'no reason codes'})", fontsize=9,
                     color=_LIGHT.get(out["traffic_light"], _COL["text"]))
        ax.set_ylabel(str(c["index"] or "index"))
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=_COL["obs"], marker="o", linestyle="", label="observed (clear)"),
                   Line2D([], [], color=_COL["curve"], label="smoothed, supported"),
                   Line2D([], [], color=_COL["interp"], linestyle="--", label="interpolated beyond tolerance"),
                   Line2D([], [], color=_COL["reject"], label="rejected acquisition")]
        ax.legend(handles=handles, fontsize=7, loc="lower center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, -0.32))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, metadata={"Software": None}, format="png")
    finally:
        plt.close(fig)
    return path


# --------------------------------------------------------------------------- HTML

_CSS = (
    "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:16px;color:#222;"
    "background:#fff;max-width:980px}"
    "h1{font-size:1.3rem;margin:0 0 4px}h2{font-size:1.05rem;margin:22px 0 6px;border-bottom:1px solid #ddd}"
    "table{border-collapse:collapse;width:100%;font-size:.86rem}td,th{border:1px solid #ddd;padding:4px 6px;"
    "text-align:left;vertical-align:top}th{background:#f4f4f4}.kv th{width:32%}.badge{display:inline-block;padding:2px 10px;"
    "border-radius:10px;color:#fff;font-weight:600}.green{background:#1a7f37}.yellow{background:#b8860b}"
    ".red{background:#c62828}.grey{background:#6b6b6b}.trig{font-weight:600;color:#c62828}"
    ".muted{color:#666;font-size:.82rem}code{font-size:.82rem;word-break:break-all}"
    "@media (prefers-color-scheme:dark){body{background:#151515;color:#e6e6e6}th{background:#262626}"
    "td,th{border-color:#3a3a3a}.muted{color:#aaa}svg{background:#fff;border-radius:4px}}"
)


def _csp() -> str:
    digest = base64.b64encode(hashlib.sha256(_CSS.encode("utf8")).digest()).decode()
    return (f"default-src 'none'; style-src 'sha256-{digest}'; img-src 'none'; base-uri 'none'; "
            "form-action 'none'")


def _kv(rows: dict) -> str:
    e = html.escape
    return '<table class="kv">' + "".join(f"<tr><th>{e(str(k))}</th><td>{e(_fmt(v))}</td></tr>" for k, v in rows.items()) \
        + "</table>"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render_html(body: dict) -> str:
    """Self-contained HTML of one package (no scripts, no external assets, CSP by style hash)."""
    e = html.escape
    b = _clean(body)
    out, cls = b["outcome"], b["classification"]
    light = out["traffic_light"] if out["traffic_light"] in _LIGHT else "grey"
    rules = "".join(
        f'<tr><td><code>{e(r["rule"])}</code></td><td>{e(r["stage"])}</td><td>{e(_fmt(r["inputs"]))}</td>'
        f'<td>{e(_fmt(r["threshold"]))}</td><td class="{"trig" if r["triggered"] else ""}">'
        f'{"yes" if r["triggered"] else "no"}</td><td>{e(_fmt(r["outcome"]))}</td>'
        f'<td>{e(_fmt(r["reason_code"]))}</td></tr>' for r in b["rules"])
    probs = {k: v for k, v in sorted(cls["probabilities"].items(), key=lambda kv: -(kv[1] or 0))}
    ds = b["data_sources"]
    obs_rows = "".join(
        f"<tr><td>{e(o['date'])}</td><td>{e(str(o['index']))}</td><td>{e(_fmt(o['mean']))}</td>"
        f"<td>{e(_fmt(o['valid_fraction']))}</td><td>{e(', '.join(s['id'] for s in o['scenes']) or '-')}</td></tr>"
        for o in ds["observed_acquisitions"] if o.get("used_for_rule_index"))
    rej_rows = "".join(
        f"<tr><td>{e(o['date'])}</td><td>{e(str(o['index']))}</td><td>{e(_fmt(o['valid_fraction']))}</td>"
        f"<td>{e(o['reason'])}</td></tr>" for o in ds["rejected_acquisitions"])
    interp = [g["date"] for g in ds["interpolated_grid"] if g["interpolated"]]
    tsupp = {k: v for k, v in b["temporal_support"].items() if k != "observed_dates"}
    markers = {k: v for k, v in b["markers"].items() if v is not None}
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{e(_csp())}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Parcel {e(b['parcel_id'])} evidence</title>
<style>{_CSS}</style></head><body>
<h1>Parcel {e(b['parcel_id'])} <span class="badge {light}">{e(out['traffic_light'])}</span></h1>
<p class="muted">Evidence package {e(b['schema'])} v{b['schema_version']} - geocrop-engine
{e(b['generator']['version'])}. Review flags support inspection prioritisation; they are not findings of
non-compliance.</p>
<h2>Outcome</h2>
{_kv({"traffic light": out['traffic_light'], "reason codes": out['traffic_light_reasons'] or '-',
      "stored traffic light": out.get('stored_traffic_light'), "consistent with table": out.get('consistent_with_table'),
      "decision": cls.get('decision'), "decision reasons": cls.get('decision_reasons') or '-',
      "declaration status": cls.get('declaration_status')})}
<h2>Classification</h2>
{_kv({"declared class": b['declared_class'], "predicted class": cls.get('predicted_class'),
      "confidence": cls.get('confidence'), "margin": cls.get('margin'), "entropy": cls.get('entropy'),
      "prediction set": cls.get('prediction_set'), "conformal status": cls.get('cp_status'),
      "conformal alpha": cls.get('conformal_alpha'), "valid pixels": cls.get('n_valid')})}
<h2>Class probabilities</h2>
{_kv(probs) if probs else '<p>-</p>'}
<h2>Area of applicability</h2>
{_kv(b['applicability'])}
<h2>Time series</h2>
{chart_svg(b)}
<p class="muted">Observed = clear acquisitions of index {e(str(b['temporal_support'].get('index')))} (points).
Solid line = Whittaker curve within {e(_fmt(b['temporal_support'].get('tolerance_days')))} days of an
observation; dashed = interpolated beyond that tolerance. Interpolated grid dates: {len(interp)} of
{len(ds['interpolated_grid'])}.</p>
<h2>Temporal support</h2>
{_kv(tsupp)}
<h2>Rules evaluated</h2>
<table><tr><th>rule</th><th>stage</th><th>inputs</th><th>threshold</th><th>triggered</th><th>outcome</th>
<th>reason code</th></tr>{rules}</table>
<h2>Markers</h2>
{_kv(markers) if markers else '<p>No marker table supplied.</p>'}
<h2>Data sources</h2>
<p class="muted">Clear acquisitions of the rule index used for the series (scene ids when a scene root was
given).</p>
<table><tr><th>date</th><th>index</th><th>parcel mean</th><th>valid fraction</th><th>scenes</th></tr>
{obs_rows}</table>
<p class="muted">Rejected acquisitions (cloud, shadow or coverage below min_coverage).</p>
<table><tr><th>date</th><th>index</th><th>valid fraction</th><th>reason</th></tr>{rej_rows}</table>
<h2>Model and provenance</h2>
{_kv({"model": b['model'].get('path'), "model sha256": b['model'].get('sha256'),
      "model sha256 (table summary)": b['model'].get('summary_sha256'),
      "model matches summary": b['model'].get('matches_summary'),
      "run manifest": (b['run_manifest'] or {}).get('path'),
      "run manifest sha256": (b['run_manifest'] or {}).get('sha256'),
      "config fingerprint": (b['run_manifest'] or {}).get('config_fingerprint')})}
<h2>Inputs</h2>
{_kv({k: (v or {}).get('sha256') if isinstance(v, dict) else v for k, v in b['inputs'].items()})}
<h2>Notes</h2><ul>{''.join(f'<li>{e(n)}</li>' for n in b['notes'])}</ul>
</body></html>
"""
    return page


# --------------------------------------------------------------------------- validation

def validate(body: dict) -> list[str]:
    """Problems of an evidence body (empty list = valid). Also checks the schema version contract."""
    problems = []
    for key, typ in REQUIRED.items():
        if key not in body:
            problems.append(f"missing {key}")
        elif not isinstance(body[key], typ) or (typ is int and isinstance(body[key], bool)):
            problems.append(f"{key} has type {type(body[key]).__name__}")
    if body.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    try:
        contracts.check("evidence", body)
    except ValueError as exc:
        problems.append(str(exc))
    for i, r in enumerate(body.get("rules", []) if isinstance(body.get("rules"), list) else []):
        missing = [k for k in RULE_KEYS if k not in r]
        if missing:
            problems.append(f"rules[{i}] lacks {missing}")
        elif not isinstance(r["triggered"], bool):
            problems.append(f"rules[{i}].triggered must be boolean")
    out = body.get("outcome", {})
    if isinstance(out, dict) and out.get("traffic_light") not in ("green", "yellow", "red", "grey"):
        problems.append("outcome.traffic_light must be green, yellow, red or grey")
    ds = body.get("data_sources", {})
    for key in ("observed_acquisitions", "rejected_acquisitions", "interpolated_grid"):
        if isinstance(ds, dict) and not isinstance(ds.get(key), list):
            problems.append(f"data_sources.{key} must be a list")
    return problems


def load(path) -> dict:
    """Read and validate an evidence.json (raises ValueError when invalid or from a newer GeoCrop)."""
    p = Path(path)
    body = json.loads(p.read_text(encoding="utf8"))
    body = contracts.check("evidence", body, p)
    problems = validate(body)
    if problems:
        raise ValueError(f"{p}: invalid evidence package: {'; '.join(problems)}")
    return body


# --------------------------------------------------------------------------- build

def _support_block(row: dict, year: int | None, params: dict) -> dict:
    out = {"index": params["index"], "tolerance_days": params["tolerance_days"]}
    for col, v in row.items():
        if col.startswith("tq_") and col != "tq_tolerance_days":
            out[col[3:]] = v
    if year is not None:
        for name in tq.PHENOLOGY_DATES:
            out[f"{name}_date"] = _doy_date(year, _get(row, f"tq_{name}_doy"))
    if "observed_dates" in out and isinstance(out["observed_dates"], str):
        out["observed_dates"] = [d for d in out["observed_dates"].split(";") if d]
    return out


def build(parcels, output_dir, ids=None, signatures=None, start: date | None = None, end: date | None = None,
          id_field: str = "parcel_id", markers=None, temporal_support=None, marker_rule_config: dict | None = None,
          temporal_rule_config: dict | None = None, model=None, run_manifest=None, scenes=None, summary=None,
          png: bool = True) -> dict:
    """Write evidence packages for `ids` (default: all parcels) of a parcel decision table.

    parcels            parcels.gpkg / .csv from `geocrop parcels` or `geocrop parcel-classify`
    signatures         CbM signature table (path or DataFrame); with `start`/`end` it provides the series,
                       observed/rejected acquisitions and (when `temporal_support` is None) the support
    markers            marker table (`cbm.markers` or `cbm_markers.event_markers`; path or DataFrame)
    temporal_support   per-parcel support (`temporal_quality.parcel_support`; path or DataFrame)
    marker_rule_config / temporal_rule_config   ``cbm.rules`` / ``cbm.temporal_support``; applied when enabled
    model              model directory or model.joblib (sha256); run_manifest: run_manifest.json
    scenes             scene root to resolve acquisition dates to scene ids
    summary            parcels summary JSON (default: ``<parcels>.summary.json`` next to the table)
    """
    parcels_path = Path(parcels)
    table, gdf = _read_table(parcels_path, id_field)
    if id_field not in table:
        raise ValueError(f"Parcel table lacks {id_field!r}")
    table[id_field] = table[id_field].astype(str)
    if table[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    selected = list(table[id_field]) if not ids else [str(i) for i in ids]
    unknown = sorted(set(selected) - set(table[id_field]))
    if unknown:
        raise ValueError(f"Unknown parcel id(s): {unknown[:10]}")
    summary_path = Path(summary) if summary else parcels_path.with_suffix(".summary.json")
    summ = json.loads(summary_path.read_text(encoding="utf8")) if summary_path.is_file() else {}
    mrules = cbm_markers.resolve_rules(marker_rule_config)
    trules = tq.resolve(temporal_rule_config)
    sig = None if signatures is None else (signatures if isinstance(signatures, pd.DataFrame)
                                           else _signatures(signatures))
    mk = None if markers is None else (markers if isinstance(markers, pd.DataFrame)
                                       else pd.read_csv(markers, dtype={"parcel_id": str}))
    if (start is None) != (end is None):
        raise ValueError("Give both start and end")
    series = None
    if sig is not None and start is not None:
        series = tq.parcel_series(sig, start, end, trules, parcel_ids=selected)
    if temporal_support is not None:
        sup = temporal_support if isinstance(temporal_support, pd.DataFrame) else \
            pd.read_csv(temporal_support, dtype={"parcel_id": str})
    elif sig is not None and start is not None:
        sup = tq.parcel_support(sig, start, end, trules, markers=mk, parcel_ids=selected)
    else:
        sup = None
    frame = table.set_index(table[id_field]).loc[selected].reset_index(drop=True)
    if mk is not None:
        m = mk.assign(parcel_id=mk.parcel_id.astype(str)).set_index("parcel_id").reindex(frame[id_field])
        for col in _MARKER_FIELDS:
            if col in m and col not in frame:
                frame[col] = m[col].to_numpy()
    if sup is not None:
        frame = tq.attach(frame, sup, id_field)
    details = cbm.traffic_light_details(frame, mrules if mrules["enabled"] else None,
                                        trules if trules["enabled"] else None)
    scene_idx = _scene_index(scenes)
    model_ref = None
    if model is not None:
        mp = Path(model)
        mp = mp / "model.joblib" if mp.is_dir() else mp
        model_ref = _file_ref(mp)
    summary_sha = summ.get("model_sha256") or summ.get("calibration_model_sha256")
    run_ref = None
    if run_manifest is not None:
        from .provenance import read_run_manifest
        rm = read_run_manifest(run_manifest)
        run_ref = {**_file_ref(run_manifest), "schema_version": rm.get("schema_version"),
                   "config_fingerprint": rm.get("config_fingerprint"), "git_commit": rm.get("git_commit"),
                   "timestamp_utc": rm.get("timestamp_utc")}
    inputs = {"parcels": _file_ref(parcels_path), "summary": _file_ref(summary_path) if summ else None,
              "signatures": None if signatures is None or isinstance(signatures, pd.DataFrame) else _file_ref(signatures),
              "markers": None if markers is None or isinstance(markers, pd.DataFrame) else _file_ref(markers),
              "temporal_support": None if temporal_support is None or isinstance(temporal_support, pd.DataFrame)
              else _file_ref(temporal_support),
              "scenes": None if scenes is None else str(scenes)}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set = set()
    index_rows = []
    year = start.year if start else None
    classes = [c[2:] for c in table.columns if c.startswith("p_")]
    for k, pid in enumerate(frame[id_field]):
        row = {c: frame.at[k, c] for c in frame.columns}
        row = {c: (None if isinstance(v, float) and math.isnan(v) else v) for c, v in row.items()}
        one = frame.iloc[[k]]
        rules = decision_rules(row, summ)
        base_colour, base_codes = cbm._base_traffic_light(one)
        rules.append({"rule": "traffic_light.base", "stage": "traffic_light",
                      "inputs": {c: row.get(c) for c in ("declaration_status", "cp_status", "calendar_consistent",
                                                         "decision")},
                      "threshold": None, "triggered": base_colour.iloc[0] != "green",
                      "outcome": base_colour.iloc[0], "reason_code": ";".join(base_codes[0]) or None})
        if mrules["enabled"]:
            rules += marker_rules(row, one, mrules)
        if trules["enabled"] or sup is not None:
            rules += temporal_rules(row, trules)
        colour = details["traffic_light"].iloc[k]
        reasons = details["traffic_light_reasons"].iloc[k]
        stored = row.get("traffic_light")
        notes = ["Review flags support inspection prioritisation; they are not findings of non-compliance."]
        if not trules["enabled"] and sup is not None:
            notes.append("Temporal-support rules were evaluated for information only (cbm.temporal_support "
                         "disabled); they did not change the outcome.")
        if stored is not None and stored != colour:
            notes.append(f"The stored traffic light ({stored}) differs from the one recomputed with the rules "
                         f"of this package ({colour}).")
        if series is None:
            notes.append("No signatures/season supplied: data sources and chart are empty.")
        entry = _series_entry(series, pid, sig, scene_idx, trules["tolerance_days"])
        body = contracts.stamp("evidence", {
            "schema": SCHEMA,
            "generator": {"name": "geocrop-engine", "version": __version__},
            "parcel_id": pid,
            "declared_class": None if row.get("declared_class") is None else str(row.get("declared_class")),
            "classification": {
                "predicted_class": row.get("predicted_class"), "decision": row.get("decision"),
                "decision_reasons": row.get("decision_reasons"), "declaration_status": row.get("declaration_status"),
                "strategy": row.get("strategy") or summ.get("strategy"), "confidence": row.get("confidence"),
                "decision_confidence": row.get("decision_confidence"), "margin": row.get("margin"),
                "decision_margin": row.get("decision_margin"), "entropy": row.get("entropy"),
                "n_valid": row.get("n_valid"),
                "probabilities": {c: row.get(f"p_{c}") for c in classes},
                "prediction_set": row.get("prediction_set"), "set_size": row.get("set_size"),
                "cp_status": row.get("cp_status"), "conformal_alpha": summ.get("conformal_alpha")},
            "applicability": {"aoa_fraction": row.get("aoa_fraction"), "aoa_coverage": row.get("aoa_coverage"),
                              "aoa_dissimilarity": row.get("aoa_dissimilarity"),
                              "min_aoa_fraction": summ.get("min_aoa_fraction"),
                              "inside": None if row.get("aoa_fraction") is None or summ.get("min_aoa_fraction") is None
                              else bool(row["aoa_fraction"] >= summ["min_aoa_fraction"])},
            "markers": {c: row.get(c) for c in _MARKER_FIELDS if c in row},
            "temporal_support": _support_block({c: v for c, v in row.items() if c.startswith("tq_")}, year, trules)
            if sup is not None else {"index": trules["index"], "tolerance_days": trules["tolerance_days"]},
            "rules": rules,
            "rule_settings": {"marker_rules": mrules, "temporal_support": trules},
            "outcome": {"traffic_light": colour, "traffic_light_reasons": reasons, "stored_traffic_light": stored,
                        "consistent_with_table": None if stored is None else stored == colour},
            "data_sources": {"index": trules["index"], "season": None if start is None else
                             {"start": start.isoformat(), "end": end.isoformat(), "step_days": trules["step_days"],
                              "lambda": trules["lambda"], "min_coverage": trules["min_coverage"]},
                             "observed_acquisitions": entry["observed"],
                             "rejected_acquisitions": entry["rejected"], "interpolated_grid": entry["grid"]},
            "model": {**(model_ref or {"path": None, "sha256": None}), "summary_sha256": summary_sha,
                      "matches_summary": None if not (model_ref and summary_sha) else model_ref["sha256"] == summary_sha},
            "run_manifest": run_ref,
            "inputs": inputs,
            "notes": notes,
        })
        body = _clean(body)
        problems = validate(body)
        if problems:
            raise ValueError(f"Evidence package for {pid} is invalid: {problems}")
        folder = out_dir / safe_name(pid, used)
        folder.mkdir(parents=True, exist_ok=True)
        text = _dump(body)
        (folder / "evidence.json").write_text(text, encoding="utf8")
        (folder / "evidence.html").write_text(render_html(body), encoding="utf8")
        chart = chart_png(body, folder / "timeseries.png") if png and entry["grid"] else None
        tsb = body["temporal_support"]
        index_rows.append({
            "parcel_id": pid, "traffic_light": colour, "traffic_light_reasons": reasons,
            "stored_traffic_light": stored, "decision": row.get("decision"),
            "declared_class": body["declared_class"], "predicted_class": row.get("predicted_class"),
            "declaration_status": row.get("declaration_status"),
            "clear_observations": tsb.get("clear_observations"), "longest_gap_days": tsb.get("longest_gap_days"),
            "peak_gap_days": tsb.get("peak_gap_days"), "interpolated_phenology": tsb.get("interpolated_phenology"),
            "rules_triggered": ";".join(r["rule"] for r in body["rules"] if r["triggered"]),
            "package": folder.name, "evidence_json": f"{folder.name}/evidence.json",
            "evidence_html": f"{folder.name}/evidence.html",
            "chart_png": f"{folder.name}/timeseries.png" if chart else None,
            "evidence_sha256": hashlib.sha256(text.encode("utf8")).hexdigest()})
    index = pd.DataFrame(index_rows)
    index.to_csv(out_dir / "index.csv", index=False)
    written = {"index_csv": str(out_dir / "index.csv")}
    if gdf is not None:
        g = gdf.assign(**{id_field: gdf[id_field].astype(str)}).set_index(id_field).loc[selected]
        import geopandas as gpd
        geo = gpd.GeoDataFrame(index.copy(), geometry=g.geometry.to_numpy(), crs=gdf.crs)
        target = out_dir / "index.gpkg"
        target.unlink(missing_ok=True)
        geo.to_file(target, driver="GPKG")
        written["index_gpkg"] = str(target)
    manifest = contracts.stamp("evidence", {
        "schema": "geocrop-evidence-batch", "generator": {"name": "geocrop-engine", "version": __version__},
        "packages": int(len(index)), "inputs": inputs,
        "model": model_ref, "run_manifest": run_ref,
        "rule_settings": {"marker_rules": mrules, "temporal_support": trules},
        "traffic_light_counts": index["traffic_light"].value_counts().sort_index().to_dict() if len(index) else {},
        "index": [{"parcel_id": r["parcel_id"], "package": r["package"], "evidence_sha256": r["evidence_sha256"]}
                  for r in index_rows]})
    (out_dir / "evidence_manifest.json").write_text(_dump(manifest), encoding="utf8")
    return {"packages": int(len(index)), "out": str(out_dir), **written,
            "traffic_light_counts": manifest["traffic_light_counts"],
            "with_series": series is not None, "with_temporal_support": sup is not None}
