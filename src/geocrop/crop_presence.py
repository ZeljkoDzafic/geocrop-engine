"""Presence of a declared crop in heterogeneous (cadastral) parcels: three comparable methods.

A cadastral parcel can hold several land-cover classes (forest, grass, built-up, maize, soybean ...), so a declaration
is not wrong because the declared crop is not the majority of the whole parcel. For every parcel with a declared crop
this module separates two questions: *is the declared crop present on the agricultural part?* and, when a declared area
exists, *does the mapped area agree with it?* (reported, never merged into one decision).

Per parcel (pixels whose centres fall inside the parcel; optional inward buffer):

    composition          area and share of every mapped class; ``agricultural_area_ha`` and ``non_agricultural_area_ha``
    A  pixel majority    dominant class of all pixels and of the agricultural part (reference method)
    B  soft vote         S(c) = sum_i A_i P_i(c) / sum_i A_i  for the declared crop c, over the whole parcel and over the
                         agricultural part (pixel area A_i; equal weights on a regular grid)
    C  evidence fusion   segments (OBIA polygons clipped to the parcel, or connected same-class pixels when none are
                         given) with mean probability, dominant class, area and homogeneity; supported segments
                         (mean P(declared) >= ``support_prob``, area >= ``min_segment_ha``), the largest connected
                         area of pixels supporting the declared crop, the declared crop's share of the agricultural
                         part and the agreement of pixel and segment majority

The preliminary status ``supported`` / ``disputed_review`` / ``not_supported`` follows explicit thresholds that are
**not validated** (``thresholds_validated: false`` in the manifest): choose them on parcels whose crop was verified in
the field and freeze them before independent validation. The status is satellite support for a declaration, never an
administrative finding. Approval of a declaration is an administrative decision, not a field verification.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
from scipy import ndimage
from shapely.geometry import shape as to_shape
from shapely.ops import unary_union

from . import __version__
from .hashing import sha256
from .raster import valid_mask
from .samples import polygon_window

AGRICULTURAL_DEFAULT = ("maize", "soybean", "sunflower", "vegetables", "winter_crops")
STATUSES = ("supported", "disputed_review", "not_supported")
EIGHT = np.ones((3, 3), dtype=int)


def _fraction(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _largest_component(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    labels, n = ndimage.label(mask, structure=EIGHT)
    return int(np.bincount(labels.ravel())[1:].max()) if n else 0


def analyse_parcel(z: np.ndarray, segment_ids: np.ndarray | None, inside: np.ndarray, classes: list[str],
                   declared: str | None, pixel_ha: float, agricultural: tuple[str, ...], support_prob: float,
                   min_segment_ha: float) -> tuple[dict, list[dict]]:
    """Metrics of one parcel from the (k, rows, cols) probabilities of its window; ``inside`` marks valid parcel pixels."""
    k = z.shape[0]
    n = int(inside.sum())
    row: dict = {"n_valid_pixels": n}
    if n == 0:
        return row, []
    pred = z.argmax(axis=0)
    zi = z[:, inside]
    counts = np.bincount(pred[inside], minlength=k)
    for i, c in enumerate(classes):
        row[f"share_{c}"] = float(counts[i] / n)
        row[f"area_ha_{c}"] = float(counts[i] * pixel_ha)
    agri_set = set(agricultural) | ({declared} if declared in classes else set())
    agri_idx = [i for i, c in enumerate(classes) if c in agri_set]
    agri = inside & np.isin(pred, agri_idx)
    n_agri = int(agri.sum())
    row["agricultural_area_ha"] = float(n_agri * pixel_ha)
    row["non_agricultural_area_ha"] = float((n - n_agri) * pixel_ha)
    row["pixel_majority"] = classes[int(counts.argmax())]
    row["pixel_majority_agricultural"] = classes[int(np.bincount(pred[agri], minlength=k).argmax())] if n_agri else None
    mean_max = float(zi.max(axis=0).mean())
    top2 = np.sort(zi, axis=0)[-2:] if k > 1 else None
    row["mean_confidence"] = mean_max
    row["mean_margin"] = float((top2[1] - top2[0]).mean()) if top2 is not None else 1.0
    segments: list[dict] = []
    if declared is None or declared not in classes:
        row["declared_class_modelled"] = False
        return row, segments
    d = classes.index(declared)
    row["declared_class_modelled"] = True
    p = z[d]
    row["soft_vote_whole"] = float(p[inside].mean())                        # area-weighted: equal pixel areas
    row["soft_vote_agricultural"] = float(p[agri].mean()) if n_agri else None
    row["a_match_whole"] = row["pixel_majority"] == declared
    row["a_match_agricultural"] = row["pixel_majority_agricultural"] == declared
    row["crop_area_ha"] = float(counts[d] * pixel_ha)
    row["crop_share_agricultural"] = _fraction(counts[d], n_agri)
    # segments: given OBIA polygons or connected same-class pixels
    if segment_ids is None:
        labels = np.zeros(pred.shape, dtype=np.int64)
        nxt = 0
        for i in range(k):
            lab, m = ndimage.label((pred == i) & inside, structure=EIGHT)
            labels[lab > 0] = lab[lab > 0] + nxt
            nxt += m
    else:
        labels = np.where(inside, segment_ids, 0)
        missing = inside & (labels == 0)
        if missing.any():                                                    # pixels outside any segment
            lab, m = ndimage.label(missing, structure=EIGHT)
            labels[lab > 0] = lab[lab > 0] + int(labels.max())
    supported_px = np.zeros(pred.shape, dtype=bool)
    seg_area = Counter()
    for s in np.unique(labels[inside]):
        m = inside & (labels == s)
        zs = z[:, m]
        mean = zs.mean(axis=1)
        dom = int(mean.argmax())
        area = float(m.sum() * pixel_ha)
        ok = bool(mean[d] >= support_prob and area >= min_segment_ha)
        if ok:
            supported_px |= m
        if classes[dom] in agri_set:
            seg_area[classes[dom]] += area
        segments.append({"segment": int(s), "area_ha": area, "mean_p_declared": float(mean[d]),
                         "dominant_class": classes[dom], "homogeneity": float(max(0.0, 1.0 - zs[d].std())),
                         "supported": ok, "agricultural": classes[dom] in agri_set, "mask": m})
    n_sup = int(supported_px.sum())
    row["supported_area_ha"] = float(n_sup * pixel_ha)
    row["supported_share_agricultural"] = _fraction(n_sup, max(n_agri, n_sup))
    row["segments"] = len(segments)
    row["segment_majority"] = max(seg_area, key=seg_area.get) if seg_area else None
    row["pixel_segment_agree"] = (row["segment_majority"] == row["pixel_majority_agricultural"]
                                  if row["segment_majority"] else None)
    row["connected_support_ha"] = float(_largest_component((p >= support_prob) & inside) * pixel_ha)
    return row, segments


def preliminary_status(row: dict, declared_area_ha: float | None, min_valid_pixels: int, min_agricultural_ha: float,
                       min_connected_ha: float, min_supported_share: float, low_score: float,
                       min_confidence: float, area_tolerance: float) -> tuple[str, list[str]]:
    """Explicit, unvalidated rule (see module docstring); returns the status and its reason codes."""
    reasons: list[str] = []
    if row.get("n_valid_pixels", 0) < min_valid_pixels:
        return "disputed_review", ["too_few_valid_pixels"]
    if not row.get("declared_class_modelled"):
        return "disputed_review", ["declared_crop_not_modelled"]
    if row["agricultural_area_ha"] < min_agricultural_ha:
        return "disputed_review", ["insufficient_agricultural_area"]
    score = row["soft_vote_agricultural"] if row["soft_vote_agricultural"] is not None else 0.0
    share = row["supported_share_agricultural"] or 0.0
    if row["connected_support_ha"] < min_connected_ha and score < low_score and row["crop_area_ha"] == 0:
        return "not_supported", ["declared_crop_not_supported"]
    if row["connected_support_ha"] >= min_connected_ha and share >= min_supported_share:
        if row.get("pixel_segment_agree") is False:
            reasons.append("pixel_segment_disagree")
        if row["mean_confidence"] < min_confidence:
            reasons.append("low_confidence")
        if declared_area_ha and declared_area_ha > 0:
            diff = abs(row["crop_area_ha"] - declared_area_ha) / declared_area_ha
            if diff > area_tolerance:
                reasons.append("area_difference")
        return ("supported" if not reasons else "disputed_review"), reasons
    if row["connected_support_ha"] < min_connected_ha:
        reasons.append("no_connected_support")
    if share < min_supported_share:
        reasons.append("low_supported_share")
    return "disputed_review", reasons


def analyse(probabilities: str | Path, parcels: str | Path | gpd.GeoDataFrame, output: str | Path,
            id_field: str = "parcel_id", declared_field: str = "declared_crop", declared_area_field: str | None = None,
            crop_mapping: dict | None = None, segments: str | Path | None = None, segment_id_field: str | None = None,
            agricultural: tuple[str, ...] = AGRICULTURAL_DEFAULT, interior_buffer_px: float = 0.0,
            support_prob: float = 0.5, min_segment_ha: float = 0.03, min_valid_pixels: int = 4,
            min_agricultural_ha: float = 0.03, min_connected_ha: float = 0.05, min_supported_share: float = 0.5,
            low_score: float = 0.1, min_confidence: float = 0.6, area_tolerance: float = 0.5) -> dict:
    """Write ``crop_presence.csv``, ``crop_presence.gpkg`` (layers ``parcels`` and ``segments``) and a manifest."""
    for name, v in (("support_prob", support_prob), ("min_supported_share", min_supported_share), ("low_score", low_score),
                    ("min_confidence", min_confidence)):
        if not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(interior_buffer_px) or interior_buffer_px < 0:
        raise ValueError("interior_buffer_px must be finite and >= 0")
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf or declared_field not in gdf:
        raise ValueError(f"Parcels need {id_field!r} and {declared_field!r}")
    if declared_area_field and declared_area_field not in gdf:
        raise ValueError(f"declared_area_field {declared_area_field!r} is not a column of the parcels")
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows, seg_rows = [], []
    with ExitStack() as stack:
        ds = stack.enter_context(rasterio.open(probabilities))
        if ds.crs is None or not ds.crs.is_projected or not math.isclose(ds.crs.linear_units_factor[1], 1):
            raise ValueError("Crop presence requires a projected CRS in metres")
        classes = list(ds.descriptions)
        if not all(classes) or len(set(classes)) != len(classes):
            raise ValueError("Probability bands need unique class-name descriptions")
        gdf = gdf.to_crs(ds.crs)
        seg = gpd.read_file(segments) if segments else None
        if seg is not None:
            if seg.crs is None:
                raise ValueError("Segments need a CRS")
            if segment_id_field is not None and segment_id_field not in seg:
                raise ValueError(f"segment_id_field {segment_id_field!r} is not a column of the segments")
            seg = seg.to_crs(ds.crs).reset_index(drop=True)
            if segment_id_field is not None:
                seg["_sid"] = pd.factorize(seg[segment_id_field])[0] + 1       # ids 1..n (0 means "no segment")
            else:
                seg["_sid"] = np.arange(1, len(seg) + 1)
            segment_id_field = "_sid"
        pixel = abs(ds.transform.a)
        pixel_ha = pixel * pixel / 1e4
        for parcel in gdf.itertuples():
            geom = parcel.geometry
            core = geom.buffer(-interior_buffer_px * pixel) if interior_buffer_px else geom
            if core.is_empty:
                core = geom
            raw = getattr(parcel, declared_field)
            declared = crop_mapping.get(str(raw), None) if crop_mapping else (None if pd.isna(raw) else str(raw))
            row = {"parcel_id": getattr(parcel, id_field), "declared_crop": declared, "parcel_area_ha": float(geom.area / 1e4),
                   "n_valid_pixels": 0}
            d_area = None
            if declared_area_field and declared_area_field in gdf:
                v = getattr(parcel, declared_area_field)
                d_area = None if pd.isna(v) else float(v)
                row["declared_area_ha"] = d_area
            win = polygon_window(core, ds)
            seg_list: list[dict] = []
            if win is not None:
                cube = ds.read(window=win)
                tr = ds.window_transform(win)
                inside = geometry_mask([core], out_shape=cube.shape[1:], transform=tr, invert=True) & valid_mask(cube, ds.nodata)
                z = cube[:, inside]
                if z.size and (np.any(z < -1e-6) or np.any(z > 1 + 1e-6)
                               or not np.allclose(z.sum(axis=0), 1, atol=1e-4, rtol=1e-5)):
                    raise ValueError("Valid probability pixels must lie in [0, 1] and sum to one")
                seg_ids = None
                if seg is not None:
                    cand = seg[seg.intersects(core)]
                    seg_ids = rasterize(((g, int(i)) for g, i in zip(cand.geometry, cand[segment_id_field])),
                                        out_shape=cube.shape[1:], transform=tr, fill=0, dtype="int64") if len(cand) else None
                metrics, seg_list = analyse_parcel(cube, seg_ids, inside, classes, declared, pixel_ha, agricultural,
                                                   support_prob, min_segment_ha)
                row.update(metrics)
                for s in seg_list:
                    shp = [g for g, v in shapes(s["mask"].astype("uint8"), mask=s["mask"], transform=tr)]
                    s["geometry"] = unary_union([to_shape(g) for g in shp])
                    s["parcel_id"] = row["parcel_id"]
                    seg_rows.append({k: v for k, v in s.items() if k != "mask"})
            if d_area and "crop_area_ha" in row:
                row["area_difference_ha"] = row["crop_area_ha"] - d_area
                row["area_difference_pct"] = 100 * row["area_difference_ha"] / d_area
            status, reasons = preliminary_status(row, d_area, min_valid_pixels, min_agricultural_ha, min_connected_ha,
                                                 min_supported_share, low_score, min_confidence, area_tolerance)
            row["preliminary_status"], row["review_reason"] = status, ";".join(reasons)
            row["geometry"] = geom
            rows.append(row)
    table = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    csv = out / "crop_presence.csv"
    table.drop(columns="geometry").to_csv(csv, index=False)
    gpkg = out / "crop_presence.gpkg"
    gpkg.unlink(missing_ok=True)
    table.to_file(gpkg, layer="parcels", driver="GPKG")
    if seg_rows:
        gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=gdf.crs).to_file(gpkg, layer="segments", driver="GPKG")
    counts = table["preliminary_status"].value_counts().to_dict()
    manifest = {
        "schema": "geocrop-crop-presence/1", "engine_version": __version__, "classes": classes,
        "inputs": {"probabilities_sha256": sha256(probabilities),
                   "parcels_sha256": sha256(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else None,
                   "segments_sha256": sha256(segments) if segments else None},
        "config": {"id_field": id_field, "declared_field": declared_field, "declared_area_field": declared_area_field,
                   "agricultural": list(agricultural), "interior_buffer_px": interior_buffer_px,
                   "support_prob": support_prob, "min_segment_ha": min_segment_ha, "min_valid_pixels": min_valid_pixels,
                   "min_agricultural_ha": min_agricultural_ha, "min_connected_ha": min_connected_ha,
                   "min_supported_share": min_supported_share, "low_score": low_score, "min_confidence": min_confidence,
                   "area_tolerance": area_tolerance, "segments": "given" if segments else "connected same-class pixels"},
        "thresholds_validated": False, "parcels": int(len(table)), "status_counts": {str(k): int(v) for k, v in counts.items()},
        "note": ("Satellite support for a declaration, not an administrative finding. Thresholds are not validated; "
                 "approval of a declaration is an administrative decision, not a field verification."),
    }
    (out / "crop_presence_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return {"csv": str(csv), "gpkg": str(gpkg), **{k: manifest[k] for k in ("parcels", "status_counts", "thresholds_validated")}}
