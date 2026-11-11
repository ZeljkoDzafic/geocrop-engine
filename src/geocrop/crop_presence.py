"""Presence of a declared crop in heterogeneous (cadastral) parcels: three comparable methods.

A cadastral parcel can hold several land-cover classes (forest, grass, built-up, maize, soybean ...), so a declaration
is not wrong because the declared crop is not the majority of the whole parcel. For every parcel with a declared crop
this module separates two questions: *is the declared crop present on the agricultural part?* and, when a declared area
exists, *does the mapped area agree with it?* (reported, never merged into one decision).

Per parcel (pixels whose centres fall inside the parcel; optional inward buffer):

    composition          area and share of every mapped class; ``agricultural_area_ha`` and ``non_agricultural_area_ha``
    P1 pixel majority    dominant class of all pixels and of the agricultural part (reference method)
    P2 soft vote         S(c) = sum_i A_i P_i(c) / sum_i A_i  for the declared crop c, over the whole parcel and over the
                         agricultural part (pixel area A_i; equal weights on a regular grid)
    P3 evidence fusion   segments (OBIA polygons clipped to the parcel, or connected same-class pixels when none are
                         given) with mean probability, dominant class, area and homogeneity; supported segments
                         (mean P(declared) >= ``support_prob``, area >= ``min_segment_ha``), the largest connected
                         area of pixels supporting the declared crop, the declared crop's share of the agricultural
                         part and the agreement of pixel and segment majority

A fourth, separate decision (``presence_status``: present / partial_presence / absent / indeterminate) is driven by the
absolute connected supported area and its reliability instead of a share of the agricultural part (see
``presence_decision``); ``mismatch_category`` and the ``flag_*`` columns separate the reasons why a declaration is not
(fully) supported (``mismatch_taxonomy``); P3 columns keep their meaning unchanged for reproducibility.

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
PRESENCE_STATUSES = ("present", "partial_presence", "absent", "indeterminate")
MISMATCH_CATEGORIES = ("too_few_pixels", "no_valid_observations", "other_crop_present", "declared_crop_partial",
                       "area_disagreement", "model_disagreement")
# residual primary reasons that are not one of the six (documented in docs/crop-presence.md)
MISMATCH_EXTRA = ("non_agricultural_dominant", "declared_crop_not_modelled", "none")
EIGHT = np.ones((3, 3), dtype=int)


def _fraction(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    labels, n = ndimage.label(mask, structure=EIGHT)
    return labels == (int(np.bincount(labels.ravel())[1:].argmax()) + 1)


def _largest_component(mask: np.ndarray) -> int:
    return int(_largest_component_mask(mask).sum())


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
    row["agricultural_share"] = float(n_agri / n)
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
    row["p1_match_whole"] = row["pixel_majority"] == declared
    row["p1_match_agricultural"] = row["pixel_majority_agricultural"] == declared
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
    comp = _largest_component_mask((p >= support_prob) & inside)             # largest 8-connected supporting area
    row["connected_support_ha"] = float(comp.sum() * pixel_ha)
    row["connected_mean_p"] = float(p[comp].mean()) if comp.any() else None   # reliability of that area (for presence_status)
    row["connected_mean_confidence"] = float(z[:, comp].max(axis=0).mean()) if comp.any() else None
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


def presence_decision(row: dict, declared_area_ha: float | None, min_valid_pixels: int = 4, min_connected_ha: float = 0.05,
                      min_mean_p: float = 0.6, min_confidence: float = 0.6, min_share: float = 0.5,
                      min_valid_fraction: float = 0.5, min_expected_pixels: float = 8.0) -> tuple[str, list[str]]:
    """Presence of the declared crop from the absolute connected supported area and its reliability (not validated).

    ``indeterminate``: too few valid pixels, a declared crop the model lacks, too small a parcel for the grid, or too
    low a valid-pixel fraction. ``present``: the largest connected area supporting the declared crop (pixel P >=
    ``support_prob``) is at least ``min_connected_ha`` and reliably classified (mean P of the declared crop and mean
    confidence in it at least ``min_mean_p`` / ``min_confidence``) and, when a declared area is given, covers at least
    ``min_share`` of it. ``partial_presence``: connected support exists but is unreliable or smaller than ``min_share``
    of the declared area, or the declared crop is mapped on at least ``min_connected_ha`` in total but fragmented.
    ``absent``: otherwise. The share of the agricultural part, which decides the P3 status, plays no role here.
    """
    if row.get("n_valid_pixels", 0) < min_valid_pixels:
        return "indeterminate", ["too_few_valid_pixels"]
    if not row.get("declared_class_modelled"):
        return "indeterminate", ["declared_crop_not_modelled"]
    frac, expected = row.get("valid_pixel_fraction"), row.get("expected_pixels")
    if expected is not None and expected < min_expected_pixels:
        return "indeterminate", ["too_few_expected_pixels"]
    if frac is not None and frac < min_valid_fraction:
        return "indeterminate", ["low_valid_fraction"]
    conn = row["connected_support_ha"]
    if conn >= min_connected_ha:
        if row["connected_mean_p"] < min_mean_p or row["connected_mean_confidence"] < min_confidence:
            return "partial_presence", ["connected_support_unreliable"]
        if declared_area_ha and declared_area_ha > 0 and conn / declared_area_ha < min_share:
            return "partial_presence", ["connected_area_below_declared_share"]
        return "present", []
    if row["crop_area_ha"] >= min_connected_ha:
        return "partial_presence", ["fragmented_support"]
    return "absent", ["no_connected_support"]


def mismatch_taxonomy(row: dict, presence_status: str, declared_area_ha: float | None, min_valid_pixels: int = 4,
                      min_valid_fraction: float = 0.5, min_expected_pixels: float = 8.0, area_tolerance: float = 0.5,
                      model_disagreement: bool | None = None) -> tuple[str, dict]:
    """Primary reason (one of ``MISMATCH_CATEGORIES`` or ``MISMATCH_EXTRA``) and independent boolean flags.

    The flags are not exclusive (each is evaluated on its own); the primary reason follows the order resolution,
    observations, other crop, partial, area, model. ``none`` means present, area within tolerance, models agree.
    Flags are satellite-consistency indicators, not diagnoses of what is wrong with the declaration.
    """
    n = row.get("n_valid_pixels", 0)
    expected = row.get("expected_pixels")
    thin = bool(row.get("inside_raster") and row.get("n_parcel_pixels", 0) < min_valid_pixels)   # thin or tiny on the grid
    res = bool((expected is not None and expected < min_expected_pixels) or thin)
    obs = bool(not res and (n < min_valid_pixels or (row.get("valid_pixel_fraction") is not None
                                                      and row["valid_pixel_fraction"] < min_valid_fraction)))
    modelled = bool(row.get("declared_class_modelled"))
    other = bool(modelled and not res and not obs and presence_status in ("absent", "partial_presence")
                 and row.get("pixel_majority_agricultural") not in (None, row.get("declared_crop")))
    partial = presence_status == "partial_presence"
    area = False
    if modelled and declared_area_ha and declared_area_ha > 0 and "crop_area_ha" in row and presence_status == "present":
        area = abs(row["crop_area_ha"] - declared_area_ha) / declared_area_ha > area_tolerance
    flags = {"too_few_pixels": res, "no_valid_observations": obs, "other_crop_present": other,
             "declared_crop_partial": partial, "area_disagreement": area,
             "model_disagreement": bool(model_disagreement)}
    if res:
        primary = "too_few_pixels"
    elif obs:
        primary = "no_valid_observations"
    elif not modelled:
        primary = "declared_crop_not_modelled"
    elif presence_status == "absent":
        primary = "other_crop_present" if other else "non_agricultural_dominant"
    elif partial:
        primary = "declared_crop_partial"
    elif area:
        primary = "area_disagreement"
    elif model_disagreement:
        primary = "model_disagreement"
    else:
        primary = "none"
    return primary, flags


def dominant_nonagricultural_class(row: dict, classes: list[str], agri_set: set[str], max_agri_share: float = 0.10) -> str | None:
    """Largest non-agricultural class of a parcel whose agricultural share is at most ``max_agri_share``."""
    share = row.get("agricultural_share")
    if share is None or share > max_agri_share:
        return None
    area = {c: row.get(f"area_ha_{c}", 0.0) for c in classes if c not in agri_set}
    return max(area, key=area.get) if area and max(area.values()) > 0 else None


def _agri_set(classes: list[str], declared: str | None, agricultural: tuple[str, ...]) -> set[str]:
    return set(agricultural) | ({declared} if declared in classes else set())


def _open_raster(stack: ExitStack, path: str | Path) -> tuple:
    ds = stack.enter_context(rasterio.open(path))
    if ds.crs is None or not ds.crs.is_projected or not math.isclose(ds.crs.linear_units_factor[1], 1):
        raise ValueError("Crop presence requires a projected CRS in metres")
    classes = list(ds.descriptions)
    if not all(classes) or len(set(classes)) != len(classes):
        raise ValueError("Probability bands need unique class-name descriptions")
    return ds, classes


def _read_segments(segments: str | Path | None, crs, segment_id_field: str | None):
    if not segments:
        return None, segment_id_field
    seg = gpd.read_file(segments)
    if seg.crs is None:
        raise ValueError("Segments need a CRS")
    if segment_id_field is not None and segment_id_field not in seg:
        raise ValueError(f"segment_id_field {segment_id_field!r} is not a column of the segments")
    seg = seg.to_crs(crs).reset_index(drop=True)
    if segment_id_field is not None:
        seg["_sid"] = pd.factorize(seg[segment_id_field])[0] + 1       # ids 1..n (0 means "no segment")
    else:
        seg["_sid"] = np.arange(1, len(seg) + 1)
    return seg, "_sid"


def _collect(ds, classes: list[str], gdf: gpd.GeoDataFrame, id_field: str, declared_field: str,
             declared_area_field: str | None, crop_mapping: dict | None, seg, segment_id_field: str | None,
             agricultural: tuple[str, ...], interior_buffer_px: float, support_prob: float, min_segment_ha: float,
             nonagri_max_share: float, keep_segments: bool = True) -> tuple[list[dict], list[dict]]:
    """Metrics (P1, P2, P3, connected support) of every parcel on one probability raster; rows carry no decisions."""
    pixel = abs(ds.transform.a)
    pixel_ha = pixel * pixel / 1e4
    rows, seg_rows = [], []
    for parcel in gdf.itertuples():
        geom = parcel.geometry
        core = geom.buffer(-interior_buffer_px * pixel) if interior_buffer_px else geom
        if core.is_empty:
            core = geom
        raw = getattr(parcel, declared_field)
        declared = crop_mapping.get(str(raw), None) if crop_mapping else (None if pd.isna(raw) else str(raw))
        row = {"parcel_id": getattr(parcel, id_field), "declared_crop": declared, "parcel_area_ha": float(geom.area / 1e4),
               "expected_pixels": float(core.area / (pixel * pixel)), "n_parcel_pixels": 0, "n_valid_pixels": 0,
               "inside_raster": False}
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
            geo = geometry_mask([core], out_shape=cube.shape[1:], transform=tr, invert=True)
            inside = geo & valid_mask(cube, ds.nodata)
            row["inside_raster"], row["n_parcel_pixels"] = True, int(geo.sum())
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
            if keep_segments:
                for s in seg_list:
                    shp = [g for g, v in shapes(s["mask"].astype("uint8"), mask=s["mask"], transform=tr)]
                    s["geometry"] = unary_union([to_shape(g) for g in shp])
                    s["parcel_id"] = row["parcel_id"]
                    seg_rows.append({k: v for k, v in s.items() if k != "mask"})
        row["valid_pixel_fraction"] = min(1.0, row["n_valid_pixels"] / row["expected_pixels"]) if row["expected_pixels"] > 0 else 0.0
        if d_area and "crop_area_ha" in row:
            row["area_difference_ha"] = row["crop_area_ha"] - d_area
            row["area_difference_pct"] = 100 * row["area_difference_ha"] / d_area
        if d_area and "connected_support_ha" in row:
            row["connected_share_declared"] = row["connected_support_ha"] / d_area
        row["dominant_nonagricultural_class"] = dominant_nonagricultural_class(
            row, classes, _agri_set(classes, declared, agricultural), nonagri_max_share)
        row["geometry"] = geom
        rows.append(row)
    return rows, seg_rows


def _decide(row: dict, cfg: dict, model_disagreement: bool | None = None) -> None:
    """Add preliminary (P3) status, presence status and mismatch taxonomy to ``row`` according to ``cfg``."""
    d_area = row.get("declared_area_ha")
    d_area = None if d_area is None or pd.isna(d_area) else d_area
    status, reasons = preliminary_status(row, d_area, cfg["min_valid_pixels"], cfg["min_agricultural_ha"], cfg["min_connected_ha"],
                                         cfg["min_supported_share"], cfg["low_score"], cfg["min_confidence"], cfg["area_tolerance"])
    row["preliminary_status"], row["review_reason"] = status, ";".join(reasons)
    pst, preasons = presence_decision(row, d_area, cfg["min_valid_pixels"], cfg["min_connected_ha"], cfg["presence_min_mean_p"],
                                      cfg["min_confidence"], cfg["presence_min_share"], cfg["min_valid_fraction"],
                                      cfg["min_expected_pixels"])
    row["presence_status"], row["presence_reason"] = pst, ";".join(preasons)
    primary, flags = mismatch_taxonomy(row, pst, d_area, cfg["min_valid_pixels"], cfg["min_valid_fraction"],
                                       cfg["min_expected_pixels"], cfg["area_tolerance"], model_disagreement)
    row["mismatch_category"] = primary
    for k, v in flags.items():
        row[f"flag_{k}"] = v


def _config(**kw) -> dict:
    for name in ("support_prob", "min_supported_share", "low_score", "min_confidence", "presence_min_mean_p",
                 "presence_min_share", "min_valid_fraction", "nonagri_max_share"):
        v = kw[name]
        if not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(kw["interior_buffer_px"]) or kw["interior_buffer_px"] < 0:
        raise ValueError("interior_buffer_px must be finite and >= 0")
    for name in ("min_expected_pixels", "min_connected_ha"):
        if not np.isfinite(kw[name]) or kw[name] < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    return kw


def _read_parcels(parcels, id_field: str, declared_field: str, declared_area_field: str | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf or declared_field not in gdf:
        raise ValueError(f"Parcels need {id_field!r} and {declared_field!r}")
    if declared_area_field and declared_area_field not in gdf:
        raise ValueError(f"declared_area_field {declared_area_field!r} is not a column of the parcels")
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    return gdf


def analyse(probabilities: str | Path, parcels: str | Path | gpd.GeoDataFrame, output: str | Path,
            id_field: str = "parcel_id", declared_field: str = "declared_crop", declared_area_field: str | None = None,
            crop_mapping: dict | None = None, segments: str | Path | None = None, segment_id_field: str | None = None,
            agricultural: tuple[str, ...] = AGRICULTURAL_DEFAULT, interior_buffer_px: float = 0.0,
            support_prob: float = 0.5, min_segment_ha: float = 0.03, min_valid_pixels: int = 4,
            min_agricultural_ha: float = 0.03, min_connected_ha: float = 0.05, min_supported_share: float = 0.5,
            low_score: float = 0.1, min_confidence: float = 0.6, area_tolerance: float = 0.5,
            presence_min_mean_p: float = 0.6, presence_min_share: float = 0.5, min_valid_fraction: float = 0.5,
            min_expected_pixels: float = 8.0, nonagri_max_share: float = 0.10,
            compare_probabilities: str | Path | None = None, review_sample_n: int = 0, seed: int = 0) -> dict:
    """Write ``crop_presence.csv``, ``crop_presence.gpkg`` (layers ``parcels`` and ``segments``) and a manifest.

    ``compare_probabilities`` is a second probability raster (another model) analysed on the same parcels with the same
    parameters; ``review_sample_n`` > 0 also writes a stratified sample per ``mismatch_category`` (see ``review_sample``).
    """
    cfg = _config(support_prob=support_prob, min_supported_share=min_supported_share, low_score=low_score,
                  min_confidence=min_confidence, interior_buffer_px=interior_buffer_px, presence_min_mean_p=presence_min_mean_p,
                  presence_min_share=presence_min_share, min_valid_fraction=min_valid_fraction,
                  min_expected_pixels=min_expected_pixels, nonagri_max_share=nonagri_max_share, min_connected_ha=min_connected_ha,
                  min_valid_pixels=min_valid_pixels, min_agricultural_ha=min_agricultural_ha, area_tolerance=area_tolerance)
    gdf = _read_parcels(parcels, id_field, declared_field, declared_area_field)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        ds, classes = _open_raster(stack, probabilities)
        gdf = gdf.to_crs(ds.crs)
        seg, sid = _read_segments(segments, ds.crs, segment_id_field)
        rows, seg_rows = _collect(ds, classes, gdf, id_field, declared_field, declared_area_field, crop_mapping, seg, sid,
                                  agricultural, interior_buffer_px, support_prob, min_segment_ha, nonagri_max_share)
        cmp_rows = None
        if compare_probabilities:
            ds2, classes2 = _open_raster(stack, compare_probabilities)
            seg2, sid2 = _read_segments(segments, ds2.crs, segment_id_field)
            cmp_rows, _ = _collect(ds2, classes2, gdf.to_crs(ds2.crs), id_field, declared_field, declared_area_field,
                                   crop_mapping, seg2, sid2, agricultural, interior_buffer_px, support_prob, min_segment_ha,
                                   nonagri_max_share, keep_segments=False)
    for i, row in enumerate(rows):
        _decide(row, cfg)
        if cmp_rows is not None:
            other = cmp_rows[i]
            _decide(other, cfg)
            for k in ("presence_status", "preliminary_status", "p1_match_agricultural", "pixel_majority_agricultural",
                      "connected_support_ha", "soft_vote_agricultural"):
                row[f"cmp_{k}"] = other.get(k)
            row["model_disagreement"] = bool(other["presence_status"] != row["presence_status"]
                                             or other.get("p1_match_agricultural") != row.get("p1_match_agricultural"))
            _decide(row, cfg, row["model_disagreement"])                 # repeat with the flag: it can be the primary reason
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
        "schema": "geocrop-crop-presence/2", "engine_version": __version__, "classes": classes,
        "inputs": {"probabilities_sha256": sha256(probabilities),
                   "compare_probabilities_sha256": sha256(compare_probabilities) if compare_probabilities else None,
                   "parcels_sha256": sha256(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else None,
                   "segments_sha256": sha256(segments) if segments else None},
        "config": {"id_field": id_field, "declared_field": declared_field, "declared_area_field": declared_area_field,
                   "agricultural": list(agricultural), "interior_buffer_px": interior_buffer_px,
                   "support_prob": support_prob, "min_segment_ha": min_segment_ha, "min_valid_pixels": min_valid_pixels,
                   "min_agricultural_ha": min_agricultural_ha, "min_connected_ha": min_connected_ha,
                   "min_supported_share": min_supported_share, "low_score": low_score, "min_confidence": min_confidence,
                   "area_tolerance": area_tolerance, "presence_min_mean_p": presence_min_mean_p,
                   "presence_min_share": presence_min_share, "min_valid_fraction": min_valid_fraction,
                   "min_expected_pixels": min_expected_pixels, "nonagri_max_share": nonagri_max_share,
                   "segments": "given" if segments else "connected same-class pixels"},
        "thresholds_validated": False, "parcels": int(len(table)), "status_counts": {str(k): int(v) for k, v in counts.items()},
        "presence_status_counts": {str(k): int(v) for k, v in table["presence_status"].value_counts().items()},
        "mismatch_category_counts": {str(k): int(v) for k, v in table["mismatch_category"].value_counts().items()},
        "note": ("Satellite support for a declaration, not an administrative finding. Thresholds are not validated; "
                 "approval of a declaration is an administrative decision, not a field verification."),
    }
    (out / "crop_presence_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    result = {"csv": str(csv), "gpkg": str(gpkg),
              **{k: manifest[k] for k in ("parcels", "status_counts", "presence_status_counts", "mismatch_category_counts",
                                          "thresholds_validated")}}
    if review_sample_n > 0:
        result["review_sample"] = review_sample(table, review_sample_n, out / "review_sample", seed=seed)
    return result


def review_sample(table: gpd.GeoDataFrame | str | Path, n: int, output: str | Path, by: str = "mismatch_category",
                  seed: int = 0) -> dict:
    """Up to ``n`` parcels per category of ``by`` for visual control (e.g. in QGIS): ``review_sample.csv`` (parcel ids,
    category, a few metrics) and ``review_sample.gpkg`` (layer ``sample``). Within a category the parcels are spread over
    the range of the parcel area (equal-size area quantile bins, then a seeded draw), not just the first rows."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not isinstance(table, gpd.GeoDataFrame):
        table = gpd.read_file(table, layer="parcels")
    if by not in table:
        raise ValueError(f"{by!r} is not a column of the table")
    rng = np.random.default_rng(seed)
    picked = []
    for _cat, g in table.groupby(by, dropna=False, sort=True):
        if len(g) <= n:
            picked.append(g.assign(sample_stratum=0))
            continue
        order = g.sort_values(["parcel_area_ha", "parcel_id"]).reset_index(drop=True)
        bins = np.array_split(np.arange(len(order)), n)                        # one parcel from each area quantile bin
        idx = [int(rng.choice(b)) for b in bins if len(b)]
        picked.append(order.iloc[idx].assign(sample_stratum=range(len(idx))))
    sample = gpd.GeoDataFrame(pd.concat(picked, ignore_index=True), geometry="geometry", crs=table.crs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("parcel_id", by, "declared_crop", "parcel_area_ha", "declared_area_ha", "crop_area_ha", "presence_status",
                        "preliminary_status", "pixel_majority_agricultural", "dominant_nonagricultural_class",
                        "sample_stratum") if c in sample]
    csv, gpkg = out.with_suffix(".csv"), out.with_suffix(".gpkg")
    sample[keep].to_csv(csv, index=False)
    gpkg.unlink(missing_ok=True)
    sample[keep + ["geometry"]].to_file(gpkg, layer="sample", driver="GPKG")
    return {"csv": str(csv), "gpkg": str(gpkg), "n": int(len(sample)),
            "per_category": {str(k): int(v) for k, v in sample[by].value_counts().items()}}


def sensitivity(probabilities: str | Path, parcels: str | Path | gpd.GeoDataFrame, output: str | Path,
                min_connected_grid: tuple[float, ...] = (0.03, 0.05, 0.1, 0.2), support_prob_grid: tuple[float, ...] = (0.4, 0.5, 0.6),
                p2_threshold: float = 0.5, id_field: str = "parcel_id", declared_field: str = "declared_crop",
                declared_area_field: str | None = None, crop_mapping: dict | None = None, segments: str | Path | None = None,
                segment_id_field: str | None = None, agricultural: tuple[str, ...] = AGRICULTURAL_DEFAULT,
                interior_buffer_px: float = 0.0, min_segment_ha: float = 0.03, min_valid_pixels: int = 4,
                min_agricultural_ha: float = 0.03, min_supported_share: float = 0.5, low_score: float = 0.1,
                min_confidence: float = 0.6, area_tolerance: float = 0.5, presence_min_mean_p: float = 0.6,
                presence_min_share: float = 0.5, min_valid_fraction: float = 0.5, min_expected_pixels: float = 8.0) -> str:
    """Sweep ``min_connected_ha`` x ``support_prob`` and write ``presence_sensitivity.csv`` (one row per grid cell).

    Columns: counts per ``presence_status``; the cross-tab with P1 on the agricultural part (``p1_agri``) and with P2
    (``p2``: ``soft_vote_agricultural >= p2_threshold``, a convention of this table only), i.e. how many parcels the
    presence criterion supports beyond P1 / P2 (``*_not_*`` columns) and how many P1 / P2 supports it does not;
    P3 ``supported`` counts and ``p3_supported_not_p1_agri`` (the subset check). Satellite consistency, not accuracy:
    there is no field truth. ``support_prob`` and ``min_connected_ha`` change P3 as well (same value in both).
    """
    gdf = _read_parcels(parcels, id_field, declared_field, declared_area_field)
    base = dict(min_valid_pixels=min_valid_pixels, min_agricultural_ha=min_agricultural_ha, min_supported_share=min_supported_share,
                low_score=low_score, min_confidence=min_confidence, area_tolerance=area_tolerance,
                presence_min_mean_p=presence_min_mean_p, presence_min_share=presence_min_share,
                min_valid_fraction=min_valid_fraction, min_expected_pixels=min_expected_pixels, nonagri_max_share=0.10,
                interior_buffer_px=interior_buffer_px)
    table = []
    with ExitStack() as stack:
        ds, classes = _open_raster(stack, probabilities)
        gdf = gdf.to_crs(ds.crs)
        seg, sid = _read_segments(segments, ds.crs, segment_id_field)
        for sp in support_prob_grid:                       # min_connected_ha does not change the metrics: one pass per support_prob
            rows, _ = _collect(ds, classes, gdf, id_field, declared_field, declared_area_field, crop_mapping, seg, sid,
                               agricultural, interior_buffer_px, sp, min_segment_ha, 0.10, keep_segments=False)
            for mc in min_connected_grid:
                cfg = _config(support_prob=sp, min_connected_ha=mc, **base)
                dec = []
                for r in rows:
                    r = dict(r)
                    _decide(r, cfg)
                    dec.append(r)
                t = pd.DataFrame(dec)
                p1 = t.get("p1_match_agricultural", pd.Series(False, index=t.index)).fillna(False).astype(bool)
                p2 = (t.get("soft_vote_agricultural", pd.Series(np.nan, index=t.index)).fillna(-1) >= p2_threshold)
                pres = t["presence_status"] == "present"
                pp = t["presence_status"].isin(["present", "partial_presence"])
                p3 = t["preliminary_status"] == "supported"
                rec = {"min_connected_ha": mc, "support_prob": sp, "parcels": int(len(t))}
                rec.update({f"presence_{s}": int((t["presence_status"] == s).sum()) for s in PRESENCE_STATUSES})
                rec.update({"p1_agri": int(p1.sum()), "p2": int(p2.sum()), "p3_supported": int(p3.sum()),
                            "present_and_p1_agri": int((pres & p1).sum()), "present_not_p1_agri": int((pres & ~p1).sum()),
                            "p1_agri_not_present": int((~pres & p1).sum()),
                            "present_and_p2": int((pres & p2).sum()), "present_not_p2": int((pres & ~p2).sum()),
                            "p2_not_present": int((~pres & p2).sum()),
                            "present_or_partial_not_p1_agri": int((pp & ~p1).sum()),
                            "p3_supported_not_p1_agri": int((p3 & ~p1).sum()),
                            "present_not_p3_supported": int((pres & ~p3).sum())})
                table.append(rec)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out, index=False)
    return str(out)
