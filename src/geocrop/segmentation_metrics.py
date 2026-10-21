"""Segmentation quality against independent reference parcels.

Raster level (boundary maps on identical grids):
    boundary precision / recall / F1 with a distance tolerance, reported in pixels AND metres;
    boundary-pixel IoU (overlap of boundary masks, not parcel IoU).
Object level (polygons):
    over-segmentation OS = 1 - |r ∩ s*| / |r|, under-segmentation US = 1 - |r ∩ s*| / |s*|,
    D = sqrt((OS² + US²) / 2) with s* the segment of largest overlap (Clinton et al., 2010);
    matched-object precision / recall / F1 at IoU >= `iou_threshold` (one-to-one matching).
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation


def boundary_scores(predicted, reference, tolerance_px: int = 2) -> dict:
    p = np.asarray(predicted, dtype=bool)
    r = np.asarray(reference, dtype=bool)
    if p.shape != r.shape or p.ndim != 2:
        raise ValueError("Expected two 2-D masks of identical shape")
    if tolerance_px < 0:
        raise ValueError("tolerance must be >= 0")
    pb = binary_dilation(p, iterations=tolerance_px) if tolerance_px else p
    rb = binary_dilation(r, iterations=tolerance_px) if tolerance_px else r
    pn, rn = int(p.sum()), int(r.sum())
    precision = float((p & rb).sum() / pn) if pn else float(rn == 0)
    recall = float((r & pb).sum() / rn) if rn else float(pn == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = int((p | r).sum())
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": float(f1),
            "boundary_pixel_iou": float((p & r).sum() / union) if union else 1.0,
            "tolerance_px": tolerance_px, "predicted_pixels": pn, "reference_pixels": rn}


def evaluate_boundary_rasters(prediction: str | Path, reference: str | Path, output: str | Path | None = None,
                              threshold: float = 0.5, tolerance_px: int = 2) -> dict:
    with rasterio.open(prediction) as p, rasterio.open(reference) as r:
        if p.shape != r.shape or p.crs != r.crs or not p.transform.almost_equals(r.transform):
            raise ValueError("Rasters must share one grid")
        x, y = p.read(1), r.read(1)
        ok = np.isfinite(x) & np.isfinite(y)
        if p.nodata is not None:
            ok &= x != p.nodata
        if r.nodata is not None:
            ok &= y != r.nodata
        result = boundary_scores((x >= threshold) & ok, (y >= 0.5) & ok, tolerance_px)
        result["tolerance_m"] = tolerance_px * abs(p.transform.a)
        result["valid_fraction"] = float(ok.mean())
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def object_scores(segments: str | Path | gpd.GeoDataFrame, reference: str | Path | gpd.GeoDataFrame,
                  iou_threshold: float = 0.5, output: str | Path | None = None) -> dict:
    seg = gpd.read_file(segments) if not isinstance(segments, gpd.GeoDataFrame) else segments
    ref = gpd.read_file(reference) if not isinstance(reference, gpd.GeoDataFrame) else reference
    if seg.crs is None or ref.crs is None:
        raise ValueError("Both layers need a CRS")
    seg = seg.to_crs(ref.crs).reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    seg_geom, ref_geom = seg.geometry.values, ref.geometry.values
    tree = seg.sindex
    os_, us_, ds_, pairs = [], [], [], []
    for i, r in enumerate(ref_geom):
        cand = tree.query(r, predicate="intersects")
        if len(cand) == 0:
            os_.append(1.0)
            us_.append(1.0)
            ds_.append(1.0)
            continue
        inter = np.array([r.intersection(seg_geom[j]).area for j in cand])
        k = int(inter.argmax())
        j, ov = int(cand[k]), inter[k]
        o, u = 1 - ov / r.area, 1 - ov / seg_geom[j].area
        os_.append(o)
        us_.append(u)
        ds_.append(float(np.sqrt((o**2 + u**2) / 2)))
        for jj, a in zip(cand, inter):
            union = r.area + seg_geom[jj].area - a
            if union > 0 and a / union >= iou_threshold:
                pairs.append((a / union, i, int(jj)))
    used_r, used_s = set(), set()
    matched_iou = []
    for iou, i, j in sorted(pairs, reverse=True):          # greedy one-to-one matching
        if i not in used_r and j not in used_s:
            used_r.add(i)
            used_s.add(j)
            matched_iou.append(iou)
    tp = len(matched_iou)
    precision = tp / len(seg) if len(seg) else 0.0
    recall = tp / len(ref) if len(ref) else 0.0
    result = {
        "reference_objects": int(len(ref)), "segments": int(len(seg)),
        "over_segmentation_mean": float(np.mean(os_)), "under_segmentation_mean": float(np.mean(us_)),
        "D_mean": float(np.mean(ds_)), "iou_threshold": iou_threshold, "matched": tp,
        "object_precision": float(precision), "object_recall": float(recall),
        "object_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "matched_iou_mean": float(np.mean(matched_iou)) if matched_iou else None,
        "method": "Clinton et al. (2010) OS/US/D; one-to-one IoU matching",
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf8")
    return result
