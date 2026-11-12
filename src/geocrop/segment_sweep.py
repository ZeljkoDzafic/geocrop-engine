"""Segment-density sweep for segmentation-assisted post-classification refinement.

OBIA segments (``segmentation.obia``) are used here as objects over which *pixel* classes or probabilities of an
existing classifier are aggregated (e.g. P3 of ``crop-presence``). This is segmentation-assisted post-classification
refinement, **not** an independently trained GEOBIA classification (no object features, no object-level training).
An independent GEOBIA random forest is a possible future feature and is not implemented.

For each requested density (``segments_per_km2``) the sweep runs ``segmentation.obia`` and reports

    segments, realised density, area quantiles (min, p10, median, p90, max in ha), share of segments below ``min_ha``
    and runtime; with ``parcels``: median parcel area and ratio to the median segment, share of parcels holding at
    least ``k`` segments, mean segments per parcel and boundary mismatch (area share of segments that overlap more than
    one parcel); with ``reference`` + ``label_field``: area-weighted object purity (under-segmentation indicator);
    with ``probabilities`` + parcels with declared crop: ``crop_presence.analyse`` per density against the
    connected-pixel baseline (status changes, shares supported).

``segments_per_km2`` is a *target* for SLIC initial centres per tile; the realised count differs (segments smaller
than ``min_area_pixels`` are dropped, SLIC adapts to the image, seam merging joins pieces), so the realised density is
reported. SLIC is deterministic given the raster and parameters (no random seed is involved).
All numbers are a recommendation for human review, not a validated conclusion.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import __version__
from .hashing import sha256

DEFAULT_DENSITIES = (100, 300, 500, 1000)
# SLIC memory grows quickly with the number of segments per tile: about 105,000 per 1024-pixel tile (1000/km2 at 10 m)
# exhausted 48 GB, about 26,000 (same density, 512-pixel tiles) ran in a few GB.
MAX_SEGMENTS_PER_TILE = 60_000
MIN_COVERAGE = 0.99
NOTE = ("Segmentation-assisted post-classification refinement: segments aggregate pixel classes/probabilities of an "
        "existing classifier. It is not an independent GEOBIA classification. Results are a recommendation for human "
        "review, not a validated conclusion.")


def segments_per_tile(density_per_km2: float, tile_size: int, pixel_size_m: float) -> float:
    """Expected number of SLIC segments in one full tile at the requested density."""
    return float(density_per_km2) * (tile_size * pixel_size_m / 1000.0) ** 2


def parcel_coverage(seg: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> float:
    """Share of the parcel area that lies inside some segment (1 = no gaps). Gaps appear when small segments are
    dropped (``min_area_pixels``) or when tiles are skipped."""
    pu = parcels.geometry.union_all()
    return float(pu.intersection(seg.geometry.union_all()).area / pu.area) if pu.area > 0 else float("nan")


def segment_stats(seg: gpd.GeoDataFrame, min_ha: float = 0.03) -> dict:
    """Count, realised density and area quantiles (ha) of segment polygons in a projected CRS in metres."""
    area = seg.geometry.area.to_numpy() / 1e4
    q = np.percentile(area, [0, 10, 50, 90, 100])
    return {"segments": int(len(area)), "total_area_ha": float(area.sum()),
            "realised_per_km2": float(len(area) / (area.sum() / 100)) if area.sum() > 0 else None,
            "area_min_ha": float(q[0]), "area_p10_ha": float(q[1]), "area_median_ha": float(q[2]),
            "area_p90_ha": float(q[3]), "area_max_ha": float(q[4]),
            "share_below_min_ha": float((area < min_ha).mean())}


def _overlay(seg: gpd.GeoDataFrame, other: gpd.GeoDataFrame) -> pd.DataFrame:
    """(seg_i, other_j, intersection area m2) for intersecting pairs; indices are positional."""
    left, right = other.sindex.query(seg.geometry, predicate="intersects")
    if len(left) == 0:
        return pd.DataFrame({"seg": [], "other": [], "area": []})
    sg, og = seg.geometry.to_numpy(), other.geometry.to_numpy()
    area = np.array([sg[i].intersection(og[j]).area for i, j in zip(left, right)])
    return pd.DataFrame({"seg": left, "other": right, "area": area})


def parcel_comparison(seg: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame, min_segments: int = 2,
                      straddle_share: float = 0.05) -> dict:
    """Segments against parcel polygons (same projected CRS).

    A segment belongs to the parcel that contains its representative point (``segments_per_parcel``). A segment
    straddles when at least two parcels each cover >= ``straddle_share`` of its area; ``boundary_mismatch_area_share``
    is the area share of such segments among the segments that touch any parcel (area-weighted)."""
    seg = seg.reset_index(drop=True)
    parcels = parcels.reset_index(drop=True)
    pts = gpd.GeoDataFrame(geometry=seg.geometry.representative_point(), crs=seg.crs)
    hit = gpd.sjoin(pts, parcels[["geometry"]], predicate="within", how="inner")
    per_parcel = np.bincount(hit["index_right"].to_numpy(), minlength=len(parcels))
    seg_area = seg.geometry.area.to_numpy()
    pairs = _overlay(seg, parcels)
    si = pairs.seg.to_numpy().astype(int)
    touched = np.bincount(si, minlength=len(seg)) > 0
    big = pairs[pairs.area.to_numpy() >= straddle_share * seg_area[si]]
    straddle = np.bincount(big.seg.to_numpy().astype(int), minlength=len(seg)) > 1
    p_area = parcels.geometry.area.to_numpy() / 1e4
    med_seg = float(np.median(seg_area) / 1e4)
    return {"parcels": int(len(parcels)), "parcel_median_ha": float(np.median(p_area)),
            "median_parcel_to_median_segment": float(np.median(p_area) / med_seg) if med_seg > 0 else None,
            "min_segments_per_parcel": int(min_segments),
            "parcels_with_min_segments_share": float((per_parcel >= min_segments).mean()),
            "segments_per_parcel_mean": float(per_parcel.mean()),
            "boundary_mismatch_area_share": float(seg_area[straddle].sum() / seg_area[touched].sum()) if touched.any() else None,
            "straddling_segments_share": float(straddle[touched].mean()) if touched.any() else None}


def purity(seg: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, label_field: str) -> dict:
    """Object purity: share of the majority reference class in the reference-covered part of each segment.

    Computed per segment from intersections with the reference polygons (assumed not to overlap each other), summarised
    area-weighted by the covered area. 1 = every covered pixel of the object has one class; low purity indicates
    segments that mix classes (under-segmentation). Segments without reference cover are skipped."""
    if label_field not in reference:
        raise ValueError(f"label_field {label_field!r} is not a column of the reference polygons")
    seg = seg.reset_index(drop=True)
    reference = reference.reset_index(drop=True)
    pairs = _overlay(seg, reference)
    if pairs.empty:
        return {"purity_segments": 0, "purity_area_weighted": None, "purity_median": None, "purity_share_below_0.8": None}
    pairs["label"] = reference[label_field].to_numpy()[pairs.other.astype(int)]
    per = pairs.groupby(["seg", "label"]).area.sum().reset_index()
    tot = per.groupby("seg").area.sum()
    top = per.groupby("seg").area.max()
    p = (top / tot)[tot > 0]
    w = tot[p.index]
    return {"purity_segments": int(len(p)), "purity_area_weighted": float(np.average(p, weights=w)),
            "purity_median": float(p.median()), "purity_share_below_0.8": float((p < 0.8).mean())}


def presence_comparison(baseline: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Parcel-level change of crop-presence results with segments against the connected-pixel baseline."""
    b = baseline.set_index("parcel_id")
    t = table.set_index("parcel_id").reindex(b.index)
    changed = b.preliminary_status != t.preliminary_status
    sup = t.preliminary_status == "supported"
    p1 = t.get("p1_match_agricultural", pd.Series(False, index=t.index)).fillna(False).astype(bool)
    out = {"presence_parcels": int(len(b)), "status_changed": int(changed.sum()),
           "supported_baseline": int((b.preliminary_status == "supported").sum()),
           "supported": int(sup.sum()),
           "supported_share": float(sup.mean()), "supported_share_baseline": float((b.preliminary_status == "supported").mean()),
           "supported_gained": int(((b.preliminary_status != "supported") & sup).sum()),
           "supported_lost": int(((b.preliminary_status == "supported") & ~sup).sum()),
           "disputed_review": int((t.preliminary_status == "disputed_review").sum()),
           "not_supported": int((t.preliminary_status == "not_supported").sum()),
           "p1_match_agricultural_share": float(p1.mean()),              # P1 does not depend on segments
           "p2_soft_vote_agricultural_mean": _mean(t.get("soft_vote_agricultural")),
           "p3_supported_share_agricultural_mean": _mean(t.get("supported_share_agricultural")),
           "p3_supported_share_agricultural_mean_baseline": _mean(b.get("supported_share_agricultural")),
           "p3_segments_per_parcel_mean": _mean(t.get("segments")),
           "supported_without_p1_match": int((sup & ~p1).sum()),
           "p1_match_not_supported": int((p1 & ~sup).sum())}
    return out


def _mean(s) -> float | None:
    if s is None:
        return None
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) else None


def sweep(raster: str | Path, out: str | Path, densities=DEFAULT_DENSITIES, bands=None, tile_size: int = 1024,
          compactness: float = 0.1, min_area_pixels: int = 9, merge_seams: bool = True, merge_distance: float = 0.08,
          workers: int | None = None, min_ha: float = 0.03, parcels: str | Path | None = None, min_segments: int = 2,
          reference: str | Path | None = None, label_field: str | None = None,
          probabilities: str | Path | None = None, id_field: str = "parcel_id", declared_field: str = "declared_crop",
          presence_options: dict | None = None, keep_segments: bool = True, aoi=None) -> dict:
    """Run ``obia`` per density and write ``segment_sweep.csv``, ``segment_sweep.md`` and ``segment_sweep_manifest.json``.

    `presence_options` are extra keyword arguments of ``crop_presence.analyse`` (e.g. ``crop_mapping``,
    ``declared_area_field``, ``support_prob``); `aoi` is an optional `geocrop.aoi.AoiMask` (only AOI pixels are segmented); the probability raster and parcels (with the declared-crop column)
    enable the crop-presence comparison. Segments are kept as ``segments_d<density>.gpkg`` unless `keep_segments` is False."""
    from .segmentation import obia
    dens = [float(d) for d in densities]
    if not dens or any(not np.isfinite(d) or d <= 0 for d in dens) or len(set(dens)) != len(dens):
        raise ValueError("densities must be unique, finite and > 0")
    if reference is not None and not label_field:
        raise ValueError("reference polygons need label_field")
    if probabilities is not None and parcels is None:
        raise ValueError("the crop-presence comparison needs parcels with a declared crop")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster) as src:
        if src.crs is None or not src.crs.is_projected or abs(src.crs.linear_units_factor[1] - 1) > 1e-9:
            raise ValueError("The sweep requires a raster in a projected CRS in metres")
        crs = src.crs
        pixel_size = abs(src.transform.a)
    warnings: list[str] = []
    for d in dens:
        n = segments_per_tile(d, tile_size, pixel_size)
        if n > MAX_SEGMENTS_PER_TILE:
            warnings.append(f"density {d:g}/km2 gives about {n:,.0f} segments per {tile_size}-pixel tile (more than "
                            f"{MAX_SEGMENTS_PER_TILE:,}): high memory use is likely; use a smaller tile_size")
    par = gpd.read_file(parcels).to_crs(crs) if parcels else None
    ref = gpd.read_file(reference).to_crs(crs) if reference else None
    opts = dict(presence_options or {})
    base = None
    if probabilities is not None:
        from . import crop_presence
        crop_presence.analyse(probabilities, par, out / "presence_baseline", id_field, declared_field, **opts)
        base = pd.read_csv(out / "presence_baseline" / "crop_presence.csv")
    rows = []
    for d in dens:
        tag = f"{d:g}"
        path = out / f"segments_d{tag}.gpkg"
        t0 = time.perf_counter()
        r = obia(raster, path, bands, tile_size, d, compactness, min_area_pixels, merge_seams, merge_distance,
                 workers=workers, aoi=aoi)
        runtime = time.perf_counter() - t0
        seg = gpd.read_file(path)
        row = {"requested_per_km2": d, **segment_stats(seg, min_ha), "runtime_s": round(runtime, 2),
               "segments_before_merge": r["segments_before_merge"], "seam_merges": r["seam_merges"]}
        if par is not None:
            row.update(parcel_comparison(seg, par, min_segments))
            row["parcel_area_covered_share"] = parcel_coverage(seg, par)
            if row["parcel_area_covered_share"] < MIN_COVERAGE:
                warnings.append(f"density {d:g}/km2: segments cover only {row['parcel_area_covered_share']:.1%} of the "
                                f"parcel area (min_area_pixels={min_area_pixels} drops small segments); the results "
                                "for this density are not comparable with a gap-free one")
        if ref is not None:
            row.update(purity(seg, ref, label_field))
        if base is not None:
            from . import crop_presence
            sub = out / f"presence_d{tag}"
            crop_presence.analyse(probabilities, par, sub, id_field, declared_field, segments=path, **opts)
            row.update(presence_comparison(base, pd.read_csv(sub / "crop_presence.csv")))
        rows.append(row)
        if not keep_segments:
            path.unlink(missing_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "segment_sweep.csv", index=False)
    (out / "segment_sweep.md").write_text(to_markdown(table), encoding="utf8")
    inputs = {"raster_sha256": sha256(raster)}
    for k, v in (("parcels", parcels), ("reference", reference), ("probabilities", probabilities)):
        inputs[f"{k}_sha256"] = sha256(v) if v else None
    manifest = {"schema": "geocrop-segment-sweep/1", "engine_version": __version__, "python": platform.python_version(),
                "inputs": inputs, "seed": None, "seed_note": "SLIC is deterministic; no random seed is used",
                "config": {"densities": dens, "bands": list(bands) if bands else None, "tile_size": tile_size,
                           "compactness": compactness, "min_area_pixels": min_area_pixels, "merge_seams": merge_seams,
                           "merge_distance": merge_distance, "workers": workers, "min_ha": min_ha,
                           "min_segments_per_parcel": min_segments, "label_field": label_field, "id_field": id_field,
                           "declared_field": declared_field, "presence_options": {k: str(v) for k, v in opts.items()}},
                "warnings": warnings, "note": NOTE}
    (out / "segment_sweep_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return {"csv": str(out / "segment_sweep.csv"), "markdown": str(out / "segment_sweep.md"),
            "manifest": str(out / "segment_sweep_manifest.json"), "densities": dens,
            "segments": [int(s) for s in table.segments], "warnings": warnings}


def to_markdown(table: pd.DataFrame) -> str:
    """Transposed table (one column per density) with the standing caveat."""
    t = table.set_index("requested_per_km2").T
    cols = [f"{c:g}/km2" for c in t.columns]
    lines = ["| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals in t.iterrows():
        cells = [("" if pd.isna(v) else f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)) for v in vals]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + f"\n\n{NOTE}\n"
