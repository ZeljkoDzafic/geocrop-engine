"""Parcel geometry suitability: is a parcel technically suitable for reliable Sentinel-2 (10 m) classification?

The assessment runs BEFORE a declaration is analysed and looks at the parcel's geometry (and, when a raster is given, at
its valid pixels) only. A parcel that fails the criteria is *insufficient spatial resolution for reliable satellite analysis*; this
says nothing about the declaration, which is neither supported nor unsupported by it. Such parcels are better reported as
a separate group than silently scored.

Per parcel (projected CRS in metres):

    area_ha              planimetric area
    width_m              representative width, ``2 x`` the largest inscribed circle radius (negative-buffer bisection,
                         the default criterion, robust for L/U shapes and strips) ; ``mean_width_m = 2 A / P`` (area over
                         half the perimeter) is also reported and can be selected (``width_method="mean"``; it is the
                         width of a long strip but underestimates compact shapes, so it is the stricter variant)
    compactness          Polsby-Popper ``4 pi A / P^2`` (1 for a circle, towards 0 for slivers)
    elongation           long / short side of the minimum rotated rectangle
    n_pixels             pixel centres inside the parcel (on the raster grid, or on a virtual grid of ``resolution`` m)
    n_inner_pixels       pixel centres inside the parcel shrunk by ``inner_buffer_m`` (default 10 m)
    n_reliable_inner     inner pixels that are also valid (finite, not nodata in any band) and, with a mask raster, not masked
    share_reliable_inner n_reliable_inner / n_inner_pixels
    inner_area_ha        area of the parcel shrunk by ``inner_buffer_m``; ``inner_area_share`` = inner_area_ha / area_ha

The rule is explicit and configurable: ``area_below_min``, ``width_below_min``, ``too_few_inner_pixels`` and
``too_few_reliable_inner_pixels`` are reason codes of ``insufficient_spatial_resolution``. The default thresholds are **not
validated**: ``sensitivity_grid`` shows how many parcels and how much area every combination would exclude, and the
thresholds should be chosen (and frozen) only after looking at that. ``compare_subsets`` repeats the P1/P2/P3 comparison of
``crop-presence`` on all declarations and on the suitable parcels only. Everything here is a recommendation for human
review, not a conclusion about any declaration.
"""
from __future__ import annotations

import itertools
import json
import math
from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

from . import __version__
from .hashing import sha256
from .raster import valid_mask
from .samples import polygon_window

STATES = ("suitable", "insufficient_spatial_resolution")
REASONS = ("area_below_min", "width_below_min", "too_few_inner_pixels", "too_few_reliable_inner_pixels")
WIDTH_METHODS = ("inscribed", "mean")
SIZE_BINS_HA = (0.2, 0.5, 1.0, 2.0, 5.0)


def max_inscribed_radius(geom, tol: float = 0.25) -> float:
    """Radius (m) of the largest circle inside ``geom``: bisection on the negative buffer (non-empty while r <= radius)."""
    if geom is None or geom.is_empty or geom.area <= 0:
        return 0.0
    xmin, ymin, xmax, ymax = geom.bounds
    lo, hi = 0.0, min(xmax - xmin, ymax - ymin) / 2
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if geom.buffer(-mid).is_empty:
            hi = mid
        else:
            lo = mid
    return float(lo)


def shape_metrics(geom) -> dict:
    """Area, perimeter, widths, compactness and elongation of a polygon (metres)."""
    area = float(geom.area) if geom is not None and not geom.is_empty else 0.0
    if area <= 0:
        return {"area_ha": 0.0, "perimeter_m": 0.0, "width_m": 0.0, "mean_width_m": 0.0, "compactness": 0.0,
                "elongation": float("nan")}
    perim = float(geom.length)
    rect = geom.minimum_rotated_rectangle
    pts = list(rect.exterior.coords) if rect.geom_type == "Polygon" else []
    sides = sorted(math.dist(pts[i], pts[i + 1]) for i in range(2)) if len(pts) >= 3 else [0.0, 0.0]
    return {"area_ha": area / 1e4, "perimeter_m": perim, "width_m": 2 * max_inscribed_radius(geom),
            "mean_width_m": 2 * area / perim, "compactness": float(4 * math.pi * area / perim ** 2),
            "elongation": float(sides[1] / sides[0]) if sides[0] > 0 else float("inf")}


def _virtual_transform(geom, resolution: float):
    xmin, _, _, ymax = geom.bounds
    return from_origin(math.floor(xmin / resolution) * resolution, math.ceil(ymax / resolution) * resolution,
                       resolution, resolution)


def pixel_counts(geom, inner, ds=None, mask_ds=None, resolution: float = 10.0) -> dict:
    """Pixel centres inside the parcel and inside its inner part; reliable ones need a raster (valid, unmasked)."""
    out = {"n_pixels": 0, "n_inner_pixels": 0, "n_reliable_inner": 0}
    if geom is None or geom.is_empty:
        return out
    if ds is None:
        tr = _virtual_transform(geom, resolution)
        x0, y0, x1, y1 = geom.bounds
        shape = (int(math.ceil((tr.f - y0) / resolution)) + 1, int(math.ceil((x1 - tr.c) / resolution)) + 1)
        valid = np.ones(shape, dtype=bool)
    else:
        win = polygon_window(geom, ds)
        if win is None:
            return out
        tr = ds.window_transform(win)
        cube = ds.read(window=win)
        shape = cube.shape[1:]
        valid = valid_mask(cube, ds.nodata)
        if mask_ds is not None:
            valid &= mask_ds.read(1, window=win) != 0
    inside = geometry_mask([geom], out_shape=shape, transform=tr, invert=True)
    out["n_pixels"] = int(inside.sum())
    if not inner.is_empty:
        core = geometry_mask([inner], out_shape=shape, transform=tr, invert=True) & inside
        out["n_inner_pixels"] = int(core.sum())
        out["n_reliable_inner"] = int((core & valid).sum())
    return out


def _metres(crs) -> bool:
    c = pyproj.CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    return bool(c.is_projected and all(math.isclose(a.unit_conversion_factor, 1) for a in c.axis_info))


def assess(parcels: str | Path | gpd.GeoDataFrame, raster: str | Path | None = None, mask: str | Path | None = None,
           id_field: str = "parcel_id", declared_field: str | None = "declared_crop", inner_buffer_m: float = 10.0,
           resolution: float = 10.0) -> pd.DataFrame:
    """Geometry and pixel metrics per parcel (no rule yet; see `classify`). ``mask``: raster on the same grid, 0 = unreliable."""
    if not np.isfinite(inner_buffer_m) or inner_buffer_m < 0:
        raise ValueError("inner_buffer_m must be finite and >= 0")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and > 0")
    if mask is not None and raster is None:
        raise ValueError("mask needs a raster on the same grid")
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf:
        raise ValueError(f"Parcels need {id_field!r}")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"{id_field} must be unique")
    if gdf.crs is None:
        raise ValueError("Parcels need a CRS")
    rows = []
    ds = mask_ds = None
    try:
        if raster is not None:
            ds = rasterio.open(raster)
            crs = ds.crs
            resolution = abs(ds.transform.a)
            if mask is not None:
                mask_ds = rasterio.open(mask)
                if mask_ds.transform != ds.transform or mask_ds.shape != ds.shape:
                    raise ValueError("mask must be on the same grid as the raster")
        else:
            crs = gdf.crs
        if crs is None or not _metres(crs):
            raise ValueError("Parcel suitability requires a projected CRS in metres")
        gdf = gdf.to_crs(crs)
        for parcel in gdf.itertuples():
            geom = parcel.geometry
            inner = geom.buffer(-inner_buffer_m) if inner_buffer_m else geom
            row = {"parcel_id": getattr(parcel, id_field)}
            if declared_field and declared_field in gdf:
                v = getattr(parcel, declared_field)
                row["declared_crop"] = None if pd.isna(v) else str(v)
            row.update(shape_metrics(geom))
            row.update(pixel_counts(geom, inner, ds, mask_ds, resolution))
            row["inner_area_ha"] = float(inner.area / 1e4) if not inner.is_empty else 0.0
            row["inner_area_share"] = row["inner_area_ha"] / (geom.area / 1e4) if geom.area > 0 else 0.0
            row["share_reliable_inner"] = row["n_reliable_inner"] / row["n_inner_pixels"] if row["n_inner_pixels"] else 0.0
            rows.append(row)
    finally:
        for d in (ds, mask_ds):
            if d is not None:
                d.close()
    return pd.DataFrame(rows)


def _failures(table: pd.DataFrame, min_area_ha: float, min_width_m: float, min_inner_pixels: int,
              min_reliable_inner_pixels: int | None, width_method: str) -> dict[str, pd.Series]:
    if width_method not in WIDTH_METHODS:
        raise ValueError(f"width_method must be one of {WIDTH_METHODS}")
    width = table["width_m" if width_method == "inscribed" else "mean_width_m"]
    few_inner = table["n_inner_pixels"] < min_inner_pixels
    min_rel = min_inner_pixels if min_reliable_inner_pixels is None else min_reliable_inner_pixels
    out = {"area_below_min": table["area_ha"] < min_area_ha, "width_below_min": width < min_width_m,
           "too_few_inner_pixels": few_inner}
    # a parcel that is already short of inner pixels is not charged a second time for unreliable ones
    out["too_few_reliable_inner_pixels"] = ~few_inner & (table["n_reliable_inner"] < min_rel)
    return out


def classify(table: pd.DataFrame, min_area_ha: float = 0.2, min_width_m: float = 50.0, min_inner_pixels: int = 10,
             min_reliable_inner_pixels: int | None = None, width_method: str = "inscribed") -> pd.DataFrame:
    """Add ``suitability`` (suitable / insufficient_spatial_resolution) and ``suitability_reasons`` (codes joined by ';')."""
    fails = _failures(table, min_area_ha, min_width_m, min_inner_pixels, min_reliable_inner_pixels, width_method)
    out = table.copy()
    bad = np.zeros(len(out), dtype=bool)
    reasons = pd.Series([""] * len(out), index=out.index)
    for code in REASONS:
        f = fails[code].to_numpy()
        bad |= f
        reasons = reasons.where(~f, reasons + np.where(reasons == "", "", ";") + code)
    out["suitability"] = np.where(bad, "insufficient_spatial_resolution", "suitable")
    out["suitability_reasons"] = reasons
    return out


def size_class(area_ha: pd.Series, bins: tuple[float, ...] = SIZE_BINS_HA) -> pd.Series:
    edges = [-np.inf, *bins, np.inf]
    labels = [f"<{bins[0]:g} ha"] + [f"{a:g}-{b:g} ha" for a, b in zip(bins[:-1], bins[1:])] + [f">={bins[-1]:g} ha"]
    return pd.cut(area_ha, edges, labels=labels, right=False).astype(str)


def sensitivity_grid(table: pd.DataFrame, areas: tuple[float, ...] = (0.2,), widths: tuple[float, ...] = (40, 50, 60),
                     inner_pixels: tuple[int, ...] = (4, 10, 16, 25), width_method: str = "inscribed",
                     by_crop: bool = True, by_size: bool = True, size_bins: tuple[float, ...] = SIZE_BINS_HA) -> pd.DataFrame:
    """Excluded parcels and area for every threshold combination (the minimum inner pixels apply to the inner and the
    reliable inner count). Long table with ``level`` all / crop / size_class; stricter thresholds never exclude fewer."""
    groups: list[tuple[str, str, pd.Series]] = [("all", "all", pd.Series(True, index=table.index))]
    if by_crop and "declared_crop" in table:
        crop = table["declared_crop"].fillna("(none)").astype(str)
        groups += [("crop", c, crop == c) for c in sorted(crop.unique())]
    if by_size:
        sc = size_class(table["area_ha"], size_bins)
        order = list(dict.fromkeys(size_class(pd.Series([*(b - 1e-9 for b in size_bins), size_bins[-1]]), size_bins)))
        groups += [("size_class", s, sc == s) for s in order if (sc == s).any()]
    rows = []
    for a, w, n in product(areas, widths, inner_pixels):
        fails = _failures(table, a, w, n, None, width_method)
        bad = pd.concat(fails, axis=1).any(axis=1)
        for level, name, sel in groups:
            total, area = int(sel.sum()), float(table.loc[sel, "area_ha"].sum())
            ex, ex_area = int((bad & sel).sum()), float(table.loc[bad & sel, "area_ha"].sum())
            rows.append({"level": level, "group": name, "min_area_ha": a, "min_width_m": w, "min_inner_pixels": n,
                         "n_parcels": total, "n_excluded": ex, "share_excluded": ex / total if total else None,
                         "area_ha": area, "excluded_area_ha": ex_area,
                         "share_excluded_area": ex_area / area if area else None})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.3f}".rstrip("0").rstrip(".")
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


NOTE = ("Technically unsuitable means unsuitable for reliable satellite analysis at 10 m, not an unsupported declaration. "
        "The thresholds are not validated: this table is a recommendation for human review, to be read before any "
        "threshold is adopted.")


def write_sensitivity(grid: pd.DataFrame, out: str | Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "suitability_sensitivity.csv", index=False)
    parts = ["# Parcel suitability: sensitivity of the exclusion to the thresholds", "", NOTE, "",
             "## All parcels", "", _md_table(grid[grid.level == "all"].drop(columns=["level", "group"]))]
    for level, title in (("crop", "By declared crop"), ("size_class", "By size class")):
        sub = grid[grid.level == level]
        if len(sub):
            parts += ["", f"## {title}", "", _md_table(sub.drop(columns=["level"]))]
    (out / "suitability_sensitivity.md").write_text("\n".join(parts) + "\n", encoding="utf8")
    return {"csv": str(out / "suitability_sensitivity.csv"), "md": str(out / "suitability_sensitivity.md")}


def _supported(df: pd.DataFrame, method: str, p2_threshold: float) -> tuple[pd.Series, pd.Series]:
    """(evaluated, supported) booleans of one method; missing evidence is not evaluated."""
    if method in ("p1_whole", "p1_agricultural"):
        col = "p1_match_whole" if method == "p1_whole" else "p1_match_agricultural"
        v = df[col].map({True: True, False: False, "True": True, "False": False})
        return v.notna(), v.fillna(False).astype(bool)
    if method == "p2_soft_vote":
        v = pd.to_numeric(df["soft_vote_agricultural"], errors="coerce")
        return v.notna(), (v >= p2_threshold).fillna(False)
    ok = df["preliminary_status"].notna()
    return ok, df["preliminary_status"].eq("supported")


METHODS = ("p1_whole", "p1_agricultural", "p2_soft_vote", "p3_status")


def compare_subsets(crop_presence_csv: str | Path | pd.DataFrame, suitability_csv: str | Path | pd.DataFrame,
                    p2_threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    """P1/P2/P3 on all declarations and on the suitable parcels only, joined on ``parcel_id``.

    Subsets: all declarations, area >= 0.2 ha, area >= 0.5 ha and geometry-suitable parcels. Returns ``comparison`` (subset x method: declarations, area, supported, share; P3 status shares), ``agreement`` (pairwise P1-agricultural / P2 / P3), ``excluded`` (number and area of the
    declarations left out, overall) and ``excluded_by_crop``. P2 supported = ``soft_vote_agricultural >= p2_threshold``;
    P3 supported = ``preliminary_status == supported``. Declarations without a suitability row count as not assessed
    and are left out of the suitable subset.
    """
    cp = pd.read_csv(crop_presence_csv) if not isinstance(crop_presence_csv, pd.DataFrame) else crop_presence_csv.copy()
    su = pd.read_csv(suitability_csv) if not isinstance(suitability_csv, pd.DataFrame) else suitability_csv.copy()
    for name, df, cols in (("crop presence", cp, ("parcel_id", "parcel_area_ha")), ("suitability", su, ("parcel_id", "suitability"))):
        miss = [c for c in cols if c not in df]
        if miss:
            raise ValueError(f"The {name} table needs columns {miss}")
    if cp["parcel_id"].duplicated().any() or su["parcel_id"].duplicated().any():
        raise ValueError("parcel_id must be unique in both tables")
    cp = cp.merge(su[["parcel_id", "suitability"]], on="parcel_id", how="left")
    cp["suitability"] = cp["suitability"].fillna("not_assessed")
    if "declared_crop" not in cp:
        cp["declared_crop"] = None
    cp["declared_crop"] = cp["declared_crop"].fillna("(none)")
    area = pd.to_numeric(cp["parcel_area_ha"], errors="coerce")
    subsets = {"all_declarations": cp, "area_ge_0.2_ha": cp[area >= 0.2], "area_ge_0.5_ha": cp[area >= 0.5],
               "geometry_suitable": cp[cp.suitability == "suitable"]}
    rows, agree = [], []
    for sname, sub in subsets.items():
        st = sub["preliminary_status"] if "preliminary_status" in sub else pd.Series(dtype=object)
        for m in METHODS:
            ev, sup = _supported(sub, m, p2_threshold)
            n, k = int(ev.sum()), int((ev & sup).sum())
            row = {"subset": sname, "method": m, "n_declarations": len(sub), "area_ha": float(sub["parcel_area_ha"].sum()),
                   "n_evaluated": n, "n_supported": k, "share_supported": k / n if n else None}
            if m == "p3_status":
                for key in ("supported", "disputed_review", "not_supported"):
                    row[f"share_{key}"] = float((st == key).sum() / n) if n else None
            rows.append(row)
        for a, b in itertools.combinations(("p1_agricultural", "p2_soft_vote", "p3_status"), 2):
            ea, sa = _supported(sub, a, p2_threshold)
            eb, sb = _supported(sub, b, p2_threshold)
            both = ea & eb
            n = int(both.sum())
            agree.append({"subset": sname, "method_a": a, "method_b": b, "n_both_evaluated": n,
                          "both_supported": int((both & sa & sb).sum()), "only_a": int((both & sa & ~sb).sum()),
                          "only_b": int((both & ~sa & sb).sum()), "neither": int((both & ~sa & ~sb).sum()),
                          "agreement": float((both & (sa == sb)).sum() / n) if n else None})
    excl = cp[cp.suitability != "suitable"]
    total_area = float(cp["parcel_area_ha"].sum())
    excluded = pd.DataFrame([{"n_declarations": len(cp), "n_excluded": len(excl), "share_excluded": len(excl) / len(cp) if len(cp) else None,
                              "area_ha": total_area, "excluded_area_ha": float(excl["parcel_area_ha"].sum()),
                              "share_excluded_area": float(excl["parcel_area_ha"].sum()) / total_area if total_area else None}])
    by = []
    for crop, g in cp.groupby("declared_crop"):
        e = g[g.suitability != "suitable"]
        by.append({"declared_crop": crop, "n_declarations": len(g), "n_excluded": len(e), "share_excluded": len(e) / len(g),
                   "area_ha": float(g["parcel_area_ha"].sum()), "excluded_area_ha": float(e["parcel_area_ha"].sum())})
    return {"comparison": pd.DataFrame(rows), "agreement": pd.DataFrame(agree), "excluded": excluded, "excluded_by_crop": pd.DataFrame(by)}


def write_comparison(result: dict[str, pd.DataFrame], out: str | Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, df in result.items():
        p = out / f"subset_{k}.csv"
        df.to_csv(p, index=False)
        paths[k] = str(p)
    md = ["# P1 / P2 / P3 on all declarations and on suitable parcels only", "", NOTE, "",
          "A smaller share on the suitable subset is not an error rate: nothing here is validated against field truth.", "",
          "## Shares supported", "", _md_table(result["comparison"]), "",
          "## Agreement of P1 (agricultural part), P2 and P3", "", _md_table(result["agreement"]), "", "## Declarations left out", "",
          _md_table(result["excluded"]), "", "## Left out by declared crop", "", _md_table(result["excluded_by_crop"])]
    (out / "subset_comparison.md").write_text("\n".join(md) + "\n", encoding="utf8")
    paths["md"] = str(out / "subset_comparison.md")
    return paths


def analyse(parcels: str | Path | gpd.GeoDataFrame, output: str | Path, raster: str | Path | None = None,
            mask: str | Path | None = None, id_field: str = "parcel_id", declared_field: str | None = "declared_crop",
            inner_buffer_m: float = 10.0, resolution: float = 10.0, min_area_ha: float = 0.2, min_width_m: float = 50.0,
            min_inner_pixels: int = 10, min_reliable_inner_pixels: int | None = None, width_method: str = "inscribed",
            grid_areas: tuple[float, ...] = (0.2,), grid_widths: tuple[float, ...] = (40, 50, 60),
            grid_inner_pixels: tuple[int, ...] = (4, 10, 16, 25), crop_presence: str | Path | None = None,
            p2_threshold: float = 0.5) -> dict:
    """Write ``parcel_suitability.csv``, the sensitivity CSV/markdown, a manifest and, with ``crop_presence``, the subsets."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    table = classify(assess(parcels, raster, mask, id_field, declared_field, inner_buffer_m, resolution), min_area_ha,
                     min_width_m, min_inner_pixels, min_reliable_inner_pixels, width_method)
    csv = out / "parcel_suitability.csv"
    table.to_csv(csv, index=False)
    grid = sensitivity_grid(table, grid_areas, grid_widths, grid_inner_pixels, width_method)
    result = {"csv": str(csv), **{f"sensitivity_{k}": v for k, v in write_sensitivity(grid, out).items()}}
    if crop_presence:
        comp = compare_subsets(crop_presence, csv, p2_threshold)
        result.update({f"subsets_{k}": v for k, v in write_comparison(comp, out).items()})
    counts = table["suitability"].value_counts().to_dict()
    manifest = {
        "schema": "geocrop-parcel-suitability/1", "engine_version": __version__,
        "inputs": {"parcels_sha256": sha256(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else None,
                   "raster_sha256": sha256(raster) if raster else None, "mask_sha256": sha256(mask) if mask else None,
                   "crop_presence_sha256": sha256(crop_presence) if crop_presence else None},
        "config": {"id_field": id_field, "inner_buffer_m": inner_buffer_m, "resolution_m": resolution,
                   "min_area_ha": min_area_ha, "min_width_m": min_width_m, "min_inner_pixels": min_inner_pixels,
                   "min_reliable_inner_pixels": min_reliable_inner_pixels, "width_method": width_method,
                   "pixel_grid": "raster" if raster else "virtual", "grid_areas": list(grid_areas),
                   "grid_widths": list(grid_widths), "grid_inner_pixels": list(grid_inner_pixels),
                   "p2_threshold": p2_threshold},
        "thresholds_validated": False, "parcels": int(len(table)), "suitability_counts": {str(k): int(v) for k, v in counts.items()},
        "note": NOTE,
    }
    (out / "parcel_suitability_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return {**result, **{k: manifest[k] for k in ("parcels", "suitability_counts", "thresholds_validated")}}
