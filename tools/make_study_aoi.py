#!/usr/bin/env python3
"""Reproducible area of interest (AOI) for the Lijevče polje case study (Republic of Srpska, Bosnia and Herzegovina).

Lijevče polje is the alluvial plain of the lower Vrbas between the Sava (north), Prosara (west), Motajica (east)
and Kozara (south-west); it belongs to the municipalities of Gradiška, Laktaši and Srbac. There is no official
polygon of the plain, so the study AOI is defined operationally and reproducibly as

    AOI = (Gradiška ∪ Laktaši ∪ Srbac) ∩ Bosnia and Herzegovina ∩ {Copernicus DEM GLO-30 < threshold}

followed by mask cleaning (morphological opening removes thin valley "fingers" that reach into the hills,
components smaller than ``--min-island-km2`` are dropped, enclosed holes smaller than ``--max-hole-km2`` -
forest patches and villages that stand out in a surface model - are filled), vectorisation, a light
simplification in UTM 33N and a final intersection with the municipality union (so the AOI never leaves BiH).

Threshold. ``--threshold auto`` (default) takes the elevation at which the histogram of the DEM inside the three
municipalities stops decreasing above its low-land mode, i.e. the first local minimum of the 5 m-bin histogram
after the mode (floodplain population vs. the continuum of the Kozara/Prosara/Motajica foothill slopes). The
value actually used is recorded in the output; a fixed value can be passed instead (``--threshold 130``).

Sources (fetched at run time, pinned):

* geoBoundaries gbOpen BIH ADM0 and ADM3 (municipalities), release commit ``f549eab`` - licence
  "Public Domain" (source: Wikimedia Commons) as stated by the geoBoundaries API; cite Runfola et al. (2020)
  PLoS ONE 15(4): e0231866, https://doi.org/10.1371/journal.pone.0231866. Files are verified by SHA-256.
* Copernicus DEM GLO-30 Public (DSM, 2021 release) as Cloud-Optimised GeoTIFFs on AWS Open Data
  (``s3://copernicus-dem-30m``, read anonymously over HTTPS, only the needed 1°x1° tiles). Licence: "Licence for
  the use of the Copernicus WorldDEM-30" (free; reproduction, distribution and adaptation allowed with the notice
  below). The SHA-256 of the mosaicked elevation array is recorded.

Output: a GeoJSON FeatureCollection in EPSG:4326 with one (multi)polygon and properties (sources, URLs, checksums,
threshold, cleaning parameters, area in km² computed in the equal-area ETRS89-LAEA projection EPSG:3035, area per
municipality, licence notices).

Usage (from the repository root)::

    python tools/make_study_aoi.py --out configs/studies/lijevce_polje/aoi.geojson
    python tools/make_study_aoi.py --threshold 130 --out /tmp/aoi.geojson --cache ~/.cache/geocrop-aoi

Needs only core dependencies (numpy, scipy, rasterio, geopandas, shapely, requests) and network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

GB_COMMIT = "f549eab"
GB_URL = "https://github.com/wmgeolab/geoBoundaries/raw/{commit}/releaseData/gbOpen/BIH/{level}/geoBoundaries-BIH-{level}.geojson"
GB_SHA256 = {
    "ADM0": "3fe0135768dd2fcf2650b884e761fa04fdad70fda0d7df5a2d9a3ed257d0bd80",
    "ADM3": "d0196a097d9d517f7db6a8013889892c0ab5aab3d7dfd2e74fd5b22152ff8857",
}
GB_LICENCE = ("geoBoundaries gbOpen BIH (boundaryLicense: Public Domain, source: Wikimedia Commons); "
              "Runfola et al. (2020) PLoS ONE 15(4): e0231866, doi:10.1371/journal.pone.0231866")
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_DEM/"
           "Copernicus_DSM_COG_10_{tile}_DEM.tif")
DEM_NOTICE = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
              "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. "
              "The organisations in charge of the Copernicus programme by law or by delegation do not incur any "
              "liability for any use of the Copernicus WorldDEM-30.")
MUNICIPALITIES = ("Gradiška", "Laktaši", "Srbac")
EQUAL_AREA_CRS = "EPSG:3035"
WORK_CRS = "EPSG:32633"


# --------------------------------------------------------------------------------------------- pure logic
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def suggest_threshold(values: np.ndarray, bin_width: float = 5.0, smooth_bins: int = 1) -> float:
    """First local minimum of the elevation histogram above its mode (upper edge of the minimum bin).

    ``values`` are elevations inside the candidate region. The histogram (``bin_width`` m bins, optional moving
    average over ``smooth_bins``) is scanned from the modal bin upwards; the threshold is the upper edge of the
    first bin whose successor is not lower, i.e. where the floodplain population gives way to the slope continuum.
    """
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite elevation values")
    lo = math.floor(v.min() / bin_width) * bin_width
    hi = math.ceil(v.max() / bin_width) * bin_width + bin_width
    counts, edges = np.histogram(v, bins=np.arange(lo, hi + bin_width / 2, bin_width))
    counts = counts.astype("float64")
    if smooth_bins > 1:
        counts = np.convolve(counts, np.ones(smooth_bins) / smooth_bins, mode="same")
    mode = int(np.argmax(counts))
    for i in range(mode, len(counts) - 1):
        if counts[i + 1] >= counts[i]:
            return float(edges[i + 1])
    return float(edges[-1])


def disk(radius_px: int) -> np.ndarray:
    r = int(radius_px)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def clean_mask(mask: np.ndarray, pixel_area_m2: np.ndarray | float, opening_px: int = 3,
               min_island_km2: float = 5.0, max_hole_km2: float = 5.0) -> np.ndarray:
    """Opening (disk of ``opening_px``), drop components < ``min_island_km2``, fill holes < ``max_hole_km2``.

    ``pixel_area_m2`` is a scalar or an array broadcastable to ``mask`` (geographic grids: area per row).
    """
    from scipy import ndimage as ndi

    m = np.asarray(mask, dtype=bool)
    area = np.broadcast_to(np.asarray(pixel_area_m2, dtype="float64"), m.shape)
    if opening_px > 0:
        m = ndi.binary_opening(m, structure=disk(opening_px))
    lab, n = ndi.label(m)
    if n:
        comp_area = ndi.sum(area, lab, index=np.arange(1, n + 1)) / 1e6
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = comp_area >= min_island_km2
        m = keep[lab]
    holes = ndi.binary_fill_holes(m) & ~m
    hlab, hn = ndi.label(holes)
    if hn:
        hole_area = ndi.sum(area, hlab, index=np.arange(1, hn + 1)) / 1e6
        fill = np.zeros(hn + 1, dtype=bool)
        fill[1:] = hole_area < max_hole_km2
        m = m | fill[hlab]
    return m


def geographic_pixel_area(transform, height: int) -> np.ndarray:
    """Approximate area (m²) of each row of a lon/lat grid, as a (height, 1) column (spherical Earth)."""
    r = 6371008.8
    rows = np.arange(height) + 0.5
    lat = transform.f + transform.e * rows
    dlat = math.radians(abs(transform.e))
    dlon = math.radians(abs(transform.a))
    return (r * r * dlon * dlat * np.cos(np.radians(lat)))[:, None]


def mask_to_geometry(mask: np.ndarray, transform):
    """Union of the True cells of ``mask`` as a shapely (Multi)Polygon in the grid's CRS."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    if not polys:
        raise ValueError("empty mask: no AOI cells")
    return unary_union(polys)


def equal_area_km2(geom, crs: str = "EPSG:4326") -> float:
    import geopandas as gpd

    return float(gpd.GeoSeries([geom], crs=crs).to_crs(EQUAL_AREA_CRS).area.iloc[0] / 1e6)


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def dem_tiles(bounds) -> list[str]:
    """Copernicus DEM 1°x1° tile ids (e.g. ``N45_00_E017_00``) covering ``(west, south, east, north)``."""
    west, south, east, north = bounds
    out = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            out.append(f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00")
    return out


def select_municipalities(gdf, names=MUNICIPALITIES, field: str = "shapeName"):
    found = gdf[gdf[field].isin(names)]
    missing = sorted(set(names) - set(found[field]))
    if missing:
        raise ValueError(f"municipalities not found in boundary data: {missing}")
    return found


def build_aoi(dem: np.ndarray, transform, municipalities, country_geom, threshold: float | str = "auto",
              opening_px: int = 3, min_island_km2: float = 5.0, max_hole_km2: float = 5.0,
              simplify_m: float = 30.0) -> dict:
    """Pure AOI construction on an in-memory lon/lat DEM grid. Returns geometry (EPSG:4326) + diagnostics."""
    import geopandas as gpd
    from rasterio.features import geometry_mask

    union = municipalities.geometry.union_all() if hasattr(municipalities.geometry, "union_all") \
        else municipalities.geometry.unary_union
    region = union.intersection(country_geom)
    inside = ~geometry_mask([region], out_shape=dem.shape, transform=transform)
    valid = inside & np.isfinite(dem)
    if not valid.any():
        raise ValueError("DEM does not overlap the municipalities")
    used = suggest_threshold(dem[valid]) if threshold == "auto" else float(threshold)
    low = valid & (dem < used)
    cleaned = clean_mask(low, geographic_pixel_area(transform, dem.shape[0]), opening_px, min_island_km2,
                         max_hole_km2)
    geom = mask_to_geometry(cleaned, transform)
    work = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(WORK_CRS)
    work = work.simplify(simplify_m).buffer(0)
    region_w = gpd.GeoSeries([region], crs="EPSG:4326").to_crs(WORK_CRS)
    aoi_w = work.intersection(region_w)
    aoi = aoi_w.to_crs("EPSG:4326").iloc[0]
    per_muni = {}
    munis = municipalities.to_crs(WORK_CRS)
    for name, g in zip(municipalities["shapeName"], munis.geometry):
        part = aoi_w.iloc[0].intersection(g)
        per_muni[name] = round(equal_area_km2(part, WORK_CRS), 1)
    return {"geometry": aoi, "threshold_m": used, "threshold_rule": "auto" if threshold == "auto" else "fixed",
            "area_km2": round(equal_area_km2(aoi), 1), "area_by_municipality_km2": per_muni,
            "dem_min_m": float(np.nanmin(dem[valid])), "dem_max_m": float(np.nanmax(dem[valid]))}


# --------------------------------------------------------------------------------------------- I/O
def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or (sha256 and sha256_bytes(dest.read_bytes()) != sha256):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    actual = sha256_bytes(dest.read_bytes())
    if sha256 and actual != sha256:
        raise RuntimeError(f"checksum mismatch for {url}: {actual} != {sha256}")
    return dest


def read_dem(bounds, pad_deg: float = 0.02):
    """Mosaic the Copernicus DEM GLO-30 COG tiles over ``bounds`` (windowed HTTP reads, no full download)."""
    import rasterio
    from rasterio.merge import merge

    west, south, east, north = bounds
    tiles = dem_tiles(bounds)
    urls = [DEM_URL.format(tile=t) for t in tiles]
    env = {"AWS_NO_SIGN_REQUEST": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
           "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env):
        srcs = [rasterio.open("/vsicurl/" + u) for u in urls]
        try:
            arr, transform = merge(srcs, bounds=(west - pad_deg, south - pad_deg, east + pad_deg, north + pad_deg),
                                   nodata=np.nan, dtype="float32")
        finally:
            for s in srcs:
                s.close()
    return arr[0], transform, urls


def main(argv: list[str] | None = None) -> int:
    import geopandas as gpd
    from shapely.geometry import mapping

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="configs/studies/lijevce_polje/aoi.geojson")
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "geocrop" / "aoi"))
    p.add_argument("--threshold", default="auto", help="'auto' (histogram rule) or metres above sea level")
    p.add_argument("--opening-px", type=int, default=3, help="opening disk radius in DEM pixels (~30 m)")
    p.add_argument("--min-island-km2", type=float, default=5.0)
    p.add_argument("--max-hole-km2", type=float, default=5.0)
    p.add_argument("--simplify-m", type=float, default=30.0)
    a = p.parse_args(argv)

    cache = Path(a.cache)
    files = {lvl: fetch(GB_URL.format(commit=GB_COMMIT, level=lvl), cache / f"geoBoundaries-BIH-{lvl}.geojson",
                        GB_SHA256[lvl]) for lvl in ("ADM0", "ADM3")}
    country = gpd.read_file(files["ADM0"]).to_crs("EPSG:4326")
    country_geom = country.geometry.iloc[0]
    munis = select_municipalities(gpd.read_file(files["ADM3"]).to_crs("EPSG:4326"))
    bounds = tuple(munis.total_bounds)
    dem, transform, urls = read_dem(bounds)
    threshold = a.threshold if a.threshold == "auto" else float(a.threshold)
    res = build_aoi(dem, transform, munis, country_geom, threshold, a.opening_px, a.min_island_km2,
                    a.max_hole_km2, a.simplify_m)
    aoi = res["geometry"]
    west, south, east, north = aoi.bounds
    zones = sorted({utm_zone(west), utm_zone(east)})
    props = {
        "name": "Lijevče polje study AOI",
        "definition": "(Gradiška ∪ Laktaši ∪ Srbac) ∩ BiH ∩ Copernicus DEM GLO-30 < threshold, cleaned",
        "municipalities": list(MUNICIPALITIES),
        "threshold_m": res["threshold_m"], "threshold_rule": res["threshold_rule"],
        "cleaning": {"opening_px": a.opening_px, "min_island_km2": a.min_island_km2,
                     "max_hole_km2": a.max_hole_km2, "simplify_m": a.simplify_m, "simplify_crs": WORK_CRS},
        "area_km2": res["area_km2"], "area_crs": EQUAL_AREA_CRS,
        "area_by_municipality_km2": res["area_by_municipality_km2"],
        "dem_range_in_municipalities_m": [round(res["dem_min_m"], 1), round(res["dem_max_m"], 1)],
        "bbox_wgs84": [round(v, 5) for v in (west, south, east, north)],
        "utm_zones": zones,
        "sources": {
            "boundaries": {"dataset": "geoBoundaries gbOpen BIH ADM0/ADM3", "commit": GB_COMMIT,
                           "urls": {k: GB_URL.format(commit=GB_COMMIT, level=k) for k in GB_SHA256},
                           "sha256": GB_SHA256, "licence": GB_LICENCE},
            "dem": {"dataset": "Copernicus DEM GLO-30 Public (DSM, 2021 release), AWS Open Data copernicus-dem-30m",
                    "tiles": urls, "mosaic_sha256": sha256_bytes(np.ascontiguousarray(dem).tobytes()),
                    "licence": "Licence for the use of the Copernicus WorldDEM-30", "notice": DEM_NOTICE},
        },
        "generator": "tools/make_study_aoi.py",
    }
    fc = {"type": "FeatureCollection", "name": "lijevce_polje_aoi",
          "features": [{"type": "Feature", "properties": props, "geometry": mapping(aoi)}]}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # coordinates rounded to ~1 cm keep the file small and the output byte-stable
    fc["features"][0]["geometry"] = json.loads(json.dumps(mapping(aoi)), parse_float=lambda s: round(float(s), 7))
    text = json.dumps(fc, ensure_ascii=False, separators=(",", ":"))
    out.write_text(text + "\n", encoding="utf8")
    summary = {k: props[k] for k in ("threshold_m", "threshold_rule", "area_km2", "area_by_municipality_km2",
                                     "bbox_wgs84", "utm_zones", "dem_range_in_municipalities_m")}
    summary["out"] = str(out)
    summary["sha256"] = sha256_bytes(out.read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
