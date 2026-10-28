"""Quicklook of REAL Sentinel-2 imagery for an area and period, ready for the map.

    geocrop quicklook --bbox 19.10 44.72 19.16 44.76 --start 2023-04-01 --end 2023-09-30 --out outputs/semberija

1. a 10 m study grid in the local UTM zone is derived from the bounding box (WGS 84);
2. Sentinel-2 L2A acquisitions are read as windows from public cloud-optimised GeoTIFFs (STAC; Earth Search
   by default, no credentials) and kept only if the area is sufficiently clear;
3. clear observations are composited per window (default: calendar months) with cloud/shadow screening.

The output holds one scene folder per acquisition date and one composite per window; the web map shows
true colour, false colour (NIR) and NDVI for each. No model or reference data are involved.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from . import s2, scenes
from .aoi import AoiMask, default_mask_path
from .raster import make_grid
from .stac_import import EARTH_SEARCH, import_season

MAX_AREA_KM2 = 2500


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def monthly_windows(start: date, end: date) -> dict[str, tuple[date, date]]:
    windows, cur = {}, date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        windows[cur.strftime("%Y-%m")] = (max(cur, start), min(nxt - timedelta(days=1), end))
        cur = nxt
    return windows


def quicklook(bbox, start: str, end: str, output: str | Path, endpoint: str = EARTH_SEARCH,
              max_cloud: float = 60, min_clear: float = 0.3, max_items: int = 60, windows: dict | None = None,
              aoi=None) -> dict:
    """Quicklook composites of a bounding box (`bbox`, unchanged behaviour) or, with `aoi` (a vector file or a
    GeoDataFrame), of an irregular AOI: the grid still covers its bounding box, but only the blocks that touch
    the AOI are imported and composited (docs/aoi-aware.md)."""
    if aoi is not None:
        frame = gpd.read_file(aoi) if not isinstance(aoi, gpd.GeoDataFrame) else aoi
        if frame.crs is None:
            raise ValueError("AOI needs a CRS")
        west, south, east, north = frame.to_crs("EPSG:4326").total_bounds
    else:
        west, south, east, north = map(float, bbox)
    if not (west < east and south < north):
        raise ValueError("bbox must be west south east north (degrees)")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    epsg = utm_epsg((west + east) / 2, (south + north) / 2)
    area = frame.to_crs(epsg) if aoi is not None else gpd.GeoDataFrame(geometry=[box(west, south, east, north)],
                                                                      crs="EPSG:4326").to_crs(epsg)
    area_km2 = float(area.geometry.area.sum() / 1e6) if aoi is not None else float(area.area.iloc[0] / 1e6)
    if area_km2 > MAX_AREA_KM2:
        raise ValueError(f"Area {area_km2:.0f} km2 is too large for a quicklook (max {MAX_AREA_KM2} km2)")
    grid = make_grid(area, f"EPSG:{epsg}", 10, out / "grid.tif", aoi_mask=aoi is not None)
    mask = AoiMask(default_mask_path(grid)) if aoi is not None else None
    imported = import_season(grid, out / "scenes", start, end, endpoint, max_cloud=max_cloud,
                             min_clear_fraction=min_clear, max_items=max_items, aoi=mask)
    usable = [r for r in imported if r["status"] in ("imported", "cached")]
    windows = windows or monthly_windows(date.fromisoformat(start), date.fromisoformat(end))
    groups = scenes.assign(scenes.discover(out / "scenes"), windows) if usable else {}
    composites = {}
    for name, members in groups.items():
        if not members:
            continue
        s2.composite([m["path"] for m in members], out / "composites" / name, reference=grid, mask_dilation_px=1,
                     temporal_screening=len(members) >= 3, aoi=mask)
        composites[name] = len(members)
    summary = {"bbox": [west, south, east, north], "crs": f"EPSG:{epsg}", "area_km2": round(area_km2, 2),
               "period": [start, end], "endpoint": endpoint, "acquisitions_found": len(imported),
               "acquisitions_used": len(usable), "composites": composites,
               "skipped": [r for r in imported if r["status"] not in ("imported", "cached")],
               "note": "Real Sentinel-2 L2A data (Copernicus), windowed reads; contains modified Copernicus Sentinel data."}
    (out / "quicklook.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary
