#!/usr/bin/env python3
"""Engineering data check for a study AOI: Sentinel-2 L2A availability and clear-observation coverage per window.

Reads ONLY the scene classification layer (SCL, 20 m COG) of every STAC item listed in a ``geocrop stac-search``
inventory CSV, warps it onto a coarse grid (``--res`` m, default 100 m) over the AOI in the study CRS, and counts
per pixel how many acquisition DATES were clear (SCL not in the engine's invalid classes; tiles of the same date are
combined). The report gives, per configured window, the number of acquisition dates, the number of dates with an
AOI-mean clear fraction >= 0.5, and the distribution of clear dates per AOI pixel (min / median / share of AOI with
at least 1, 2 and 3 clear dates). This is a planning check on metadata-grade masks, not a study result.

Usage (from the repository root)::

    geocrop stac-search --endpoint https://earth-search.aws.element84.com/v1 --collection sentinel-2-l2a \\
        --bbox <aoi bbox> --start 2023-04-01 --end 2023-09-30 --max-items 1000 --cloud-max 100 --out inv.csv
    python tools/study_data_check.py --inventory inv.csv --aoi configs/studies/lijevce_polje/aoi.geojson \\
        --config configs/studies/lijevce_polje/study_2023.yaml --end 2023-09-30 --out data_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)   # mirrors geocrop.s2.DEFAULT_SCL_INVALID


def window_of(day: date, windows: dict[str, tuple[date, date]]) -> str | None:
    for name, (a, b) in windows.items():
        if a <= day <= b:
            return name
    return None


def clear_from_scl(scl: np.ndarray, invalid=DEFAULT_SCL_INVALID) -> tuple[np.ndarray, np.ndarray]:
    """(observed, clear) boolean arrays from an SCL array (0 = no data / outside the tile)."""
    observed = scl > 0
    clear = observed & ~np.isin(scl, invalid)
    return observed, clear


def summarise(clear_by_date: dict[date, np.ndarray], inside: np.ndarray, windows: dict) -> dict:
    """Per window: dates, dates with AOI-mean clear fraction >= 0.5, clear-date counts per AOI pixel."""
    out = {}
    n_in = int(inside.sum())
    for name, (a, b) in windows.items():
        days = sorted(d for d in clear_by_date if a <= d <= b)
        count = np.zeros(inside.shape, dtype="int32")
        fractions = []
        for d in days:
            c = clear_by_date[d] & inside
            count += c
            fractions.append(float(c.sum() / n_in) if n_in else 0.0)
        v = count[inside]
        out[name] = {
            "start": a.isoformat(), "end": b.isoformat(), "dates": len(days),
            "dates_mostly_clear": int(sum(f >= 0.5 for f in fractions)),
            "aoi_clear_fraction_by_date": {d.isoformat(): round(f, 3) for d, f in zip(days, fractions)},
            "clear_dates_per_pixel": ({"min": int(v.min()), "median": float(np.median(v)), "max": int(v.max()),
                                       "share_ge1": round(float((v >= 1).mean()), 3),
                                       "share_ge2": round(float((v >= 2).mean()), 3),
                                       "share_ge3": round(float((v >= 3).mean()), 3)} if v.size else {}),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import yaml
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling

    from geocrop.stac_import import GDAL_ENV

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inventory", required=True)
    p.add_argument("--aoi", required=True)
    p.add_argument("--config", required=True, help="study config (grid.crs and windows)")
    p.add_argument("--end", help="clip the last window to this date (inventory end)")
    p.add_argument("--res", type=float, default=100.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf8"))
    crs = cfg["grid"]["crs"]
    windows = {k: (date.fromisoformat(v["start"]), date.fromisoformat(v["end"])) for k, v in cfg["windows"].items()}
    if a.end:
        end = date.fromisoformat(a.end)
        windows = {k: (s, min(e, end)) for k, (s, e) in windows.items() if s <= end}
    aoi = gpd.read_file(a.aoi).to_crs(crs)
    w, s, e, n = aoi.total_bounds
    width, height = int(np.ceil((e - w) / a.res)), int(np.ceil((n - s) / a.res))
    transform = from_origin(w, n, a.res, a.res)
    inside = ~geometry_mask(aoi.geometry, out_shape=(height, width), transform=transform)

    inv = pd.read_csv(a.inventory)
    scl = inv[inv["asset_key"] == "scl"].drop_duplicates("item_id")
    clear_by_date: dict[date, np.ndarray] = {}
    with rasterio.Env(**GDAL_ENV):
        for _, row in scl.sort_values("datetime").iterrows():
            day = date.fromisoformat(str(row["datetime"])[:10])
            if window_of(day, windows) is None:
                continue
            with rasterio.open("/vsicurl/" + row["href"]) as src, WarpedVRT(
                    src, crs=crs, transform=transform, width=width, height=height,
                    resampling=Resampling.nearest, nodata=0) as vrt:
                arr = vrt.read(1)
            _, clear = clear_from_scl(arr)
            clear_by_date[day] = clear_by_date.get(day, np.zeros_like(clear)) | clear
    report = {"inventory": str(a.inventory), "aoi": str(a.aoi), "crs": crs, "grid_res_m": a.res,
              "aoi_pixels": int(inside.sum()), "scl_items": int(len(scl)),
              "scl_invalid_classes": list(DEFAULT_SCL_INVALID), "windows": summarise(clear_by_date, inside, windows),
              "note": "Planning check from SCL at coarse resolution; contains modified Copernicus Sentinel data."}
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf8")
    for k, v in report["windows"].items():
        print(k, v["start"], v["end"], "dates", v["dates"], "mostly clear", v["dates_mostly_clear"],
              "clear dates/pixel", v["clear_dates_per_pixel"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
