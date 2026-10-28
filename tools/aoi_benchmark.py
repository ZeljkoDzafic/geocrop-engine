"""Measured saving of AOI-aware Sentinel-2 import on a real irregular AOI (docs/aoi-aware.md, docs/performance.md).

    PYTHONPATH=src python tools/aoi_benchmark.py --out /tmp/aoi_bench --start 2023-07-01 --end 2023-07-31

Creates the study grid and its AOI mask for the AOI (default: the Lijevce polje case study), then runs the same
windowed STAC import (`stac_import.import_season`, what `geocrop stac-import` calls) once WITHOUT and once WITH
the mask, each into its own folder, and measures per run (helpers of tools/scale_benchmark.py): wall time,
process CPU time, network bytes received (lower bound from per-connection nettop sampling on macOS, upper bound
from the host interface counters), bytes allocated on disk and peak RSS. Finally every scene imported by both
runs is compared pixel by pixel inside the AOI. Needs psutil and network access; it downloads one month of
windowed COG reads, so run it on a connection where that is acceptable. `--order masked unmasked` reverses the
order (the second run may profit from caches of the first, so repeat in both orders to see the effect).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_benchmark as sb  # noqa: E402  (tools/scale_benchmark.py: Sampler, _dir_bytes)

DEFAULT_AOI = Path(__file__).resolve().parents[1] / "configs" / "studies" / "lijevce_polje" / "aoi.geojson"


def measure(label: str, grid: Path, mask, out: Path, args) -> dict:
    import pandas as pd

    from geocrop import stac_import
    out.mkdir(parents=True, exist_ok=True)
    sampler = sb.Sampler(out)
    sampler.start()
    stages: list[dict] = []
    try:
        with sampler.stage(f"s2_import_{label}", stages) as rec:
            results = stac_import.import_season(grid, out / "s2", args.start, args.end, max_cloud=args.max_cloud,
                                                min_clear_fraction=args.min_clear, max_items=args.max_items, aoi=mask)
            rec["items"] = len(results)
            rec["status_counts"] = pd.Series([r["status"].split(":")[0] + (": " + r["status"].split(": ", 1)[1]
                                                                        if r["status"].startswith("skipped: outside")
                                                                        else "") for r in results]).value_counts().to_dict()
    finally:
        sampler.stop_event.set()
    return {"label": label, "stage": stages[0], "results": results}


def compare(grid: Path, mask, plain: list[dict], masked: list[dict]) -> dict:
    """Scenes imported by both runs: bit-for-bit equality inside the AOI-active blocks, no-data elsewhere."""
    import numpy as np
    import rasterio

    from geocrop.indices import S2_BANDS
    from geocrop.raster import NODATA
    inside = mask.dense()
    active = np.zeros_like(inside)
    for w in mask.windows():
        active[int(w.row_off):int(w.row_off + w.height), int(w.col_off):int(w.col_off + w.width)] = True
    a = {Path(r["path"]).name: Path(r["path"]) for r in plain if r["status"] in ("imported", "cached")}
    b = {Path(r["path"]).name: Path(r["path"]) for r in masked if r["status"] in ("imported", "cached")}
    common = sorted(set(a) & set(b))
    worst, rasters, outside_ok = 0.0, 0, True
    for name in common:
        for band in (*S2_BANDS, "SCL"):
            with rasterio.open(a[name] / f"{band}.tif") as x, rasterio.open(b[name] / f"{band}.tif") as y:
                u, v = x.read(1).astype("float64"), y.read(1).astype("float64")
            same_nan = np.array_equal(np.isnan(u[active]), np.isnan(v[active]))
            diff = float(np.nanmax(np.abs(u[active] - v[active]))) if active.any() else 0.0
            worst = max(worst, diff if same_nan else float("inf"))
            fill = 0 if band == "SCL" else NODATA
            outside_ok &= bool(np.all(v[~active] == fill))
            rasters += 1
    return {"scenes_in_both": len(common), "only_unmasked": sorted(set(a) - set(b)), "only_masked": sorted(set(b) - set(a)),
            "rasters_compared": rasters, "max_abs_difference_in_active_blocks": worst,
            "masked_outside_active_blocks_is_nodata": outside_ok, "aoi_pixels": int(inside.sum())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--aoi", default=str(DEFAULT_AOI))
    p.add_argument("--crs", default="EPSG:32633")
    p.add_argument("--start", default="2023-07-01")
    p.add_argument("--end", default="2023-07-31")
    p.add_argument("--max-cloud", type=float, default=60)
    p.add_argument("--min-clear", type=float, default=0.5)
    p.add_argument("--max-items", type=int, default=200)
    p.add_argument("--order", nargs=2, choices=["masked", "unmasked"], default=["unmasked", "masked"])
    p.add_argument("--halo-px", type=int, default=0)
    args = p.parse_args(argv)

    from geocrop.aoi import AoiMask, default_mask_path
    from geocrop.raster import make_grid
    out = Path(args.out)
    grid = make_grid(args.aoi, args.crs, 10, out / "grid.tif", aoi_mask=True, halo_px=args.halo_px)
    mask = AoiMask(default_mask_path(grid), args.halo_px)
    summary = mask.summary()
    print(json.dumps(summary), flush=True)
    runs = {}
    for label in args.order:
        runs[label] = measure(label, grid, mask if label == "masked" else None, out / label, args)
        print(json.dumps({k: v for k, v in runs[label]["stage"].items()}, default=str), flush=True)
    check = compare(grid, mask, runs["unmasked"]["results"], runs["masked"]["results"])
    keys = ("wall_s", "cpu_s", "net_in_mb_lower", "net_in_mb_host_upper", "dir_size_end_mb", "peak_rss_mb")
    ratios = {}
    for k in keys:
        u, m = runs["unmasked"]["stage"].get(k), runs["masked"]["stage"].get(k)
        ratios[k] = {"unmasked": u, "masked": m, "masked_over_unmasked": round(m / u, 3) if u and m is not None else None}
    result = {"schema": "geocrop-aoi-benchmark/1", "aoi": str(args.aoi), "period": [args.start, args.end],
              "grid": summary, "order": args.order, "ratios": ratios, "comparison": check,
              "unmasked": {k: runs["unmasked"]["stage"].get(k) for k in ("items", "status_counts")},
              "masked": {k: runs["masked"]["stage"].get(k) for k in ("items", "status_counts")}}
    (out / "aoi_benchmark.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf8")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
