"""Performance at scale: the real chain on nested real areas, with measured wall/CPU time, RAM, disk and network.

    PYTHONPATH=src python tools/scale_benchmark.py run --out outputs/scale --sizes 25 100 500 1000
    PYTHONPATH=src python tools/scale_benchmark.py report --out outputs/scale --images docs/images

`run` executes one child process per area (so that peak RSS is per area), smallest first, each a nested square
centred in the Semberija plain (northern Bosnia and Herzegovina, Republic of Srpska; grid EPSG:32634, 10 m):

    grid -> Sentinel-2 L2A import (Earth Search STAC, windowed COG reads) -> Sentinel-1 RTC import (Planetary
    Computer STAC) -> S2/S1 window composites (4 monthly windows) -> feature stack -> OBIA segmentation of the peak
    window -> PSEUDO-labels (k-means of segment NDVI trajectories, as in `geocrop acceptance`; NOT crop types)
    -> Random-Forest training (spatial CV, calibration, conformal, AOA) -> prediction (classes, probabilities,
    uncertainty, AOA) -> parcel analytics on the segments.

A background sampler records per stage: wall time, process CPU time (user + system, incl. finished children)
and average busy cores, sampled peak RSS of the process tree and the process high-water mark, bytes added to
the output directory and its peak size, and network bytes received. Network bytes come from two sources:
a LOWER bound from per-connection `nettop` sampling of this process (macOS; flows that open and close between
two samples are missed) and an UPPER bound from the host's interface counters (includes all other traffic on
the machine). System load is recorded because other jobs on the host inflate wall time (not CPU time).

Before an area is started its runtime is predicted from the previous area (linear in area); an area predicted
to exceed `--max-hours` is not started and is recorded as skipped. Each child is also killed at `--max-hours`.
The labels are pseudo-classes: the training/prediction cost is representative, the accuracy is not reported.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CENTRE_UTM34 = (364000.0, 4960000.0)        # Semberija plain west of Bijeljina (~19.10 E, 44.78 N), EPSG:32634
CRS = "EPSG:32634"
START, END = "2023-06-01", "2023-09-30"     # one season, four monthly windows (as `geocrop acceptance`)
RF = {"n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
MB = 2**20


# ----------------------------------------------------------------------------------------- sampling

def _dir_bytes(path: Path) -> int:
    """Allocated bytes below `path` (st_blocks, i.e. what the volume really loses)."""
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _nettop_flows(pid: int) -> dict[str, int] | None:
    """bytes_in per live TCP/UDP flow of `pid` (macOS nettop); None where nettop is unavailable."""
    if sys.platform != "darwin" or not shutil.which("nettop"):
        return None
    try:
        out = subprocess.run(["nettop", "-L", "1", "-p", str(pid), "-x", "-J", "bytes_in"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.splitlines()
    if not lines or "bytes_in" not in lines[0].split(","):
        return None
    col = lines[0].split(",").index("bytes_in")
    flows = {}
    for line in lines[1:]:
        cols = line.split(",")
        name = next((c for c in cols[:col] if c), "")
        if len(cols) > col and name.startswith(("tcp", "udp")):
            try:
                flows[name] = int(cols[col])
            except ValueError:
                pass
    return flows


class Sampler(threading.Thread):
    """Samples the current process tree every `interval` s; disk/network less often (they cost more)."""

    def __init__(self, out: Path, interval: float = 0.5, disk_every: float = 5.0, net_every: float = 2.0):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process()
        self.out, self.interval, self.disk_every, self.net_every = out, interval, disk_every, net_every
        self.flows: dict[str, int] = {}
        self.nettop_ok = True
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cur: dict | None = None
        self.disk_now = _dir_bytes(out)
        psutil.cpu_percent(None)

    def tree_rss(self) -> int:
        rss = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except self.psutil.Error:
                pass
        return rss

    def net_lower(self) -> int:
        flows = _nettop_flows(os.getpid())
        if flows is None:
            self.nettop_ok = False
        else:
            for k, v in flows.items():
                self.flows[k] = max(v, self.flows.get(k, 0))
        return sum(self.flows.values())

    def run(self):
        last_disk = last_net = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            rss = self.tree_rss()
            if now - last_disk >= self.disk_every:
                self.disk_now = _dir_bytes(self.out)
                last_disk = now
            if now - last_net >= self.net_every:
                self.net_lower()
                last_net = now
            cpu = self.psutil.cpu_percent(None)
            with self.lock:
                c = self.cur
                if c is not None:
                    c["peak_rss"] = max(c["peak_rss"], rss)
                    c["rss_sum"] += rss
                    c["n"] += 1
                    c["peak_disk"] = max(c["peak_disk"], self.disk_now)
                    c["sys_cpu_sum"] += cpu
                    c["load_sum"] += os.getloadavg()[0]
            self.stop_event.wait(self.interval)

    def counters(self) -> dict:
        t = self.proc.cpu_times()
        return {"cpu": t.user + t.system + t.children_user + t.children_system,
                "net_upper": self.psutil.net_io_counters().bytes_recv, "net_lower": self.net_lower(),
                "disk": _dir_bytes(self.out), "wall": time.perf_counter()}

    @contextmanager
    def stage(self, name: str, records: list):
        start = self.counters()
        with self.lock:
            self.cur = {"peak_rss": self.tree_rss(), "rss_sum": 0, "n": 0, "peak_disk": start["disk"],
                        "sys_cpu_sum": 0.0, "load_sum": 0.0}
        rec = {"stage": name, "status": "ok"}
        try:
            yield rec
        except BaseException as exc:
            rec["status"] = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            end = self.counters()
            with self.lock:
                c, self.cur = self.cur, None
            wall = end["wall"] - start["wall"]
            cpu = end["cpu"] - start["cpu"]
            rec.update({
                "wall_s": round(wall, 2), "cpu_s": round(cpu, 2),
                "avg_busy_cores": round(cpu / wall, 2) if wall > 0 else None,
                "cpu_utilisation_of_host": round(cpu / wall / os.cpu_count(), 3) if wall > 0 else None,
                "peak_rss_mb": round(max(c["peak_rss"], self.tree_rss()) / MB, 1),
                "mean_rss_mb": round(c["rss_sum"] / c["n"] / MB, 1) if c["n"] else None,
                "process_hwm_rss_mb": _hwm_mb(),
                "disk_written_mb": round((end["disk"] - start["disk"]) / MB, 1),
                "dir_size_end_mb": round(end["disk"] / MB, 1),
                "dir_peak_mb": round(max(c["peak_disk"], end["disk"]) / MB, 1),
                "net_in_mb_lower": round((end["net_lower"] - start["net_lower"]) / MB, 1) if self.nettop_ok else None,
                "net_in_mb_host_upper": round((end["net_upper"] - start["net_upper"]) / MB, 1),
                "host_cpu_busy_pct_mean": round(c["sys_cpu_sum"] / c["n"], 1) if c["n"] else None,
                "host_load1_mean": round(c["load_sum"] / c["n"], 1) if c["n"] else None,
            })
            records.append(rec)
            print(f"  [{name}] {rec['status']} wall {rec['wall_s']} s, cpu {rec['cpu_s']} s, peak RSS "
                  f"{rec['peak_rss_mb']} MB, +disk {rec['disk_written_mb']} MB, net >= {rec['net_in_mb_lower']} MB",
                  flush=True)


def _hwm_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (MB if sys.platform == "darwin" else 1024), 1)


# ----------------------------------------------------------------------------------------- one area

def square(km2: float) -> tuple[float, float, float, float]:
    half = math.sqrt(km2) * 1000 / 2
    x, y = CENTRE_UTM34
    return (x - half, y - half, x + half, y + half)


def run_one(km2: float, out: Path, s1: bool, seed: int = 42) -> dict:  # noqa: C901 - linear chain
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    from geocrop import config as cfgmod
    from geocrop import features, model, parcels, preflight, s1_stac, s2, scenes, segmentation, stac_import
    from geocrop import s1 as s1mod
    from geocrop.acceptance import pseudo_reference, windows_for
    from geocrop.raster import NODATA, make_grid
    from geocrop.samples import extract

    out.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(out)
    sampler.start()
    stages: list[dict] = []
    windows = windows_for(START, END)
    periods = list(windows)
    info = {"km2_requested": km2, "bbox_utm": square(km2), "crs": CRS, "period": [START, END],
            "windows": {p: [a.isoformat(), b.isoformat()] for p, (a, b) in windows.items()}, "s1": s1}
    t_all = time.perf_counter()
    error = None
    try:
        with sampler.stage("grid", stages):
            grid = make_grid(gpd.GeoDataFrame(geometry=[box(*square(km2))], crs=CRS), CRS, 10, out / "grid.tif")
            with rasterio.open(grid) as g:
                info.update({"shape": list(g.shape), "pixels": g.width * g.height,
                             "area_km2": round(g.width * g.height / 1e4, 2),
                             "bbox_wgs84": [round(v, 5) for v in transform_bounds(g.crs, "EPSG:4326", *g.bounds)]})
        with sampler.stage("s2_import", stages) as rec:
            res = stac_import.import_season(grid, out / "scenes" / "s2", START, END, max_cloud=60,
                                            min_clear_fraction=0.5, max_items=200)
            rec["items"] = len(res)
            rec["status_counts"] = pd.Series([r["status"].split(":")[0] for r in res]).value_counts().to_dict()
        s2_dirs = [Path(r["path"]) for r in res if r["status"] in ("imported", "cached")]
        s1_dirs = []
        if s1:
            with sampler.stage("s1_import", stages) as rec:
                res1 = s1_stac.import_season(grid, out / "scenes" / "s1", START, END, max_items=200)
                rec["items"] = len(res1)
                rec["status_counts"] = pd.Series([r["status"].split(":")[0] for r in res1]).value_counts().to_dict()
            s1_dirs = [Path(r["path"]) for r in res1 if r["status"] in ("imported", "cached")]
        info["scenes"] = {"s2": len(s2_dirs), "s1": len(s1_dirs)}
        comp = out / "composites"
        if comp.exists():
            shutil.rmtree(comp)
        with sampler.stage("s2_composites", stages) as rec:
            g2 = scenes.assign([scenes.read(d) for d in s2_dirs], windows)
            rec["per_window"] = {p: len(v) for p, v in g2.items()}
            for p in periods:
                s2.composite([m["path"] for m in g2[p]], comp / p, reference=grid, mask_dilation_px=1,
                             temporal_screening=len(g2[p]) >= 3)
        if s1:
            with sampler.stage("s1_composites", stages) as rec:
                g1 = scenes.assign([scenes.read(d) for d in s1_dirs], windows)
                rec["per_window"] = {p: len(v) for p, v in g1.items()}
                for p in periods:
                    if g1[p]:
                        s1mod.composite([m["path"] for m in g1[p]], comp / p, reference=grid)
        stack = out / "features" / "stack.tif"
        with sampler.stage("features", stages) as rec:
            features.build_stack(comp, stack)
            rec["features"] = len(features.load_manifest(stack))
        info["features"] = rec["features"]
        with sampler.stage("segmentation", stages) as rec:
            with rasterio.open(stack) as ds:
                names = list(ds.descriptions)
                med = {}
                for p in periods:
                    v = ds.read(names.index(f"{p}_NDVI") + 1)
                    med[p] = float(np.median(v[v != NODATA])) if (v != NODATA).any() else -1.0
            peak = max(med, key=med.get)
            seg = segmentation.obia(stack, out / "segmentation" / "segments.gpkg",
                                    bands=[f"{peak}_B04", f"{peak}_B08", f"{peak}_B11", f"{peak}_NDVI"],
                                    segments_per_km2=50)
            segs = gpd.read_file(seg["path"])
            rec.update({"peak_window": peak, "segments": len(segs)})
        with sampler.stage("pseudo_labels", stages) as rec:
            ref, pinfo = pseudo_reference(stack, segs, periods, k=4, seed=seed)
            rec.update({k: pinfo[k] for k in ("k", "segments_used", "train_segments", "holdout_segments")})
            train_path = out / "reference" / "pseudo_training.gpkg"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.unlink(missing_ok=True)
            ref[ref.split == "train"].assign(polygon_id=lambda d: "S" + d.segment_id.astype(str),
                                             class_name=lambda d: d.pseudo_class)[
                ["polygon_id", "class_name", "geometry"]].to_file(train_path, driver="GPKG")
        with sampler.stage("training", stages) as rec:
            samples = extract(stack, train_path, max_per_polygon=50, seed=seed)
            rep = model.train(samples, out / "model", params=RF, n_splits=3, spatial_block_m=2000, seed=seed,
                              bootstrap=100)
            rec.update({"samples": rep["samples"], "features": rep["features"], "trees": RF["n_estimators"],
                        "folds": 3})
            info["samples"], info["classes"] = rep["samples"], len(rep.get("class_counts", {})) or pinfo["k"]
        pred = out / "prediction"
        with sampler.stage("prediction", stages):
            model.predict(stack, out / "model" / "model.joblib", pred)
        with sampler.stage("parcels", stages) as rec:
            pin = ref.assign(parcel_id=lambda d: "S" + d.segment_id.astype(str),
                             declared_crop=lambda d: d.pseudo_class)[["parcel_id", "declared_crop", "geometry"]]
            ps = parcels.aggregate(pred / "probabilities.tif", pin, out / "parcels" / "parcels.gpkg",
                                   strategy="mean_probability", conformal=out / "model" / "conformal.json", alpha=0.1,
                                   aoa=pred / "aoa.tif")
            rec["parcels"] = ps["parcels"]
        # preflight estimate for the same layout (scenes, grid, features, samples) to compare with the measurement
        cfg = cfgmod.load(None, {
            "grid": {"crs": CRS, "resolution_m": 10, "reference": str(grid)},
            "windows": {p: {"start": a, "end": b} for p, (a, b) in info["windows"].items()},
            "paths": {"s2_scenes": str(out / "scenes" / "s2"), "s1_scenes": str(out / "scenes" / "s1"),
                      "composites": str(comp), "features": str(stack), "output": str(out / "run"),
                      "training": str(train_path)},
            "compositing": {"mask_dilation_px": 1},
            "model": {"n_estimators": RF["n_estimators"], "cv_folds": 3}})
        layout = preflight.plan(cfg)
        info["preflight_estimate"] = preflight.estimate(cfg, layout, n_samples=info["samples"],
                                                        n_classes=info["classes"])
        info["preflight_layout"] = {"window_features": layout["window_features"],
                                    "s2_scenes": sum(len(v) for v in layout["s2_by_window"].values()),
                                    "s1_scenes": sum(len(v) for v in layout["s1_by_window"].values()),
                                    "pixels": layout["pixels"]}
        info["output_sizes_mb"] = {name: round(_dir_bytes(out / name) / MB, 1) for name in
                                   ("scenes", "composites", "features", "segmentation", "model", "prediction",
                                    "parcels")}
    except Exception as exc:  # recorded; the report shows which stage failed
        import traceback
        error = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc(limit=8)
    sampler.stop_event.set()
    result = {"schema": "geocrop-scale-benchmark/1", "area": km2, "error": error, "stages": stages, "info": info,
              "total_wall_s": round(time.perf_counter() - t_all, 1), "process_hwm_rss_mb": _hwm_mb(),
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "host": {"platform": platform.platform(), "python": platform.python_version(),
                       "cpu_count": os.cpu_count(), "ram_gb": round(_ram_bytes() / 1e9, 1)},
              "network_bytes_method": {"lower": "nettop per-flow sampling of this process (macOS)" if
                                       sampler.nettop_ok else None,
                                       "upper": "host interface counters (psutil.net_io_counters; all traffic)"}}
    (out / "benchmark.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf8")
    return result


def _ram_bytes() -> int:
    import psutil
    return psutil.virtual_memory().total


# ----------------------------------------------------------------------------------------- orchestration

def run_all(out: Path, sizes, s1_sizes, max_hours: float, python: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "runs.json"
    runs = json.loads(log.read_text()) if log.exists() else {}
    prev = None
    for km2 in sorted(sizes):
        key = f"{km2:g}"
        target = out / f"area_{key}km2"
        done = target / "benchmark.json"
        if done.exists() and not json.loads(done.read_text()).get("error"):
            prev = json.loads(done.read_text())
            print(f"{key} km2: already measured, reused", flush=True)
            continue
        if prev and prev.get("total_wall_s"):
            predicted_h = prev["total_wall_s"] * km2 / prev["area"] / 3600
            if predicted_h > max_hours:
                runs[key] = {"status": "skipped", "reason": f"predicted {predicted_h:.1f} h (linear in area from "
                             f"{prev['area']:g} km2) exceeds {max_hours} h"}
                print(f"{key} km2: {runs[key]['reason']}", flush=True)
                log.write_text(json.dumps(runs, indent=2))
                continue
        print(f"{key} km2: starting", flush=True)
        cmd = [python, str(Path(__file__).resolve()), "one", "--km2", str(km2), "--out", str(target)]
        if km2 in s1_sizes:
            cmd.append("--s1")
        t0 = time.perf_counter()
        try:
            rc = subprocess.run(cmd, timeout=max_hours * 3600).returncode
            status = "ok" if rc == 0 else f"exit {rc}"
        except subprocess.TimeoutExpired:
            status = f"killed after {max_hours} h"
        runs[key] = {"status": status, "wall_s": round(time.perf_counter() - t0, 1)}
        log.write_text(json.dumps(runs, indent=2))
        if done.exists():
            prev = json.loads(done.read_text())


# ----------------------------------------------------------------------------------------- report

STAGE_ORDER = ["grid", "s2_import", "s1_import", "s2_composites", "s1_composites", "features", "segmentation",
               "pseudo_labels", "training", "prediction", "parcels"]


def load_results(out: Path) -> list[dict]:
    res = []
    for f in sorted(out.glob("area_*km2/benchmark.json")):
        res.append(json.loads(f.read_text()))
    return sorted(res, key=lambda r: r["area"])


def _fit(x, y):
    """Least-squares y = a + b x (b only when < 2 points)."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return 0.0, float(y[0] / x[0]) if len(x) else 0.0
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def report(out: Path, images: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = [r for r in load_results(out) if not r.get("error")]
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    rows = []
    for r in res:
        st = {s["stage"]: s for s in r["stages"]}
        i = r["info"]
        rows.append({"area": i["area_km2"], "mpx": i["pixels"] / 1e6, "s2": i["scenes"]["s2"], "s1": i["scenes"]["s1"],
                     "features": i.get("features"), "samples": i.get("samples"), "wall": r["total_wall_s"],
                     "cpu": sum(s["cpu_s"] for s in r["stages"]),
                     "peak_rss": max(s["peak_rss_mb"] for s in r["stages"]),
                     "disk": max(s["dir_peak_mb"] for s in r["stages"]),
                     "net_lo": sum(s["net_in_mb_lower"] or 0 for s in r["stages"]),
                     "net_hi": sum(s["net_in_mb_host_upper"] for s in r["stages"]), "stages": st,
                     "est": i.get("preflight_estimate", {}), "sizes": i.get("output_sizes_mb", {})})
    lines.append("| Area km² | Grid Mpx | S2 / S1 scenes | Features | Train samples | Wall min | CPU min | "
                 "Peak RSS GB | Output peak GB | Network in GB (lower–upper) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in rows:
        lines.append(f"| {w['area']:g} | {w['mpx']:.2f} | {w['s2']} / {w['s1']} | {w['features']} | {w['samples']} | "
                     f"{w['wall'] / 60:.1f} | {w['cpu'] / 60:.1f} | {w['peak_rss'] / 1024:.2f} | "
                     f"{w['disk'] / 1024:.2f} | {w['net_lo'] / 1024:.2f}–{w['net_hi'] / 1024:.2f} |")
    lines.append("")
    for w in rows:
        lines.append(f"**{w['area']:g} km²** (grid {w['mpx']:.2f} Mpx)")
        lines.append("")
        lines.append("| Stage | Wall s | CPU s | Busy cores | Peak RSS MB | Disk written MB | Net in MB (lower–upper) | "
                     "Host load (1 min) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name in STAGE_ORDER:
            s = w["stages"].get(name)
            if s:
                lo = "n/a" if s["net_in_mb_lower"] is None else f"{s['net_in_mb_lower']:g}"
                lines.append(f"| {name} | {s['wall_s']:g} | {s['cpu_s']:g} | {s['avg_busy_cores']} | "
                             f"{s['peak_rss_mb']:g} | {s['disk_written_mb']:g} | {lo}–{s['net_in_mb_host_upper']:g} | "
                             f"{s['host_load1_mean']} |")
        lines.append("")
    # preflight comparison
    lines.append("| Area km² | Quantity | Preflight estimate | Measured | Measured / estimate |")
    lines.append("|---|---|---|---|---|")
    comparisons = []
    for w in rows:
        e, st, sz = w["est"], w["stages"], w["sizes"]
        if not e.get("disk"):
            continue
        rt = e["runtime_hours_rough"]
        meas = {
            "composite runtime s": (rt["composite"] * 3600, sum(st[n]["wall_s"] for n in ("s2_composites", "s1_composites") if n in st)),
            "features runtime s": (rt["features"] * 3600, st["features"]["wall_s"]),
            "train runtime s": (rt["train"] * 3600, st["training"]["wall_s"]),
            "predict runtime s": (rt["predict"] * 3600, st["prediction"]["wall_s"]),
            "composites disk GB": (e["disk"]["composite"]["expected_gb"], sz.get("composites", 0) * MB / 1e9),
            "feature stack disk GB": (e["disk"]["features"]["expected_gb"], sz.get("features", 0) * MB / 1e9),
            "prediction disk GB": (e["disk"]["predict"]["expected_gb"], sz.get("prediction", 0) * MB / 1e9),
            "peak RAM GB (max stage)": (e["peak_ram_gb"]["max"], w["peak_rss"] * MB / 1e9),
        }
        for q, (est, m) in meas.items():
            ratio = m / est if est else float("nan")
            comparisons.append({"area": w["area"], "quantity": q, "estimate": est, "measured": m, "ratio": ratio})
            lines.append(f"| {w['area']:g} | {q} | {est:.3g} | {m:.3g} | {ratio:.2f} |")
    lines.append("")
    # plots
    if rows:
        a = [w["area"] for w in rows]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        ax = axes[0]
        for name in STAGE_ORDER:
            ys = [w["stages"].get(name, {}).get("wall_s") for w in rows]
            if any(v is not None and v > 1 for v in ys):
                ax.plot(a, [v if v is not None else float("nan") for v in ys], marker="o", label=name)
        ax.set(xscale="log", yscale="log", xlabel="area (km²)", ylabel="wall time (s)", title="Wall time per stage")
        ax.legend(fontsize=7)
        ax = axes[1]
        ax.plot(a, [w["peak_rss"] / 1024 for w in rows], marker="o", label="peak RSS (GB)")
        ax.plot(a, [w["disk"] / 1024 for w in rows], marker="s", label="output dir peak (GB)")
        ax.plot(a, [w["net_lo"] / 1024 for w in rows], marker="^", label="network in, lower bound (GB)")
        ax.set(xscale="log", yscale="log", xlabel="area (km²)", title="Memory, disk, network")
        ax.legend(fontsize=7)
        ax = axes[2]
        ax.plot(a, [w["wall"] / 60 for w in rows], marker="o", label="wall (min)")
        ax.plot(a, [w["cpu"] / 60 for w in rows], marker="s", label="CPU (min)")
        ax.set(xscale="log", yscale="log", xlabel="area (km²)", title="Total time")
        ax.legend(fontsize=7)
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(images / "scale_benchmark.png", dpi=130)
        plt.close(fig)
    # extrapolation: per stage, wall and CPU linear in area (fit over measured areas)
    extra = {}
    if len(rows) >= 2:
        for name in STAGE_ORDER:
            pts = [(w["area"], w["stages"][name]) for w in rows if name in w["stages"]]
            if len(pts) >= 2:
                extra[name] = {"wall": _fit([p[0] for p in pts], [p[1]["wall_s"] for p in pts]),
                               "cpu": _fit([p[0] for p in pts], [p[1]["cpu_s"] for p in pts]),
                               "disk_mb": _fit([p[0] for p in pts], [p[1]["disk_written_mb"] for p in pts]),
                               "net_mb": _fit([p[0] for p in pts], [p[1]["net_in_mb_lower"] or 0 for p in pts])}
        extra["peak_rss_mb"] = _fit([w["area"] for w in rows], [w["peak_rss"] for w in rows])
        extra["dir_peak_mb"] = _fit([w["area"] for w in rows], [w["disk"] for w in rows])
    summary = {"rows": [{k: v for k, v in w.items() if k != "stages"} for w in rows], "comparisons": comparisons,
               "linear_fits_a_plus_b_area": extra}
    (out / "scale_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    (out / "scale_tables.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure all areas (one child process per area)")
    r.add_argument("--out", required=True)
    r.add_argument("--sizes", type=float, nargs="+", default=[25, 100, 500, 1000])
    r.add_argument("--s1-sizes", type=float, nargs="*", default=None, help="areas that also import Sentinel-1 "
                   "(default: all)")
    r.add_argument("--max-hours", type=float, default=3.0)
    o = sub.add_parser("one", help="measure one area (used by `run`)")
    o.add_argument("--km2", type=float, required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--s1", action="store_true")
    p = sub.add_parser("report", help="tables, plots and linear fits from measured areas")
    p.add_argument("--out", required=True)
    p.add_argument("--images", default="docs/images")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_all(Path(a.out), a.sizes, a.sizes if a.s1_sizes is None else a.s1_sizes, a.max_hours, sys.executable)
    elif a.cmd == "one":
        res = run_one(a.km2, Path(a.out), a.s1)
        sys.exit(1 if res["error"] else 0)
    else:
        report(Path(a.out), Path(a.images))


if __name__ == "__main__":
    main()
