# Performance at scale (measured)

Engineering evidence for sizing: the real chain run on nested real areas of the Semberija plain (northern
Bosnia and Herzegovina, Republic of Srpska; grid EPSG:32634, 10 m), one season (2023-06-01 to 2023-09-30, four
monthly windows), with wall time, CPU time, memory, disk and network measured per stage. These numbers document
the engine; they are not results of the crop-mapping study.

```bash
pip install psutil matplotlib                     # measurement and plot dependencies of the tool only
PYTHONPATH=src python tools/scale_benchmark.py run --out outputs/scale --sizes 25 100 500 1000 --max-hours 3
PYTHONPATH=src python tools/scale_benchmark.py report --out outputs/scale --images docs/images
```

## What runs

Each area is a square centred at 364 000 E / 4 960 000 N (west of Bijeljina) and runs in its own process, so
that peak memory is per area:

grid → **Sentinel-2 L2A import** (Earth Search STAC, windowed COG reads, cloud filter 60 %, ≥ 50 % clear AOI) →
**Sentinel-1 RTC import** (Planetary Computer STAC, signed COGs, overpass mosaics) → S2 and S1 window composites
(S2 with 1 px mask dilation and temporal screening) → feature stack (97 features) → OBIA segmentation of the
peak-NDVI window (50 segments/km²) → **pseudo-labels** → Random Forest training (100 trees, 3 spatial folds,
calibration, conformal, AOA) → prediction (classes, probabilities, uncertainty, AOA map) → parcel analytics on
the segments.

**Labels are synthetic pseudo-classes** (k-means of per-segment NDVI trajectories, exactly as in
`geocrop acceptance`; 50 sampled pixels per training segment). They make training and prediction cost
realistic for this many polygons; **no accuracy is reported** and none should be inferred. Training cost depends
on the number of reference samples, not on area: with real reference polygons it is usually far smaller than here.

## How it is measured

A sampling thread records the process tree every 0.5 s. Per stage:

| Quantity | Method |
|---|---|
| Wall / CPU time | `time.perf_counter`; process CPU user + system incl. finished children (psutil); busy cores = CPU / wall |
| Peak RSS | sampled maximum of the process tree RSS (the process high-water mark is also stored) |
| Disk | allocated bytes (`st_blocks`) added to the output directory, and its peak size |
| Network in | **lower bound**: bytes received per TCP flow of this process from macOS `nettop`, sampled every 2 s (flows that open and close between samples are missed); **upper bound**: host interface counters (all traffic of the machine; marked n/a where a 32-bit counter wrapped) |
| Host load | 1-min load average and host CPU busy %, because other jobs inflate wall (not CPU) time |

MB/GB in the tables are binary (MiB/GiB). Machine: Apple M5 Pro laptop (15 cores, 48 GB RAM, internal NVMe SSD), macOS, Python 3.12; ordinary
broadband connection. **The host was shared** with other heavy jobs (the 25 km² run and the 500 km² training/prediction ran at 1-min
load 17–25 on 15 cores; the 1000 km² run mostly at 3–7): CPU seconds are reliable, wall seconds are pessimistic.

## Results

![Scaling of wall time per stage, memory/disk/network and total time with area](images/scale_benchmark.png)

| Area km² | Grid Mpx | S2 / S1 scenes | Features | Train samples | Wall min | CPU min | Peak RSS GB | Output peak GB | Network in GB (lower–upper) |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 0.25 | 14 / 38 | 97 | 28600 | 5.6 | 2.7 | 1.50 | 0.33 | 0.59–1.76 |
| 100 | 1.00 | 15 / 38 | 97 | 112500 | 8.0 | 6.7 | 2.41 | 1.20 | 1.34–1.93 |
| 501 | 5.01 | 15 / 38 | 97 | 551450 | 43.7 | 50.5 | 5.92 | 5.80 | 3.67–n/a |
| 1001 | 10.01 | 16 / 38 | 97 | 1159069 | 60.2 | 106.4 | 7.38 | 11.48 | 6.75–n/a |

All four areas completed (the 3 h cap was not reached; 1000 km² took 60 min). A first 1000 km² attempt hung for
more than an hour in the Sentinel-1 import on a dead HTTPS connection; GDAL had no HTTP timeout, which is fixed in
this release (connect/total timeouts and a low-speed abort, see CHANGELOG) and the area was re-measured from
scratch. Areas are the snapped grid extents; scene counts are acquisitions that passed the filters (S1: overpass
mosaics of both orbit directions; Sentinel-1B was not operating in 2023).

<details><summary>Per-stage measurements (all areas)</summary>

**25 km²** (grid 0.25 Mpx)

| Stage | Wall s | CPU s | Busy cores | Peak RSS MB | Disk written MB | Net in MB (lower–upper) | Host load (1 min) |
|---|---|---|---|---|---|---|---|
| grid | 0.08 | 0.06 | 0.69 | 222.8 | 0 | 0–0 | None |
| s2_import | 191.93 | 18.09 | 0.09 | 301.6 | 98.8 | 349.8–986.6 | 25.3 |
| s1_import | 46.49 | 7.6 | 0.16 | 276.5 | 59.8 | 249.7–328.4 | 21.6 |
| s2_composites | 5.73 | 4.2 | 0.73 | 467.4 | 28.8 | 0–36.3 | 21.7 |
| s1_composites | 1.7 | 1.04 | 0.61 | 503.9 | 6.1 | 0–15 | 22.4 |
| features | 2.47 | 2.27 | 0.92 | 703.9 | 80.5 | 0–6.7 | 22.4 |
| segmentation | 2.21 | 2.06 | 0.93 | 771.6 | 1.1 | 0–5.8 | 22.1 |
| pseudo_labels | 0.78 | 0.86 | 1.1 | 794.9 | 0.8 | 0–2.3 | 21.9 |
| training | 11.58 | 52.14 | 4.5 | 1122.6 | 56.9 | 0–28.9 | 22.8 |
| prediction | 70.62 | 73.95 | 1.05 | 1533.1 | 7 | 0–387.8 | 21.9 |
| parcels | 0.59 | 0.53 | 0.9 | 1167.2 | 1.7 | 0–7.4 | 22.2 |

**100 km²** (grid 1.00 Mpx)

| Stage | Wall s | CPU s | Busy cores | Peak RSS MB | Disk written MB | Net in MB (lower–upper) | Host load (1 min) |
|---|---|---|---|---|---|---|---|
| grid | 0.07 | 0.04 | 0.6 | 225.1 | 0 | 0–0.8 | None |
| s2_import | 255.97 | 34.23 | 0.13 | 366.2 | 422.3 | 821.8–1403.7 | 19.8 |
| s1_import | 71.53 | 15.78 | 0.22 | 368.1 | 237.3 | 545.4–572.3 | 13.3 |
| s2_composites | 10.7 | 9.85 | 0.92 | 681.9 | 122.7 | 0–0.1 | 8.1 |
| s1_composites | 2.87 | 2.42 | 0.84 | 700.9 | 24.2 | 0–0 | 7.6 |
| features | 7.82 | 7.58 | 0.97 | 1147.2 | 288 | 0–0.2 | 7.3 |
| segmentation | 14.24 | 14.12 | 0.99 | 1491 | 4.3 | 0–0.2 | 6.4 |
| pseudo_labels | 1.83 | 2.66 | 1.45 | 1173.2 | 3 | 0–0 | 5.8 |
| training | 25 | 214.81 | 8.59 | 1923.2 | 98.9 | 0–0.3 | 8.9 |
| prediction | 89.71 | 99.87 | 1.11 | 2467.5 | 26.4 | 0–1.3 | 6.9 |
| parcels | 1.22 | 1.17 | 0.96 | 2034.7 | 6.7 | 0–0 | 4.6 |

**501 km²** (grid 5.01 Mpx)

| Stage | Wall s | CPU s | Busy cores | Peak RSS MB | Disk written MB | Net in MB (lower–upper) | Host load (1 min) |
|---|---|---|---|---|---|---|---|
| grid | 0.07 | 0.04 | 0.55 | 234.1 | 0 | 0–0 | None |
| s2_import | 581.21 | 107.95 | 0.19 | 719.5 | 2103.8 | 1523.6–n/a | 2.0 |
| s1_import | 948.1 | 112.2 | 0.12 | 707.4 | 1183 | 2234.7–2357.1 | 1.6 |
| s2_composites | 46.74 | 46.2 | 0.99 | 1599.7 | 581.1 | 0–0.8 | 1.2 |
| s1_composites | 12 | 11.54 | 0.96 | 1617.7 | 123.6 | 0–0.2 | 1.2 |
| features | 36.3 | 36.68 | 1.01 | 3088.1 | 1424 | 0–0.6 | 1.2 |
| segmentation | 62.66 | 63.01 | 1.01 | 3583.5 | 21.9 | 0–1 | 1.7 |
| pseudo_labels | 7.76 | 8.65 | 1.11 | 3442 | 15.4 | 0–0.1 | 2.1 |
| training | 181.31 | 1789.1 | 9.87 | 6046.4 | 332.2 | 0–4.6 | 17.6 |
| prediction | 741.23 | 846.04 | 1.14 | 6067 | 121.5 | 0–1294.8 | 20.9 |
| parcels | 6.04 | 5.98 | 0.99 | 1859.8 | 33.8 | 0–51.8 | 6.6 |

**1001 km²** (grid 10.01 Mpx)

| Stage | Wall s | CPU s | Busy cores | Peak RSS MB | Disk written MB | Net in MB (lower–upper) | Host load (1 min) |
|---|---|---|---|---|---|---|---|
| grid | 0.09 | 0.05 | 0.54 | 239.8 | 0 | 0–0 | 30.1 |
| s2_import | 832.19 | 174.83 | 0.21 | 1121.7 | 4200.7 | 2371.8–n/a | 4.7 |
| s1_import | 838.31 | 141.15 | 0.17 | 1169.7 | 2373 | 4538.3–n/a | 2.9 |
| s2_composites | 108.33 | 106.19 | 0.98 | 2574.8 | 1142.8 | 0–125.4 | 3.6 |
| s1_composites | 27.09 | 25.69 | 0.95 | 2605 | 244.8 | 0–14.7 | 5.0 |
| features | 77.38 | 79.79 | 1.03 | 2885.1 | 2816 | 0–29.6 | 4.9 |
| segmentation | 238.27 | 238.95 | 1.0 | 3619.1 | 45.9 | 0–1726.7 | 4.5 |
| pseudo_labels | 45.15 | 48.19 | 1.07 | 3780 | 32.2 | 0–115 | 2.9 |
| training | 438.74 | 4312.62 | 9.83 | 7555.4 | 575.1 | 0–213.8 | 31.1 |
| prediction | 989.99 | 1246.99 | 1.26 | 6961 | 250.4 | 0–164.3 | 6.7 |
| parcels | 12.41 | 12.3 | 0.99 | 4012.9 | 70.6 | 0–0.4 | 3.6 |

</details>

### Observations

* **Imports are network-bound and sequential.** S2 and S1 imports use 0.2 busy cores and dominate small areas
  (latency) and stay ~45 % of wall time at 1000 km² (14 min each). Per 1000 km² and season they received at
  least 2.3 GB (S2) and 4.4 GB (S1) and wrote 4.1 GB and 2.3 GB of scene folders. Scene folders are the largest
  item on disk (56 % of the 11.5 GiB output at 1000 km²). Several import processes in parallel (one per
  tile/orbit or window) are the obvious speed-up on a VM with good bandwidth.
* **Compositing, features, segmentation and parcels run on one core** (busy cores ≈ 1.0); they scale linearly
  with area except **OBIA segmentation, which grew 3.8x from 500 to 1000 km²** (63 → 238 s, ~1 core, host
  quiet) after growing linearly up to 500 km² — the seam reconciliation over a single global segment list is the
  likely cause; for regional runs segment per tile/municipality (listed as an open issue).
* **Prediction is dominated by the area-of-applicability map**: a single-core nearest-neighbour search of every
  pixel among 20 000 training points in 97 dimensions (98 s of 101 s for 1 Mpx when timed separately; 99–148 s
  per Mpx across runs). The Random-Forest probabilities themselves take a few seconds per Mpx on all cores.
* **Training** (pseudo-labels: 50 pixels per segment, so samples grow with area — 1.16 M samples at 1000 km²)
  is the only stage using many cores (~10 busy) and the memory peak (7.4 GiB at 1000 km²). With real reference
  polygons the sample count is set by the reference data, not by the area.
* **Memory is modest**: peak RSS 1.5 → 7.4 GiB from 25 to 1000 km², of which ~1 GiB is interpreter/GDAL base.
  Window-wise compositing, features and prediction keep their working sets block-sized; the RSS values per stage
  include memory retained from earlier stages of the same process.
* **Disk** grows linearly: ~11.7 MiB per km² of output for this configuration (4 windows, S2 + S1, 97 features).

## Parallel imports and blockwise processes: before and after

Re-measured on the same macOS laptop (15 cores, native, no Docker) with `tools/scale_benchmark.py`, same areas and
season as above, Sentinel-1 included. **Baseline** = the sequential code paths of earlier releases
(`--workers 1 --import-workers 1 --aoa-search kdtree`); **new** = the defaults (4 import threads, 4 block/tile
processes, matrix-product AOA search). Each variant was run once per area:

```bash
PYTHONPATH=src python tools/scale_benchmark.py run --out outputs/scale-base --sizes 100 500 --workers 1 --import-workers 1 --aoa-search kdtree
PYTHONPATH=src python tools/scale_benchmark.py run --out outputs/scale-new  --sizes 100 500
```

**The host was shared and heavily loaded** (1-min load 5-100 on 15 cores, differing between the two variants, see
the load columns), so wall times of the unchanged single-core stages vary by up to ~1.3x between runs; speed-ups
below about 1.3x are noise. Peak RSS is the sampled sum over the process tree; in this tool all stages of an area
run in one process, so it includes memory retained from earlier stages.

**100 km2**

| Stage | Baseline wall s | New wall s | Speed-up | Baseline load1 | New load1 | Baseline peak RSS MB | New peak RSS MB |
|---|---|---|---|---|---|---|---|
| s2_import | 235.49 | 74.38 | 3.17x | 12.8 | 11.7 | 367.6 | 563.6 |
| s1_import | 78.76 | 44.15 | 1.78x | 16.1 | 6.8 | 373.1 | 496.5 |
| s2_composites | 11.71 | 10.91 | 1.07x | 16.3 | 5.8 | 705.0 | 753.9 |
| s1_composites | 3.21 | 2.97 | 1.08x | 16.7 | 5.5 | 724.1 | 715.2 |
| features | 8.8 | 8.31 | 1.06x | 16.7 | 5.3 | 1184.7 | 1155.8 |
| segmentation | 15.82 | 15.76 | 1.00x | 16.4 | 5.2 | 1572.5 | 1542.0 |
| pseudo_labels | 2.01 | 1.89 | 1.06x | 15.9 | 5.6 | 1207.3 | 1167.5 |
| training | 30.88 | 29.22 | 1.06x | 23.0 | 8.2 | 1497.5 | 1839.0 |
| prediction | 117.39 | 16.52 | 7.11x | 23.5 | 9.5 | 2372.8 | 5743.0 |
| parcels | 1.43 | 1.31 | 1.09x | 19.4 | 9.2 | 2142.4 | 1668.9 |
| total | 505 | 205 | 2.46x | | | | |

**500 km2**

| Stage | Baseline wall s | New wall s | Speed-up | Baseline load1 | New load1 | Baseline peak RSS MB | New peak RSS MB |
|---|---|---|---|---|---|---|---|
| s2_import | 548.16 | 154.61 | 3.55x | 36.8 | 5.5 | 747.2 | 1532.4 |
| s1_import | 408.52 | 137.0 | 2.98x | 15.1 | 9.6 | 765.6 | 1170.3 |
| s2_composites | 67.42 | 55.57 | 1.21x | 29.4 | 11.9 | 1795.1 | 2132.4 |
| s1_composites | 17.2 | 13.43 | 1.28x | 25.9 | 11.2 | 1773.0 | 1636.3 |
| features | 47.07 | 42.34 | 1.11x | 22.1 | 9.5 | 3067.5 | 3087.6 |
| segmentation | 83.44 | 41.92 | 1.99x | 19.1 | 45.3 | 3558.1 | 6877.5 |
| pseudo_labels | 9.83 | 10.94 | 0.90x | 21.8 | 60.1 | 3676.0 | 3296.8 |
| training | 210.38 | 226.1 | 0.93x | 80.4 | 100.0 | 6456.3 | 5867.8 |
| prediction | 998.17 | 67.86 | 14.71x | 28.1 | 56.2 | 6030.0 | 13111.8 |
| parcels | 13.69 | 6.11 | 2.24x | 23.0 | 32.2 | 1421.7 | 3631.5 |
| total | 2404 | 756 | 3.18x | | | | |

What changed and why (measured):

* **Imports** (thread pool over STAC items, `resources.import_workers`, default 4): 3.2-3.6x faster for Sentinel-2,
  1.8-3.0x for Sentinel-1. The gain is bounded by the machine's bandwidth: sequential Sentinel-1 reads already ran at
  ~7 MB/s, which looks like the link limit (not verified), so threads help most where the reads are latency-bound.
* **Prediction**: 7.1x at 100 km² and 14.7x at 500 km² (998 s to 68 s). Two parts: the AOA search is exact but uses a
  float32 matrix product instead of a k-d tree (35 s vs 146 s per Mpx on one core, same 97 features and 20 000
  training points, measured on 40 000 real pixels of the 100 km² stack), and blocks run in 4 processes. **Memory grows
  with workers**: each holds a model copy (287 MB for the 500 km² model) and its block buffers; the prediction stage
  peaked at 13.1 GB (6.0 GB before) at 500 km² and 5.7 GB (2.4 GB) at 100 km², including ~3-6 GB retained by the
  benchmark process itself. Set `resources.workers: 2` (or 1) in a small container; `geocrop preflight` adds the
  per-worker memory.
* **OBIA segmentation**: tiles run in processes. Profiling a synthetic 500 km² stack (4 bands, field-like blocks,
  **synthetic data**) gave 63.9 s sequential and 21.5 s with 4 workers, identical segment tables; 1000 km² took 146 s
  sequentially. The profile does **not** support the seam reconciliation as the cause of the earlier super-linear
  growth: it takes about 1 s of 146 s. Over 90 % of the time is the mask-centroid initialisation inside
  scikit-image's SLIC (a k-means over the tile's pixels with one centroid per segment: quadratic in the tile size,
  linear in the number of tiles). The step between 500 and 1000 km² follows from the count of full 1024-px tiles
  (4 to 9); total cost is linear in area for a fixed tile size. Real 1000 km² data was not re-measured.
* **Compositing, features, pseudo-labels, training, parcels** are unchanged (single core or already parallel);
  per-block parallel composites and features need edits in modules outside this change and were not attempted.

**Equality with the sequential run** (real data, 19.10-19.16 E, 44.72-44.76 N, June 2023, Earth Search and Planetary
Computer): Sentinel-2, 3 scenes imported with 1 and 4 workers: all 36 files byte-identical (SHA-256), result lists
identical, 37.5 s vs 12.4 s (3.0x). Sentinel-1, 9 overpasses: all 27 files byte-identical, 10.1 s vs 9.0 s (1.1x; small
area). Rasters from `map_aoa` with 1 and 3 processes are byte-identical; `predict` classes and AOA are identical and
probabilities differ by at most 1e-6 (synthetic data, `tests/test_parallel.py`); segment tables from 1 and 4 tile
workers are identical. The new AOA search equals the k-d tree distances to 3e-17 absolute on the real 40 000-pixel
sample, with identical float32 DI for all pixels.

## Preflight estimator vs measurement

`geocrop preflight` estimates were computed for each measured layout (same grid, scenes, windows, features,
sample count) with the coefficients before and after the recalibration in this release (full table:
`outputs/scale/scale_tables.md` from the `report` command):

| Area km² | Predict s: before / after / measured | Peak RAM GB: before / after / measured | Composite s: estimate / measured | Stack GB: estimate / measured |
|---|---|---|---|---|
| 25 | 18 / 40 / 71 | 0.55 / 1.55 / 1.61 | 7.2 / 7.4 | 0.078 / 0.084 |
| 100 | 68 / 155 / 90 | 1.58 / 2.58 / 2.59 | 25 / 14 | 0.31 / 0.30 |
| 501 | 349 / 778 / 741 | 7.31 / 8.31 / 6.36 | 133 / 59 | 1.55 / 1.49 |
| 1001 | 702 / 1560 / 990 | 15.4 / 16.4 / 7.92 | 274 / 135 | 3.11 / 2.95 |

* Disk estimates (composites, stack, prediction) were within ±15 % at every size — no change needed.
* **Prediction runtime was under-estimated up to 4x** (2.1x at 500 km²) because the AOA map was not modelled. The
  estimate now adds 150 s per Mpx per 100 features for it; it is within 0.6–1.8x of the measurements (the 25 km²
  value was measured on a host with load ~22).
* **Peak RAM was under-estimated ~3x for small areas** (no interpreter/GDAL base). ~1 GB is now added per stage.
  At large sample counts the training estimate (upper bound for the Random-Forest size) is ~2x conservative.
* Composite, feature and training runtimes are conservative by 1.0–2.7x; they are left unchanged because they
  are upper bounds on a busier machine. Scene import (network) is not estimated by preflight; use the rates below.

The Linux wrapper returns a non-zero status if any requested area fails, times out, is skipped for the time
budget, or has incomplete stage evidence. It still exports partial reports and `exit_status.txt` for diagnosis.
Each invocation uses a fresh Docker volume by default; `--volume <name>` explicitly resumes an existing run.
A successful report-generation step does not convert a failed benchmark into a successful run.

## Linux (container) measurements

The same tool on Ubuntu, as a proxy for a VM (no Ubuntu VM was available). **This is a resource-limited container
on a shared laptop, not a dedicated server**: read the numbers as "what a 2-CPU / 4 GB Linux box does", not as
the speed of Linux versus macOS.

```bash
tools/linux_benchmark.sh --sizes 100 500 --cpus 2 --memory 4g --max-hours 2 --out outputs/scale-linux
```

`tools/linux_benchmark.sh` builds an image from `ubuntu:26.04` with the README installation (Python 3.12 from uv,
`pip install -c constraints/py312.txt -e '.[all,stac,web]'`, plus psutil) and runs `tools/scale_benchmark.py`
with `docker run --cpus 2 --memory 4g --memory-swap 4g` (no swap); outputs go to a Docker volume (the VM's ext4
disk, not a bind mount of the host).

| Item | Value |
|---|---|
| Container | Ubuntu 26.04.1 LTS, aarch64, CPython 3.12.15 (uv), cgroup v2 `cpu.max 200000 100000` (2 CPUs), `memory.max` 4 GiB, no swap |
| Docker VM | Rancher Desktop (Docker Engine 29.5.3, Lima, kernel 6.18 `-virt`): 2 vCPUs, 5.8 GiB, shared with ~1.5 GiB of other running containers |
| Host | Apple M5 Pro laptop (15 cores, 48 GB), macOS; other jobs running (host 1-min load 4–57 during run 2) |
| Network | ordinary broadband; inside the container `psutil.net_io_counters` counts only the container's interface, so the "upper" bound is the container's own received bytes (the macOS per-flow lower bound is unavailable on Linux) |

**100 km² (grid 1.00 Mpx, 15 S2 / 38 S1 scenes, 97 features, 112 500 training samples), two runs:**

| Run | Code | Wall min | CPU min | Peak RSS GB | Output peak GB | Network in GB | macOS (same area) |
|---|---|---|---|---|---|---|---|
| 1 | 3a05477 | 10.2 | 8.6 | 1.64 | 1.18 | 1.08 | wall 8.0, CPU 6.7, RSS 2.41, output 1.20, net 1.34–1.93 |
| 2 | 0075e35 | 12.6 | 12.2 | 1.59 | 1.18 | 1.09 | |

| Stage | Run 1 wall s | Run 1 CPU s | Run 2 wall s | Run 2 CPU s | Peak RSS MB (run 1) | Disk written MB | macOS wall s |
|---|---|---|---|---|---|---|---|
| s2_import | 179.7 | 14.0 | 183.7 | 24.9 | 316 | 421.5 | 256.0 |
| s1_import | 51.2 | 10.3 | 43.8 | 17.4 | 308 | 236.5 | 71.5 |
| s2_composites | 10.8 | 10.6 | 17.3 | 17.1 | 608 | 112.9 | 10.7 |
| s1_composites | 2.4 | 2.4 | 4.7 | 4.6 | 553 | 24.1 | 2.9 |
| features | 6.8 | 6.7 | 13.0 | 12.6 | 666 | 278.5 | 7.8 |
| segmentation | 21.6 | 21.5 | 36.9 | 35.9 | 645 | 4.3 | 14.2 |
| pseudo_labels | 5.4 | 5.3 | 7.3 | 7.3 | 599 | 3.0 | 1.8 |
| training | 145.4 | 250.9 | 222.0 | 376.8 | 1258 | 98.4 | 25.0 |
| prediction | 189.1 | 192.1 | 226.0 | 229.7 | 1680 | 23.8 | 89.7 |
| parcels | 1.9 | 1.9 | 3.8 | 3.6 | 907 | 6.7 | 1.2 |

* Run 2 ran while the macOS host was heavily loaded (1-min load 27–39 on 15 cores at its start); its CPU
  seconds rose with its wall time (a vCPU that the host does not schedule still counts as busy inside the guest),
  so **run 1 is the better figure and run-to-run spread on a shared host is up to ~1.7x per CPU-bound stage**.
  Segmentation changed between the runs (see below) but is a 1-tile area at 100 km², where the change is minor.
* Disk and output sizes are identical on Linux and macOS (±2 %); network volume is the same order (1.1 GB measured
  on the container interface vs a 1.34 GB lower bound on macOS; the difference was not investigated).
* Peak RSS is lower than on macOS (1.6 vs 2.4 GiB), probably because fewer trees are built concurrently on 2 CPUs
  and the operating systems account RSS differently (not investigated). A 4 GB container is ample at 100 km².
* Training is the stage that depends on cores: 251 CPU s took 145 s on 2 CPUs (1.7 busy cores) vs 25 s on the
  15-core laptop. Prediction is dominated by the single-core AOA map and ran 2.1x slower than on the laptop's
  faster cores.

**500 km², earlier attempt: not completed** (completed in the release-candidate runs below). A first attempt (code 3a05477) stalled in OBIA segmentation: its global stretch made
a decimated whole-raster read of the 97-band pixel-interleaved stack, and with GDAL's block cache at 5 % of the
container memory (~205 MB) every output row re-decompressed a whole row of 25 MB tiles; after 39 min the stage
had not finished and was stopped. This is fixed in this release (block-wise regular subsample, ~10 s for that
stack, identical percentiles to three decimals; the whole OBIA stage then took 146 s in a 1-CPU repro). The
re-run (code 0075e35) completed every stage up to the pseudo-labels:

| Stage (500 km², 5.0 Mpx) | Wall s | CPU s | Peak RSS MB | Disk written MB | macOS wall s |
|---|---|---|---|---|---|
| s2_import | 433.2 | 95.4 | 673 | 2099.4 | 581.2 |
| s1_import | 418.0 | 60.8 | 580 | 1179.1 | 948.1 |
| s2_composites | 78.8 | 77.9 | 673 | 562.0 | 46.7 |
| s1_composites | 20.6 | 20.3 | 673 | 120.4 | 12.0 |
| features | 44.7 | 44.2 | 774 | 1390.5 | 36.3 |
| segmentation | 161.6 | 159.6 | 797 | 21.9 | 62.7 |
| pseudo_labels | 33.3 | 33.0 | 960 | 15.4 | 7.8 |

During its training stage the Docker daemon of the laptop's VM stopped (the client lost the connection; no
container exit status or OOM record survived), so training, prediction and parcels at 500 km² were **not
measured on Linux**. The macOS run peaked at 6.0 GiB in training at this size and the recalibrated preflight
estimates 8.3 GB, i.e. above this container's 4 GiB: preflight now reports that as an error inside the container
(see below), and a 500 km² run should get ≥ 8 GB.

### Release candidate 1.0.0rc1: two resource profiles, 25 to 500 km²

Measured on the committed release candidate (code `16f44ad`, 2026-10-28) with `tools/linux_benchmark.sh` in an
`ubuntu:26.04` container (CPython 3.12 via uv, pinned install). The Docker VM was a Rancher Desktop VM on the same
Apple M5 Pro laptop as above (15 cores, 48 GB), so these are **resource-limited containers on a shared laptop, not a
dedicated Linux server**, and the "CPU" limit is a CFS quota on VM vCPUs. Two profiles:

| Profile | `docker run` limits | Docker VM | Areas |
|---|---|---|---|
| small | `--cpus 2 --memory 5g` (no swap) | 2 vCPUs, 5.8 GiB | 25, 100 km² |
| large | `--cpus 8 --memory 14g` (no swap) | 8 vCPUs, 15.6 GiB | 100, 500 km² |

```bash
tools/linux_benchmark.sh --sizes 25 100 --cpus 2 --memory 5g --out outputs/scale-linux-2cpu-5g
tools/linux_benchmark.sh --sizes 100 500 --cpus 8 --memory 14g --max-hours 3 --out outputs/scale-linux-8cpu-14g
```

All stages of all four runs completed (`exit_status.txt` 0). Training labels are pseudo-labels (software
measurement, not accuracy).

| Profile | Area km² | Wall min | CPU min | Peak RSS GiB | Output peak GiB |
|---|---|---|---|---|---|
| small | 25 | 6.0 | 1.3 | 1.12 | 0.32 |
| small | 100 | 9.1 | 5.3 | 1.60 | 1.18 |
| large | 100 | 10.4 | 8.1 | 1.96 | 1.18 |
| large | 500 | 37.5 | 54.6 | 4.90 | 5.59 |

Per stage (wall seconds; CPU seconds in brackets):

| Stage | small 25 km² | small 100 km² | large 100 km² | large 500 km² |
|---|---|---|---|---|
| s2_import | 181 (8) | 298 (15) | 374 (17) | 563 (66) |
| s1_import | 131 (5) | 40 (10) | 45 (12) | 414 (58) |
| s2_composites | 2 (2) | 8 (8) | 13 (13) | 58 (58) |
| s1_composites | 1 (1) | 2 (2) | 3 (3) | 14 (14) |
| features | 1 (1) | 5 (5) | 6 (6) | 31 (31) |
| segmentation | 2 (2) | 18 (18) | 22 (22) | 122 (122) |
| pseudo_labels | 0 (0) | 4 (4) | 2 (2) | 25 (28) |
| training | 22 (35) | 108 (188) | 57 (302) | 403 (2186) |
| prediction | 22 (23) | 64 (67) | 100 (109) | 613 (704) |
| parcels | 0 (0) | 1 (1) | 2 (2) | 9 (9) |

Reading the two profiles (all figures from the tables above):

* **Extra cores help only the stage that uses them.** Training used 1.75 busy cores in the small profile and 5.3 to
  5.4 in the large one: at 100 km² it took 108 s instead of 57 s, and at 500 km² 403 s. Prediction (dominated by the
  single-core AOA map), OBIA segmentation and the composites ran on about 1.0 to 1.15 cores in both profiles, so
  they did not speed up (prediction at 100 km² was even slower in the large profile, 100 s vs 64 s, within the
  run-to-run spread of a shared laptop).
* **The two imports dominate small areas** (62 % of the wall time at 100 km² in the small profile, 67 % in the
  large one, 43 % at 500 km²). They are network-bound and sequential, which is why the large profile was not
  faster in total at 100 km² (10.4 vs 9.1 min).
* **500 km² now completes on Linux**: 37.5 min wall, 54.6 CPU-min, 4.9 GiB peak RSS, 5.6 GiB output peak, about
  3.7 GiB received. After the imports, the largest stages are prediction (613 s, single core) and training (403 s).
  The earlier attempt that did not finish (below) hit the segmentation stall fixed in this release and a Docker
  daemon stop; neither occurred in these runs. The measured peak (4.9 GiB) is below the preflight RAM estimate
  (8.3 GB), so the estimate is conservative.
* Wall times depend on the network and on other jobs on the laptop (another container ran in the same VM): treat
  the minutes as roughly ±30 %. Linux (9.1 and 10.4 min) and macOS (8.0 min) at 100 km² agree within that spread.

### Preflight on Linux

The estimates stored with run 2 (code with the changes below), measured / estimate:

| Quantity (100 km²) | Estimate | Measured (run 1 / run 2) | Ratio |
|---|---|---|---|
| composites, feature stack, prediction disk | 0.16 / 0.31 / 0.03 GB | 0.14 / 0.29 / 0.025 GB | 0.83–0.94 |
| composite runtime | 25 s | 13 / 22 s | 0.5–0.9 |
| features runtime | 11 s | 7 / 13 s | 0.6–1.2 |
| train runtime | 270 s (2 CPUs) | 145 / 222 s | 0.5–0.8 |
| predict runtime | 155 s | 189 / 226 s | 1.2–1.5 |
| peak RAM | 2.58 GB | 1.76 / 1.71 GB | 0.66–0.68 |

* Disk estimates hold on Linux as on macOS (within 17 %).
* **Training was under-estimated 2.1x** before this release (68 s estimated, 145 s measured): the coefficient is an
  8-core wall time. The estimate now scales by 8 / usable CPUs below 8, where usable CPUs honour the CPU affinity
  and the cgroup quota (`docker run --cpus`, systemd `CPUQuota`).
* **The RAM check compared with the VM's 5.8 GiB, not the container's 4 GiB**: `SC_PHYS_PAGES` ignores cgroup
  limits. It now uses the cgroup v2/v1 memory limit when that is smaller (Docker `--memory`, systemd `MemoryMax`,
  Kubernetes limits).
* Prediction stays within 1.5x (single-core AOA on slower cores); RAM is conservative.

## Extrapolation to the 25-municipality region

Assumptions (state them with any number you quote): **AOI ≈ 10 000 km²** (the planning figure of
[deployment](deployment.md); rasters cover the full grid rectangle, which for a 16.3–19.3 E band can be about
twice the AOI — the right-hand column), **one season, 4 windows**, the same chain and inputs as measured (S2 + S1,
97 features, no time-series features or embeddings), the per-km² rates of the 1000 km² run, linear scaling except
where noted, the same laptop-class CPU and connection, everything run sequentially in one process per stage.

| Item | per 1000 km² (measured) | 10 000 km² | 20 000 km² (grid rectangle) |
|---|---|---|---|
| S2 import (sequential) | 14 min, ≥ 2.3 GB in, 4.1 GB on disk | 2.3 h, ≥ 23 GB, 41 GB | 4.6 h, ≥ 46 GB, 82 GB |
| S1 RTC import (sequential) | 14 min, ≥ 4.4 GB in, 2.3 GB on disk | 2.3 h, ≥ 44 GB, 23 GB | 4.7 h, ≥ 89 GB, 46 GB |
| Composites + features | 3.5 min, 4.1 GB | 35 min, 41 GB | 1.2 h, 82 GB |
| OBIA segmentation | 4 min | 0.7 h if linear; up to ~6 h at the 500→1000 km² growth rate (segment per tile) | 1.3 h linear |
| Prediction incl. AOA | 16.5 min | 2.8 h (single core) | 5.5 h |
| Training | depends on reference samples, not area: 1.16 M samples → 7.3 min on 10 cores, 7.4 GiB | – | – |
| **Total wall time (excl. training)** | ~52 min | **~9 h (linear) to ~14 h** | ~17 h (linear), more with segmentation growth |
| **Peak disk** (scenes + derived) | 11.5 GiB | **~115 GiB** | ~230 GiB |

Three seasons (2023–2025) multiply imports, composites, features and prediction by three (~30–40 h, ~350 GiB at
10 000 km²); a longer season with more windows (the case-study config uses 4 windows over April–October plus
time-series features) adds scenes and features roughly in proportion. These are planning figures from four
measured points on one machine, not guarantees.

## VM sizing

Cross-checked against [deployment](deployment.md) and the recalibrated preflight:

| Use | vCPU | RAM | Disk (fast scratch) | Network | Measured basis |
|---|---|---|---|---|---|
| Pilot ≤ 1 000 km², one season | 8 | 16–32 GB | 100 GB NVMe (+ archive) | ≥ 100 Mbit/s | 1000 km² run: 7.4 GiB peak, 11.5 GiB output, 60 min on a shared 15-core laptop |
| Region ~10 000 km², one season | 16 | 64 GB | 500 GB–1 TB NVMe | ≥ 500 Mbit/s, parallel imports | extrapolation above: ~115 GiB per season, 9–14 h sequential |
| Region, three seasons kept online | 16 | 64 GB | 1–2 TB NVMe + archive | as above | ~350 GiB of scenes and products |

* The measurements confirm the deployment table (8 cores / 32 GB pilot; 12–16 cores / 64 GB regional) and show
  it is **not memory-bound**: 64 GB leaves room for regional training sets (a few million samples) and for
  running 3–4 imports or tiles in parallel. 128 GB is only needed for many parallel SNAP jobs (not used here:
  Sentinel-1 RTC comes pre-processed from Planetary Computer).
* The deployment table's 2 TB scratch for one regional season is generous for this chain (~115–230 GiB); keep it
  if SAFE downloads/SNAP processing or time-series features are used. Keep 20–25 % free space.
* The wall-time levers are I/O parallelism (imports) and the single-core AOA and segmentation stages, not more
  RAM or a GPU. More cores help training and parallel tiles; a GPU is not needed for this chain.
* Run `geocrop preflight` on the real configuration before a regional run; its disk figures matched within
  ±15 % here and its runtime/RAM figures are now conservative upper bounds.


### AOI-aware import measured on the Lijevče polje AOI

Setup: the real AOI (`configs/studies/lijevce_polje/aoi.geojson`, EPSG:32633, 10 m): grid 26.44 Mpx (2643.8 km²),
AOI 7.19 Mpx (719.4 km², 27.2 % of the grid); 58 of the 117 blocks of 512 px touch the AOI (55.6 % of the grid
pixels are inside active blocks, because the AOI is a diagonal band). `tools/aoi_benchmark.py` runs the same
`stac_import.import_season` once without and once with the mask (macOS laptop, shared host with a high load average
during the runs, Earth Search COGs; wall time and the network counters are therefore noisy).
Reproduce: `python tools/aoi_benchmark.py --out <dir> --start 2023-07-14 --end 2023-07-18 --order masked unmasked`
(and the reverse order).

Final implementation, 8 scenes (14-18 July 2023), both orders, masked / unmasked:

| Metric | masked first | unmasked first |
|---|---|---|
| wall time | 282 s / 254 s | 248 s / 279 s |
| process CPU time | 68 s / 107 s | 63 s / 109 s |
| bytes on disk of the scene folders | 1170 MB / 1905 MB | 1170 MB / 1904 MB |
| peak RSS | 2143 / 2116 MB | 2152 / 2296 MB |

Every one of the 88 rasters equals the unmasked scene bit for bit inside the active blocks (max abs difference
0.0) and is no-data elsewhere.

* **Disk**: 61 % of the unmasked size; **CPU**: 58-63 %.
* **Wall time**: no consistent change (0.89x and 1.11x depending on order); the import is network-bound and the
  host was loaded, so no wall-time saving is claimed.
* **Network**: the interface counters of the benchmark (nettop sampling, host counters) disagreed with each other
  and with the order of the runs (masked / unmasked between 0.90 and 1.34 for the lower bound), so they are not
  used. A deterministic count of the HTTP range requests of GDAL (`CPL_CURL_VERBOSE`) for one scene
  (`S2A_33TXK_20230716_0_L2A`, 41 of the 58 active blocks contain scene data) gave 54 requests / 137.5 MB masked
  against 69 requests / 154.3 MB unmasked (89 %). The saving is small here because the unmasked import already
  reads only the COG window of the bounding box, and the active blocks are 56 % of it; AOIs that are more compact
  in blocks save proportionally more.
* An earlier version of the reader (one boundless read per 512 px block, before adjacent blocks were merged into
  runs and windows inside the raster were read without a temporary VRT) fetched 236.4 MB for the same scene because
  neighbouring blocks re-fetched shared source tiles; the final numbers above are from the corrected reader.

A full month (1-31 July 2023, 39 STAC items) was run once with that earlier reader: unmasked 1946 s wall, 755 s CPU,
7.2 GB on disk; masked 2286 s wall, 838 s CPU, 4.4 GB on disk (28 scenes imported against 29: a scene near the
clear-fraction threshold is admitted by the AOI-pixel criterion differently, see the limitations in
[aoi-aware](aoi-aware.md)). It was not repeated with the final reader.
