# Deployment and sizing

Step-by-step VM runbook: [`deploy/README.md`](../deploy/README.md) (setup script, systemd template, Docker).

The engine runs on one machine; the bottleneck is I/O and concurrent raster work, not the GPU.

| Component | Pilot (≤ 1 000 km²) | Regional (~10 000 km², one season) |
|---|---|---|
| CPU | 8 cores | 12–16 cores |
| RAM | 32 GB | 64 GB (128 GB if many parallel SNAP jobs) |
| Fast scratch (NVMe) | 500 GB–1 TB | 2 TB |
| Archive | 1–2 TB | 4 TB + independent copy of key results |
| GPU | not needed | optional 12–16 GB VRAM for U-Net training only |
| OS | Ubuntu 24.04 LTS (or macOS/Windows for development) | Ubuntu 24.04 LTS |

**Data volume model** (10 000 km² at 10 m ≈ 10⁸ pixels): one float32 layer ≈ 0.4 GB; a 125-feature stack
≈ 50 GB uncompressed (less with deflate). Typical one-season planning ranges: S2 input 150–400 GB, S1 input
100–300 GB, derived rasters 100–350 GB, scratch 100–300 GB — confirm with `geocrop catalog` totals and a pilot.
Keep 20–25 % free space on scratch.

**Layout** (example): `/srv/geocrop/{code,configs}`, `/data/geocrop/{raw,interim,processed,reference}`,
`/scratch/geocrop` (SNAP temp, unpacked SAFE), `/data/geocrop/outputs`. Run long jobs in `tmux`/`systemd`;
SNAP: set `-c` cache (e.g. 8–16 GB) and `snap.dataio.reader.tileWidth/Height`, put `java.io.tmpdir` on NVMe.

**Pilot before scaling**: one S2 tile and a few S1 scenes over a representative 500–1 000 km² (plain + fragmented
parcels); measure download size, scratch peak, RAM peak, runtime per stage (`stages.json` records seconds); then
extrapolate by area and scene count.

**Parallelism**: scenes are independent (run several `s2-import`/`snap` processes); compositing and features are
block-wise; Random Forest uses all cores (`n_jobs: -1`).
