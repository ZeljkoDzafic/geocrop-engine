# Preflight, disk guard and cleanup

A regional run (for example 25 municipalities, one season) takes days. `geocrop preflight` answers two
questions before you start it: **can the run succeed with these inputs and settings**, and **roughly what
will it cost** in disk, memory and time. It changes no data.

```bash
geocrop preflight --config configs/study.yaml --out preflight.json --markdown preflight.md \
    [--inventory manifests/s2.csv manifests/s1.csv] [--strict]
```

The Markdown report is printed to the console as well. Exit code `2` means **blocked** (with `--strict`,
also *ready with warnings*), so the command can gate a systemd unit or a CI job.

## Verdict

| Verdict | Meaning | What to do |
|---|---|---|
| `ready` | every check passed | start the run |
| `ready-with-warnings` | no blocking problem, but something limits quality or margin | read every warning; decide deliberately |
| `blocked` | at least one check would make a stage fail or the result meaningless | fix the errors first |

Every check has `code`, `section`, `status` (`pass` / `warn` / `error`), a message and, where useful,
`details` (counts, windows, classes).

## Checks

### Scenes per window (`scenes.*`)
Scene folders (`scene.json` sidecars under `paths.s2_scenes` / `paths.s1_scenes`) are assigned to windows
exactly as the composite stage does. Inventory CSVs from `geocrop catalog` or `geocrop stac-search`
(`--inventory`, or `preflight.inventories` in the config) add the number of *planned* scenes, with S1
counts per orbit direction. When a sensor has no imported scene folders yet, its inventory counts are
used instead (and `scenes.not_imported` warns).

| Code | Status | Condition |
|---|---|---|
| `scenes.window_empty` | error | a window has no S2 and no S1 scene (the composite stage would stop) |
| `scenes.s2_every_window` | error | S2 is present in some windows but not all (the feature stage requires all or none) |
| `scenes.s1_layout` | error | S1 missing in a window, or different orbit directions per window with `s1.separate_orbits` |
| `scenes.files` | error | scene folders without their required rasters (`B02…B12`, `SCL`; `VV`, `VH`) |
| `scenes.sidecar` / `inventory.readable` | error | unreadable `scene.json` or inventory CSV |
| `scenes.window_minimum` | warn | fewer scenes than `preflight.min_s2_scenes_per_window` / `min_s1_scenes_per_window` |

### Clear-observation coverage (`coverage.clear_observations`)
For each window the S2 masks (SCL, plus CLDPRB when `compositing.cloud_probability_max` is set, plus
`CLOUDMASK.tif`/`SHADOW.tif` when present) are read **decimated** onto a coarse version of the reference
grid (at most `preflight.coverage_max_px` pixels on the long side) and same-day tiles are merged, as in
compositing. Reported: AOI fraction with zero clear observations, AOI fraction below
`compositing.min_valid_observations` (these pixels become no-data in the composite) and the median number
of clear observations. The AOI is the `1` area of a `geocrop grid` raster; any other reference raster is
treated as all-AOI.

* error: no AOI pixel reaches the minimum (the window composite would be empty);
* warn: the fraction below the minimum exceeds `preflight.max_unobserved_fraction`.

Mask dilation and temporal screening are not simulated, so real coverage can be slightly lower.

### Grid and CRS compatibility (`grid.*`, `<layer>.*`)
* `grid.metric_crs` (error): the grid must be a projected CRS in metres.
* `grid.declared_crs` (error): `grid.crs` must match the reference grid; `grid.resolution` (warn) compares pixel size.
* `grid.available` / `grid.readable`: no grid could be determined (from `grid.reference` or the first S2 scene).
* `training.present` (error) / `parcels.present`, `observations.present` (warn: dependent stages are skipped).
* `<layer>.overlap`: features are reprojected to the grid CRS and intersected with its footprint. No training
  polygon inside → error (typically a wrong CRS definition); partial overlap → warn.
* `audit.*`: the full [input audit](classification-validation.md) (schema, geometry validity, unique IDs,
  conflicting overlaps, training/field leakage within `validation.training_exclusion_m`) runs on the configured
  layers; its errors and warnings are included unchanged.

### Training class support (`classes.*`)
Computed on polygon **centroids** with the training code itself (`model.spatial_blocks`,
`model.split_final_holdout`, `model.selection_samples`, the fold planner) at `model.spatial_block_m`,
`model.cv_folds`, `model.holdout_fraction`, `model.calibration_mode` and `model.spatial_buffer_m`. No raster is read.

| Code | Status | Condition |
|---|---|---|
| `classes.spatial_groups` | error | fewer than two spatial blocks (CV impossible) |
| `classes.holdout` / `classes.folds` | error | the locked holdout, calibration split or (buffered) folds cannot be built |
| `classes.development` | error | the locked holdout takes every polygon of a class (it cannot be trained) |
| `classes.blocks` | warn | class in a single block (never both trained and tested) |
| `classes.blocks_per_fold` | warn | class in fewer blocks than `cv_folds` (cannot appear in every test fold) |
| `classes.fold_coverage` / `classes.fold_training` | warn | class absent from some test folds / training folds |
| `classes.holdout_coverage` | warn | class absent from the locked holdout (no holdout accuracy for it) |

Training samples pixel centres rather than centroids, so a buffered fold plan can differ slightly.

### Resources (`resources.*`) — rough planning figures
| Figure | How it is estimated |
|---|---|
| grid pixels, area | width × height of the reference grid (rasters cover the full extent, not only the AOI) |
| feature count | window features with the naming rules of `features.plan` (bands, indices, S1 features per orbit prefix, deltas, seasonal) + time-series features (`ts`, phenology, harmonics) |
| disk | composites: 42 B per pixel and S2 window (10 float32 bands + uint16 count) + 4 B per S1 layer; features: 4 B per pixel and feature (+ parts kept next to a merged stack, + QA layers); prediction: (classes + 5) float32 + uint8; samples/model: samples × features × 4 B × 3 + RF size |
| expected size | uncompressed × `resources.compression_ratio` (default 0.8; measured 0.77–0.82 on noisy synthetic data with deflate + floating-point predictor; real data with no-data outside the AOI usually compresses better) |
| peak RAM | largest per-stage working set from block sizes (512 px; 256 px for time series) and scene counts, plus model size; Python/GDAL overhead and parallel tree building are not included |
| RF model size | upper bound: trees × 2 × 0.63 × samples / `min_samples_leaf` nodes × (64 + 8 × classes) bytes |
| runtime | coefficients in `geocrop.preflight.RUNTIME` (seconds per megapixel and scene / feature / tree), measured on the synthetic demo on an 8-core laptop and rounded up |

`resources.disk` compares, **per volume**, the free space with the cumulative need of all stages writing to
it: error when `expected × safety_factor + min_free_gb` exceeds free space, warn when only the compressed
estimate fits. `resources.ram` compares the rough peak with installed memory (warn above 70 %, error above 100 %).

Treat runtimes as an order of magnitude. After a pilot, compare with `<output>/stages.json` (`seconds`,
`peak_rss_mb` per stage) and `du -sh`, then scale by area and scene count.

## Disk guard during runs
Before `composite`, `features` (including time series) and `predict` actually run (not when skipped), the
pipeline checks the free space on the stage's output volume:

    need = estimate × compression_ratio × safety_factor − size of this stage's existing outputs + min_free_gb

If free space is lower, the stage fails **before writing anything** with `DiskSpaceError` and a message
naming the volume, free and needed GB; `run.log` records it. Successful checks are logged as `[disk]` lines.
Configure with `resources.disk_guard` (true), `resources.min_free_gb` (5), `resources.safety_factor` (1.25)
and `resources.compression_ratio` (0.8). Disable only deliberately (for example on a volume that reports
free space incorrectly).

Parallelism is not capped by a separate `max_workers` setting: compositing, features and prediction are
block-wise and single-process, and model training uses `model.n_jobs` (−1 = all cores). Lower `model.n_jobs`
on shared machines; run several `s2-import`/`snap` processes only as far as RAM allows.

## Partial outputs and cleanup
When a stage fails or is interrupted, every declared output written or modified during that attempt is moved
to `<output>/.quarantine/<stage>-<time>-<pid>/` (with `quarantine.json` listing original paths), so a partial
raster can never be mistaken for a final product. A rename costs no space; across volumes the partial file is
deleted instead. The stage is recorded as `failed` and reruns on the next `geocrop run`.

```bash
geocrop clean --config configs/study.yaml --dry-run            # list paths and sizes only
geocrop clean --config configs/study.yaml                       # temporary files + quarantine
geocrop clean --config configs/study.yaml --all-intermediates   # + composites, feature stacks, samples, hash cache
geocrop clean --config configs/study.yaml --include-results     # + model/prediction/parcels/cbm/validation/experiments
```

* **Always removed**: `*.tmp`, `*.part` and `.stages-*.tmp` in the output and composite directories (and next to
  the feature stack), and `<output>/.quarantine`.
* **`--all-intermediates`**: known composite products in each window folder (`B02…B12`, `valid_count`,
  `composite*.json`, `[ASC_|DESC_]VV/VH`), the feature stack and its `_ts`/`_windows` parts, manifests and QA
  files, `samples.npz` and `.hashcache.json`. The affected stages rerun automatically.
* **Never removed**: scene folders (`paths.s2_scenes`, `paths.s1_scenes`), the grid, training/parcel/observation
  layers, crop mapping, crop calendar, inventories and the config. Any candidate inside (or containing) one of
  these is skipped and listed under `skipped_protected`. Unknown files are never removed; `stages.json`,
  `run.log` and `run_manifest.json` are kept even with `--include-results`.
* `clean` takes the run lock, so it refuses to run while a `geocrop run` is active on the same output.

## Configuration keys
| Key | Default | Meaning |
|---|---|---|
| `resources.disk_guard` | true | check free space before heavy stages |
| `resources.min_free_gb` | 5 | space that must remain free on each output volume |
| `resources.safety_factor` | 1.25 | multiplier (≥ 1) on the estimated stage size |
| `resources.compression_ratio` | 0.8 | expected compressed / uncompressed size, in (0, 1] |
| `preflight.min_s2_scenes_per_window` | 3 | warn below this many S2 scenes per window |
| `preflight.min_s1_scenes_per_window` | 2 | warn below this many S1 scenes per window |
| `preflight.max_unobserved_fraction` | 0.05 | warn when more of the AOI lacks the minimum clear observations |
| `preflight.coverage_max_px` | 512 | long side of the decimated grid used for coverage (≥ 16) |
| `preflight.inventories` | [] | inventory CSVs (`catalog` / `stac-search`) counted per window |
