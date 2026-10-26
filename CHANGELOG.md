# Changelog

## Unreleased
### Added
- `geocrop preflight`: pass/warn/error checks and a ready / ready-with-warnings / blocked verdict (JSON and
  Markdown) before long runs: scenes per window and S1 orbit (scene folders and inventory CSVs), decimated
  clear-observation coverage, class support under the configured spatial blocks, folds and locked holdout,
  grid/layer CRS and AOI overlap, the input audit, and rough disk/RAM/runtime estimates against free space.
- Disk guard before `composite`, `features` and `predict` (`resources.disk_guard`, `min_free_gb`,
  `safety_factor`, `compression_ratio`); stops with a clear message before writing.
- Partial outputs of failed or interrupted stages are moved to `<output>/.quarantine`; `geocrop clean`
  (`--dry-run`, `--all-intermediates`, `--include-results`) removes temporary files, quarantine and
  regenerable intermediates while never touching scene folders, reference layers or configs.
- Versioned REST API: every endpoint is also served under `/api/v1/...`, the documented stable contract;
  unversioned `/api/...` paths remain as compatibility aliases. `EngineClient` uses `/api/v1` and falls back to
  `/api` on older servers.
- Security gate (`tests/test_security.py`), threat model ([docs/security.md](docs/security.md)) and `SECURITY.md`.
- CI: branch coverage with `coverage.xml` artifact and a fail-under threshold, `pip-audit` of the pinned and the
  latest dependency sets, test run on the pinned set, Docker image build; Dependabot for pip and GitHub Actions.
- Tag-driven `release` workflow (version consistency check, tests, sdist + wheel, GitHub Release with CHANGELOG
  notes and checksums); `tools/check_versions.py`; pinned `constraints/py312.txt`;
  [release and rollback policy](docs/release.md).
- `geocrop ftw-boundaries` and the `ftw` segmenter: pretrained Fields of The World field-boundary models on
  two-window GeoCrop composites (CC-BY checkpoints by default, non-commercial ones only with
  `--allow-noncommercial`), SHA-256-pinned runtime download (`geocrop.ftw_weights`), seamless blockwise CPU
  inference and watershed parcels; `benchmark-segmentation --methods` adds registry segmenters from the CLI;
  `boundaries_to_parcels(min_seed_distance=...)`. See `docs/field-boundaries.md`.
- Layered architecture (core / adapters / application / interface) declared in `geocrop._layers` and
  enforced by `tests/test_architecture.py` (AST import rules, no network libraries in core); ADRs in
  `docs/adr/`.
- Plugin registries (`geocrop.plugins`) with lazy entry-point discovery for classifiers (`geocrop.estimators`),
  indices (`geocrop.indices`), segmenters (`geocrop.segmenters`), exporters and validators; `model.algorithm`
  and index names accept plugins; broken plugins only warn. Example package in `examples/plugin_example`.
- `geocrop.segmenters.Segmenter` protocol with registered OBIA and U-Net + watershed segmenters;
  `benchmark.segmentation(..., methods=...)` benchmarks any registered segmenter.
- `schema_version` in `scene.json`, the `model.joblib` payload and `run_manifest.json`, validated on read
  (`geocrop.contracts`): unversioned files still load, files from a newer version fail with an upgrade hint.
- `geocrop quicklook`: real Sentinel-2 imagery for a bounding box and period (windowed COG reads, monthly
  cloud-screened composites), available from the CLI and the web workspace.
- Map: true colour, false colour and NDVI for every composite and scene, compact imagery control, Sentinel-2
  cloudless (EOX) and OpenStreetMap basemaps.
- Web workspace: the CLI console is now a macOS Terminal-style session with live syntax colouring,
  editor-style completion (commands, flags, choices, workspace paths), ghost-text hints, signature help,
  history with reverse search, streamed job output with coloured exit status, Ctrl+C cancellation and the
  built-ins `help`, `clear`, `history` and `jobs`.
- `GET /api/files?prefix=...`: authenticated workspace path completion (at most 200 entries; no absolute
  paths, `..`, hidden files, web state directory or symlinks leaving the workspace).
- Reliability gate `tests/test_reliability.py`: SIGKILL inside features/train/predict of a real `geocrop run`
  with recovery checked against an uninterrupted reference; corrupt-input matrix; reproducibility of two
  independent runs; (`slow`) memory bounded by the block size at 2048² and roughly linear runtime.
- `geocrop.faultinject`: test-only pause points, inert unless `GEOCROP_FAULT_PAUSE` is set.
- `raster.RasterInputError` with checked raster access (`open_input`, `read_block`, `check_grid`,
  `feature_names`) at the features, samples and predict boundaries. Errors name the file and the problem.

### Security
- Web API: `manifest --snap-gpt` could be passed in an abbreviated spelling (`--snap-gp=...`), letting a token
  holder run an arbitrary executable; options are now checked after parsing.
- Web API: every `/api` path requires the token before routing (including unknown paths); the Host allow-list
  accepts `[::1]:port` independently of the Starlette version; all security headers (now also
  `X-Frame-Options`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, `Permissions-Policy`,
  CSP `object-src 'none'`) are sent on rejections and errors too; `Sec-Fetch-Site: cross-site` API requests are
  refused; request bodies are limited to 1 MiB and job arguments to 128 KiB in total.
- Web API: unreadable or malformed `run` configurations return HTTP 400 without echoing file content (previously
  HTTP 500); unexpected errors return a generic HTTP 500 that keeps the security headers.
- Docker: job state, `.env` files and virtual environments are excluded from the build context.

### Changed
- SNAP `gpt` execution moved to the adapter `geocrop.snap` (`geocrop.s1.run_snap` remains); `sha256` moved
  to `geocrop.hashing` (`geocrop.provenance.sha256` remains), so core modules no longer depend on
  subprocess-running code.

### Fixed
- CatBoost predictions (a column vector) broke cross-validation, benchmarks and ablation; labels are now
  flattened through `estimators.predict_labels`.
- Downstream stages no longer consume the partial products of an upstream stage that failed or was killed
  (e.g. `--stages validate` after an interrupted predict used to evaluate a half-written class raster). They
  now stop and ask for the upstream stage to be rerun.
- Truncated or corrupt GeoTIFFs failed with GDAL's anonymous "Read failed"; the error now names the file and window.
- A composite band that is entirely nodata/NaN/inf, or a stack without one valid pixel, used to produce an
  empty classification without an error. Both are now rejected and the empty products removed. Tests confirm
  that isolated NaN/±inf input pixels end up unclassified.
- Grid mismatch errors name the offending file and say whether CRS, shape or transform differ.
- Missing or duplicate stack band descriptions are reported with the file. Previously predict reported
  "lacks model features".
- Training polygons: empty layers, all-empty geometries, blank labels or ids (a missing label used to become
  the class "nan"), duplicate ids (now listed) and polygons that miss the stack all fail with the file named.
- `scene.json`: invalid JSON, missing keys, unknown sensor or bad datetime raise errors naming the file.
- Sentinel-2 compositing rejects SCL masks with values outside 0–11 or a non-integer dtype, instead of treating
  unknown values as clear sky.
- Run lock is acquired atomically, so two runs starting together cannot both take it. Stale locks with an
  empty or garbled pid are taken over. A damaged `.hashcache.json` is rebuilt instead of aborting the run.
- Docker image build: the conda environment step tried to install the not-yet-copied checkout.
- Packaging: SPDX license expression (`license = "Apache-2.0"`), removing setuptools deprecation warnings.
- `geocrop benchmark` now removes the configured locked holdout before model selection (same split as `train`).
- `benchmark-models` fits the tuned parameters (selected on development data) before holdout scoring and timing.
- `stac-import` reuses cached scenes only for the same study grid and with all bands present.
- Map responses of a previously selected job are discarded.

## 0.9.0 — 2026-10-26
Hardening release: leakage-free validation for ML and DL, reliable resumable execution, seam-free
segmentation, reproducible benchmarks, real-data integration tests and VM deployment; plus the web
workspace, QGIS integration and audit tooling.

### Added
- Nested spatial cross-validation for tuned models and an optional locked spatial test holdout in training,
  ablation and benchmarks (scored once, McNemar on the holdout).
- U-Net validation split by spatial patch groups (random patch split refused unless explicitly allowed);
  normalisation statistics from training patches only; `segment-patches` with spatial group ids.
- OBIA with one global stretch, seam reconciliation across tiles (spectral similarity + shared border,
  union-find) and stable global segment ids.
- Content-addressed stage fingerprints (SHA-256, cached for large files, small files always re-hashed),
  upstream dependency chain, run lock and `run.log` with peak memory per stage.
- `benchmark-models` (repeated seeds, nested tuning, holdout, timing, McNemar) and `benchmark-segmentation`
  (OBIA with/without seam merging vs U-Net, object and boundary metrics), alongside the strict
  fingerprinted `benchmark` report.
- Cloud-native Sentinel-2 import from STAC COGs (`stac-import`): only the study-grid window is read,
  provider-applied BOA offsets are honoured, implausible radiometry is rejected. `s2-import --reference`
  writes SAFE products directly on the study grid (AOI crop).
- Test tiers: golden-value feature regression, bounded-memory benchmark (`slow`), integration tests on real
  Sentinel-2 data over Semberija, July 2023 (`network`, weekly workflow).
- VM deployment: `deploy/setup_ubuntu.sh`, systemd run template, Dockerfile, runbook.
- Web workspace fully in English, with a results map (bundled Leaflet): crop classes, uncertainty and
  area-of-applicability overlays, parcels outlined by traffic light, layer switcher, legend and popups;
  `/api/jobs/{id}/map` endpoint. GitHub Pages deployment is manual-only.
- Standalone QGIS Processing plugin with command/demo/job workflows, authenticated API access and GIS-layer downloads.
- Optional HTML workspace, authenticated HTTP job API, persistent job logs/cancellation and notebook client.
- Per-parcel non-exclusive decision reasons and separate evidence warnings, with summary counts.
- Separate temporal-support rasters/summaries, optional coverage/gap masking and explicit temporal-grid validation.
- Input audit, optional spatially buffered CV, and selected-class parcel confidence/margin evidence.
- Frozen-predictor calibration on held-out spatial groups with conservative sparse-class inclusion,
  explicit calibration roles and saved-model SHA-256 binding through prediction and parcel aggregation.
- Multi-estimator benchmark on identical spatial folds with sample/fold hashes, explicit failures and timings.
- Richer STAC raster/projection/classification metadata, asset/grid/legend/date validation and correct relative paths.
- Primary-source ecosystem review with exact commits and a real-data benchmark protocol.

### Fixed
- Missing requested AOA evidence now forces review; partial AOA support uses all valid probability pixels
  as its denominator. Misaligned AOA grids and malformed probability vectors are rejected.
- Atomic, versioned pipeline running/failure/success states; complete output contracts and dependency identities
  prevent reuse of partial reruns, missing sidecars and stale sensor/orbit composites.
- Unsupported OOF split-conformal guarantee removed; calibration-fit diagnostics separated from independent evidence.
- Parcel decisions use the selected class's evidence; explicit review outcomes retain yellow traffic lights.
- U-Net train/validation overlap when only a few patches were available.
- Earth Search COGs with `earthsearch:boa_offset_applied` would otherwise receive the -0.1 offset twice.

## 0.8.0 — 2026-10-21
Second development wave, informed by a review of WorldCereal, Sen4CAP, sits, eo-learn, JRC CbM, FORCE and
Fields of The World.

### Added
- Cloud/shadow screening: sun-angle shadow projection at import, external `CLOUDMASK.tif` (OmniCloudMask runner),
  multi-temporal outlier screening in compositing.
- Regular-grid time series: weighted Whittaker smoother with upper envelope, phenology metrics, harmonic
  regression; `ts-features` stack builder and `merge-stacks`; time-series features in the pipeline.
- Estimator factory: ExtraTrees, HistGradientBoosting, LightGBM, CatBoost and a TempCNN in addition to RF.
- Temperature scaling of probabilities; class-conditional conformal prediction sets with pooled fallback for
  small classes; label-noise screening; Area of Applicability raster; variogram tool for block size.
- Checks-by-Monitoring: parcel signatures, markers (peak, harvest, bare soil), crop-calendar consistency,
  traffic light; field-campaign ranking.
- Export: COG, GeoParquet, STAC items with classification extension, HCAT mapping template.
- Synthetic per-scene generator (clouds, S1) and an end-to-end scene-to-traffic-light test.

### Fixed
- uint8 masks are reprojected with fill 255 so GDAL never rewrites valid zeros.

## 0.7.0 — 2026-10-21
First public release of GeoCrop Engine (formerly the internal "Open GeoCrop Engine" prototype, package `etfarm_ml`).

### Added
- Package renamed to `geocrop`; single `geocrop` CLI; YAML configuration with strict validation and fingerprints.
- Resumable config-driven pipeline (`geocrop run`) with stage cache and run manifests.
- Reference grid builder; scene metadata (`scene.json`) and window assignment/inventory.
- Sentinel-2: metadata-driven SAFE import incl. cloud probability; SCL + probability masking with halo-safe
  dilation; same-day tile mosaicking; median/mean/percentile composites; `valid_count`.
- Sentinel-1: SNAP graphs for σ⁰ and terrain-flattened γ⁰ (corrected step order, Copernicus DEM, projected output);
  scene import; linear-domain composites, optionally per orbit direction.
- Index registry (15 optical/radar indices with formulas and references); feature stack with deltas, seasonal
  statistics and feature manifest; feature selection by group/window.
- Windowed sample extraction; class-stratified spatial-block CV; grid tuning; bootstrap CIs; calibration (ECE);
  uncertainty raster (max probability, margin, entropy).
- Ablation experiments with McNemar tests; recursive feature reduction; group importance.
- Parcel aggregation with interior buffer, three decision strategies, declaration status, second-stage parcel model.
- Independent validation with season filter, leakage exclusion, window vote, reject curve; Olofsson area estimation.
- CDSE inventory with product attributes and size forecast; token refresh; resumable download with checksums.
- OBIA tiling by object density; U-Net with BCE+Dice, normalisation contract and overlapping inference;
  object-level segmentation metrics.
- Neutral parcel adapter with pseudonymisation; GIS-inspectable fold assignment; input quality checks.
- Documentation set, CITATION.cff, CI on Python 3.10–3.12.
