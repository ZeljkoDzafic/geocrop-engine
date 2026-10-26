# Changelog

## Unreleased
### Added
- `geocrop quicklook`: real Sentinel-2 imagery for a bounding box and period (windowed COG reads, monthly
  cloud-screened composites), available from the CLI and the web workspace.
- Map: true colour, false colour and NDVI for every composite and scene, compact imagery control, Sentinel-2
  cloudless (EOX) and OpenStreetMap basemaps.

### Fixed
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
