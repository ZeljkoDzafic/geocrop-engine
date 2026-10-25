# Changelog

## Unreleased

### Added
- Separate temporal-support rasters/summaries, optional coverage/gap masking and explicit temporal-grid validation.
- Input audit, optional spatially buffered CV, and selected-class parcel confidence/margin evidence.
- Frozen-predictor calibration on held-out spatial groups with conservative sparse-class inclusion,
  explicit calibration roles and saved-model SHA-256 binding through prediction and parcel aggregation.
- Multi-estimator benchmark on identical spatial folds with sample/fold hashes, explicit failures and timings.
- Richer STAC raster/projection/classification metadata, asset/grid/legend/date validation and correct relative paths.
- Primary-source ecosystem review with exact commits and a real-data benchmark protocol.

### Fixed
- Atomic, versioned pipeline running/failure/success states; complete output contracts and dependency identities
  prevent reuse of partial reruns, missing sidecars and stale sensor/orbit composites.
- Unsupported OOF split-conformal guarantee removed; calibration-fit diagnostics separated from independent evidence.
- Parcel decisions use the selected class's evidence; explicit review outcomes retain yellow traffic lights.

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
