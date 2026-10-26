# Changelog

## Unreleased
### Added
- Temporal quality (`geocrop.temporal_quality`, docs/temporal-quality.md): observed vs interpolated support per
  pixel (`ts-features --temporal-support` writes `<stack>_temporal_support.tif` + versioned manifest: clear
  observations, longest gap, season coverage, distance of SOS/peak/EOS to the nearest clear acquisition, gap at the
  peak, interpolation-dependent feature share) and per parcel (`geocrop temporal-support`, `tq_*` columns). Optional
  `cbm.temporal_support` rules route weakly supported parcels to review with reason codes `LOW_OBS`, `LONG_GAP`,
  `LONG_GAP_AT_PEAK`, `INTERPOLATED_PHENOLOGY`, `INTERPOLATED_FEATURES`, `NO_TEMPORAL_SUPPORT`
  (`cbm.traffic_light_details(..., temporal_rules=...)`, `parcels.aggregate(..., temporal_support=, temporal_rules=)`);
  disabled by default, existing traffic lights unchanged.
- Parcel evidence packages (`geocrop evidence`, `geocrop.evidence`, docs/evidence.md): per parcel a versioned
  `evidence.json` (contract `evidence`) with classification, conformal set, AOA, markers, temporal support, every
  evaluated rule with inputs/thresholds/outcome/reason code, observed vs rejected acquisitions and the interpolated
  grid, model sha256 and run-manifest reference; a self-contained strict-CSP HTML page with an inline SVG chart; a
  PNG chart; batch `index.csv`/`index.gpkg` and `evidence_manifest.json`. Byte-deterministic for identical inputs.
- Presentation website (`website/`) redesigned in the web workspace's design language (shared colour tokens, system
  fonts, light/dark appearance with toggle, phone layout), with sections reflecting the current engine and new
  real-imagery captures of the workspace (map light/dark, New analysis, command palette, terminal, phone).
- External data-cube adapter (`geocrop.datacube`, docs/datacube.md): `geocrop cube load` reads Sentinel-2 L2A from
  any STAC API through odc-stac directly onto the study grid with the same SCL screening and BOA-offset handling as
  `stac-import` (identical scene folders on a real Semberija July 2023 run); `geocrop cube import` validates
  xarray/Zarr/NetCDF or Open Data Cube cubes (band names, CRS, grid lattice, units/scale, nodata, UTC time) and
  writes scene folders or window composites; `geocrop cube export` writes scenes, composites or a feature stack as
  CF-1.8 Zarr/NetCDF. Optional `data:` config section (`source: local|odc-stac`). xarray, odc-stac, zarr and
  netCDF4 are optional.
- Local model and benchmark registry (`geocrop.registry`, docs/model-registry.md): `geocrop models
  register|list|show|compare|promote|verify` stores trained models as immutable, hash-sealed entries (content
  SHA-256, engine version and git commit, algorithm and parameters, feature manifest, input contract, classes and
  HCAT mapping, training-data summary, CV/holdout/transfer/conformal metrics, licences of pretrained components)
  with verified split fingerprints; `compare` refuses results scored on different folds or holdouts; stages
  staging/production/archived in an append-only log; `geocrop predict --model registry:<name>@<version|stage>`
  with an input-contract check against the feature stack. FTW and Presto checkpoints appear as `pretrained`
  entries from their existing URL+SHA-256 pins. `benchmark`, `benchmark-models`, `benchmark-public`,
  `benchmark-segmentation`, `ablation` and `transfer` runs are recorded automatically (`$GEOCROP_REGISTRY`,
  `off` disables); `geocrop benchmarks list|show|leaderboard` ranks only runs with identical split fingerprints
  (Markdown/CSV/JSON). `registry.overview()` gives the web layer a read-only JSON snapshot.
- Optional foundation-model embeddings as feature group `emb` (gaps G2/G3, docs/embeddings.md):
  `geocrop embeddings` reads AlphaEarth Satellite Embedding V1 (64-d, CC-BY-4.0, bottom-up COG windows) and
  TESSERA v1 (128-d, `.npy` row-range reads) onto the study grid, or computes Presto embeddings on the CPU from
  GeoCrop monthly S2/S1 composites (pixel and parcel level; official MIT checkpoint pinned by SHA-256).
  `features.embeddings` config key, `features.attach_embeddings`, and ablation rows E5–E7 (handcrafted vs
  embeddings vs both) when the stack carries `emb`.
- `geocrop active-learning select|ingest|simulate` (`geocrop.active_learning`, gap G9): batch selection among
  top-uncertain parcels with k-center diversity in standardised feature space, per-predicted-class quotas,
  minimum spacing and trip grouping; ingestion of field/analyst labels with per-round provenance, conflict
  reporting and enforced validation separation; retrospective label-budget simulation (uncertainty+diversity vs
  uncertainty vs random on one fixed spatial test set, seeded repeats, 95 % CIs, CSV + PNG). Docs in
  `docs/active-learning.md`. `fieldwork.priority_scores` exposes the existing visit priority.
- Transfer validation (gap G8): leave-one-region-out, leave-one-year-out and region-year CV
  (`cv_strategy` in `model.cross_validate` / `nested_cross_validate` / `train`, `experiments.run`;
  `geocrop train|ablation --cv-strategy`, `geocrop transfer`) with per-held-out-group metrics and the transfer
  gap to spatial-block CV; sample region/year ids (`geocrop samples --region-field/--year-field/--year`) and a
  multi-year sample store (`geocrop samples-merge`) that refuses mismatched feature manifests; config sections
  `cv` and `predict`; docs/validation-protocols.md.
- Optional edge-aware probability smoothing before argmax and parcel aggregation (gap G15,
  `geocrop predict --smooth`, `predict.smoothing`) and a synthetic accuracy/conformal-coverage comparison
  (`geocrop smoothing-experiment`).
- Optional openEO back-end (`geocrop.openeo_adapter`, gap G10): `geocrop openeo-extract` runs Sentinel-2/-1 window
  composites, scene extraction or parcel signatures on CDSE openEO and writes them in GeoCrop's composite/scene/
  signature contracts on the study grid; `geocrop openeo-graph` (and `--dry-run`) exports the recipe as openEO
  process graphs without credentials. New optional `openeo:` configuration section; see `docs/openeo.md`.
- `geocrop acceptance`: one versioned, automatically executable acceptance run of the whole chain on a ~2 × 2 km
  real area in Semberija (2023, June–September), with real Sentinel-2 and Sentinel-1 data; 21 pass/fail checks
  (grid identity, radiometry and backscatter plausibility, S1/S2 co-registration, coverage, no-data leakage,
  index and probability ranges, reproducibility, parcel schema, held-out recovery of documented pseudo-classes,
  COG/STAC/GeoParquet validity, run manifest), runtime and memory per stage, and a schema-versioned
  `acceptance_report.json` + `acceptance_report.md`. `--offline` runs the same chain on synthetic scenes (CI).
  First published report: `docs/acceptance/acceptance-0.10-dev-2026-10-26.md`.
- `geocrop.s1_stac`: Sentinel-1 RTC (terrain-flattened γ⁰, linear) import from Microsoft Planetary Computer as
  windowed COG reads onto the study grid, without SNAP; slices of an overpass are mosaicked; SAS signing via the
  optional `planetary-computer` package or the public token endpoint; physical plausibility statistics.
- Sentinel-1 to Sentinel-2 co-registration: `s1_stac.estimate_offsets` (sub-pixel NDVI / VH cross-correlation per
  acquisition geometry) and `s1_stac.apply_offsets` (corrected copies of scene folders). The acceptance run found
  systematic, orbit-dependent offsets of the Planetary Computer RTC scenes over Semberija (up to ≈ 50 m).

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
- Event and homogeneity markers for Checks by Monitoring (`geocrop.cbm_markers`, docs/markers.md):
  harvest from Sentinel-2 + Sentinel-1 (separates harvest from senescence when S1 is available), tillage
  after harvest, mowing events with dates, within-parcel homogeneity (std + Sarle's bimodality), explicit
  confidence and data-sufficiency fields; S1 series analysed per orbit group. New `cbm.s1_signatures`,
  `parcel_signatures(moments=True)`, `timeseries.drop_events`; CLI `s1-signatures`, `event-markers`,
  `traffic-light`, `signatures --moments`.
- Optional traffic-light marker rules (`cbm.rules`, off by default) with `traffic_light_reasons` reason codes
  (`cbm.traffic_light_details`, `parcels.aggregate(marker_rules=...)`); `cbm.events` holds marker parameters.
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
- `geocrop benchmark-public` (G1): frozen, versioned benchmark protocols (`configs/benchmarks/`) on public
  EuroCrops parcels with Sentinel-2 from Earth Search; parcel-level OA / macro-F1 / per-class F1 with
  block-bootstrap CIs, split-conformal coverage and set size, review burden at a target accuracy, wall time and
  peak RSS; results JSON + Markdown (`docs/benchmarks/`). EuroCrops data are downloaded at run time (md5-pinned).
- `geocrop.hcat` and `configs/hcat/` (G12): populated HCAT mapping and harmonisation table for the RS/BiH classes
  (CC-BY-SA-4.0 data, attribution in `configs/hcat/LICENSE-HCAT.md`).
- `geocrop calendar derive|worldcereal` (G11): data-driven crop calendars and suggested phenological windows from
  parcel markers; WorldCereal crop-calendar priors (PANGAEA, runtime download, SHA-256-pinned). See
  `docs/calendars.md`.
- GitHub issue templates (bug report, feature request).
- `geocrop parcel-classify` / `geocrop.parcel_classify` (gap G4): parcel-level classification on parcel
  signatures (Whittaker-smoothed mean/std series, temporal percentiles, window means, shape descriptors)
  through the same parcel-grouped spatial CV, locked holdout, calibration, conformal sets and AOA as
  `geocrop train`; output uses the `geocrop parcels` decision/traffic-light schema; `--compare-pixels` scores it
  against pixel -> parcel aggregation on identical parcels and folds. `geocrop parcel-series-demo` writes
  SYNTHETIC parcel series; `tools/parcel_vs_pixel_experiment.py` reproduces the documented synthetic comparison.
- `ltae` estimator (gap G13): Lightweight Temporal Attention Encoder (Garnot & Landrieu, 2020) in
  `geocrop.temporal_models`, seeded, early stopping, picklable; `tempcnn` gains optional early stopping and the
  scikit-learn estimator API. `estimators.sequence_params` derives `n_channels`/day-of-year positions from
  `ts_` feature names; `benchmark-models --models tempcnn ltae` (and any registered algorithm name) works.
- Redesigned web workspace: the map as the main view, a guided *New analysis* sheet (area by coordinates or drawn on
  the map, period presets, command preview) that runs `quicklook`, one-click Semberija and demo examples, an imagery
  bar (composites/scenes, product, date stepping), layer switches and colour-ramp legends, a searchable command
  palette (⌘K / Ctrl K), grouped command forms with a live command-line preview, a job history with durations and
  relative times, file icons and sizes, notifications for finished and failed jobs, keyboard shortcuts (?),
  light and dark appearance, and a phone layout with a tab bar. The Terminal view hosts the CLI console component.
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
- `docs/limitations.md` no longer states a fixed (stale) number of tests.
- Cross-validation accepts estimators whose `predict` returns an `(n, 1)` array (CatBoost).
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
