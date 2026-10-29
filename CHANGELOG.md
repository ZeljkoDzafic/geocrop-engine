# Changelog

## Unreleased

- Parallel I/O and blockwise processes (new module `geocrop.parallel`; keys `resources.import_workers`,
  `resources.workers`; flags `stac-import --workers`, `predict --workers`, `segment-obia --workers`): the Sentinel-2 and
  Sentinel-1 STAC imports read scenes with a bounded thread pool (default 4, at most 8) with exponential backoff on
  HTTP 429/503; the AOA map and the model blocks of `predict` run in a process pool (default min(usable CPUs, 4)); OBIA
  tiles are segmented in parallel. Outputs are identical to the sequential run (files, scene.json, rasters).
- AOA nearest-neighbour search: in high dimension it uses a float32 matrix product to find candidates and recomputes the
  distance exactly in float64 over every point rounding could have made the nearest (same distances as the k-d tree to
  ~1e-15, ~4x faster on one core; `GEOCROP_AOA_SEARCH=kdtree` selects the old search). `geocrop preflight` credits block workers.
- `tools/scale_benchmark.py run --workers N --import-workers N --aoa-search kdtree` reproduces the sequential baseline.

- `docs/performance.md`: Linux container measurements of the release candidate in two resource profiles (2 CPU / 5 GB
  at 25 and 100 km², 8 CPU / 14 GB at 100 and 500 km²); 500 km² now completes on Linux.

- `tools/linux_benchmark.sh`: `docker run` now attaches stdin (`-i`); without it the container ran nothing and the
  script reported success. A missing `exit_status.txt` now fails the run.
### Validation fixes
- Ablation and model-benchmark McNemar comparisons use one majority-vote prediction per reference polygon, not per pixel; reports
  retain unit counts, deterministic tie handling and raw/Holm-adjusted p-values for separate OOF/holdout families.
- Independent validation supports `--group-field` / `validation.group_field` for cluster bootstrap. Window-majority
  reject curves use the window vote share, while central-pixel probability calibration is explicitly separate.
  Probability grids/band names are checked and invalid probabilities are excluded from calibration with counts.

## 1.0.0rc1 — 2026-10-27
Release candidate for 1.0. Highlights: layered architecture with plugin registries and file contracts, reliability and
release gate, preflight and disk guard, web workspace redesign with terminal and parcel review, S1+S2 acceptance run,
foundation-model feature providers, FTW boundaries, active learning, domain-shift safeguards, model/benchmark
registry, data-cube and openEO adapters, Lijevče polje study configuration. Field-verified accuracy in the Republic of
Srpska is still open; see docs/limitations.md.
### Validation fixes
- Web jobs use a separate POSIX process group when the Python/libc build cannot provide `posix_spawn(setsid)`;
  cancellation remains group-scoped and the worker never falls back to unsafe fork handlers.
- Fresh-user checks restore host ownership of their reports, and distinguish skipped steps from a full PASS.
- Optical E2 ablations now use `sensors: [S2]` so radar-derived delta features cannot enter the optical baseline.
  The selector, default ladder and both 2023 study configurations enforce this separation; recorded feature lists
  and spatial fold signatures are regression-tested.
- Scale benchmark orchestration returns a failing status for failed, incomplete, timed-out or budget-skipped
  requested areas. The Linux wrapper exports partial evidence without hiding failures, records `exit_status.txt`
  and starts with a fresh Docker volume unless resumption is explicitly requested with `--volume`.
- Fresh-user checks preserve proxy/trust settings across the ordinary-user login; `--ca-bundle` supports a
  read-only combined CA bundle for environments with an HTTPS inspection proxy.

### Tests
- Branch-coverage tests for error and edge paths of the result-critical core modules, review/registry and the
  data adapters (mocked HTTP, no network): every 1.0 coverage-goal module is at or above 90 % branch coverage on the
  minimal CI install, whose total rose from 72.6 % to 78.6 % (CI `--cov-fail-under` 71 -> 76).

### Fixed
- `applicability.fit` raises a clear `ValueError` for a single CV fold, identical or non-finite training features
  and mismatched fold ids instead of an opaque NumPy/SciPy error; `map_aoa` names AOA features missing from the stack.
- `stac_import.cached_scene_valid` treats a `scene.json` that is valid JSON but not an object as not cached
  (re-import) instead of crashing with `AttributeError`.

### Added
- `tools/fresh_user_check.sh`: independent-user installation check in a pristine Ubuntu container (ordinary sudo
  user, README steps only: apt packages, Python 3.12 via apt or uv, pinned pip install, demo, offline acceptance,
  real Sentinel-2 quicklook, `serve` + `/api/v1/health` with a token); record in
  docs/release-gates/fresh-user-ubuntu.md. `tools/linux_benchmark.sh`: `tools/scale_benchmark.py` in an Ubuntu
  image with explicit `--cpus/--memory`; Linux (container) measurements in docs/performance.md.
- Domain-shift safeguards (`geocrop.shift`, docs/domain-shift.md). Training writes `model/domain_reference.npz`.
  Prediction writes `prediction/shift_report.json` with the verdict `in-domain | mild | severe` and its reasons:
  AOA share and dissimilarity shift, class-mix adjusted domain-classifier AUC with grouped folds, standardised
  mean differences, predicted class-prior shift, and year/region metadata. Under any detected shift, conformal
  coverage is reported as "not established" in the prediction manifest, the parcel table/summary
  (`conformal_coverage`, `domain_shift`), the evidence packages and the web review. New config section
  `uncertainty`: `shift_policy: warn | review | abstain` (yellow `DOMAIN_SHIFT`; grey and
  `model_decision_withheld` for out-of-AOA parcels under a severe shift) and
  `shift_conformal: none | conservative | weighted` (weighted = covariate-shift weighted Mondrian split
  conformal after Tibshirani et al. 2019). The default `warn` leaves decisions and traffic lights unchanged, and
  the section is fingerprint-neutral at its defaults. `geocrop benchmark-transfer` reports the shift verdicts and
  coverage per threshold mode; result in docs/benchmarks/si-transfer-pomurje.v1.1.md (same protocol hash as v1).
- Lijevče polje case study (docs/studies/lijevce-polje.md): reproducible AOI script `tools/make_study_aoi.py`
  (geoBoundaries municipalities Gradiška, Laktaši, Srbac ∩ Copernicus DEM GLO-30 below a histogram-derived
  threshold, cleaned; pinned sources and checksums; area in EPSG:3035), the generated
  `configs/studies/lijevce_polje/aoi.geojson` (719.4 km², UTM zone 33 only), the 2023 study configuration
  `configs/studies/lijevce_polje/study_2023.yaml` (EPSG:32633), the planning check `tools/study_data_check.py`
  (Sentinel-2 dates and SCL clear-observation coverage per window) and a "Lijevče polje, Jun–Aug 2023" preset in
  the web workspace's New-analysis sheet.
- `tools/check_docs.py` (run by `tests/test_check_docs.py` in the default tier): every relative link and anchor in
  the Markdown documents and `website/index.html` resolves, every `geocrop <command>`/`--option` written in the
  living docs exists, `docs/configuration.md` lists exactly the keys of `geocrop.config.DEFAULTS`,
  `docs/architecture.md` names every module of the layer map, and `docs/cli.md` matches the parser.
- Release-gate findings (docs/release-gates/release-gate-c091962.md): `geocrop samples --groups/--feature-names`
  (`samples.extract(features=...)`) extracts only the selected features, so NaN in other groups (phenology of a
  season that starts before the series) no longer removes polygons; sample files record the classes of the
  reference polygons (`reference_classes`) and `train`, `benchmark` and `benchmark-models` refuse a sample set
  that lost a whole class (`MissingClassError`, non-zero exit) unless `--allow-missing-classes`; pipeline keys
  `model.feature_groups` and `model.allow_missing_classes` (fingerprint-neutral at their defaults).
- `geocrop.downloads`: one download helper for every run-time download (FTW, Presto/Galileo/AnySat, OmniCloudMask,
  AlphaEarth index, EuroCrops, WorldCereal) with retry + exponential backoff, HTTP Range resume and size/checksum
  verification; one cache root `$GEOCROP_CACHE` with fixed subfolders (`geocrop.cachedirs`).
- Copernicus attribution (`geocrop.attribution`): "Contains modified Copernicus Sentinel data <year>" as GeoTIFF
  tag `GEOCROP_ATTRIBUTION` on S2/S1 scene imports (Earth Search, CDSE SAFE, SNAP, Planetary Computer, openEO),
  carried through composites, stacks and predictions into COG exports and STAC items (`license`, `attribution`,
  `providers`, licence link).
- OmniCloudMask weights pinned by Hugging Face revision + SHA-256 (`geocrop cloudmask` downloads them into the
  cache; `CLOUDMASK.tif` records model, weights checksums, licence and citation); Galileo, AnySat and
  OmniCloudMask pins checked by `tools/check_weights.py`.
- `benchmark-models --params '{"ltae": {"epochs": 30}}'` (per-model overrides) and `--epochs` for TempCNN/L-TAE;
  per-model fold signatures (`fold_signatures_per_model`, `identical_folds_across_models`) computed from the folds
  each run actually used; `fold_signature` in every cross-validation result.
- Extras `ftw` (torch + segmentation-models-pytorch), `openeo`, `plots` (matplotlib), `cube` (xarray, odc-stac,
  zarr, netCDF4); `stac` adds planetary-computer.
- `geocrop benchmark-transfer` (`geocrop.public_transfer`) and the frozen protocol
  `configs/benchmarks/si_transfer_pomurje.v1.yaml`: public region/year transfer benchmark (Slovenia; train in
  Pomurje 2021, test in Podravje/Posavje/Gorenjska 2021 and Pomurje 2025) with spatial-block bootstrap CIs,
  per-class drops, conformal coverage under shift, AOA shares, local baselines on identical target parcels and
  the pooled leave-one-region/year-out CV; result in docs/benchmarks/si-transfer-pomurje.v1.md. Labels:
  EuroCrops SI 2021 and MKGP KMRS 2025 (`public_data.url_layer`, checksum-pinned runtime download).
- `tools/scale_benchmark.py` and docs/performance.md: the real chain on 25–1000 km² of Semberija with measured
  wall/CPU time, RSS, disk and network per stage, scaling plots, regional extrapolation and VM sizing.
- `plugin-contracts/2`: adds the `feature_providers` registry (entry-point group `geocrop.feature_providers`) and the
  `evidence`/feature-manifest file-contract versions; existing plugin groups are unchanged (no migration needed).
- `tools/deploy_website.sh`: manual deployment of the presentation site to Cloudflare static assets, with security headers in `website/_headers`.
- Analyst parcel review in the web workspace (gap G7, docs/web-interface.md#review-analyst-parcel-decisions): a
  Review view with a filterable, sortable queue (traffic light, declared class, model decision, reason code,
  uncertainty, review state; J/K to move), a parcel card with declared vs predicted class, probabilities, conformal
  set, reason pills, markers, the parcel outlined on the map, an inline-SVG time series (S2 indices, S1 dB from
  signature tables, or parcel means from the job's imagery) and true-colour/NDVI chips per date; A/D/U record
  agree / disagree with corrected class / can't tell with an explicit training or validation purpose into an
  append-only `review/decisions.csv` (analyst, UTC time, model SHA-256), exported as an `active-learning ingest`
  label table. Core logic in `geocrop.review`, routes in `geocrop.web_review` (authenticated, `/api/v1` aliases,
  workspace-confined, no client-supplied paths). Parcels with a `geocrop evidence` package get its triggered rules
  as reason pills, a download of the package page and, without signatures, its observed/interpolated series.
- Feature-provider adapter interface for foundation-model embeddings (`geocrop.feature_providers`, plugin
  registry `geocrop.plugins.feature_providers`, entry-point group `geocrop.feature_providers`): name, licence,
  pinned weights (URL + SHA-256), input contract, `embed_pixels` / `embed_parcels`. AlphaEarth, TESSERA and
  Presto now implement it (unchanged output); new providers `galileo` (Galileo-nano, MIT, encoder port verified
  against the official code, max abs diff 1.2e-7) and `anysat` (AnySat base, MIT, s2 inference port, max abs
  diff 0.0 per tile). Features are namespaced `emb_<provider>_NN` so several providers fit in one stack
  (`features.embeddings.source` accepts a list; `namespace`, `allow_license` keys; `geocrop embeddings
  --source a b`, `--namespace`, `--allow-license`, `--list`); legacy `emb_NN` stacks stay readable and a second
  legacy source is attached under namespaced names. Parcel tables for pre-computed sources (parcel means).
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
- Release gate `tools/release_gate.py` (docs/release-gate.md): verifies a commit from a pristine clone in a
  fresh venv (all extras, pinned constraints) - ruff, default/slow/network test tiers, real and offline
  acceptance, crash recovery, Docker clean-machine proxy, combined integration study
  (`tools/gate_integration.py`), public benchmark rerun, licence/weights audit (`tools/check_weights.py`,
  docs/licences-and-weights.md) and plugin contracts - and writes `release_gate_report.{json,md}`;
  `deploy/vm_acceptance.sh` runs the same gate on a fresh Ubuntu VM. Reports under docs/release-gates/.
- Frozen plugin contract snapshot `plugin-contracts/1` (`tests/golden/plugin_contracts.json`,
  `tests/test_plugin_contracts.py`): entry-point groups, built-in names, `Registry` API, `Segmenter` protocol,
  `Index` record, exporter/validator signatures and file-contract versions. Changing any of them requires a new
  snapshot version and a CHANGELOG line.
- `geocrop run` wires in the standalone capabilities, all off by default (docs/configuration.md#optional-parts-of-geocrop-run):
  `cv.strategy` with sample region/year ids and `model/transfer_report.json`; `predict.smoothing`; an
  `embeddings` stage (`features.embeddings`, attached to the stack; network adapters imported only when
  configured); `cbm.event_markers` (Sentinel-1 signatures and event markers) and `cbm.rules` in the cbm and
  parcels stages; an optional `parcel_classify` stage (`decision: additional | replace`). Optional stages and
  options enter fingerprints only when enabled (`pipeline.OPTIONAL_DEPENDS`, `config.FINGERPRINT_NEUTRAL`), so
  existing configurations keep their configuration and stage fingerprints. Temporal support is wired in too:
  `timeseries.temporal_support` (per-pixel raster in the features stage) and `cbm.temporal_support`
  (`cbm/temporal_support.csv` in the cbm stage, routing in the parcels/parcel_classify stages).
- `model.params`: estimator-specific hyperparameters (e.g. of plugin estimators) passed through unchanged and
  validated against the selected estimator's constructor (`model.config_params`, `model.check_params`); used by
  `geocrop run`, `ablation`, `reduce-features` and `transfer`. `--algorithm`/`--algorithms` accept registered
  plugin estimators (`plugins.Registry.declared_names`).
- Feature manifests (`<stack>.features.json`) carry `schema_version` (`features.write_manifest`) and are checked on
  read (`features.read_manifest`); unversioned manifests still load (ADR 0003).

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
- CI and release workflows run on `ubuntu-26.04` (pinned instead of `ubuntu-latest`); VM scripts and docs target Ubuntu 26.04
  LTS (24.04 still supported; Python 3.12 from uv when the distribution has no `python3.12` package).
- `matplotlib` is optional everywhere: the scale-benchmark report and evidence packages skip their PNG without it.
- `constraints/py312.txt` pins every optional extra (156 packages, previous pins kept); the Docker image now uses
  `python:3.12-slim-bookworm` and installs from the constraints file (`--build-arg EXTRAS=...`), runs as an
  unprivileged user and keeps `/usr/local/bin/_entrypoint.sh`.
- `$GEOCROP_FTW_CACHE` and `$GEOCROP_DATA_CACHE` are deprecated aliases (FutureWarning) of `$GEOCROP_CACHE/ftw`
  and `$GEOCROP_CACHE/public`; with only `$GEOCROP_CACHE` set, public datasets now go to `$GEOCROP_CACHE/public`.
- Optional backends: a missing or unloadable (native-library OSError) LightGBM, CatBoost or torch is reported the
  same way everywhere (`estimators.backend_error`): skipped in benchmarks, a `model.backend` error in preflight.
- `NOTICE` lists Leaflet (BSD-2), the MIT code adapted from Presto/Galileo/AnySat, the CC-BY-SA HCAT tables and
  the EOX non-commercial basemap; docs/licences-and-weights.md updated.
- SNAP `gpt` execution moved to the adapter `geocrop.snap` (`geocrop.s1.run_snap` remains); `sha256` moved
  to `geocrop.hashing` (`geocrop.provenance.sha256` remains), so core modules no longer depend on
  subprocess-running code.

### Fixed
- OBIA segmentation: the global 2–98 % stretch is computed from a block-wise regular subsample; the former
  decimated whole-raster read of the pixel-interleaved feature stack thrashed GDAL's block cache (5 % of the
  container's memory) and had not finished after 39 min at 500 km² in a 4 GB Ubuntu container (now ~10 s).
- `geocrop preflight`: the RAM check uses the cgroup memory limit when it is below installed RAM (Docker
  `--memory`, systemd `MemoryMax`, Kubernetes); the training runtime estimate grows by 8 / usable CPUs below 8
  (CPU affinity or cgroup quota), after training ran 2.1x the 8-core estimate on 2 CPUs.
- README: the exact commands for a fresh Ubuntu 24.04/26.04 (system packages incl. `libexpat1`, without which
  rasterio fails with uv's Python, Python 3.12 via uv on 26.04, pinned pip install) and an authenticated
  `/api/v1/health` request after `geocrop serve`.
- `deploy/vm_acceptance.sh --system` installs `libexpat1` explicitly (rasterio's GDAL needs it with uv's Python);
  `deploy/setup_ubuntu.sh` gives the data/scratch directories to the run user (`GEOCROP_USER`, default the sudo
  user) instead of leaving them root-owned with mode 750.
- Documentation matches the code (docs/limitations.md, docs/testing.md and README rewritten): OBIA seam
  reconciliation was described as missing in limitations/segmentation; limitations now separate synthetic,
  real-data and public-benchmark evidence (field accuracy in the study region open; conformal coverage below
  target under shift); configuration table gained `cache.hash`, `model.holdout_fraction`, `model.inner_splits`
  and `openeo.*`; architecture/ADR module maps, ablation ladder, embedding names, marker confidence formula,
  contract list, deploy runbook (user/ownership/paths), Docker user and loopback-only serving, Ubuntu 26.04,
  Python 3.10–3.12, and stale CLI help (`segment-obia`, `benchmark-transfer --cache`, `cube load`) corrected;
  `geocrop run` embeddings limitation (single AlphaEarth/TESSERA/Presto source) documented.
- OBIA and watershed post-processing raise a clear error naming the `segmentation` extra when scikit-image is
  missing (previously a bare `ModuleNotFoundError`, e.g. in `geocrop acceptance --offline` on a core install).
- The default test tier passes on a minimal install (`.[dev,stac,web]`, no torch/pyarrow/matplotlib), as in CI;
  openEO parcel geometries are rounded to 1e-7° so process graphs are identical across platforms.
- STAC imports (Sentinel-2 Earth Search, Sentinel-1 RTC Planetary Computer) set GDAL HTTP connect/total
  timeouts and a low-speed abort, so a stalled HTTPS connection fails that item instead of hanging the import
  (observed: > 1 h hang in a 1000 km² Sentinel-1 import).
- `geocrop preflight`: the prediction runtime now includes the single-core area-of-applicability map (~150 s
  per megapixel with ~100 features; it was under-estimated about 2x at 500 km²) and peak RAM includes ~1 GB of
  interpreter/GDAL overhead (it was under-estimated about 3x at 25 km²); see docs/performance.md.
- LightGBM/CatBoost installed but with an unloadable native library (e.g. LightGBM without `libomp` on macOS)
  raised `OSError` and aborted `geocrop benchmark-models`; the model is now reported as skipped with the reason.
- `samples` silently skipped reference polygons without a pixel valid in every feature, so a whole class could
  vanish (e.g. NaN phenology features of low-amplitude classes from `ts-features`); it now emits a
  `SampleCoverageWarning` with the skipped polygons per class and the classes left without samples.
- `hcat.harmonise` returned NaN instead of None for unmapped codes under pandas 3 (the version pinned in
  `constraints/py312.txt`), failing `tests/test_public_benchmark.py` in a clean pinned install.
- `docs/limitations.md` no longer states a fixed (stale) number of tests.
- Cross-validation accepts estimators whose `predict` returns an `(n, 1)` array (CatBoost).
- CatBoost predictions (a column vector) broke cross-validation, benchmarks and ablation; labels are now
  flattened through `estimators.predict_labels`.
- Downstream stages no longer consume the partial products of an upstream stage that failed or was killed
  (e.g. `--stages validate` after an interrupted predict used to evaluate a half-written class raster). They
  now stop and ask for the upstream stage to be rerun.
- Web workspace: after a map preview had used PROJ, later jobs could crash with exit code -11 before starting
  (PROJ's fork handler closed its cached proj.db handle in the forked child; reproduced on macOS when proj.db
  had been replaced on disk). Jobs are now started with `posix_spawn`, which runs no fork handlers.
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
- `geocrop run` embeddings stage: every on-VM provider (Presto, Galileo, AnySat, plugin `MonthlyProvider`s) now gets
  the monthly composites through one shared path (`features.embeddings.monthly`, or `<dir>/monthly` built once
  from the scene folders); before, Galileo and AnySat failed because only Presto's composites were built.
- `features.embeddings.source` as a list no longer raises `TypeError` in the stage: each provider writes its own
  raster (namespaced `emb_<provider>_NN`) and the features stage attaches all of them.
- The stage honours `namespace` and `allow_license`; a provider that needs a licence acknowledgement is refused
  before anything runs, exactly as `geocrop embeddings --allow-license`. `parcels: true` applies to every provider.
- The embeddings stage fingerprint includes the provider set (with the SHA-256 of pinned weights and plugin
  source files), the resolved namespace, a local checkpoint and the monthly composites' content; with embeddings
  off, no fingerprint changes. `checkpoint` with several sources is rejected by config validation.
- Model registry: `licences` records every embedding provider found in the feature manifest (Galileo, AnySat,
  plugins) from the licence and weights it declares, and `pretrained` entries list every provider's pinned weights
  (`anysat/default`, `galileo/<file>`, plugins), not only FTW and Presto.
- `tools/check_weights.py` also reads the transfer protocols' `sources:` key (EuroCrops, KMRS 2025), so every
  pinned download is re-checked; identical pins are checked once.

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
