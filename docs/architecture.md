# Architecture

## Principles
1. **Configuration, not code, defines a study.** AOI, season, windows, classes, features, model and decision
   rules live in YAML; the package contains no country, year or platform assumptions.
2. **Files are the interface.** Every stage reads/writes GeoTIFF, GeoPackage, CSV or JSON with documented
   contracts ([data-contracts.md](data-contracts.md)); any stage can be replaced or run elsewhere.
3. **Validity over convenience.** Leakage is prevented structurally (polygon groups, spatial blocks,
   exclusion of field points near training data); outputs say *review* when evidence is insufficient.
4. **Bounded memory.** Raster stages work block-wise (memory ∝ block size, not AOI); training samples are
   read per polygon window.
5. **Reproducibility.** Seeds everywhere, resumable stages with fingerprints, run manifests with versions and hashes.

## Layers

Every module belongs to one layer; arrows mean "may import". The map lives in `src/geocrop/_layers.py` and is
enforced by `tests/test_architecture.py` ([ADR 0001](adr/0001-layered-architecture.md)).

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ interface     cli · web · web_jobs · web_map · web_review   (web* imported only by    │
 │               cli and by each other)                                                  │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ application   pipeline · config · provenance · cache · cleanup · audit · preflight ·  │
 │               experiments · benchmark · public_benchmark · public_transfer ·          │
 │               acceptance · evidence · registry · demo                                 │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ adapters      catalog (CDSE) · stac · stac_import · s1_stac · quicklook · export ·    │
 │               adapters (LPIS) · snap (SNAP gpt) · client · downloads · public_data ·  │
 │               embedding_sources · ftw_weights · openeo_adapter · datacube             │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ core          raster · scenes · s2 · s1 · clouds · indices · features · timeseries ·  │
 │ (local files, tsfeatures · temporal_quality · samples · model · estimators ·          │
 │  no network)  temporal_models · metrics · uncertainty · applicability · spatial ·     │
 │               splits · parcels · parcel_classify · validation · fieldwork · cbm ·     │
 │               cbm_markers · calendars · hcat · segmentation · segmenters · unet ·     │
 │               segmentation_metrics · ftw · embeddings · feature_providers ·           │
 │               presto_encoder · galileo_encoder · anysat_encoder · active_learning ·   │
 │               review · attribution · cachedirs · faultinject · plugins · contracts ·  │
 │               hashing · _layers                                                       │
 └──────────────────────────────────────────────────────────────────────────────────────┘
        ▲ plugins (entry points: geocrop.estimators / indices / segmenters / exporters / validators / feature_providers)
```

Decisions: [ADR index](adr/README.md) — [layers](adr/0001-layered-architecture.md),
[plugins](adr/0002-plugin-entry-points.md), [file contracts and schema versions](adr/0003-file-contracts-and-schema-versions.md),
[single-machine execution](adr/0004-single-machine-execution.md).

## Modules

| Module | Responsibility |
|---|---|
| `catalog`, `stac` | CDSE OData inventory, token handling, resumable download; generic STAC inventory |
| `scenes` | `scene.json` metadata, product-name parsing, assignment of scenes to windows, availability tables |
| `s2` | SAFE import with metadata-driven radiometry; SCL/cloud-probability masking with dilation; same-day tile mosaicking; per-window median/percentile composites with `valid_count` |
| `s1` | SNAP graph generation (σ⁰ or terrain-flattened γ⁰), scene import, linear-domain per-window composites (optionally per orbit direction) |
| `snap` | adapter running SNAP `gpt` (`s1.run_snap` remains as an alias) |
| `raster` | reference grid, block iteration, aligned reads, GeoTIFF profiles, checked input access (`RasterInputError` naming file and problem) |
| `indices` | single registry of spectral/radar indices with formulas and references |
| `features` | multi-temporal stack (bands, indices, radar, deltas, seasonal statistics) + feature manifest; feature selection for experiments |
| `samples` | windowed extraction of reference pixels, cached `.npz` |
| `model` | spatial blocks, stratified group folds, cross-validation, grid tuning, final RF, block-wise prediction with uncertainty |
| `experiments` | ablation on identical folds with McNemar tests; recursive feature reduction; group importance |
| `metrics` | OA, κ, F1, UA/PA, bootstrap CIs, Olofsson area estimation, calibration, reject curves, McNemar |
| `parcels` | interior-buffered aggregation, decision strategies, declaration status, second-stage parcel model |
| `validation` | independent point validation with year filter, leakage exclusion and window vote; mapped area |
| `clouds` | sun-angle shadow projection, external cloud masks (OmniCloudMask), multi-temporal outlier screening |
| `timeseries`, `tsfeatures` | regular-grid binning, weighted Whittaker smoothing, phenology, harmonics; time-series feature stack, stack merging |
| `estimators` | RF, ExtraTrees, HistGradientBoosting, LightGBM, CatBoost, TempCNN, L-TAE and plugin classifiers behind one factory |
| `plugins` | registries for estimators, indices, segmenters, exporters, validators and feature providers; lazy entry-point discovery |
| `contracts`, `hashing` | `schema_version` stamping/validation/migration of JSON contracts; SHA-256 of files |
| `uncertainty`, `applicability`, `spatial` | temperature scaling, Mondrian conformal sets, label-noise screening; Area of Applicability; variograms |
| `cbm`, `fieldwork` | parcel signatures, markers, crop-calendar consistency, traffic light; field-campaign ranking |
| `export` | COG, GeoParquet, STAC items, HCAT mapping |
| `segmentation`, `unet`, `segmentation_metrics` | OBIA, U-Net boundaries, watershed objects; boundary and object metrics |
| `segmenters` | `Segmenter` protocol and registered OBIA, U-Net + watershed and Fields of The World (`ftw`) segmenters |
| `adapters`, `splits` | neutral schema mapping and pseudonymisation; GIS-inspectable fold assignment |
| `intake` | reference-data intake report (`geocrop intake`): readability, encodings, crop-mapping coverage, season, AOI overlap, class/CV support, leakage; reuses `audit` and `preflight` ([guide](priprema-podataka.md)) |
| `audit` | read-only input audit: schema, geometry, class support, contradictory overlaps and field/training proximity |
| `pipeline`, `config`, `provenance`, `cli` | orchestration, configuration, manifests and checks, command line |
| `s1_stac`, `stac_import`, `quicklook` | Sentinel-1 RTC (Planetary Computer) and Sentinel-2 L2A (Earth Search) windowed COG import; quicklooks |
| `downloads`, `cachedirs`, `public_data` | run-time downloads with retry, Range resume and checksums; cache root `$GEOCROP_CACHE`; public reference datasets |
| `attribution` | Copernicus attribution tags carried from scene import to exports |
| `temporal_quality` | observed-vs-interpolated temporal support rasters and review routing ([temporal quality](temporal-quality.md)) |
| `shift` | prediction-time domain-shift check (`shift_report.json`), coverage statements, `DOMAIN_SHIFT` routing, conservative/weighted conformal thresholds ([domain shift](domain-shift.md)) |
| `temporal_models` | TempCNN and L-TAE sequence classifiers (PyTorch, optional) |
| `cbm_markers`, `calendars`, `hcat` | event/homogeneity markers and marker rules; data-driven crop calendars; EuroCrops HCAT harmonisation |
| `parcel_classify` | parcel-level classifier on parcel signatures ([parcel classification](parcel-classification.md)) |
| `ftw`, `ftw_weights` | Fields of The World boundary models and their pinned checkpoint download ([field boundaries](field-boundaries.md)) |
| `embeddings`, `feature_providers`, `embedding_sources`, `presto_encoder`, `galileo_encoder`, `anysat_encoder` | foundation-model feature providers and encoders ([embeddings](embeddings.md)) |
| `active_learning` | batch selection, label ingestion and budget simulation ([active learning](active-learning.md)) |
| `review`, `evidence` | analyst parcel review queue and decisions; per-parcel evidence packages ([evidence](evidence.md)) |
| `openeo_adapter`, `datacube` | optional openEO back-end ([openEO](openeo.md)); external xarray/Zarr/NetCDF cubes and odc-stac ([datacube](datacube.md)) |
| `preflight`, `cleanup`, `cache` | pre-run checks and resource estimates; removal of temporary/partial outputs; content hash cache |
| `benchmark`, `public_benchmark`, `public_transfer`, `registry` | classifier/segmentation benchmarks; frozen public protocols; local model and benchmark registry |
| `acceptance`, `demo`, `faultinject` | real-data and offline acceptance run; synthetic demo; test-only fault-injection hook |
| `web`, `web_jobs`, `web_map`, `web_review` | optional local HTML workspace, job queue, map layers and review API ([web interface](web-interface.md)) |

## Stage graph

```
inventory ─► download ─► s2-import ─┐
                         snap ─► s1-import ─┤
                                            ▼
grid ─────────────────────────────► composite (per window) ─► features ─► samples ─► train ─► predict
                         scenes ─► ts-features (Whittaker, phenology) ─┘          (calibration, conformal, AOA)
                         scenes ─► signatures ─► markers ─────────────────────────────────► parcels (traffic light)
                                                                                 │                │
                                                                           experiments      parcels ─► parcel-model
                                                                                                  │
                                                                                              validate ─► area
segment-obia │ segment-unet-train ─► segment-unet-infer ─► boundaries-to-parcels ─► evaluate-segments
```

### `geocrop run` (resumable pipeline)

```
composite ─► [embeddings] ─► features ─► samples ─► train ─► predict ──────────────┐
                                                                                   ▼
             [cbm: signatures, markers, (S1 signatures, event markers)] ─► [parcel_classify] ─► parcels ─► validate
                                                          samples + features ─► experiments
```

`pipeline.DEPENDS` lists the upstream stages whose fingerprints enter a stage's fingerprint;
`pipeline.OPTIONAL_DEPENDS` marks the edges that exist only while an option is on (`features ← embeddings`
with `features.embeddings.attach`, `parcels ← parcel_classify` with `parcel_classify.decision: replace`).
Options inside a stage (`cv.strategy`, `predict.smoothing`, `cbm.event_markers`, `cbm.rules`, `model.params`)
are added to that stage's fingerprint only when they differ from their default, and keys added to the
configuration later are omitted from `config.fingerprint` while at their default
(`config.FINGERPRINT_NEUTRAL`). Enabling an option therefore re-runs exactly the stages that depend on it.

## Extending
Extensions never require changes to the core ([ADR 0002](adr/0002-plugin-entry-points.md);
complete example: [`examples/plugin_example`](../examples/plugin_example)):

* **Another classifier:** register a factory `(params, seed) -> estimator` (with `fit`, `predict_proba`,
  `classes_`) under the entry-point group `geocrop.estimators`; select it with `model: {algorithm: <name>}`.
  Spatial CV, calibration, conformal sets, AOA, prediction and benchmarks work unchanged.
* **Another index:** publish a `geocrop.indices.Index` under `geocrop.indices` (or add one entry to
  `indices.INDICES` in a fork); it becomes available in configuration, features, CBM and the index table.
* **Another segmentation algorithm:** implement `geocrop.segmenters.Segmenter` (`name`,
  `segment(raster, output, **params) -> dict`) under `geocrop.segmenters`; compare it with
  `benchmark.segmentation(..., methods={"mine": {"segmenter": "<name>", ...}})`.
* **Another sensor, data cube or data provider:** write an adapter that produces scene folders
  (`scene.json` + bands on the reference grid) or window folders following the composite contract
  (`<window>/<band>.tif`); `features` does not care where they came from.
* **Exporters and validators:** callables registered under `geocrop.exporters` / `geocrop.validators`.

`python -m geocrop.plugins` lists every registered extension and where it came from.
