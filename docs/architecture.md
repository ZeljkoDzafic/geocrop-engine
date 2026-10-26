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
 │ interface     cli · web · web_jobs · web_map         (web imported only by cli)      │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ application   pipeline · config · provenance · cache · audit · experiments ·         │
 │               benchmark · demo                                                       │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ adapters      catalog (CDSE) · stac · stac_import · quicklook · export ·             │
 │               adapters (LPIS) · snap (SNAP gpt) · client                             │
 └──────────────────────────────────────┬───────────────────────────────────────────────┘
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ core          raster · scenes · s2 · s1 · clouds · indices · features · timeseries ·  │
 │ (local files, tsfeatures · samples · model · estimators · metrics · uncertainty ·     │
 │  no network)  applicability · spatial · splits · parcels · validation · fieldwork ·   │
 │               cbm · segmentation · segmenters · unet · segmentation_metrics ·         │
 │               plugins · contracts · hashing                                           │
 └──────────────────────────────────────────────────────────────────────────────────────┘
        ▲ plugins (entry points: geocrop.estimators / indices / segmenters / exporters / validators)
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
| `estimators` | RF, ExtraTrees, HistGradientBoosting, LightGBM, CatBoost, TempCNN and plugin classifiers behind one factory |
| `plugins` | registries for estimators, indices, segmenters, exporters and validators; lazy entry-point discovery |
| `contracts`, `hashing` | `schema_version` stamping/validation/migration of JSON contracts; SHA-256 of files |
| `uncertainty`, `applicability`, `spatial` | temperature scaling, Mondrian conformal sets, label-noise screening; Area of Applicability; variograms |
| `cbm`, `fieldwork` | parcel signatures, markers, crop-calendar consistency, traffic light; field-campaign ranking |
| `export` | COG, GeoParquet, STAC items, HCAT mapping |
| `segmentation`, `unet`, `segmentation_metrics` | OBIA, U-Net boundaries, watershed objects; boundary and object metrics |
| `segmenters` | `Segmenter` protocol and registered OBIA / U-Net + watershed segmenters |
| `adapters`, `splits` | neutral schema mapping and pseudonymisation; GIS-inspectable fold assignment |
| `audit` | read-only input audit: schema, geometry, class support, contradictory overlaps and field/training proximity |
| `pipeline`, `config`, `provenance`, `cli` | orchestration, configuration, manifests and checks, command line |

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
