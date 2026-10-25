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

## Modules

| Module | Responsibility |
|---|---|
| `catalog`, `stac` | CDSE OData inventory, token handling, resumable download; generic STAC inventory |
| `scenes` | `scene.json` metadata, product-name parsing, assignment of scenes to windows, availability tables |
| `s2` | SAFE import with metadata-driven radiometry; SCL/cloud-probability masking with dilation; same-day tile mosaicking; per-window median/percentile composites with `valid_count` |
| `s1` | SNAP graph generation (σ⁰ or terrain-flattened γ⁰), `gpt` execution, scene import, linear-domain per-window composites (optionally per orbit direction) |
| `raster` | reference grid, block iteration, aligned reads, GeoTIFF profiles |
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
| `estimators` | RF, ExtraTrees, HistGradientBoosting, LightGBM, CatBoost, TempCNN behind one factory |
| `uncertainty`, `applicability`, `spatial` | temperature scaling, Mondrian conformal sets, label-noise screening; Area of Applicability; variograms |
| `cbm`, `fieldwork` | parcel signatures, markers, crop-calendar consistency, traffic light; field-campaign ranking |
| `export` | COG, GeoParquet, STAC items, HCAT mapping |
| `segmentation`, `unet`, `segmentation_metrics` | OBIA, U-Net boundaries, watershed objects; boundary and object metrics |
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
* **Another classifier:** `model.make_rf` is the single factory; any scikit-learn estimator with
  `predict_proba` works with `cross_validate`, `train` and `predict`.
* **Another index:** add one `Index(...)` entry to `indices.INDICES`; it becomes available in configuration,
  features and the documentation table.
* **Another sensor or data cube:** produce window folders following the composite contract
  (`<window>/<band>.tif` on the reference grid); `features` does not care where they came from.
