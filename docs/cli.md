# CLI reference

Generated from `geocrop --help`. Every command prints JSON or a path; run `geocrop <command> --help` for details.

## `geocrop serve`

Start the local HTML workspace and HTTP API (optional web extra)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--host` |  | `127.0.0.1` |  |
| `--port` |  | `8765` |  |
| `--workspace` |  | `.` | working directory for engine jobs |
| `--state-dir` |  |  | job database/log directory; default <workspace>/.geocrop-web |
| `--token-env` |  | `GEOCROP_WEB_TOKEN` | name of the variable containing the access token |

## `geocrop demo`

Run the full pipeline on synthetic data (software check, not accuracy)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--root` |  | `outputs/demo` |  |
| `--quick` |  |  |  |

## `geocrop run`

Run the config-driven pipeline (resumable)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |
| `--stages` |  |  | subset of stages, default all |
| `--force` |  |  |  |

## `geocrop config-check`

Validate a configuration file and print the resolved config

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |

## `geocrop preflight`

Check whether a configured run can succeed and estimate its cost (no data changed)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |
| `--inventory` |  | `[]` | catalog/stac-search CSVs to count planned scenes |
| `--out` |  |  | save the JSON report |
| `--markdown` |  |  | save a Markdown report |
| `--strict` |  |  | exit 2 also on warnings (default: only when blocked) |

## `geocrop clean`

Remove temporary files, quarantined partial outputs and (optionally) regenerable intermediates

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |
| `--dry-run` |  |  | list paths and sizes without removing anything |
| `--all-intermediates` |  |  | also composites, feature stacks, samples and hash cache (regenerated on the next run) |
| `--include-results` |  |  | also model/prediction/parcels/cbm/validation/experiments outputs of the run directory |

## `geocrop indices`

Print the spectral/radar index table (name, formula, reference)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--format` |  | `markdown` |  |

## `geocrop catalog`

CDSE OData inventory for an AOI and season

| Option | Required | Default | Notes |
|---|---|---|---|
| `--aoi-wkt-file` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--collection` |  | `SENTINEL-2` |  |
| `--product-type` |  |  |  |
| `--cloud-max` |  |  |  |
| `--out` | yes |  |  |

## `geocrop stac-search`

Inventory from any STAC API (default: CDSE STAC)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--endpoint` |  | `https://stac.dataspace.copernicus.eu/v1` |  |
| `--collection` | yes |  |  |
| `--bbox` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--max-items` |  | `1000` |  |
| `--cloud-max` |  |  |  |
| `--out` | yes |  |  |

## `geocrop download`

Resumable authenticated CDSE download of an inventory CSV

| Option | Required | Default | Notes |
|---|---|---|---|
| `--manifest` | yes |  |  |
| `--out` | yes |  |  |
| `--limit` |  |  |  |

## `geocrop grid`

Create the reference analysis grid from an AOI

| Option | Required | Default | Notes |
|---|---|---|---|
| `--aoi` | yes |  |  |
| `--crs` | yes |  |  |
| `--resolution` |  | `10` |  |
| `--buffer-m` |  |  |  |
| `--out` | yes |  |  |

## `geocrop s2-import`

Sentinel-2 L2A SAFE -> scene folder (reflectance, SCL, CLDPRB, scene.json)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--safe` | yes |  |  |
| `--out` | yes |  |  |
| `--reference` |  |  | write directly on the study grid (crop to AOI, saves disk) |

## `geocrop stac-import`

Windowed import of S2 L2A COGs from a STAC API onto the study grid (no full downloads)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--reference` | yes |  | study grid GeoTIFF (geocrop grid) |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  | scene root, one folder per acquisition |
| `--endpoint` |  | `https://earth-search.aws.element84.com/v1` |  |
| `--collection` |  | `sentinel-2-l2a` |  |
| `--max-cloud` |  | `80` | tile cloud cover pre-filter (%%) |
| `--min-clear` |  | `0.2` | minimum clear fraction inside the grid |
| `--max-items` |  | `500` |  |

## `geocrop quicklook`

Real Sentinel-2 imagery for an area and period: monthly cloud-free composites for the map

| Option | Required | Default | Notes |
|---|---|---|---|
| `--bbox` | yes |  | degrees, e.g. 19.10 44.72 19.16 44.76 (Semberija) |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  |  |
| `--max-cloud` |  | `60` |  |
| `--min-clear` |  | `0.3` |  |
| `--max-items` |  | `60` |  |

## `geocrop acceptance`

End-to-end acceptance run on real Sentinel-1/2 data (or --offline synthetic) with a pass/fail report

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  | output directory (acceptance_report.json/.md) |
| `--bbox` |  | `[19.12, 44.73, 19.145, 44.748]` | degrees; default ~2 x 2 km in Semberija |
| `--start` |  | `2023-06-01` |  |
| `--end` |  | `2023-09-30` | monthly windows between start and end |
| `--offline` |  |  | synthetic scenes, no network (CI) |
| `--seed` |  | `42` |  |
| `--k` |  | `4` | number of pseudo-classes (k-means) |
| `--no-s1-coregistration` |  |  | keep Sentinel-1 geolocation as delivered (no offset correction to Sentinel-2) |

## `geocrop s2-composite`

Cloud-masked temporal composite of S2 scene folders

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  |  |
| `--out` | yes |  |  |
| `--reference` |  |  |  |
| `--statistic` |  | `median` |  |
| `--cloud-prob-max` |  |  |  |
| `--dilate` |  |  |  |
| `--min-obs` |  | `1` |  |

## `geocrop snap-graph`

Write a SNAP GPT graph for S1 GRD preprocessing

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--backscatter` |  | `sigma0` |  |
| `--crs` |  | `EPSG:32634` |  |
| `--dem` |  | `Copernicus 30m Global DEM` |  |
| `--pixel-spacing` |  | `10` |  |
| `--speckle` |  |  | e.g. 'Refined Lee' or 'Lee' |

## `geocrop snap`

Run SNAP gpt with a graph on one product

| Option | Required | Default | Notes |
|---|---|---|---|
| `--gpt` | yes |  |  |
| `--graph` | yes |  |  |
| `--input` | yes |  |  |
| `--output` | yes |  |  |
| `--cache` |  | `8G` |  |

## `geocrop s1-import`

SNAP GeoTIFF -> S1 scene folder (VV.tif, VH.tif, scene.json)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--geotiff` | yes |  |  |
| `--product-name` | yes |  |  |
| `--orbit` | yes |  |  |
| `--backscatter` |  | `sigma0` |  |
| `--out` | yes |  |  |

## `geocrop s1-composite`

Linear-domain temporal composite of S1 scene folders on a reference grid

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  |  |
| `--reference` | yes |  |  |
| `--out` | yes |  |  |
| `--separate-orbits` |  |  |  |
| `--statistic` |  | `mean` |  |

## `geocrop inventory`

Scene availability per window from imported scene folders

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  |  |
| `--config` | yes |  |  |

## `geocrop features`

Build the multi-temporal feature stack

| Option | Required | Default | Notes |
|---|---|---|---|
| `--composites` | yes |  |  |
| `--out` | yes |  |  |
| `--config` |  |  | use the features: section of a config |

## `geocrop check`

Quality checks for rasters and vector reference data

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` |  | `[]` |  |
| `--training` |  |  |  |
| `--parcels` |  |  |  |
| `--observations` |  |  |  |

## `geocrop audit`

Audit study inputs, conflicting labels and training/field leakage without changing data

| Option | Required | Default | Notes |
|---|---|---|---|
| `--training` |  |  |  |
| `--parcels` |  |  |  |
| `--observations` |  |  |  |
| `--features` |  |  | feature stack GeoTIFF |
| `--label-field` |  | `class_name` |  |
| `--id-field` |  | `polygon_id` |  |
| `--parcel-id-field` |  | `parcel_id` |  |
| `--observation-label-field` |  |  |  |
| `--exclusion-m` |  |  |  |
| `--distance-crs` |  |  | projected metric CRS for field exclusion distances |
| `--out` |  |  | save the JSON report; exit 2 when errors are found |

## `geocrop folds`

Assign class-stratified spatial-block folds to reference polygons (GIS-inspectable)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--polygons` | yes |  |  |
| `--out` | yes |  |  |
| `--label-field` |  | `class_name` |  |
| `--block-m` |  | `5000` |  |
| `--folds` |  | `5` |  |

## `geocrop samples`

Extract training samples from a stack and reference polygons

| Option | Required | Default | Notes |
|---|---|---|---|
| `--features` | yes |  |  |
| `--polygons` | yes |  |  |
| `--out` | yes |  |  |
| `--label-field` |  | `class_name` |  |
| `--id-field` |  | `polygon_id` |  |
| `--max-per-polygon` |  | `250` |  |
| `--inner-buffer-m` |  |  |  |
| `--region-field` |  |  | polygon attribute stored as sample region id (leave-one-region-out) |
| `--year-field` |  |  | polygon attribute stored as sample year id (leave-one-year-out) |
| `--year` |  |  | constant year id for all samples of this single-season stack |

## `geocrop train`

Spatial CV + final Random Forest

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--block-m` |  | `5000` |  |
| `--buffer-m` |  |  | purge training polygons near held-out sampled pixels |
| `--folds` |  | `5` |  |
| `--params` |  |  | JSON, e.g. {"n_estimators": 500} |
| `--tune` |  |  | JSON grid, e.g. {"max_features": ["sqrt", 0.3]} |
| `--groups` |  |  | restrict to feature groups |
| `--manifest` |  |  | feature stack (for --groups) |
| `--algorithm` |  |  | built-in or registered plugin estimator |
| `--no-calibrate` |  |  | skip temperature scaling |
| `--holdout` |  |  | fraction of spatial blocks locked as final test |
| `--calibration-mode` |  | `oof` | holdout freezes a predictor without fitting on calibration polygons |
| `--calibration-fraction` |  | `0.2` | fraction of spatial groups held out |
| `--cv-strategy` |  | `spatial_blocks` | add leave-one-region/year-out transfer validation to the report |

## `geocrop benchmark`

Compare classifiers on identical spatial folds; internal model selection only

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--algorithms` |  | `['rf', 'extra_trees', 'hist_gb']` |  |
| `--config` |  |  | inherit model settings and spatial validation policy |
| `--params` |  |  | JSON estimator settings shared across candidates |
| `--block-m` |  |  |  |
| `--buffer-m` |  |  |  |
| `--folds` |  |  |  |
| `--seed` |  |  |  |

## `geocrop predict`

Blockwise prediction: classes, probabilities, uncertainty

| Option | Required | Default | Notes |
|---|---|---|---|
| `--features` | yes |  |  |
| `--model` | yes |  |  |
| `--out` | yes |  |  |
| `--config` |  |  | use the predict.smoothing section |
| `--smooth` |  |  | edge-aware smoothing of probabilities before argmax |
| `--smooth-window` |  |  | odd neighbourhood size in pixels (default 7) |
| `--smoothness` |  |  | smoothing strength; larger smooths more (default 10) |

## `geocrop ablation`

Feature-group ablation experiments on identical folds

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--features` | yes |  | feature stack (manifest source) |
| `--out` | yes |  |  |
| `--config` |  |  | experiments: and model: sections |
| `--targets` |  | `[]` |  |
| `--block-m` |  |  | override config model.spatial_block_m (default 5000) |
| `--folds` |  |  | override config model.cv_folds (default 5) |
| `--buffer-m` |  |  | override config model.spatial_buffer_m (default 0) |
| `--seed` |  |  | override config seed (default 42) |
| `--cv-strategy` |  |  | override config cv.strategy: add leave-one-group-out transfer columns |

## `geocrop reduce-features`

Recursive feature elimination curve under spatial CV

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--block-m` |  |  |  |
| `--buffer-m` |  |  |  |
| `--config` |  |  | inherit model and holdout calibration policy |
| `--step` |  | `0.2` |  |

## `geocrop parcels`

Parcel aggregation, decision and declaration comparison

| Option | Required | Default | Notes |
|---|---|---|---|
| `--probabilities` | yes |  |  |
| `--parcels` | yes |  |  |
| `--out` | yes |  |  |
| `--id-field` |  | `parcel_id` |  |
| `--declared-field` |  | `declared_crop` |  |
| `--strategy` |  | `mean_probability` |  |
| `--buffer-px` |  | `1` |  |
| `--min-pixels` |  | `4` |  |
| `--min-confidence` |  |  |  |
| `--min-majority-share` |  |  |  |
| `--min-margin` |  |  |  |
| `--crop-mapping` |  |  | CSV source_code,class_name |
| `--conformal` |  |  | conformal.json from training |
| `--alpha` |  | `0.1` |  |
| `--aoa` |  |  | aoa.tif from prediction |
| `--min-aoa-fraction` |  | `0.5` | minimum inside-AOA share of valid probability pixels |
| `--markers` |  |  | markers.csv from geocrop markers |
| `--calendar` |  |  | crop calendar YAML |

## `geocrop parcel-model`

Second-stage parcel classifier on aggregated descriptors

| Option | Required | Default | Notes |
|---|---|---|---|
| `--parcels` | yes |  | output of geocrop parcels with a reference label column |
| `--label-field` | yes |  |  |
| `--out` | yes |  |  |
| `--block-m` |  | `5000` |  |

## `geocrop parcel-classify`

Classify parcels directly from their signatures (parcel-grouped spatial CV, calibration, conformal sets, traffic light)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--signatures` | yes |  | signatures .csv/.parquet from geocrop signatures |
| `--parcels` | yes |  | parcel polygons with ids (and reference labels for training) |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  | output directory |
| `--label-field` |  |  | independent reference label column; required unless --model is given |
| `--model` |  |  | existing parcel model directory (skip training) |
| `--id-field` |  | `parcel_id` |  |
| `--declared-field` |  | `declared_crop` |  |
| `--indices` |  |  | signature indices to use (default: all in the table) |
| `--stats` |  | `['mean', 'std']` | signature columns per date |
| `--groups` |  | `['ts', 'stats', 'windows', 'shape']` | feature groups for tabular models (temporal models use ts only) |
| `--step-days` |  | `5` |  |
| `--output-step-days` |  | `15` |  |
| `--window-days` |  | `30` |  |
| `--lam` |  | `20.0` | Whittaker smoothing parameter |
| `--algorithm` |  | `rf` | any registered estimator, e.g. rf, hist_gb, tempcnn, ltae |
| `--params` |  |  | JSON estimator settings |
| `--block-m` |  | `5000` |  |
| `--buffer-m` |  |  |  |
| `--folds` |  | `5` |  |
| `--seed` |  | `42` |  |
| `--holdout` |  |  | fraction of spatial blocks locked as final test |
| `--calibration-mode` |  | `oof` |  |
| `--calibration-fraction` |  | `0.2` |  |
| `--bootstrap` |  | `200` |  |
| `--alpha` |  | `0.1` | conformal level 1-alpha for prediction sets |
| `--min-pixels` |  | `4` |  |
| `--min-confidence` |  |  |  |
| `--min-margin` |  |  |  |
| `--min-aoa-fraction` |  | `0.5` |  |
| `--crop-mapping` |  |  | CSV source_code,class_name |
| `--markers` |  |  | markers.csv from geocrop markers |
| `--calendar` |  |  | crop calendar YAML |
| `--compare-pixels` |  |  | pixel samples .npz (polygon = parcel id): compare with pixel->parcel aggregation on the same parcels and folds |
| `--pixel-algorithm` |  | `rf` |  |
| `--pixel-params` |  |  |  |

## `geocrop parcel-series-demo`

Write SYNTHETIC parcel signatures, parcels and pixel samples (software tests)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--n-side` |  | `12` | parcels per grid side |
| `--seed` |  | `11` |  |
| `--pixel-noise` |  | `0.08` |  |
| `--cloud-fraction` |  | `0.3` |  |

## `geocrop validate`

Independent validation with field observations

| Option | Required | Default | Notes |
|---|---|---|---|
| `--prediction` | yes |  | prediction directory |
| `--observations` | yes |  |  |
| `--out` | yes |  |  |
| `--label-field` |  | `class_name` |  |
| `--date-field` |  | `observed_at` |  |
| `--year` |  |  |  |
| `--training` |  |  |  |
| `--exclusion-m` |  |  |  |
| `--window-px` |  |  |  |
| `--label-mapping` |  |  | CSV source,class_name |

## `geocrop area`

Mapped area per class and Olofsson et al. (2014) error-adjusted estimates

| Option | Required | Default | Notes |
|---|---|---|---|
| `--classes` | yes |  | classes.tif |
| `--validation` |  |  | validation_report.json (stratified sample) |
| `--out` | yes |  |  |

## `geocrop normalize-parcels`

Map external parcel fields to the neutral schema

| Option | Required | Default | Notes |
|---|---|---|---|
| `--input` | yes |  |  |
| `--out` | yes |  |  |
| `--id-field` | yes |  |  |
| `--declared-field` |  |  |  |
| `--crop-mapping` |  |  |  |
| `--pseudonymise` |  |  |  |

## `geocrop segment-obia`

SLIC superpixel segmentation (tile-bounded)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` | yes |  |  |
| `--out` | yes |  |  |
| `--bands` |  |  |  |
| `--segments-per-km2` |  | `50` |  |
| `--compactness` |  | `0.1` |  |
| `--tile-size` |  | `1024` |  |
| `--no-merge-seams` |  |  | keep segments split at tile borders |
| `--merge-distance` |  | `0.08` | max RMS spectral difference for seam merges |

## `geocrop segment-unet-train`

Train the boundary U-Net (requires torch)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--images` | yes |  |  |
| `--masks` | yes |  |  |
| `--model` | yes |  |  |
| `--epochs` |  | `30` |  |
| `--batch-size` |  | `8` |  |
| `--groups` |  |  | .npy/.txt with one spatial group id per patch (required for a fair split) |
| `--allow-random-split` |  |  | accept an optimistic random patch split |

## `geocrop segment-patches`

Cut image/boundary patches with spatial group ids for U-Net training

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` | yes |  |  |
| `--mask` | yes |  | rasterised reference boundaries (1 = boundary) |
| `--out` | yes |  | output prefix; writes _X.npy, _Y.npy, _groups.npy |
| `--size` |  | `128` |  |
| `--block-m` |  | `2000` |  |
| `--bands` |  |  |  |

## `geocrop segment-unet-infer`

Boundary probability with overlapping tiles

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` | yes |  |  |
| `--model` | yes |  |  |
| `--out` | yes |  |  |
| `--bands` |  |  |  |
| `--overlap` |  | `32` |  |

## `geocrop boundaries-to-parcels`

Watershed post-processing of boundary probability

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` | yes |  |  |
| `--out` | yes |  |  |
| `--threshold` |  | `0.5` |  |

## `geocrop evaluate-segments`

Object-level segmentation quality vs reference parcels

| Option | Required | Default | Notes |
|---|---|---|---|
| `--segments` | yes |  |  |
| `--reference` | yes |  |  |
| `--out` | yes |  |  |
| `--iou` |  | `0.5` |  |

## `geocrop evaluate-boundaries`

Boundary raster precision/recall/F1 with tolerance

| Option | Required | Default | Notes |
|---|---|---|---|
| `--prediction` | yes |  |  |
| `--reference` | yes |  |  |
| `--out` | yes |  |  |
| `--threshold` |  | `0.5` |  |
| `--tolerance-px` |  | `2` |  |

## `geocrop ts-features`

Time-series features (Whittaker, phenology, harmonics) from scene folders

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  | scene root folder(s) |
| `--reference` | yes |  | reference grid GeoTIFF |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  |  |
| `--s2-indices` |  | `['NDVI', 'NDRE', 'NDMI']` |  |
| `--s1-indices` |  | `['VHVV_dB']` |  |
| `--step-days` |  | `5` |  |
| `--output-step-days` |  | `15` |  |
| `--lam` |  | `50.0` |  |
| `--envelope` |  |  | upper-envelope iterations |
| `--min-observed-fraction` |  |  | minimum occupied temporal-bin fraction |
| `--max-gap-days` |  |  | maximum consecutive empty-bin duration, including edges |

## `geocrop merge-stacks`

Concatenate feature stacks on one grid

| Option | Required | Default | Notes |
|---|---|---|---|
| `--stacks` | yes |  |  |
| `--out` | yes |  |  |

## `geocrop embeddings`

Foundation-model embeddings (AlphaEarth, TESSERA, on-VM Presto) on the study grid

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` |  |  | features.embeddings, grid.reference and paths sections |
| `--source` |  |  | override features.embeddings.source |
| `--year` |  |  | override features.embeddings.year |
| `--reference` |  |  | study grid GeoTIFF (default grid.reference) |
| `--out` |  |  | output directory (default <features dir>/embeddings) |
| `--monthly` |  |  | Presto: monthly composites root <YYYY-MM>/B02.tif ... VV.tif |
| `--start` |  |  | Presto: first month YYYY-MM (default <year>-01) |
| `--months` |  |  | Presto: number of months (default 12) |
| `--parcels` |  |  | Presto: also write parcel-level embeddings for this layer |
| `--id-field` |  |  | parcel id field (default parcels.id_field) |
| `--checkpoint` |  |  | Presto: local checkpoint instead of the pinned download |
| `--make-monthly` |  |  | Presto: first build calendar-month composites from paths.s2_scenes / paths.s1_scenes |

## `geocrop signatures`

Parcel-mean time series (CbM signatures) from S2 scene folders

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  |  |
| `--parcels` | yes |  |  |
| `--reference` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  | .csv or .parquet |
| `--indices` |  | `['NDVI', 'BSI']` |  |
| `--id-field` |  | `parcel_id` |  |
| `--moments` |  |  | add skewness and bimodality (homogeneity marker) |

## `geocrop markers`

Phenology, harvest and bare-soil markers from parcel signatures

| Option | Required | Default | Notes |
|---|---|---|---|
| `--signatures` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  |  |
| `--index` |  | `NDVI` |  |

## `geocrop variogram`

Spatial autocorrelation range of reference objects (for spatial_block_m)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--max-distance` |  |  |  |

## `geocrop fieldwork`

Rank parcels for the next field campaign (uncertainty + AOA, spatially spread)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--parcels` | yes |  | output of geocrop parcels |
| `--out` | yes |  |  |
| `--n` |  | `50` |  |
| `--min-spacing-m` |  | `1000` |  |
| `--per-class` |  |  |  |

## `geocrop cloudmask`

Write CLOUDMASK.tif for S2 scene folders with OmniCloudMask (optional dependency)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  |  |

## `geocrop export`

Publish a prediction directory: COGs, STAC item, optional GeoParquet parcels

| Option | Required | Default | Notes |
|---|---|---|---|
| `--prediction` | yes |  |  |
| `--out` | yes |  |  |
| `--id` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--parcels` |  |  | parcels.gpkg to convert to GeoParquet |
| `--hcat` |  |  | CSV class_name,hcat_code,hcat_name |

## `geocrop benchmark-models`

Compare classifiers on identical spatial folds (nested tuning, repeats, holdout)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--models` |  | `['rf', 'extra_trees', 'hist_gb', 'lightgbm', 'catboost']` | model names; tempcnn and ltae need --groups ts (any registered algorithm name works) |
| `--features` |  |  | feature stack (manifest) to restrict --groups |
| `--groups` |  |  |  |
| `--grids` |  |  | JSON {"rf": {"max_features": ["sqrt", 0.3]}} |
| `--block-m` |  | `5000` |  |
| `--folds` |  | `5` |  |
| `--repeats` |  | `3` |  |
| `--holdout` |  |  |  |
| `--targets` |  | `[]` |  |

## `geocrop benchmark-segmentation`

OBIA (with/without seam merging) vs U-Net against reference parcels

| Option | Required | Default | Notes |
|---|---|---|---|
| `--raster` | yes |  |  |
| `--reference` | yes |  |  |
| `--out` | yes |  |  |
| `--bands` |  |  |  |
| `--unet-model` |  |  |  |
| `--unet-bands` |  |  |  |
| `--tile-size` |  | `1024` |  |
| `--segments-per-km2` |  | `50` |  |
| `--methods` |  |  | extra registry segmenters as JSON, e.g. '{"ftw": {"segmenter": "ftw", "model": "FTW_PRUE_EFNET_B3_CCBY"}}' |

## `geocrop ftw-boundaries`

Pretrained Fields of The World field boundaries (CC-BY checkpoints) -> parcels

| Option | Required | Default | Notes |
|---|---|---|---|
| `--window-a` |  |  | early-season composite/scene folder (B02..B08 reflectance) |
| `--window-b` |  |  | late-season composite/scene folder |
| `--stack` |  |  | existing 8-band FTW stack instead of --window-a/--window-b |
| `--units` |  |  | scaling of --stack if it has no GeoCrop tag |
| `--grid` |  |  | study grid raster (default: window A B04 grid) |
| `--out` |  |  | output folder |
| `--model` |  | `FTW_PRUE_EFNET_B3_CCBY` | FTW registry name (see --list-models) |
| `--checkpoint` |  |  | local checkpoint file instead of a registry download |
| `--allow-noncommercial` |  |  | permit FULL (non-commercial training data) checkpoints; prints a licence warning |
| `--cache-dir` |  |  | checkpoint cache (default $GEOCROP_FTW_CACHE or ~/.cache/geocrop/ftw) |
| `--no-download` |  |  | fail if the checkpoint is not cached |
| `--download-only` |  |  |  |
| `--list-models` |  |  |  |
| `--patch` |  | `256` |  |
| `--overlap` |  | `32` |  |
| `--threads` |  |  |  |
| `--threshold` |  | `0.5` | not-field probability below = field interior |
| `--min-area-pixels` |  | `16` |  |
| `--seed-percentile` |  |  |  |
| `--min-seed-distance` |  | `1.0` | watershed seed distance from boundary (px) |
| `--reference` |  |  | reference parcels: also write object-level segment metrics |
| `--iou` |  | `0.5` |  |

## `geocrop benchmark-public`

Run a frozen public benchmark protocol (EuroCrops parcels + Sentinel-2 from STAC)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--protocol` | yes |  | versioned protocol YAML, e.g. configs/benchmarks/*.yaml |
| `--work` | yes |  | working directory (scenes, composites, results.json/.md) |
| `--cache` |  |  | download cache; default $GEOCROP_DATA_CACHE or ~/.cache/geocrop/public |
| `--models` |  |  | subset of the protocol's models (default all) |
| `--workers` |  | `4` | parallel scene imports |
| `--layer` |  |  | local copy of the EuroCrops layer instead of the pinned download |
| `--publish` |  |  | directory to copy results.md/.json to (named after the protocol) |

## `geocrop calendar`

Derive a crop calendar and phenological windows from parcel markers, or read WorldCereal calendar priors

| Option | Required | Default | Notes |
|---|---|---|---|
| `action` | yes |  |  |
| `--markers` |  |  | derive: markers table from geocrop markers (CSV/Parquet) |
| `--signatures` |  |  | derive: parcel signatures (CSV/Parquet); markers computed with --start/--end |
| `--start` |  |  | derive with --signatures: season start (ISO date) |
| `--end` |  |  | derive with --signatures: season end (ISO date) |
| `--labels` |  |  | derive: parcels with labels (GPKG/GeoJSON/CSV) |
| `--label-field` |  | `class_name` |  |
| `--id-field` |  | `parcel_id` |  |
| `--year` |  |  | derive: year of the suggested windows |
| `--percentiles` |  | `[10, 50, 90]` |  |
| `--min-parcels` |  | `5` |  |
| `--min-days` |  | `30` | derive: shortest suggested window |
| `--max-windows` |  | `6` |  |
| `--out` |  |  | derive: calendar YAML; worldcereal: JSON (default: print) |
| `--windows-out` |  |  | derive: suggested windows YAML snippet |
| `--lon` |  |  | worldcereal: longitude (WGS84) |
| `--lat` |  |  | worldcereal: latitude (WGS84) |
| `--cache` |  |  | worldcereal: download cache directory |

## `geocrop manifest`

Write a reproducibility manifest (versions, git, hashes)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--config` |  |  |  |
| `--inputs` |  | `[]` |  |
| `--snap-gpt` |  |  |  |

## `geocrop active-learning`

Active-learning loop: batch selection, label ingestion, label-budget simulation

| Option | Required | Default | Notes |
|---|---|---|---|
| `al_cmd` | yes |  |  |

### `geocrop active-learning select`

Select a diverse, spatially spread batch of uncertain parcels to label

| Option | Required | Default | Notes |
|---|---|---|---|
| `--parcels` | yes |  | output of geocrop parcels (entropy, p_<class>) |
| `--out` | yes |  | selected parcels (.gpkg/.geojson); also writes *_label_form.csv |
| `--budget` | yes |  | parcels to visit in this round |
| `--features` |  |  | table keyed by parcel id (CSV/Parquet/GPKG) spanning the diversity space |
| `--feature-columns` |  |  | columns of --features (or parcels) to use |
| `--id-field` |  | `parcel_id` |  |
| `--candidate-factor` |  | `3.0` | top-uncertain pool = factor x budget |
| `--uncertainty-weight` |  | `0.3` | 0 = pure diversity, 1 = pure uncertainty |
| `--min-spacing-m` |  | `1000` |  |
| `--stratify` |  | `equal` | quota per predicted class |
| `--trips` |  |  | group the batch into N field trips by area |
| `--training` |  |  | current training polygons: their parcels are excluded and act as centres |
| `--exclude` |  |  | parcels never to propose (e.g. validation sites): ids or geometries |
| `--round` |  | `1` |  |
| `--seed` |  | `42` |  |

### `geocrop active-learning ingest`

Merge field observations / analyst decisions into training with provenance

| Option | Required | Default | Notes |
|---|---|---|---|
| `--labels` | yes |  | CSV/GPKG: parcel_id, label, source, date, confidence[, purpose] |
| `--parcels` | yes |  | parcel geometries with --id-field |
| `--training` | yes |  | current training polygons (polygon_id, class_name) |
| `--out` | yes |  | updated training polygons |
| `--round` | yes |  |  |
| `--validation` |  |  | validation observations kept out of training (points/polygons or ids) |
| `--validation-buffer-m` |  |  |  |
| `--provenance` |  |  | provenance CSV (default <out dir>/al_provenance.csv) |
| `--min-confidence` |  |  |  |
| `--conflict-policy` |  | `exclude` |  |
| `--existing-policy` |  | `keep` |  |
| `--inner-buffer-m` |  |  |  |
| `--id-field` |  | `parcel_id` |  |
| `--label-field` |  | `class_name` |  |

### `geocrop active-learning simulate`

Label-budget curves: uncertainty+diversity vs uncertainty vs random

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--samples` |  |  | samples.npz (aggregated to one unit per polygon); default SYNTHETIC pool |
| `--strategies` |  | `['uncertainty_diversity', 'uncertainty', 'random']` |  |
| `--initial` |  | `12` |  |
| `--batch` |  | `12` |  |
| `--rounds` |  | `8` |  |
| `--repeats` |  | `5` |  |
| `--seed` |  | `42` |  |
| `--block-m` |  | `1000` |  |
| `--test-fraction` |  | `0.25` |  |
| `--test-buffer-m` |  |  |  |
| `--candidate-factor` |  | `3.0` |  |
| `--uncertainty-weight` |  | `0.3` |  |
| `--min-spacing-m` |  |  |  |
| `--stratify` |  | `none` |  |
| `--params` |  |  | classifier JSON, e.g. {"algorithm": "rf", "n_estimators": 200} |
| `--synthetic-side` |  | `24` | SYNTHETIC pool: parcels per side |
| `--synthetic-noise` |  | `0.1` |  |
| `--no-figure` |  |  |  |

## `geocrop s1-signatures`

Parcel-mean Sentinel-1 VV/VH/VH-VV (dB) per acquisition with orbit geometry

| Option | Required | Default | Notes |
|---|---|---|---|
| `--scenes` | yes |  | root of S1 scene folders (VV.tif, VH.tif, scene.json) |
| `--parcels` | yes |  |  |
| `--reference` | yes |  |  |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  | .csv or .parquet |
| `--id-field` |  | `parcel_id` |  |

## `geocrop event-markers`

Harvest, tillage, mowing, homogeneity and data-sufficiency markers (S2 + S1)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--signatures` |  |  | S2 signatures (geocrop signatures, ideally with --moments) |
| `--s1-signatures` |  |  | S1 signatures (geocrop s1-signatures) |
| `--start` | yes |  |  |
| `--end` | yes |  |  |
| `--out` | yes |  |  |
| `--config` |  |  | use the cbm.events section of a config |

## `geocrop traffic-light`

Re-evaluate the traffic light of a parcels layer with marker rules and reason codes

| Option | Required | Default | Notes |
|---|---|---|---|
| `--parcels` | yes |  | parcels.gpkg from geocrop parcels |
| `--markers` | yes |  | event markers CSV from geocrop event-markers |
| `--config` |  |  | cbm.rules section (applied even when rules.enabled is false) |
| `--rules` |  |  | JSON rules overriding the config, e.g. '{"harvest_expected_classes": ["maize"]}' |
| `--out` | yes |  | .gpkg (a .csv is written alongside) |
| `--id-field` |  | `parcel_id` |  |

## `geocrop samples-merge`

Combine per-season sample files into one multi-year sample store

| Option | Required | Default | Notes |
|---|---|---|---|
| `--inputs` | yes |  | samples .npz files, one per season |
| `--years` |  |  | year id per input (default: year metadata stored in each file) |
| `--stacks` |  |  | feature stack per input; their feature manifests must match |
| `--out` | yes |  |  |

## `geocrop transfer`

Leave-one-region-out / leave-one-year-out validation vs spatial CV

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--strategies` |  | `['region', 'year']` |  |
| `--config` |  |  | inherit model settings |
| `--params` |  |  | JSON estimator settings |
| `--block-m` |  |  |  |
| `--buffer-m` |  |  |  |
| `--folds` |  |  |  |
| `--seed` |  |  |  |

## `geocrop smoothing-experiment`

SYNTHETIC comparison of accuracy and conformal coverage with/without smoothing

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--seed` |  | `42` |  |
| `--alpha` |  | `0.1` |  |

## `geocrop openeo-extract`

Run composites, scenes or parcel signatures on an openEO back-end (CDSE) and write them in GeoCrop's file contracts (optional openeo extra)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |
| `--products` |  |  | default: openeo.products of the config |
| `--dry-run` |  |  | only write the process graphs; needs no credentials |
| `--graphs` |  |  | directory for the process graph JSON; default <openeo.work_dir>/graphs |
| `--auth` |  | `auto` | auto: CDSE_OPENEO_CLIENT_ID/SECRET if set, else the OIDC device flow |

## `geocrop openeo-graph`

Export the configured GeoCrop recipe as openEO process graphs (no credentials needed)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--config` | yes |  |  |
| `--products` |  |  | default: openeo.products of the config |
| `--out` |  |  | output directory; default <openeo.work_dir>/graphs |
| `--stdout` |  |  | print the openEO process objects as one JSON document |
