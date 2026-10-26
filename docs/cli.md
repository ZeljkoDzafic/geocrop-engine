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
| `--algorithm` |  |  |  |
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
| `--models` |  | `['rf', 'extra_trees', 'hist_gb', 'lightgbm', 'catboost']` |  |
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

## `geocrop manifest`

Write a reproducibility manifest (versions, git, hashes)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--config` |  |  |  |
| `--inputs` |  | `[]` |  |
| `--snap-gpt` |  |  |  |

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
