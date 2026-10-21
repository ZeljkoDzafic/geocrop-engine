# CLI reference

Generated from `geocrop --help`. Every command prints JSON or a path; run `geocrop <command> --help` for details.

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

## `geocrop train`

Spatial CV + final Random Forest

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--block-m` |  | `5000` |  |
| `--folds` |  | `5` |  |
| `--params` |  |  | JSON, e.g. {"n_estimators": 500} |
| `--tune` |  |  | JSON grid, e.g. {"max_features": ["sqrt", 0.3]} |
| `--groups` |  |  | restrict to feature groups |
| `--manifest` |  |  | feature stack (for --groups) |
| `--algorithm` |  |  |  |
| `--no-calibrate` |  |  | skip temperature scaling |

## `geocrop predict`

Blockwise prediction: classes, probabilities, uncertainty

| Option | Required | Default | Notes |
|---|---|---|---|
| `--features` | yes |  |  |
| `--model` | yes |  |  |
| `--out` | yes |  |  |

## `geocrop ablation`

Feature-group ablation experiments on identical folds

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--features` | yes |  | feature stack (manifest source) |
| `--out` | yes |  |  |
| `--config` |  |  | experiments: and model: sections |
| `--targets` |  | `[]` |  |
| `--block-m` |  | `5000` |  |
| `--folds` |  | `5` |  |

## `geocrop reduce-features`

Recursive feature elimination curve under spatial CV

| Option | Required | Default | Notes |
|---|---|---|---|
| `--samples` | yes |  |  |
| `--out` | yes |  |  |
| `--block-m` |  | `5000` |  |
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

## `geocrop segment-unet-train`

Train the boundary U-Net (requires torch)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--images` | yes |  |  |
| `--masks` | yes |  |  |
| `--model` | yes |  |  |
| `--epochs` |  | `30` |  |
| `--batch-size` |  | `8` |  |

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

## `geocrop manifest`

Write a reproducibility manifest (versions, git, hashes)

| Option | Required | Default | Notes |
|---|---|---|---|
| `--out` | yes |  |  |
| `--config` |  |  |  |
| `--inputs` |  | `[]` |  |
| `--snap-gpt` |  |  |  |
