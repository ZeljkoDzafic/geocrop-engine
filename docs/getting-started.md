# Getting started

## 1. Install

```bash
git clone https://github.com/ZeljkoDzafic/geocrop-engine.git
cd geocrop-engine
conda env create -f environment.yml && conda activate geocrop     # recommended (GDAL from conda-forge)
pytest                                                               # ~40 tests, < 1 min
```
Alternatives: `pip install -e '.[all]'` in any Python ≥ 3.10 environment with working rasterio/GDAL wheels;
`pip install -e '.[deep]'` adds PyTorch for the U-Net segmentation path.

For Sentinel-1 install [ESA SNAP](https://step.esa.int/main/download/snap-download/) (9 or newer) and note the
path of its `gpt` executable. Put SNAP's cache/temp directories on a local SSD.

## 2. Verify with synthetic data

```bash
geocrop demo --root outputs/demo
```
Outputs under `outputs/demo/`: `features/stack.tif` (+ `stack.features.json`), `model/train_report.json`,
`prediction/{classes,probabilities,uncertainty}.tif`, `parcels/parcels.gpkg|csv|summary.json`,
`validation/validation_report.json`, `experiments/ablation.csv`. Open the GeoTIFF/GeoPackage files in QGIS.

## 3. Real study, step by step

### 3.1 Area, grid and inventory
```bash
# AOI polygon(s) in any CRS; one projected CRS for the whole study grid
geocrop grid --aoi data/reference/aoi.gpkg --crs EPSG:32634 --resolution 10 --out data/reference/grid.tif

# WKT (EPSG:4326) of the AOI for catalogue queries
geocrop catalog --aoi-wkt-file data/reference/aoi.wkt --start 2023-03-15 --end 2023-10-31 \
    --collection SENTINEL-2 --out manifests/s2_2023.csv
geocrop catalog --aoi-wkt-file data/reference/aoi.wkt --start 2023-03-15 --end 2023-10-31 \
    --collection SENTINEL-1 --out manifests/s1_2023.csv
```
Each inventory writes a CSV (product, date, tile/relative orbit, orbit direction, cloud cover, size) and a JSON
summary with the total download volume. **Inspect these before choosing window dates or buying storage.**
A provider-neutral alternative is `geocrop stac-search` (CDSE STAC by default, any STAC API works).

### 3.2 Download
```bash
export CDSE_USERNAME=... CDSE_PASSWORD=...          # never commit credentials
geocrop download --manifest manifests/s2_2023.csv --out data/raw/s2 --limit 3   # pilot first
```
Downloads resume after interruption (`.part` files, HTTP Range), refresh expired tokens, and record status and
SHA-256 in `download_status.json`.

### 3.3 Sentinel-2
```bash
unzip data/raw/s2/S2A_MSIL2A_20230612T093041_....zip -d data/raw/s2/
geocrop s2-import --safe data/raw/s2/S2A_MSIL2A_20230612T093041_....SAFE --out data/interim/s2/20230612_T34TCQ
```
`s2-import` applies the product's own `BOA_QUANTIFICATION_VALUE` and `BOA_ADD_OFFSET` (fails if missing),
aligns the 20 m bands to the 10 m grid and keeps SCL and the cloud-probability mask.

### 3.4 Sentinel-1
```bash
geocrop snap-graph --out configs/s1_sigma0.xml --crs EPSG:32634 --backscatter sigma0
geocrop snap --gpt /opt/snap/bin/gpt --graph configs/s1_sigma0.xml \
    --input data/raw/s1/S1A_IW_GRDH_1SDV_20230615....zip --output data/interim/s1_snap/20230615.tif
geocrop s1-import --geotiff data/interim/s1_snap/20230615.tif \
    --product-name S1A_IW_GRDH_1SDV_20230615T164512_... --orbit ASCENDING --out data/interim/s1/20230615_A
```

### 3.5 Windows, configuration, run
```bash
cp configs/example.yaml configs/my_study.yaml       # edit paths, windows, classes
geocrop inventory --scenes data/interim --config configs/my_study.yaml
geocrop check --training data/reference/training.gpkg --parcels data/reference/parcels.gpkg \
    --observations data/reference/observations.gpkg
geocrop folds --polygons data/reference/training.gpkg --out outputs/folds.gpkg --block-m 5000
geocrop run --config configs/my_study.yaml
```
`geocrop folds` shows (and maps) which classes would be missing from a test fold — adjust `spatial_block_m`
or `cv_folds` before training if rare classes disappear.

`geocrop run` executes `composite → features → samples → train → predict → parcels → validate → experiments`.
Re-running skips unchanged stages; `--stages train predict` runs a subset; `--force` recomputes.

### 3.6 Read the results
| File | Content |
|---|---|
| `model/train_report.json` | sample/polygon/block counts, tuning trials, spatial-CV metrics per fold, bootstrap CIs, calibration |
| `model/feature_importance.csv` | impurity importance of the final model |
| `prediction/uncertainty.tif` | max probability, top-2 margin, normalised entropy |
| `parcels/parcels.gpkg` | per-parcel class, confidence, homogeneity, decision and declaration status |
| `validation/validation_report.json` | independent accuracy, selection log (year filter, leakage exclusions), CIs, calibration, reject curve |
| `experiments/ablation.csv` | manuscript-ready comparison table with McNemar p-values |
| `run_manifest.json` | versions, git commit, configuration fingerprint, input hashes |
