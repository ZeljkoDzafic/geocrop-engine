# GeoCrop Engine

**Open, reproducible Sentinel-1/Sentinel-2 crop-type classification, parcel-level aggregation and
declaration verification — fully local, provider-neutral, Apache-2.0.**

[![tests](https://github.com/ZeljkoDzafic/geocrop-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/ZeljkoDzafic/geocrop-engine/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

GeoCrop Engine turns Copernicus Sentinel time series into crop maps with class probabilities,
summarises them per agricultural parcel and compares the result with farmer declarations (LPIS/IACS)
— with the validation discipline a scientific publication or an inspection workflow needs:
spatially blocked cross-validation, independent field validation, area-adjusted accuracy,
calibration and explicit *review* outcomes instead of forced decisions.

It runs on a workstation or a single server, needs no cloud platform account and no GPU, and every
stage reads and writes plain GeoTIFF / GeoPackage / CSV / JSON so any GIS (QGIS, ArcGIS, PostGIS,
custom platforms) can consume the results.

```
 CDSE / STAC inventory ─► download ─► S2 L2A import ──► cloud/shadow-masked window composites ─┐
                                      S1 GRD (SNAP) ──► linear-domain window composites ────────┤
                                                                                                ▼
   reference polygons ─► samples ◄─ multi-temporal feature stack (bands, indices, radar, Δ, season)
                            │
                            ▼
      spatial-block CV · tuning · ablation (McNemar) ─► Random Forest ─► classes · probabilities · uncertainty
                                                                              │
             parcels (LPIS) ─► interior-buffered aggregation (majority | mean probability | combined + reject)
                                                                              │
                     agreement · discrepancy · uncertain · insufficient resolution  ─►  GIS / platform
                                                                              │
       independent field points ─► leakage-checked validation · bootstrap CIs · calibration · reject curve
```

## Why another crop-mapping tool?

| Need | GeoCrop Engine |
|---|---|
| Local, reproducible processing without a proprietary cloud | Python + GDAL stack; optional ESA SNAP for Sentinel-1; resumable stages with fingerprints and run manifests |
| Honest accuracy | Spatial-block, class-stratified CV; leakage assertions; independent point validation that automatically excludes observations near training polygons; bootstrap CIs; Olofsson et al. (2014) area-adjusted estimates; calibration (ECE) |
| Parcel-level decisions | Interior buffer against mixed pixels; majority vs. mean-probability vs. combined-with-reject strategies; second-stage parcel classifier; declaration status that is a *review aid*, never a verdict |
| Fragmented landscapes | Pixel counts per parcel, `insufficient_resolution` status, shape index, within-parcel variability |
| Publishable experiments | One-line ablations (S2 / S1 / fusion / window subsets) on identical folds with McNemar tests, feature-reduction curves, group importance, index table with formulas and references |
| Missing parcel boundaries | OBIA (SLIC) and an optional U-Net boundary model with watershed post-processing, scored with boundary F1 and object-level OS/US/IoU |
| Clouds and shadows | SCL + cloud probability, geometric shadow projection from sun angles, external masks (OmniCloudMask, s2cloudless, Fmask), halo dilation, multi-temporal outlier screening, Whittaker gap filling |
| Crop timing | Regular-grid time series (weighted Whittaker, upper envelope), phenology metrics (SOS, peak, EOS, LOS, slopes, integral), harmonic coefficients |
| Trustworthy decisions | Temperature-scaled probabilities, class-conditional conformal prediction sets, Area of Applicability (extrapolation flag), label-noise screening |
| Checks-by-Monitoring | Parcel signatures (Sen4CAP/JRC style), harvest / bare-soil / peak markers, crop-calendar consistency, green-yellow-red traffic light, field-campaign planning |
| Choice of model | Random Forest, ExtraTrees, HistGradientBoosting, LightGBM, CatBoost, TempCNN — all behind the same spatial CV, ablation and prediction |
| Interoperability | COG rasters, GeoParquet parcels, STAC items with classification extension, EuroCrops HCAT mapping |

## Install

```bash
conda env create -f environment.yml      # Python 3.11, GDAL stack from conda-forge
conda activate geocrop
# or: pip install -e '.[all]'   (Python >= 3.10)
pip install -e '.[deep]'                 # optional: U-Net / TempCNN (PyTorch)
pip install -e '.[boost,cloud,parquet]'  # optional: LightGBM/CatBoost, OmniCloudMask, GeoParquet
```
ESA SNAP (for Sentinel-1 preprocessing) is installed separately: <https://step.esa.int/main/download/snap-download/>.

## Quick start (synthetic data, ~10 s)

```bash
geocrop demo --root outputs/demo
```
Generates a synthetic six-class scene with phenological NDVI trajectories and Sentinel-1 backscatter,
then runs features → spatial CV → prediction → parcels → independent validation → ablation.
**Synthetic metrics verify the software only; they say nothing about real-world accuracy.**

## Real data in eight commands

```bash
geocrop catalog --aoi-wkt-file aoi.wkt --start 2023-04-01 --end 2023-10-15 --out manifests/s2.csv
geocrop download --manifest manifests/s2.csv --out data/raw/s2            # CDSE_USERNAME / CDSE_PASSWORD
geocrop grid --aoi aoi.gpkg --crs EPSG:32634 --out data/reference/grid.tif
geocrop stac-import --reference data/reference/grid.tif --start 2023-03-15 --end 2023-10-31 --out data/interim/s2
#   (or: geocrop download + geocrop s2-import --safe ... --reference data/reference/grid.tif)
geocrop snap-graph --out configs/s1_sigma0.xml --crs EPSG:32634            # then: geocrop snap / s1-import
geocrop inventory --scenes data/interim --config configs/example.yaml      # check availability per window
geocrop check --training data/reference/training.gpkg --parcels data/reference/parcels.gpkg
geocrop run --config configs/example.yaml                                  # resumable full pipeline
```
See [docs/getting-started.md](docs/getting-started.md) for the full walkthrough.

## Audit inputs and separate training from test samples

```bash
geocrop audit --training training.gpkg --parcels parcels.gpkg --observations observations.gpkg \
  --features stack.tif --exclusion-m 50 --out outputs/input-audit.json
geocrop train --samples samples.npz --out outputs/model --block-m 5000 --buffer-m 100
```

The read-only audit reports invalid schemas/geometries, conflicting training labels, class support,
field/training proximity and stack coverage; errors return exit status 2. Optional buffered CV removes
whole training polygons near held-out sampled pixels and reports exclusions and retained distances.
Set `model.spatial_buffer_m` in YAML to apply the same policy to pipeline training and ablations.
These are study-design safeguards; independent real-data validation remains necessary. See
[classification and validation](docs/classification-validation.md) for assumptions and limits.

## Documentation

An English [presentation website](website/README.md) introduces the workflow and interfaces with
actual web, CLI and QGIS captures. Preview it with `python3 -m http.server 8080 --directory website`.

### HTML workspace and Jupyter

```bash
pip install -e '.[web]'
# Set GEOCROP_WEB_TOKEN securely to a random token of at least 32 characters.
geocrop serve --workspace /path/to/your/study --port 8765
```

The optional local web interface adds command forms, an engine CLI console, queued jobs, logs,
cancellation and artifact downloads. Existing CLI commands and Python APIs remain available.
An authenticated HTTP API and `geocrop.client.EngineClient` support Jupyter Notebook, JupyterLab
and external Python tools. See [web/API setup](docs/web-interface.md) and the
[runnable notebook](examples/web_notebook.ipynb). No frontend build tools are required.

### QGIS Processing integration

The standalone QGIS plugin adds `geocrop:runcommand`, `geocrop:demo` and `geocrop:loadjob` to the
Processing Toolbox. It submits jobs to the HTTP API, streams logs, supports cancellation/reconnection
and downloads GIS results into fresh local directories for loading as layers. It does not require
engine scientific dependencies in QGIS's Python. See [QGIS installation, authentication and examples](docs/qgis.md).

```bash
python tools/build_qgis_plugin.py --out /path/to/new/geocrop-qgis.zip
```

The current development additions include frozen-predictor holdout calibration, complete-output
resume checks, reproducible classifier benchmarking and stricter STAC exports. They follow a
[source review of operational and open EO systems](docs/open-source-review.md); no comparative
real-data superiority is claimed.

Time-series processing additionally writes separate observation-support rasters and summaries, with
optional coverage/gap thresholds for reconstructed features. See [temporal QA](docs/timeseries.md).

```bash
geocrop train --samples samples.npz --out outputs/model --calibration-mode holdout --calibration-fraction 0.2
geocrop benchmark --samples samples.npz --out outputs/benchmark --config study.yaml --algorithms rf extra_trees hist_gb
```

Holdout calibration reserves entire spatial groups and deploys the same frozen model used for scoring
them. It requires enough class/group support and leaves less data for fitting. The default remains
empirical OOF calibration; independent field evaluation is required for both modes.

| Topic | |
|---|---|
| [Getting started](docs/getting-started.md) | installation, demo, real-data walkthrough |
| [Architecture](docs/architecture.md) | modules, stages, design principles |
| [Data contracts](docs/data-contracts.md) | every input and output file |
| [Preprocessing](docs/preprocessing.md) | Sentinel-2 masking & compositing, Sentinel-1 SNAP chain, grids |
| [Features](docs/features.md) | feature groups, naming, index formulas and references |
| [Classification & validation](docs/classification-validation.md) | spatial CV, tuning, ablation, independent validation, area estimation |
| [Parcels & declarations](docs/parcels.md) | aggregation, decision strategies, declaration status, parcel model |
| [Clouds & time series](docs/timeseries.md) | cloud/shadow screening, Whittaker smoothing, phenology, harmonics |
| [Uncertainty & applicability](docs/uncertainty.md) | calibration, conformal prediction, AOA, label noise, variogram |
| [Checks-by-Monitoring](docs/cbm.md) | parcel signatures, markers, traffic light, field-campaign planning |
| [Segmentation](docs/segmentation.md) | OBIA, U-Net, boundary/object metrics |
| [Interoperability](docs/interoperability.md) | COG, GeoParquet, STAC, HCAT, comparison with other systems |
| [Configuration](docs/configuration.md) | all YAML keys |
| [CLI reference](docs/cli.md) | all commands |
| [Reproducibility & privacy](docs/reproducibility.md) | manifests, caching, data protection |
| [Deployment](docs/deployment.md) | hardware sizing, server layout |
| [Scope & limitations](docs/limitations.md) | what is and is not validated |
| [Acceptance run](docs/acceptance.md) | `geocrop acceptance`: whole chain on real Sentinel-1/2 data, pass/fail report |

## Status

Version 0.9 — research software. The complete pipeline is unit-tested, runs end-to-end on synthetic
data, and its Sentinel-2 ingestion, compositing and features are integration-tested on real 2023 Sentinel-2
L2A data (public COGs over Semberija). CDSE access, SNAP graphs and SAFE import follow the official product
specifications but must be verified on real scenes for each study (see [limitations](docs/limitations.md)).
The engine was developed for a 2023 soybean / sunflower declaration-verification study in northern Bosnia
and Herzegovina; nothing about that study is hard-coded — area, season, classes, windows and rules are
configuration.

## Authors

- **Željko Džafić** — HTEC Group; Faculty of Technical Sciences, University of Novi Sad
- **Dragutin Adžić** — Faculty of Natural Sciences and Mathematics, University of Banja Luka

## Citation

If you use GeoCrop Engine, please cite it via [CITATION.cff](CITATION.cff):

> Džafić, Ž.; Adžić, D. *GeoCrop Engine: open Sentinel-1/Sentinel-2 crop-type classification and
> parcel-level declaration verification* (version 0.9.0) [Software]. 2026.
> https://github.com/ZeljkoDzafic/geocrop-engine

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copernicus Sentinel data are provided
under the Copernicus data policy; ESA SNAP and Python dependencies keep their own licences. No satellite
data, parcel data or model weights are distributed with this repository.
