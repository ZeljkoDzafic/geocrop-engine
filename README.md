# GeoCrop Engine

**Open, reproducible Sentinel-1/Sentinel-2 crop-type classification, parcel-level aggregation and
declaration verification — fully local, provider-neutral, Apache-2.0.**

[![tests](https://github.com/ZeljkoDzafic/geocrop-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/ZeljkoDzafic/geocrop-engine/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

[Source](https://github.com/ZeljkoDzafic/geocrop-engine) ·
[Website](https://geocrop.calm-silence-b9f5.workers.dev) ·
[Documentation](docs/index.md) ·
[CLI reference](docs/cli.md) ·
[Limitations](docs/limitations.md)

GeoCrop Engine turns Copernicus Sentinel time series into crop maps with class probabilities,
summarises them per agricultural parcel and compares the result with farmer declarations (LPIS/IACS),
with the validation discipline a scientific publication or an inspection workflow needs:
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

## Status

Version 0.9.0, research software. The whole chain is tested on synthetic data and runs on real
Sentinel-2 L2A and Sentinel-1 RTC data from public STAC services. Crop-type accuracy has been measured only on
public Slovenian data with declared labels (parcel OA 86.9–88.5 %, [public benchmarks](docs/benchmarks/README.md));
**field-verified accuracy in the study region (northern Bosnia and Herzegovina) is not yet established**, and
conformal coverage drops below its target under region/year shift. [Scope and limitations](docs/limitations.md)
lists what is validated on real data and what only on synthetic data. Nothing about the original 2023 soybean /
sunflower study is hard-coded: area, season, classes, windows and rules are configuration.

## Install

Python 3.10–3.12 (CI tests all three). GDAL, PROJ and GEOS come inside the rasterio/pyogrio/pyproj/shapely
wheels, so a plain virtual environment is enough:

```bash
git clone https://github.com/ZeljkoDzafic/geocrop-engine.git
cd geocrop-engine
python3.12 -m venv .venv && source .venv/bin/activate     # or python3.10 / python3.11
python -m pip install -e '.[all,web]'                      # add -c constraints/py312.txt for pinned versions
```

On a fresh Ubuntu 26.04 or 24.04 machine, as an ordinary user with `sudo` (this sequence and the quick start
below are checked end to end by `tools/fresh_user_check.sh`, see
[fresh-user check](docs/release-gates/fresh-user-ubuntu.md)):

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 libexpat1
# Python 3.12, the interpreter of constraints/py312.txt:
sudo apt-get install -y --no-install-recommends python3.12 python3.12-venv      # Ubuntu 24.04
# Ubuntu 26.04 ships Python 3.14 and no python3.12 package; install 3.12 with uv instead (no root needed):
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv python install 3.12
git clone https://github.com/ZeljkoDzafic/geocrop-engine.git && cd geocrop-engine
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -c constraints/py312.txt -e '.[all,web]'
```
`libexpat1` is needed by the GDAL inside the rasterio wheel (minimal images lack it when Python does not come from
the distribution, e.g. with uv: `ImportError: libexpat.so.1`); `libgomp1` by the LightGBM wheel of `boost`.
Activate the environment (`source .venv/bin/activate`) in every new shell. `.[all,web]` adds scikit-image (segmentation), pystac-client (STAC search/import), pyarrow (GeoParquet) and the
local web workspace. The core install (`pip install -e .`) already runs `geocrop demo`. Further optional extras:

| Extra | Adds |
|---|---|
| `deep` | PyTorch: U-Net boundaries, TempCNN / L-TAE, Presto / Galileo / AnySat embeddings |
| `ftw` | Fields of The World boundary models (PyTorch + segmentation-models-pytorch) |
| `boost`, `cloud`, `parquet` | LightGBM / CatBoost, OmniCloudMask, GeoParquet |
| `stac`, `openeo`, `cube`, `plots` | STAC + Planetary Computer signing, openEO back-end, xarray/odc-stac/Zarr/NetCDF cubes, figures |
| `dev` | pytest, pytest-cov, ruff, build, pip-audit |

Alternatives:

* **conda:** `conda env create -f environment.yml && conda activate geocrop` (Python 3.11, GDAL stack from
  conda-forge), then `pip install -e '.[web]'` for the web workspace.
* **Docker** (CPython 3.12, pinned dependencies, runs as an unprivileged user; ESA SNAP not included):

  ```bash
  docker build -t geocrop .                        # extras all,stac,web; --build-arg EXTRAS=... to change
  mkdir -p work && docker run --rm -v "$PWD/work:/work" geocrop demo --root /work/demo
  ```
  On Linux the container user (uid 1000) must be able to write the mounted directory
  (or add `--user "$(id -u):$(id -g)"`).

Run-time downloads (checkpoints, weights, public datasets) are cached under `$GEOCROP_CACHE` (default
`~/.cache/geocrop`), resumed and checksum-verified; see [licences and weights](docs/licences-and-weights.md).
ESA SNAP (only for the Sentinel-1 GRD route) is installed separately:
<https://step.esa.int/main/download/snap-download/>.

## Five-minute quick start

From a fresh clone with the virtual environment above (times measured on a laptop):

```bash
# 1. Synthetic end-to-end run (~10 s; ~35 s on the very first start): features → spatial CV → prediction → parcels →
#    independent validation → ablation. Outputs: outputs/demo/{model,prediction,parcels,validation,...}
geocrop demo --root outputs/demo

# 2. The acceptance chain on synthetic Sentinel-1/2 scenes with its 21 pass/fail checks (~25 s, no network)
geocrop acceptance --offline --out outputs/acceptance

# 3. Real Sentinel-2 L2A imagery, no account needed (~2 min, network): monthly cloud-screened composites
geocrop quicklook --bbox 19.10 44.72 19.16 44.76 --start 2023-06-01 --end 2023-08-31 --out outputs/semberija-2023

# 4. Browse everything in the local web workspace (http://127.0.0.1:8765, token-protected)
export GEOCROP_WEB_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
geocrop serve --workspace . --port 8765
#    from a second shell with the same GEOCROP_WEB_TOKEN: the API answers only with the token
curl -H "Authorization: Bearer $GEOCROP_WEB_TOKEN" http://127.0.0.1:8765/api/v1/health
```

**Synthetic metrics verify the software only; they say nothing about real-world accuracy.** Open
`outputs/demo/parcels/parcels.gpkg` and `outputs/demo/prediction/classes.tif` in QGIS, or the workspace's map
view, to see the product layout. `geocrop <command> --help` documents every command.

## Real data

```bash
geocrop catalog --aoi-wkt-file aoi.wkt --start 2023-04-01 --end 2023-10-15 --out manifests/s2.csv
geocrop download --manifest manifests/s2.csv --out data/raw/s2            # CDSE_USERNAME / CDSE_PASSWORD
geocrop grid --aoi aoi.gpkg --crs EPSG:32634 --out data/reference/grid.tif
geocrop stac-import --reference data/reference/grid.tif --start 2023-03-15 --end 2023-10-31 --out data/interim/s2
#   (or: geocrop download, then geocrop s2-import --safe ... --reference data/reference/grid.tif)
geocrop snap-graph --out configs/s1_sigma0.xml --crs EPSG:32634            # then: geocrop snap / s1-import
geocrop inventory --scenes data/interim --config configs/example.yaml      # check availability per window
geocrop check --training data/reference/training.gpkg --parcels data/reference/parcels.gpkg
geocrop preflight --config configs/example.yaml --markdown preflight.md   # can the run succeed? rough cost
geocrop run --config configs/example.yaml                                  # resumable full pipeline
geocrop clean --config configs/example.yaml --dry-run                      # temp files, partial outputs, intermediates
```

[`configs/example.yaml`](configs/example.yaml) is the annotated configuration; every key is listed in
[configuration](docs/configuration.md). [Getting started](docs/getting-started.md) walks through a study, and
`geocrop audit` / `geocrop preflight` check inputs and resources before a long run
([preflight](docs/preflight.md)). For a VM, see the [runbook](deploy/README.md).

## Why another crop-mapping tool?

| Need | GeoCrop Engine |
|---|---|
| Local, reproducible processing without a proprietary cloud | Python + GDAL stack; optional ESA SNAP for Sentinel-1; resumable stages with fingerprints and run manifests |
| Honest accuracy | Spatial-block, class-stratified CV; leakage assertions; independent point validation that automatically excludes observations near training polygons; bootstrap CIs; Olofsson et al. (2014) area-adjusted estimates; calibration (ECE); frozen public benchmarks |
| Parcel-level decisions | Interior buffer against mixed pixels; majority vs. mean-probability vs. combined-with-reject strategies; second-stage parcel classifier; declaration status that is a *review aid*, never a verdict |
| Fragmented landscapes | Pixel counts per parcel, `insufficient_resolution` status, shape index, within-parcel variability |
| Publishable experiments | One-line ablations (S2 / S1 / fusion / window subsets) on identical folds with McNemar tests, feature-reduction curves, group importance, index table with formulas and references |
| Missing parcel boundaries | OBIA (SLIC, tiled with seam reconciliation), an optional U-Net boundary model and Fields of The World checkpoints with watershed post-processing, scored with boundary F1 and object-level OS/US/IoU |
| Clouds and shadows | SCL + cloud probability, geometric shadow projection from sun angles, external masks (OmniCloudMask, s2cloudless, Fmask), halo dilation, multi-temporal outlier screening, Whittaker gap filling |
| Crop timing | Regular-grid time series (weighted Whittaker, upper envelope), phenology metrics (SOS, peak, EOS, LOS, slopes, integral), harmonic coefficients, temporal-support QA |
| Trustworthy decisions | Temperature-scaled probabilities, class-conditional conformal prediction sets, Area of Applicability (extrapolation flag), label-noise screening |
| Checks-by-Monitoring | Parcel signatures (Sen4CAP/JRC style), harvest / bare-soil / mowing / homogeneity markers, crop-calendar consistency, green-yellow-red traffic light with reason codes, field-campaign planning, analyst review |
| Choice of model | Random Forest, ExtraTrees, HistGradientBoosting, LightGBM, CatBoost, TempCNN, L-TAE and plugin estimators, all behind the same spatial CV, ablation and prediction; optional foundation-model embeddings |
| Interoperability | COG rasters, GeoParquet parcels, STAC items with classification extension, EuroCrops HCAT mapping, openEO, xarray/Zarr cubes, QGIS plugin, HTTP API |

## Interfaces

* **CLI**: `geocrop <command>`; [reference](docs/cli.md) generated from the parser.
* **Web workspace and HTTP API** (`pip install -e '.[web]'`, `geocrop serve`): command forms, a terminal, queued
  jobs with logs and cancellation, a results map and analyst parcel review; `geocrop.client.EngineClient` for
  Jupyter ([guide](docs/web-interface.md), [notebook](examples/web_notebook.ipynb)). Single-owner and
  loopback-only by design ([security model](docs/security.md)).
* **QGIS**: a Processing plugin that submits jobs to the HTTP API and loads the results as layers
  ([installation](docs/qgis.md)); build it with `python tools/build_qgis_plugin.py --out /path/to/new/geocrop-qgis.zip`.
* **Python** and **plugins**: estimators, indices, segmenters, exporters, validators and feature providers can be
  added through entry points ([architecture](docs/architecture.md#extending), [example](examples/plugin_example)).

The [presentation website](website/README.md) introduces the workflow with actual web, CLI and QGIS captures.

## Documentation

| Topic | |
|---|---|
| [Documentation index](docs/index.md) | every page |
| [Getting started](docs/getting-started.md) | installation, demo, real-data walkthrough |
| [Scope & limitations](docs/limitations.md) | what is validated on real data, what only on synthetic data, what is open |
| [Architecture](docs/architecture.md) | layers, modules, stages, extension points |
| [Data contracts](docs/data-contracts.md) | every input and output file |
| [Configuration](docs/configuration.md) | all YAML keys |
| [Preprocessing](docs/preprocessing.md) · [Features](docs/features.md) · [Clouds & time series](docs/timeseries.md) | Sentinel-2/1 chains, feature groups and formulas, smoothing and phenology |
| [Classification & validation](docs/classification-validation.md) · [Validation protocols](docs/validation-protocols.md) | spatial CV, tuning, ablation, independent validation, transfer CV |
| [Uncertainty & applicability](docs/uncertainty.md) | calibration, conformal prediction, AOA, label noise |
| [Parcels & declarations](docs/parcels.md) · [Checks-by-Monitoring](docs/cbm.md) · [Markers](docs/markers.md) | aggregation, decision strategies, signatures, traffic light |
| [Segmentation](docs/segmentation.md) · [Field boundaries (FTW)](docs/field-boundaries.md) | OBIA, U-Net, pretrained boundary models, metrics |
| [Embeddings](docs/embeddings.md) | AlphaEarth, TESSERA, Presto, Galileo, AnySat feature providers |
| [Public benchmarks](docs/benchmarks/README.md) · [Acceptance run](docs/acceptance.md) · [Performance](docs/performance.md) | measured accuracy (public data), real-data chain check, scaling |
| [Testing](docs/testing.md) | test tiers, reliability and security gates, coverage |
| [Reproducibility & privacy](docs/reproducibility.md) · [Deployment](docs/deployment.md) | manifests, caching, data protection, hardware sizing |
| [Security model](docs/security.md) ([reporting](SECURITY.md)) · [Releases & rollback](docs/release.md) | web API threat model, dependency audits, SemVer, pinned installs |

## Authors

- **Željko Džafić** — HTEC Group; Faculty of Technical Sciences, University of Novi Sad
- **Dragutin Adžić** — Faculty of Natural Sciences and Mathematics, University of Banja Luka

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use GeoCrop Engine, please cite it via [CITATION.cff](CITATION.cff):

> Džafić, Ž.; Adžić, D. *GeoCrop Engine: open Sentinel-1/Sentinel-2 crop-type classification and
> parcel-level declaration verification* (version 0.9.0) [Software]. 2026.
> https://github.com/ZeljkoDzafic/geocrop-engine

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copernicus Sentinel data are provided
under the Copernicus data policy; ESA SNAP and Python dependencies keep their own licences. No satellite
data, parcel data or model weights are distributed with this repository; data and weights downloaded at run time
keep their own licences ([licences and weights](docs/licences-and-weights.md)).
