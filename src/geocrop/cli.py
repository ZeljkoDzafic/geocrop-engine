"""`geocrop` command-line interface. Every stage is also importable as a Python function."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _json_arg(text):
    return json.loads(text) if text else None


def build_parser(parser_class=argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = parser_class(prog="geocrop", description="GeoCrop Engine: open Sentinel-1/2 crop classification "
                                "and parcel-level declaration verification.")
    p.add_argument("--version", action="version", version=f"geocrop-engine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def cmd(name, help_):
        return sub.add_parser(name, help=help_, description=help_)

    x = cmd("serve", "Start the local HTML workspace and HTTP API (optional web extra)")
    x.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    x.add_argument("--port", type=int, default=8765)
    x.add_argument("--workspace", default=".", help="working directory for engine jobs")
    x.add_argument("--state-dir", help="job database/log directory; default <workspace>/.geocrop-web")
    x.add_argument("--token-env", default="GEOCROP_WEB_TOKEN", help="name of the variable containing the access token")

    x = cmd("demo", "Run the full pipeline on synthetic data (software check, not accuracy)")
    x.add_argument("--root", default="outputs/demo")
    x.add_argument("--quick", action="store_true")

    x = cmd("run", "Run the config-driven pipeline (resumable)")
    x.add_argument("--config", required=True)
    x.add_argument("--stages", nargs="*", help="subset of stages, default all")
    x.add_argument("--force", action="store_true")

    x = cmd("config-check", "Validate a configuration file and print the resolved config")
    x.add_argument("--config", required=True)

    x = cmd("indices", "Print the spectral/radar index table (name, formula, reference)")
    x.add_argument("--format", choices=["json", "markdown"], default="markdown")

    # data acquisition
    x = cmd("catalog", "CDSE OData inventory for an AOI and season")
    x.add_argument("--aoi-wkt-file", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--collection", choices=["SENTINEL-2", "SENTINEL-1"], default="SENTINEL-2")
    x.add_argument("--product-type")
    x.add_argument("--cloud-max", type=float)
    x.add_argument("--out", required=True)

    x = cmd("stac-search", "Inventory from any STAC API (default: CDSE STAC)")
    x.add_argument("--endpoint", default="https://stac.dataspace.copernicus.eu/v1")
    x.add_argument("--collection", required=True)
    x.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--max-items", type=int, default=1000)
    x.add_argument("--cloud-max", type=float)
    x.add_argument("--out", required=True)

    x = cmd("download", "Resumable authenticated CDSE download of an inventory CSV")
    x.add_argument("--manifest", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--limit", type=int)

    # preprocessing
    x = cmd("grid", "Create the reference analysis grid from an AOI")
    x.add_argument("--aoi", required=True)
    x.add_argument("--crs", required=True)
    x.add_argument("--resolution", type=float, default=10)
    x.add_argument("--buffer-m", type=float, default=0)
    x.add_argument("--out", required=True)

    x = cmd("s2-import", "Sentinel-2 L2A SAFE -> scene folder (reflectance, SCL, CLDPRB, scene.json)")
    x.add_argument("--safe", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--reference", help="write directly on the study grid (crop to AOI, saves disk)")

    x = cmd("stac-import", "Windowed import of S2 L2A COGs from a STAC API onto the study grid (no full downloads)")
    x.add_argument("--reference", required=True, help="study grid GeoTIFF (geocrop grid)")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help="scene root, one folder per acquisition")
    x.add_argument("--endpoint", default="https://earth-search.aws.element84.com/v1")
    x.add_argument("--collection", default="sentinel-2-l2a")
    x.add_argument("--max-cloud", type=float, default=80, help="tile cloud cover pre-filter (%%)")
    x.add_argument("--min-clear", type=float, default=0.2, help="minimum clear fraction inside the grid")
    x.add_argument("--max-items", type=int, default=500)

    x = cmd("quicklook", "Real Sentinel-2 imagery for an area and period: monthly cloud-free composites for the map")
    x.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"),
                   help="degrees, e.g. 19.10 44.72 19.16 44.76 (Semberija)")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--max-cloud", type=float, default=60)
    x.add_argument("--min-clear", type=float, default=0.3)
    x.add_argument("--max-items", type=int, default=60)

    x = cmd("s2-composite", "Cloud-masked temporal composite of S2 scene folders")
    x.add_argument("--scenes", nargs="+", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--reference")
    x.add_argument("--statistic", default="median")
    x.add_argument("--cloud-prob-max", type=float)
    x.add_argument("--dilate", type=int, default=0)
    x.add_argument("--min-obs", type=int, default=1)

    x = cmd("snap-graph", "Write a SNAP GPT graph for S1 GRD preprocessing")
    x.add_argument("--out", required=True)
    x.add_argument("--backscatter", choices=["sigma0", "gamma0"], default="sigma0")
    x.add_argument("--crs", default="EPSG:32634")
    x.add_argument("--dem", default="Copernicus 30m Global DEM")
    x.add_argument("--pixel-spacing", type=float, default=10)
    x.add_argument("--speckle", help="e.g. 'Refined Lee' or 'Lee'")

    x = cmd("snap", "Run SNAP gpt with a graph on one product")
    x.add_argument("--gpt", required=True)
    x.add_argument("--graph", required=True)
    x.add_argument("--input", required=True)
    x.add_argument("--output", required=True)
    x.add_argument("--cache", default="8G")

    x = cmd("s1-import", "SNAP GeoTIFF -> S1 scene folder (VV.tif, VH.tif, scene.json)")
    x.add_argument("--geotiff", required=True)
    x.add_argument("--product-name", required=True)
    x.add_argument("--orbit", required=True, choices=["ASCENDING", "DESCENDING"])
    x.add_argument("--backscatter", default="sigma0")
    x.add_argument("--out", required=True)

    x = cmd("s1-composite", "Linear-domain temporal composite of S1 scene folders on a reference grid")
    x.add_argument("--scenes", nargs="+", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--separate-orbits", action="store_true")
    x.add_argument("--statistic", choices=["mean", "median"], default="mean")

    x = cmd("inventory", "Scene availability per window from imported scene folders")
    x.add_argument("--scenes", required=True)
    x.add_argument("--config", required=True)

    # modelling
    x = cmd("features", "Build the multi-temporal feature stack")
    x.add_argument("--composites", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--config", help="use the features: section of a config")

    x = cmd("check", "Quality checks for rasters and vector reference data")
    x.add_argument("--raster", nargs="*", default=[])
    x.add_argument("--training")
    x.add_argument("--parcels")
    x.add_argument("--observations")

    x = cmd("audit", "Audit study inputs, conflicting labels and training/field leakage without changing data")
    x.add_argument("--training")
    x.add_argument("--parcels")
    x.add_argument("--observations")
    x.add_argument("--features", help="feature stack GeoTIFF")
    x.add_argument("--label-field", default="class_name")
    x.add_argument("--id-field", default="polygon_id")
    x.add_argument("--parcel-id-field", default="parcel_id")
    x.add_argument("--observation-label-field")
    x.add_argument("--exclusion-m", type=float, default=0)
    x.add_argument("--distance-crs", help="projected metric CRS for field exclusion distances")
    x.add_argument("--out", help="save the JSON report; exit 2 when errors are found")

    x = cmd("folds", "Assign class-stratified spatial-block folds to reference polygons (GIS-inspectable)")
    x.add_argument("--polygons", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--label-field", default="class_name")
    x.add_argument("--block-m", type=float, default=5000)
    x.add_argument("--folds", type=int, default=5)

    x = cmd("samples", "Extract training samples from a stack and reference polygons")
    x.add_argument("--features", required=True)
    x.add_argument("--polygons", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--label-field", default="class_name")
    x.add_argument("--id-field", default="polygon_id")
    x.add_argument("--max-per-polygon", type=int, default=250)
    x.add_argument("--inner-buffer-m", type=float, default=0)

    x = cmd("train", "Spatial CV + final Random Forest")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--block-m", type=float, default=5000)
    x.add_argument("--buffer-m", type=float, default=0, help="purge training polygons near held-out sampled pixels")
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--params", type=_json_arg, help='JSON, e.g. {"n_estimators": 500}')
    x.add_argument("--tune", type=_json_arg, help='JSON grid, e.g. {"max_features": ["sqrt", 0.3]}')
    x.add_argument("--groups", nargs="*", help="restrict to feature groups")
    x.add_argument("--manifest", help="feature stack (for --groups)")
    x.add_argument("--algorithm", choices=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost", "tempcnn"])
    x.add_argument("--no-calibrate", action="store_true", help="skip temperature scaling")
    x.add_argument("--holdout", type=float, default=0.0, help="fraction of spatial blocks locked as final test")
    x.add_argument("--calibration-mode", choices=["oof", "holdout"], default="oof",
                   help="holdout freezes a predictor without fitting on calibration polygons")
    x.add_argument("--calibration-fraction", type=float, default=0.2, help="fraction of spatial groups held out")

    x = cmd("benchmark", "Compare classifiers on identical spatial folds; internal model selection only")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--algorithms", nargs="+", choices=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost", "tempcnn"],
                   default=["rf", "extra_trees", "hist_gb"])
    x.add_argument("--config", help="inherit model settings and spatial validation policy")
    x.add_argument("--params", type=_json_arg, help="JSON estimator settings shared across candidates")
    x.add_argument("--block-m", type=float)
    x.add_argument("--buffer-m", type=float)
    x.add_argument("--folds", type=int)
    x.add_argument("--seed", type=int)

    x = cmd("predict", "Blockwise prediction: classes, probabilities, uncertainty")
    x.add_argument("--features", required=True)
    x.add_argument("--model", required=True)
    x.add_argument("--out", required=True)

    x = cmd("ablation", "Feature-group ablation experiments on identical folds")
    x.add_argument("--samples", required=True)
    x.add_argument("--features", required=True, help="feature stack (manifest source)")
    x.add_argument("--out", required=True)
    x.add_argument("--config", help="experiments: and model: sections")
    x.add_argument("--targets", nargs="*", default=[])
    x.add_argument("--block-m", type=float, help="override config model.spatial_block_m (default 5000)")
    x.add_argument("--folds", type=int, help="override config model.cv_folds (default 5)")
    x.add_argument("--buffer-m", type=float, help="override config model.spatial_buffer_m (default 0)")
    x.add_argument("--seed", type=int, help="override config seed (default 42)")

    x = cmd("reduce-features", "Recursive feature elimination curve under spatial CV")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--block-m", type=float)
    x.add_argument("--buffer-m", type=float)
    x.add_argument("--config", help="inherit model and holdout calibration policy")
    x.add_argument("--step", type=float, default=0.2)

    x = cmd("parcels", "Parcel aggregation, decision and declaration comparison")
    x.add_argument("--probabilities", required=True)
    x.add_argument("--parcels", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--id-field", default="parcel_id")
    x.add_argument("--declared-field", default="declared_crop")
    x.add_argument("--strategy", choices=["majority", "mean_probability", "combined"], default="mean_probability")
    x.add_argument("--buffer-px", type=float, default=1)
    x.add_argument("--min-pixels", type=int, default=4)
    x.add_argument("--min-confidence", type=float)
    x.add_argument("--min-majority-share", type=float)
    x.add_argument("--min-margin", type=float)
    x.add_argument("--crop-mapping", help="CSV source_code,class_name")
    x.add_argument("--conformal", help="conformal.json from training")
    x.add_argument("--alpha", type=float, default=0.1)
    x.add_argument("--aoa", help="aoa.tif from prediction")
    x.add_argument("--min-aoa-fraction", type=float, default=0.5,
                   help="minimum inside-AOA share of valid probability pixels")
    x.add_argument("--markers", help="markers.csv from geocrop markers")
    x.add_argument("--calendar", help="crop calendar YAML")

    x = cmd("parcel-model", "Second-stage parcel classifier on aggregated descriptors")
    x.add_argument("--parcels", required=True, help="output of geocrop parcels with a reference label column")
    x.add_argument("--label-field", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--block-m", type=float, default=5000)

    x = cmd("validate", "Independent validation with field observations")
    x.add_argument("--prediction", required=True, help="prediction directory")
    x.add_argument("--observations", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--label-field", default="class_name")
    x.add_argument("--date-field", default="observed_at")
    x.add_argument("--year", type=int)
    x.add_argument("--training")
    x.add_argument("--exclusion-m", type=float, default=0)
    x.add_argument("--window-px", type=int, default=0)
    x.add_argument("--label-mapping", help="CSV source,class_name")

    x = cmd("area", "Mapped area per class and Olofsson et al. (2014) error-adjusted estimates")
    x.add_argument("--classes", required=True, help="classes.tif")
    x.add_argument("--validation", help="validation_report.json (stratified sample)")
    x.add_argument("--out", required=True)

    x = cmd("normalize-parcels", "Map external parcel fields to the neutral schema")
    x.add_argument("--input", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--id-field", required=True)
    x.add_argument("--declared-field")
    x.add_argument("--crop-mapping")
    x.add_argument("--pseudonymise", action="store_true")

    # segmentation
    x = cmd("segment-obia", "SLIC superpixel segmentation (tile-bounded)")
    x.add_argument("--raster", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--bands", nargs="*")
    x.add_argument("--segments-per-km2", type=float, default=50)
    x.add_argument("--compactness", type=float, default=0.1)
    x.add_argument("--tile-size", type=int, default=1024)
    x.add_argument("--no-merge-seams", action="store_true", help="keep segments split at tile borders")
    x.add_argument("--merge-distance", type=float, default=0.08, help="max RMS spectral difference for seam merges")

    x = cmd("segment-unet-train", "Train the boundary U-Net (requires torch)")
    x.add_argument("--images", required=True)
    x.add_argument("--masks", required=True)
    x.add_argument("--model", required=True)
    x.add_argument("--epochs", type=int, default=30)
    x.add_argument("--batch-size", type=int, default=8)
    x.add_argument("--groups", help=".npy/.txt with one spatial group id per patch (required for a fair split)")
    x.add_argument("--allow-random-split", action="store_true", help="accept an optimistic random patch split")

    x = cmd("segment-patches", "Cut image/boundary patches with spatial group ids for U-Net training")
    x.add_argument("--raster", required=True)
    x.add_argument("--mask", required=True, help="rasterised reference boundaries (1 = boundary)")
    x.add_argument("--out", required=True, help="output prefix; writes _X.npy, _Y.npy, _groups.npy")
    x.add_argument("--size", type=int, default=128)
    x.add_argument("--block-m", type=float, default=2000)
    x.add_argument("--bands", nargs="*")

    x = cmd("segment-unet-infer", "Boundary probability with overlapping tiles")
    x.add_argument("--raster", required=True)
    x.add_argument("--model", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--bands", nargs="*")
    x.add_argument("--overlap", type=int, default=32)

    x = cmd("boundaries-to-parcels", "Watershed post-processing of boundary probability")
    x.add_argument("--raster", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--threshold", type=float, default=0.5)

    x = cmd("evaluate-segments", "Object-level segmentation quality vs reference parcels")
    x.add_argument("--segments", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--iou", type=float, default=0.5)

    x = cmd("evaluate-boundaries", "Boundary raster precision/recall/F1 with tolerance")
    x.add_argument("--prediction", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--threshold", type=float, default=0.5)
    x.add_argument("--tolerance-px", type=int, default=2)

    x = cmd("ts-features", "Time-series features (Whittaker, phenology, harmonics) from scene folders")
    x.add_argument("--scenes", nargs="+", required=True, help="scene root folder(s)")
    x.add_argument("--reference", required=True, help="reference grid GeoTIFF")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--s2-indices", nargs="*", default=["NDVI", "NDRE", "NDMI"])
    x.add_argument("--s1-indices", nargs="*", default=["VHVV_dB"])
    x.add_argument("--step-days", type=int, default=5)
    x.add_argument("--output-step-days", type=int, default=15)
    x.add_argument("--lam", type=float, default=50.0)
    x.add_argument("--envelope", type=int, default=0, help="upper-envelope iterations")
    x.add_argument("--min-observed-fraction", type=float, default=0, help="minimum occupied temporal-bin fraction")
    x.add_argument("--max-gap-days", type=float, help="maximum consecutive empty-bin duration, including edges")

    x = cmd("merge-stacks", "Concatenate feature stacks on one grid")
    x.add_argument("--stacks", nargs="+", required=True)
    x.add_argument("--out", required=True)

    x = cmd("signatures", "Parcel-mean time series (CbM signatures) from S2 scene folders")
    x.add_argument("--scenes", required=True)
    x.add_argument("--parcels", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help=".csv or .parquet")
    x.add_argument("--indices", nargs="*", default=["NDVI", "BSI"])
    x.add_argument("--id-field", default="parcel_id")

    x = cmd("markers", "Phenology, harvest and bare-soil markers from parcel signatures")
    x.add_argument("--signatures", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--index", default="NDVI")

    x = cmd("variogram", "Spatial autocorrelation range of reference objects (for spatial_block_m)")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--max-distance", type=float)

    x = cmd("fieldwork", "Rank parcels for the next field campaign (uncertainty + AOA, spatially spread)")
    x.add_argument("--parcels", required=True, help="output of geocrop parcels")
    x.add_argument("--out", required=True)
    x.add_argument("--n", type=int, default=50)
    x.add_argument("--min-spacing-m", type=float, default=1000)
    x.add_argument("--per-class", type=int)

    x = cmd("cloudmask", "Write CLOUDMASK.tif for S2 scene folders with OmniCloudMask (optional dependency)")
    x.add_argument("--scenes", nargs="+", required=True)

    x = cmd("export", "Publish a prediction directory: COGs, STAC item, optional GeoParquet parcels")
    x.add_argument("--prediction", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--id", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--parcels", help="parcels.gpkg to convert to GeoParquet")
    x.add_argument("--hcat", help="CSV class_name,hcat_code,hcat_name")

    x = cmd("benchmark-models", "Compare classifiers on identical spatial folds (nested tuning, repeats, holdout)")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--models", nargs="*", default=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost"])
    x.add_argument("--features", help="feature stack (manifest) to restrict --groups")
    x.add_argument("--groups", nargs="*")
    x.add_argument("--grids", type=_json_arg, help='JSON {"rf": {"max_features": ["sqrt", 0.3]}}')
    x.add_argument("--block-m", type=float, default=5000)
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--repeats", type=int, default=3)
    x.add_argument("--holdout", type=float, default=0.0)
    x.add_argument("--targets", nargs="*", default=[])

    x = cmd("benchmark-segmentation", "OBIA (with/without seam merging) vs U-Net against reference parcels")
    x.add_argument("--raster", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--bands", nargs="*")
    x.add_argument("--unet-model")
    x.add_argument("--unet-bands", nargs="*")
    x.add_argument("--tile-size", type=int, default=1024)
    x.add_argument("--segments-per-km2", type=float, default=50)

    # public benchmark and crop calendars (G1, G11)
    x = cmd("benchmark-public", "Run a frozen public benchmark protocol (EuroCrops parcels + Sentinel-2 from STAC)")
    x.add_argument("--protocol", required=True, help="versioned protocol YAML, e.g. configs/benchmarks/*.yaml")
    x.add_argument("--work", required=True, help="working directory (scenes, composites, results.json/.md)")
    x.add_argument("--cache", help="download cache; default $GEOCROP_DATA_CACHE or ~/.cache/geocrop/public")
    x.add_argument("--models", nargs="*", help="subset of the protocol's models (default all)")
    x.add_argument("--workers", type=int, default=4, help="parallel scene imports")
    x.add_argument("--layer", help="local copy of the EuroCrops layer instead of the pinned download")
    x.add_argument("--publish", help="directory to copy results.md/.json to (named after the protocol)")

    x = cmd("calendar", "Derive a crop calendar and phenological windows from parcel markers, or read "
                        "WorldCereal calendar priors")
    x.add_argument("action", choices=["derive", "worldcereal"])
    x.add_argument("--markers", help="derive: markers table from geocrop markers (CSV/Parquet)")
    x.add_argument("--signatures", help="derive: parcel signatures (CSV/Parquet); markers computed with --start/--end")
    x.add_argument("--start", help="derive with --signatures: season start (ISO date)")
    x.add_argument("--end", help="derive with --signatures: season end (ISO date)")
    x.add_argument("--labels", help="derive: parcels with labels (GPKG/GeoJSON/CSV)")
    x.add_argument("--label-field", default="class_name")
    x.add_argument("--id-field", default="parcel_id")
    x.add_argument("--year", type=int, help="derive: year of the suggested windows")
    x.add_argument("--percentiles", nargs=3, type=float, default=[10, 50, 90], metavar=("LOW", "MID", "HIGH"))
    x.add_argument("--min-parcels", type=int, default=5)
    x.add_argument("--min-days", type=int, default=30, help="derive: shortest suggested window")
    x.add_argument("--max-windows", type=int, default=6)
    x.add_argument("--out", help="derive: calendar YAML; worldcereal: JSON (default: print)")
    x.add_argument("--windows-out", help="derive: suggested windows YAML snippet")
    x.add_argument("--lon", type=float, help="worldcereal: longitude (WGS84)")
    x.add_argument("--lat", type=float, help="worldcereal: latitude (WGS84)")
    x.add_argument("--cache", help="worldcereal: download cache directory")

    x = cmd("manifest", "Write a reproducibility manifest (versions, git, hashes)")
    x.add_argument("--out", required=True)
    x.add_argument("--config")
    x.add_argument("--inputs", nargs="*", default=[])
    x.add_argument("--snap-gpt")
    return p


def _calendar(a):
    """`geocrop calendar derive|worldcereal` (kept out of `main` to keep the dispatch flat)."""
    from . import calendars
    if a.action == "worldcereal":
        if a.lon is None or a.lat is None:
            raise SystemExit("worldcereal needs --lon and --lat")
        from .public_data import WORLDCEREAL_CITATION, worldcereal_calendar_files
        priors = {"lon": a.lon, "lat": a.lat, "unit": "day of year (first season)",
                  "priors": calendars.worldcereal_priors(worldcereal_calendar_files(cache=a.cache), a.lon, a.lat),
                  "source": WORLDCEREAL_CITATION}
        if a.out:
            Path(a.out).write_text(json.dumps(priors, indent=2), encoding="utf8")
        _print(priors)
        return
    import pandas as pd
    if not a.labels or not (a.markers or a.signatures):
        raise SystemExit("derive needs --labels and --markers (or --signatures with --start/--end)")
    if a.markers:
        markers = pd.read_parquet(a.markers) if a.markers.endswith(".parquet") else pd.read_csv(a.markers)
        source = a.markers
    else:
        from datetime import date

        from .cbm import markers as compute_markers
        if not a.start or not a.end:
            raise SystemExit("--signatures needs --start and --end")
        sig = pd.read_parquet(a.signatures) if a.signatures.endswith(".parquet") else pd.read_csv(a.signatures)
        markers = compute_markers(sig, date.fromisoformat(a.start), date.fromisoformat(a.end))
        source = a.signatures
    if a.labels.lower().endswith(".csv"):
        labels = pd.read_csv(a.labels, dtype={a.id_field: str})
    else:
        import geopandas as gpd
        labels = pd.DataFrame(gpd.read_file(a.labels).drop(columns="geometry"))
    labels = labels.rename(columns={a.label_field: "class_name", a.id_field: "parcel_id"})
    derived = calendars.derive(markers, labels[["parcel_id", "class_name"]], tuple(a.percentiles), a.min_parcels)
    cal = calendars.to_calendar(derived)
    text = calendars.calendar_yaml(cal, derived, source)
    if a.out:
        Path(a.out).write_text(text, encoding="utf8")
    result = {"calendar": a.out or text, "classes": {k: v["n_parcels"] for k, v in derived["classes"].items()}}
    if a.windows_out or a.year:
        year = a.year or (int(a.start[:4]) if a.start else None)
        if year is None:
            raise SystemExit("Suggested windows need --year")
        windows = calendars.suggest_windows(derived, year, a.min_days, a.max_windows)
        snippet = calendars.windows_yaml(windows, derived, source)
        if a.windows_out:
            Path(a.windows_out).write_text(snippet, encoding="utf8")
        result["windows"] = windows
    _print(result)


def main(argv=None):  # noqa: C901 - flat dispatch is the clearest form here
    a = build_parser().parse_args(argv)
    c = a.cmd
    if c == "serve":
        import os

        try:
            import uvicorn

            from .web import create_app
        except ImportError as exc:
            raise RuntimeError('Install the web interface: pip install -e ".[web]"') from exc
        token = os.environ.get(a.token_env)
        if not token or len(token) < 32:
            raise ValueError(f"Set {a.token_env} to a random access token of at least 32 characters")
        app = create_app(a.workspace, token, a.state_dir, token_env=a.token_env)
        uvicorn.run(app, host=a.host, port=a.port, proxy_headers=False)
        return
    if c == "demo":
        from .demo import run
        _print(run(a.root, quick=a.quick))
    elif c == "run":
        from .pipeline import run
        _print(run(a.config, a.stages, a.force))
    elif c == "config-check":
        from .config import fingerprint, load
        cfg = load(a.config)
        _print({"fingerprint": fingerprint(cfg), "config": cfg})
    elif c == "indices":
        from .indices import table
        rows = table()
        if a.format == "json":
            _print(rows)
        else:
            print("| Name | Sensor | Formula | Reference |\n|---|---|---|---|")
            for r in rows:
                print(f"| {r['name']} | {r['sensor']} | `{r['formula']}` | {r['reference']} |")
    elif c == "catalog":
        from .catalog import inventory
        _print(inventory(Path(a.aoi_wkt_file).read_text().strip(), a.start, a.end, a.out, a.collection,
                         a.product_type, a.cloud_max))
    elif c == "stac-search":
        from .stac import discover
        _print(discover(a.endpoint, a.collection, a.bbox, a.start, a.end, a.out, a.max_items, a.cloud_max))
    elif c == "download":
        from .catalog import download_manifest
        _print(download_manifest(a.manifest, a.out, a.limit))
    elif c == "grid":
        from .raster import make_grid
        print(make_grid(a.aoi, a.crs, a.resolution, a.out, a.buffer_m))
    elif c == "s2-import":
        from .s2 import import_safe
        print(import_safe(a.safe, a.out, reference=a.reference))
    elif c == "stac-import":
        from .stac_import import import_season
        res = import_season(a.reference, a.out, a.start, a.end, a.endpoint, a.collection, a.max_cloud, a.min_clear,
                            a.max_items)
        counts = {}
        for r in res:
            key = r["status"].split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        _print({"items": len(res), "status": counts, "out": a.out})
    elif c == "quicklook":
        from .quicklook import quicklook
        r = quicklook(a.bbox, a.start, a.end, a.out, max_cloud=a.max_cloud, min_clear=a.min_clear,
                      max_items=a.max_items)
        _print({k: r[k] for k in ("crs", "area_km2", "acquisitions_found", "acquisitions_used", "composites")})
    elif c == "s2-composite":
        from .s2 import composite
        print(composite(a.scenes, a.out, a.reference, a.statistic, cloud_probability_max=a.cloud_prob_max,
                        mask_dilation_px=a.dilate, min_valid_observations=a.min_obs))
    elif c == "snap-graph":
        from .s1 import write_snap_graph
        print(write_snap_graph(a.out, a.backscatter, a.dem, a.pixel_spacing, a.crs, a.speckle))
    elif c == "snap":
        from .s1 import run_snap
        sys.exit(run_snap(a.gpt, a.graph, a.input, a.output, a.cache))
    elif c == "s1-import":
        from .s1 import import_geotiff
        print(import_geotiff(a.geotiff, a.out, a.product_name, a.orbit, a.backscatter))
    elif c == "s1-composite":
        from .s1 import composite
        print(composite(a.scenes, a.out, a.reference, a.separate_orbits, a.statistic))
    elif c == "inventory":
        from . import config, scenes
        windows = config.parse_windows(config.load(a.config)["windows"])
        _print(scenes.inventory(scenes.assign(scenes.discover(a.scenes), windows)))
    elif c == "features":
        from .features import build_stack
        opts = {}
        if a.config:
            from .config import load, parse_windows
            cfg = load(a.config)
            opts = {k: cfg["features"][k] for k in ("s2_bands", "s2_indices", "s1", "deltas", "seasonal")}
            opts["periods"] = list(parse_windows(cfg["windows"])) or None
        print(build_stack(a.composites, a.out, **opts))
    elif c == "check":
        from .provenance import check_raster, check_vector
        out = {"rasters": [check_raster(r) for r in a.raster]}
        if a.training:
            out["training"] = check_vector(a.training, ["polygon_id", "class_name"], ["polygon_id"], "class_name")
        if a.parcels:
            out["parcels"] = check_vector(a.parcels, ["parcel_id"], ["parcel_id"], "declared_crop")
        if a.observations:
            out["observations"] = check_vector(a.observations, ["class_name"], (), "class_name")
        _print(out)
    elif c == "folds":
        from .splits import assign_folds
        _print(assign_folds(a.polygons, a.out, a.label_field, a.block_m, a.folds))
    elif c == "audit":
        from .audit import audit_inputs
        if a.out:
            target = Path(a.out)
            for source in (a.training, a.parcels, a.observations, a.features):
                if source and (target.resolve() == Path(source).resolve() or
                               (target.exists() and Path(source).exists() and target.samefile(source))):
                    raise ValueError("Audit output must differ from every input path")
        report = audit_inputs(training=a.training, parcels=a.parcels, observations=a.observations,
                              stack=a.features, label_field=a.label_field, id_field=a.id_field,
                              parcel_id_field=a.parcel_id_field, observation_label_field=a.observation_label_field,
                              training_exclusion_m=a.exclusion_m, distance_crs=a.distance_crs)
        if a.out:
            target = Path(a.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
        _print(report)
        if not report["ok"]:
            sys.exit(2)
    elif c == "samples":
        from .samples import extract
        s = extract(a.features, a.polygons, a.label_field, a.id_field, a.max_per_polygon, a.inner_buffer_m)
        print(s.save(a.out))
    elif c == "train":
        from .features import load_manifest, select
        from .model import train
        from .samples import Samples
        s = Samples.load(a.samples)
        feats = select([f for f in load_manifest(a.manifest) if f.name in s.features], groups=a.groups) if a.groups else None
        params = dict(a.params or {"n_estimators": 500, "max_features": "sqrt",
                                   "class_weight": "balanced_subsample", "n_jobs": -1})
        if a.algorithm:
            params["algorithm"] = a.algorithm
        r = train(s, a.out, feats, params, a.folds, a.block_m, tune_grid=a.tune, calibrate=not a.no_calibrate,
                  holdout_fraction=a.holdout, spatial_buffer_m=a.buffer_m, calibration_mode=a.calibration_mode,
                  calibration_fraction=a.calibration_fraction)
        _print({k: r["cross_validation"][k] for k in ("overall_accuracy", "kappa", "macro_f1", "fold_macro_f1_std")})
    elif c == "benchmark":
        from .benchmark import run
        from .config import load
        from .model import RF_KEYS
        from .samples import Samples
        cfg = load(a.config)
        m = cfg["model"]
        params = {k: m[k] for k in RF_KEYS if k in m and m[k] is not None}
        params.update(a.params or {})
        if len(set(a.algorithms)) != len(a.algorithms):
            raise ValueError("Each benchmark algorithm must be listed once")
        report = run(Samples.load(a.samples), a.out,
                     {name: {**params, "algorithm": name} for name in a.algorithms},
                     n_splits=a.folds if a.folds is not None else m["cv_folds"],
                     spatial_block_m=a.block_m if a.block_m is not None else m["spatial_block_m"],
                     seed=a.seed if a.seed is not None else cfg["seed"],
                     spatial_buffer_m=a.buffer_m if a.buffer_m is not None else m["spatial_buffer_m"],
                     calibration_mode=m["calibration_mode"], calibration_fraction=m["calibration_fraction"],
                     holdout_fraction=m["holdout_fraction"])
        _print({"status": report["status"], "summary": report["summary"], "note": report["note"]})
        if report["status"] != "complete":
            sys.exit(2)
    elif c == "predict":
        from .model import predict
        _print(predict(a.features, a.model, a.out))
    elif c == "ablation":
        from . import experiments
        from .config import load
        from .features import load_manifest
        from .model import RF_KEYS
        from .samples import Samples
        cfg = load(a.config)
        exps = cfg["experiments"] or None
        m = cfg["model"]
        params = {k: m[k] for k in RF_KEYS if k in m and m[k] is not None}
        n_splits = a.folds if a.folds is not None else m["cv_folds"]
        block_m = a.block_m if a.block_m is not None else m["spatial_block_m"]
        buffer_m = a.buffer_m if a.buffer_m is not None else m["spatial_buffer_m"]
        seed = a.seed if a.seed is not None else cfg["seed"]
        t = experiments.run(Samples.load(a.samples), load_manifest(a.features), a.out, exps, params, n_splits,
                            block_m, seed=seed, targets=a.targets or cfg["classes"]["targets"],
                            spatial_buffer_m=buffer_m, calibration_mode=m["calibration_mode"],
                            calibration_fraction=m["calibration_fraction"])
        print(t.to_string(index=False))
    elif c == "reduce-features":
        from .config import load
        from .experiments import feature_reduction
        from .model import RF_KEYS
        from .samples import Samples
        s = Samples.load(a.samples)
        cfg = load(a.config)
        m = cfg["model"]
        params = {k: m[k] for k in RF_KEYS if k in m and m[k] is not None}
        t = feature_reduction(s, s.features, params, a.out,
                              a.block_m if a.block_m is not None else m["spatial_block_m"],
                              n_splits=m["cv_folds"], seed=cfg["seed"], step=a.step,
                              spatial_buffer_m=a.buffer_m if a.buffer_m is not None else m["spatial_buffer_m"],
                              calibration_mode=m["calibration_mode"], calibration_fraction=m["calibration_fraction"])
        print(t[["n_features", "macro_f1", "overall_accuracy"]].to_string(index=False))
    elif c == "parcels":
        import pandas as pd
        import yaml

        from .adapters import read_mapping
        from .parcels import aggregate
        _print(aggregate(a.probabilities, a.parcels, a.out, a.id_field, a.declared_field, a.strategy, a.buffer_px,
                         a.min_pixels, a.min_confidence, a.min_majority_share, a.min_margin,
                         read_mapping(a.crop_mapping) or None, a.conformal, a.alpha, a.aoa,
                         min_aoa_fraction=a.min_aoa_fraction,
                         markers=pd.read_csv(a.markers) if a.markers else None,
                         calendar=yaml.safe_load(Path(a.calendar).read_text()) if a.calendar else None))
    elif c == "parcel-model":
        from .parcels import train_parcel_model
        r = train_parcel_model(a.parcels, a.label_field, a.out, spatial_block_m=a.block_m)
        _print({"parcel_model_macro_f1": r["cross_validation"]["macro_f1"],
                "majority_macro_f1": r["baselines"]["majority"]["macro_f1"],
                "mean_probability_macro_f1": r["baselines"]["mean_probability"]["macro_f1"]})
    elif c == "validate":
        from .adapters import read_mapping
        from .validation import evaluate_points
        r = evaluate_points(a.prediction, a.observations, a.out, a.label_field, a.date_field, a.year, a.training,
                            a.exclusion_m, a.window_px, read_mapping(a.label_mapping) or None)
        _print({"selection": r["selection"], "overall_accuracy": r["accuracy"]["overall_accuracy"],
                "macro_f1": r["accuracy"]["macro_f1"], "bootstrap": r.get("bootstrap")})
    elif c == "area":
        from .metrics import area_adjusted
        from .validation import mapped_area
        areas = mapped_area(a.classes)
        out = {"mapped_area": areas}
        if a.validation:
            acc = json.loads(Path(a.validation).read_text())["accuracy"]
            labels = acc["labels"]
            out["adjusted"] = area_adjusted(acc["confusion_matrix"], labels, {k: areas.get(k, 0.0) for k in labels})
            out["warning"] = "Valid only if observations are a probability sample stratified by map class."
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf8")
        _print(out)
    elif c == "normalize-parcels":
        from .adapters import normalize_parcels, read_mapping
        _print(normalize_parcels(a.input, a.out, a.id_field, a.declared_field, read_mapping(a.crop_mapping),
                                 a.pseudonymise))
    elif c == "segment-obia":
        from .segmentation import obia
        _print(obia(a.raster, a.out, a.bands, a.tile_size, a.segments_per_km2, a.compactness,
                    merge_seams=not a.no_merge_seams, merge_distance=a.merge_distance))
    elif c == "segment-unet-train":
        from .segmentation import unet_train
        r = unet_train(a.images, a.masks, a.model, a.epochs, a.batch_size, groups=a.groups,
                       allow_random_split=a.allow_random_split)
        _print({"best_val_loss": r["best_val_loss"], "epochs": r["epochs"], "split": r["split"]})
    elif c == "segment-patches":
        import numpy as np

        from .segmentation import extract_patches
        X, Y, G = extract_patches(a.raster, a.mask, a.size, block_m=a.block_m, bands=a.bands)
        for suffix, arr in (("_X", X), ("_Y", Y), ("_groups", G)):
            np.save(f"{a.out}{suffix}.npy", arr)
        _print({"patches": int(len(X)), "groups": int(len(set(G)))})
    elif c == "segment-unet-infer":
        from .segmentation import unet_infer
        print(unet_infer(a.raster, a.model, a.out, overlap=a.overlap, bands=a.bands))
    elif c == "boundaries-to-parcels":
        from .segmentation import boundaries_to_parcels
        _print(boundaries_to_parcels(a.raster, a.out, a.threshold))
    elif c == "evaluate-segments":
        from .segmentation_metrics import object_scores
        _print(object_scores(a.segments, a.reference, a.iou, a.out))
    elif c == "evaluate-boundaries":
        from .segmentation_metrics import evaluate_boundary_rasters
        _print(evaluate_boundary_rasters(a.prediction, a.reference, a.out, a.threshold, a.tolerance_px))
    elif c == "ts-features":
        from datetime import date

        from .tsfeatures import build_timeseries_stack
        print(build_timeseries_stack(a.scenes, a.reference, a.out, date.fromisoformat(a.start), date.fromisoformat(a.end),
                                     a.s2_indices, a.s1_indices, a.step_days, a.output_step_days, a.lam, a.envelope,
                                     min_observed_fraction=a.min_observed_fraction, max_gap_days=a.max_gap_days))
    elif c == "merge-stacks":
        from .tsfeatures import merge_stacks
        print(merge_stacks(a.stacks, a.out))
    elif c == "signatures":
        from datetime import date

        from .cbm import parcel_signatures
        t = parcel_signatures(a.scenes, a.parcels, a.reference, a.out, date.fromisoformat(a.start),
                              date.fromisoformat(a.end), a.indices, a.id_field)
        _print({"rows": len(t), "parcels": int(t.parcel_id.nunique()), "out": a.out})
    elif c == "markers":
        from datetime import date

        import pandas as pd

        from .cbm import markers
        sig = pd.read_parquet(a.signatures) if a.signatures.endswith(".parquet") else pd.read_csv(a.signatures)
        m = markers(sig, date.fromisoformat(a.start), date.fromisoformat(a.end), a.index)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(a.out, index=False)
        _print({"parcels": len(m), "with_peak": int(m.peak_doy.notna().sum()), "out": a.out})
    elif c == "variogram":
        from .samples import Samples
        from .spatial import object_table, variogram
        t = object_table(Samples.load(a.samples))
        r = variogram(t.x, t.y, t.pc1, a.max_distance)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))
        _print({k: r[k] for k in ("model", "range_m", "max_distance_m")})
    elif c == "fieldwork":
        from .fieldwork import rank
        r = rank(a.parcels, a.out, a.n, a.min_spacing_m, a.per_class)
        _print({"selected": len(r), "out": a.out})
    elif c == "cloudmask":
        from .clouds import omnicloudmask_scene
        for sdir in a.scenes:
            print(omnicloudmask_scene(sdir))
    elif c == "export":
        from .export import read_hcat, stac_item, to_cog, to_geoparquet
        out = Path(a.out)
        assets = {}
        for name, resampling in (("classes", "nearest"), ("probabilities", "average"), ("uncertainty", "average"),
                                 ("aoa", "nearest")):
            src = Path(a.prediction) / f"{name}.tif"
            if src.exists():
                to_cog(src, out / f"{name}.tif", resampling)
        (out / "class_legend.json").write_text((Path(a.prediction) / "class_legend.json").read_text())
        if a.parcels:
            to_geoparquet(a.parcels, out / "parcels.parquet")
            assets["parcels"] = {"href": "parcels.parquet", "type": "application/vnd.apache.parquet", "roles": ["data"]}
        item = stac_item(out, a.id, a.start, a.end, out / "item.json", read_hcat(a.hcat) if a.hcat else None, assets)
        _print({"stac_item": str(out / "item.json"), "assets": list(item["assets"])})
    elif c == "benchmark-models":
        from . import benchmark
        from .samples import Samples
        s = Samples.load(a.samples)
        feats = None
        if a.groups:
            from .features import load_manifest, select
            feats = select([f for f in load_manifest(a.features) if f.name in s.features], groups=a.groups)
        cands = {m: benchmark.DEFAULT_MODELS[m] for m in a.models}
        t = benchmark.models(s, a.out, feats, cands, a.grids, a.folds, a.block_m, repeats=a.repeats,
                             holdout_fraction=a.holdout, targets=a.targets)
        print(t.to_string(index=False))
    elif c == "benchmark-segmentation":
        from . import benchmark
        t = benchmark.segmentation(a.raster, a.reference, a.out, a.bands, a.unet_model, a.unet_bands, a.tile_size,
                                   a.segments_per_km2)
        print(t.to_string(index=False))
    elif c == "benchmark-public":
        import shutil

        from .public_benchmark import run as run_public
        report = run_public(a.protocol, a.work, a.cache, a.models, a.workers, a.layer)
        if a.publish:
            dest = Path(a.publish)
            dest.mkdir(parents=True, exist_ok=True)
            stem = f"{report['protocol']['id']}.v{report['protocol']['version']}"
            for suffix in ("md", "json"):
                shutil.copyfile(Path(a.work) / f"results.{suffix}", dest / f"{stem}.{suffix}")
        _print({"results": str(Path(a.work) / "results.json"), "protocol_sha256": report["protocol"]["sha256"],
                "models": {k: v["status"] for k, v in report["models"].items()}})
    elif c == "calendar":
        _calendar(a)
    elif c == "manifest":
        from .provenance import run_manifest
        cfg = None
        if a.config:
            from .config import load
            cfg = load(a.config)
        print(run_manifest(a.out, cfg, a.inputs, a.snap_gpt))


if __name__ == "__main__":
    main()
