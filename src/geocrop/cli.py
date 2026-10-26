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

    x = cmd("preflight", "Check whether a configured run can succeed and estimate its cost (no data changed)")
    x.add_argument("--config", required=True)
    x.add_argument("--inventory", nargs="*", default=[], help="catalog/stac-search CSVs to count planned scenes")
    x.add_argument("--out", help="save the JSON report")
    x.add_argument("--markdown", help="save a Markdown report")
    x.add_argument("--strict", action="store_true", help="exit 2 also on warnings (default: only when blocked)")

    x = cmd("clean", "Remove temporary files, quarantined partial outputs and (optionally) regenerable intermediates")
    x.add_argument("--config", required=True)
    x.add_argument("--dry-run", action="store_true", help="list paths and sizes without removing anything")
    x.add_argument("--all-intermediates", action="store_true",
                   help="also composites, feature stacks, samples and hash cache (regenerated on the next run)")
    x.add_argument("--include-results", action="store_true",
                   help="also model/prediction/parcels/cbm/validation/experiments outputs of the run directory")

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

    x = cmd("acceptance", "End-to-end acceptance run on real Sentinel-1/2 data (or --offline synthetic) with a "
                          "pass/fail report")
    x.add_argument("--out", required=True, help="output directory (acceptance_report.json/.md)")
    x.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   default=[19.120, 44.730, 19.145, 44.748], help="degrees; default ~2 x 2 km in Semberija")
    x.add_argument("--start", default="2023-06-01")
    x.add_argument("--end", default="2023-09-30", help="monthly windows between start and end")
    x.add_argument("--offline", action="store_true", help="synthetic scenes, no network (CI)")
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--k", type=int, default=4, help="number of pseudo-classes (k-means)")
    x.add_argument("--no-s1-coregistration", action="store_true",
                   help="keep Sentinel-1 geolocation as delivered (no offset correction to Sentinel-2)")

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
    x.add_argument("--region-field", help="polygon attribute stored as sample region id (leave-one-region-out)")
    x.add_argument("--year-field", help="polygon attribute stored as sample year id (leave-one-year-out)")
    x.add_argument("--year", help="constant year id for all samples of this single-season stack")

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
    x.add_argument("--algorithm", choices=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost", "tempcnn", "ltae"])
    x.add_argument("--no-calibrate", action="store_true", help="skip temperature scaling")
    x.add_argument("--holdout", type=float, default=0.0, help="fraction of spatial blocks locked as final test")
    x.add_argument("--calibration-mode", choices=["oof", "holdout"], default="oof",
                   help="holdout freezes a predictor without fitting on calibration polygons")
    x.add_argument("--calibration-fraction", type=float, default=0.2, help="fraction of spatial groups held out")
    x.add_argument("--cv-strategy", choices=["spatial_blocks", "region", "year", "region_year"],
                   default="spatial_blocks", help="add leave-one-region/year-out transfer validation to the report")

    x = cmd("benchmark", "Compare classifiers on identical spatial folds; internal model selection only")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--algorithms", nargs="+", choices=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost", "tempcnn",
                                                       "ltae"],
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
    x.add_argument("--config", help="use the predict.smoothing section")
    x.add_argument("--smooth", action="store_true", help="edge-aware smoothing of probabilities before argmax")
    x.add_argument("--smooth-window", type=int, help="odd neighbourhood size in pixels (default 7)")
    x.add_argument("--smoothness", type=float, help="smoothing strength; larger smooths more (default 10)")

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
    x.add_argument("--cv-strategy", choices=["spatial_blocks", "region", "year", "region_year"],
                   help="override config cv.strategy: add leave-one-group-out transfer columns")

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

    # parcel-level classification on signatures (gaps G4/G13)
    x = cmd("parcel-classify", "Classify parcels directly from their signatures (parcel-grouped spatial CV, "
                               "calibration, conformal sets, traffic light)")
    x.add_argument("--signatures", required=True, help="signatures .csv/.parquet from geocrop signatures")
    x.add_argument("--parcels", required=True, help="parcel polygons with ids (and reference labels for training)")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help="output directory")
    x.add_argument("--label-field", help="independent reference label column; required unless --model is given")
    x.add_argument("--model", help="existing parcel model directory (skip training)")
    x.add_argument("--id-field", default="parcel_id")
    x.add_argument("--declared-field", default="declared_crop")
    x.add_argument("--indices", nargs="*", help="signature indices to use (default: all in the table)")
    x.add_argument("--stats", nargs="*", default=["mean", "std"], help="signature columns per date")
    x.add_argument("--groups", nargs="*", default=["ts", "stats", "windows", "shape"],
                   help="feature groups for tabular models (temporal models use ts only)")
    x.add_argument("--step-days", type=int, default=5)
    x.add_argument("--output-step-days", type=int, default=15)
    x.add_argument("--window-days", type=int, default=30)
    x.add_argument("--lam", type=float, default=20.0, help="Whittaker smoothing parameter")
    x.add_argument("--algorithm", default="rf", help="any registered estimator, e.g. rf, hist_gb, tempcnn, ltae")
    x.add_argument("--params", type=_json_arg, help="JSON estimator settings")
    x.add_argument("--block-m", type=float, default=5000)
    x.add_argument("--buffer-m", type=float, default=0)
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--holdout", type=float, default=0.0, help="fraction of spatial blocks locked as final test")
    x.add_argument("--calibration-mode", choices=["oof", "holdout"], default="oof")
    x.add_argument("--calibration-fraction", type=float, default=0.2)
    x.add_argument("--bootstrap", type=int, default=200)
    x.add_argument("--alpha", type=float, default=0.1, help="conformal level 1-alpha for prediction sets")
    x.add_argument("--min-pixels", type=int, default=4)
    x.add_argument("--min-confidence", type=float)
    x.add_argument("--min-margin", type=float)
    x.add_argument("--min-aoa-fraction", type=float, default=0.5)
    x.add_argument("--crop-mapping", help="CSV source_code,class_name")
    x.add_argument("--markers", help="markers.csv from geocrop markers")
    x.add_argument("--calendar", help="crop calendar YAML")
    x.add_argument("--compare-pixels", help="pixel samples .npz (polygon = parcel id): compare with pixel->parcel "
                                            "aggregation on the same parcels and folds")
    x.add_argument("--pixel-algorithm", default="rf")
    x.add_argument("--pixel-params", type=_json_arg)

    x = cmd("parcel-series-demo", "Write SYNTHETIC parcel signatures, parcels and pixel samples (software tests)")
    x.add_argument("--out", required=True)
    x.add_argument("--n-side", type=int, default=12, help="parcels per grid side")
    x.add_argument("--seed", type=int, default=11)
    x.add_argument("--pixel-noise", type=float, default=0.08)
    x.add_argument("--cloud-fraction", type=float, default=0.3)

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
    x.add_argument("--temporal-support", action="store_true",
                   help="also write <out>_temporal_support.tif: observed vs interpolated support per pixel")
    x.add_argument("--support-tolerance-days", type=float, default=10,
                   help="a value farther than this from a clear acquisition counts as interpolated")

    x = cmd("merge-stacks", "Concatenate feature stacks on one grid")
    x.add_argument("--stacks", nargs="+", required=True)
    x.add_argument("--out", required=True)

    x = cmd("embeddings", "Foundation-model embeddings (AlphaEarth, TESSERA, on-VM Presto) on the study grid")
    x.add_argument("--config", help="features.embeddings, grid.reference and paths sections")
    x.add_argument("--source", choices=["alphaearth", "tessera", "presto"], help="override features.embeddings.source")
    x.add_argument("--year", type=int, help="override features.embeddings.year")
    x.add_argument("--reference", help="study grid GeoTIFF (default grid.reference)")
    x.add_argument("--out", help="output directory (default <features dir>/embeddings)")
    x.add_argument("--monthly", help="Presto: monthly composites root <YYYY-MM>/B02.tif ... VV.tif")
    x.add_argument("--start", help="Presto: first month YYYY-MM (default <year>-01)")
    x.add_argument("--months", type=int, help="Presto: number of months (default 12)")
    x.add_argument("--parcels", help="Presto: also write parcel-level embeddings for this layer")
    x.add_argument("--id-field", default=None, help="parcel id field (default parcels.id_field)")
    x.add_argument("--checkpoint", help="Presto: local checkpoint instead of the pinned download")
    x.add_argument("--make-monthly", action="store_true",
                   help="Presto: first build calendar-month composites from paths.s2_scenes / paths.s1_scenes")

    x = cmd("signatures", "Parcel-mean time series (CbM signatures) from S2 scene folders")
    x.add_argument("--scenes", required=True)
    x.add_argument("--parcels", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help=".csv or .parquet")
    x.add_argument("--indices", nargs="*", default=["NDVI", "BSI"])
    x.add_argument("--id-field", default="parcel_id")
    x.add_argument("--moments", action="store_true", help="add skewness and bimodality (homogeneity marker)")

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
    x.add_argument("--models", nargs="*", default=["rf", "extra_trees", "hist_gb", "lightgbm", "catboost"],
                   help="model names; tempcnn and ltae need --groups ts (any registered algorithm name works)")
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
    x.add_argument("--methods", type=_json_arg, help='extra registry segmenters as JSON, e.g. '
                   '\'{"ftw": {"segmenter": "ftw", "model": "FTW_PRUE_EFNET_B3_CCBY"}}\'')

    x = cmd("ftw-boundaries", "Pretrained Fields of The World field boundaries (CC-BY checkpoints) -> parcels")
    x.add_argument("--window-a", help="early-season composite/scene folder (B02..B08 reflectance)")
    x.add_argument("--window-b", help="late-season composite/scene folder")
    x.add_argument("--stack", help="existing 8-band FTW stack instead of --window-a/--window-b")
    x.add_argument("--units", choices=["reflectance", "dn"], help="scaling of --stack if it has no GeoCrop tag")
    x.add_argument("--grid", help="study grid raster (default: window A B04 grid)")
    x.add_argument("--out", help="output folder")
    x.add_argument("--model", default="FTW_PRUE_EFNET_B3_CCBY", help="FTW registry name (see --list-models)")
    x.add_argument("--checkpoint", help="local checkpoint file instead of a registry download")
    x.add_argument("--allow-noncommercial", action="store_true",
                   help="permit FULL (non-commercial training data) checkpoints; prints a licence warning")
    x.add_argument("--cache-dir", help="checkpoint cache (default $GEOCROP_FTW_CACHE or ~/.cache/geocrop/ftw)")
    x.add_argument("--no-download", action="store_true", help="fail if the checkpoint is not cached")
    x.add_argument("--download-only", action="store_true")
    x.add_argument("--list-models", action="store_true")
    x.add_argument("--patch", type=int, default=256)
    x.add_argument("--overlap", type=int, default=32)
    x.add_argument("--threads", type=int)
    x.add_argument("--threshold", type=float, default=0.5, help="not-field probability below = field interior")
    x.add_argument("--min-area-pixels", type=int, default=16)
    x.add_argument("--seed-percentile", type=float, default=0)
    x.add_argument("--min-seed-distance", type=float, default=1.0, help="watershed seed distance from boundary (px)")
    x.add_argument("--reference", help="reference parcels: also write object-level segment metrics")
    x.add_argument("--iou", type=float, default=0.5)

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

    # --- active learning (gap G9): select | ingest | simulate ---
    x = cmd("active-learning", "Active-learning loop: batch selection, label ingestion, label-budget simulation")
    al = x.add_subparsers(dest="al_cmd", required=True, metavar="action")
    y = al.add_parser("select", help="Select a diverse, spatially spread batch of uncertain parcels to label",
                      description="Select a diverse, spatially spread batch of uncertain parcels to label")
    y.add_argument("--parcels", required=True, help="output of geocrop parcels (entropy, p_<class>)")
    y.add_argument("--out", required=True, help="selected parcels (.gpkg/.geojson); also writes *_label_form.csv")
    y.add_argument("--budget", type=int, required=True, help="parcels to visit in this round")
    y.add_argument("--features", help="table keyed by parcel id (CSV/Parquet/GPKG) spanning the diversity space")
    y.add_argument("--feature-columns", nargs="*", help="columns of --features (or parcels) to use")
    y.add_argument("--id-field", default="parcel_id")
    y.add_argument("--candidate-factor", type=float, default=3.0, help="top-uncertain pool = factor x budget")
    y.add_argument("--uncertainty-weight", type=float, default=0.3, help="0 = pure diversity, 1 = pure uncertainty")
    y.add_argument("--min-spacing-m", type=float, default=1000)
    y.add_argument("--stratify", choices=["equal", "proportional", "none"], default="equal",
                   help="quota per predicted class")
    y.add_argument("--trips", type=int, help="group the batch into N field trips by area")
    y.add_argument("--training", help="current training polygons: their parcels are excluded and act as centres")
    y.add_argument("--exclude", help="parcels never to propose (e.g. validation sites): ids or geometries")
    y.add_argument("--round", type=int, default=1)
    y.add_argument("--seed", type=int, default=42)
    y = al.add_parser("ingest", help="Merge field observations / analyst decisions into training with provenance",
                      description="Merge field observations / analyst decisions into training with provenance")
    y.add_argument("--labels", required=True, help="CSV/GPKG: parcel_id, label, source, date, confidence[, purpose]")
    y.add_argument("--parcels", required=True, help="parcel geometries with --id-field")
    y.add_argument("--training", required=True, help="current training polygons (polygon_id, class_name)")
    y.add_argument("--out", required=True, help="updated training polygons")
    y.add_argument("--round", type=int, required=True)
    y.add_argument("--validation", help="validation observations kept out of training (points/polygons or ids)")
    y.add_argument("--validation-buffer-m", type=float, default=0.0)
    y.add_argument("--provenance", help="provenance CSV (default <out dir>/al_provenance.csv)")
    y.add_argument("--min-confidence", type=float, default=0.0)
    y.add_argument("--conflict-policy", choices=["exclude", "highest_confidence", "latest", "majority"],
                   default="exclude")
    y.add_argument("--existing-policy", choices=["keep", "replace"], default="keep")
    y.add_argument("--inner-buffer-m", type=float, default=0.0)
    y.add_argument("--id-field", default="parcel_id")
    y.add_argument("--label-field", default="class_name")
    y = al.add_parser("simulate", help="Label-budget curves: uncertainty+diversity vs uncertainty vs random",
                      description="Label-budget curves: uncertainty+diversity vs uncertainty vs random")
    y.add_argument("--out", required=True)
    y.add_argument("--samples", help="samples.npz (aggregated to one unit per polygon); default SYNTHETIC pool")
    y.add_argument("--strategies", nargs="*", default=["uncertainty_diversity", "uncertainty", "random"])
    y.add_argument("--initial", type=int, default=12)
    y.add_argument("--batch", type=int, default=12)
    y.add_argument("--rounds", type=int, default=8)
    y.add_argument("--repeats", type=int, default=5)
    y.add_argument("--seed", type=int, default=42)
    y.add_argument("--block-m", type=float, default=1000)
    y.add_argument("--test-fraction", type=float, default=0.25)
    y.add_argument("--test-buffer-m", type=float, default=0.0)
    y.add_argument("--candidate-factor", type=float, default=3.0)
    y.add_argument("--uncertainty-weight", type=float, default=0.3)
    y.add_argument("--min-spacing-m", type=float, default=0.0)
    y.add_argument("--stratify", choices=["equal", "proportional", "none"], default="none")
    y.add_argument("--params", type=_json_arg, help='classifier JSON, e.g. {"algorithm": "rf", "n_estimators": 200}')
    y.add_argument("--synthetic-side", type=int, default=24, help="SYNTHETIC pool: parcels per side")
    y.add_argument("--synthetic-noise", type=float, default=0.10)
    y.add_argument("--no-figure", action="store_true")

    # --- CbM event markers (Sentinel-1 + Sentinel-2) --------------------------------------------------
    x = cmd("s1-signatures", "Parcel-mean Sentinel-1 VV/VH/VH-VV (dB) per acquisition with orbit geometry")
    x.add_argument("--scenes", required=True, help="root of S1 scene folders (VV.tif, VH.tif, scene.json)")
    x.add_argument("--parcels", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help=".csv or .parquet")
    x.add_argument("--id-field", default="parcel_id")
    x = cmd("event-markers", "Harvest, tillage, mowing, homogeneity and data-sufficiency markers (S2 + S1)")
    x.add_argument("--signatures", help="S2 signatures (geocrop signatures, ideally with --moments)")
    x.add_argument("--s1-signatures", help="S1 signatures (geocrop s1-signatures)")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--config", help="use the cbm.events section of a config")
    x = cmd("traffic-light", "Re-evaluate the traffic light of a parcels layer with marker rules and reason codes")
    x.add_argument("--parcels", required=True, help="parcels.gpkg from geocrop parcels")
    x.add_argument("--markers", required=True, help="event markers CSV from geocrop event-markers")
    x.add_argument("--config", help="cbm.rules section (applied even when rules.enabled is false)")
    x.add_argument("--rules", type=_json_arg, help="JSON rules overriding the config, e.g. "
                   "'{\"harvest_expected_classes\": [\"maize\"]}'")
    x.add_argument("--out", required=True, help=".gpkg (a .csv is written alongside)")
    x.add_argument("--id-field", default="parcel_id")

    # --- transfer validation and probability smoothing (G8, G15) ---
    x = cmd("samples-merge", "Combine per-season sample files into one multi-year sample store")
    x.add_argument("--inputs", nargs="+", required=True, help="samples .npz files, one per season")
    x.add_argument("--years", nargs="*", help="year id per input (default: year metadata stored in each file)")
    x.add_argument("--stacks", nargs="*", help="feature stack per input; their feature manifests must match")
    x.add_argument("--out", required=True)

    x = cmd("transfer", "Leave-one-region-out / leave-one-year-out validation vs spatial CV")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--strategies", nargs="+", choices=["region", "year", "region_year"], default=["region", "year"])
    x.add_argument("--config", help="inherit model settings")
    x.add_argument("--params", type=_json_arg, help="JSON estimator settings")
    x.add_argument("--block-m", type=float)
    x.add_argument("--buffer-m", type=float)
    x.add_argument("--folds", type=int)
    x.add_argument("--seed", type=int)

    x = cmd("smoothing-experiment", "SYNTHETIC comparison of accuracy and conformal coverage with/without smoothing")
    x.add_argument("--out", required=True)
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--alpha", type=float, default=0.1)

    # optional openEO back-end (CDSE); local processing stays the default
    openeo_products = ["s2_composites", "s1_composites", "s2_scenes", "signatures"]
    x = cmd("openeo-extract", "Run composites, scenes or parcel signatures on an openEO back-end (CDSE) and write "
                              "them in GeoCrop's file contracts (optional openeo extra)")
    x.add_argument("--config", required=True)
    x.add_argument("--products", nargs="+", choices=openeo_products, help="default: openeo.products of the config")
    x.add_argument("--dry-run", action="store_true", help="only write the process graphs; needs no credentials")
    x.add_argument("--graphs", help="directory for the process graph JSON; default <openeo.work_dir>/graphs")
    x.add_argument("--auth", choices=["auto", "client-credentials", "device"], default="auto",
                   help="auto: CDSE_OPENEO_CLIENT_ID/SECRET if set, else the OIDC device flow")
    x = cmd("openeo-graph", "Export the configured GeoCrop recipe as openEO process graphs (no credentials needed)")
    x.add_argument("--config", required=True)
    x.add_argument("--products", nargs="+", choices=openeo_products, help="default: openeo.products of the config")
    x.add_argument("--out", help="output directory; default <openeo.work_dir>/graphs")
    x.add_argument("--stdout", action="store_true", help="print the openEO process objects as one JSON document")

    # --- temporal quality and parcel evidence packages (observed vs interpolated, audit trail) ---
    x = cmd("temporal-support", "Per-parcel temporal support (observed vs interpolated) from signatures; "
                                "optionally route a parcels layer with cbm.temporal_support rules")
    x.add_argument("--signatures", required=True, help="S2 signatures CSV/Parquet (geocrop signatures)")
    x.add_argument("--start", required=True)
    x.add_argument("--end", required=True)
    x.add_argument("--out", required=True, help="per-parcel support CSV (tq_* columns)")
    x.add_argument("--markers", help="markers CSV (adds the harvest-date distance)")
    x.add_argument("--config", help="use the cbm.temporal_support (and cbm.rules) sections of a config")
    x.add_argument("--params", type=_json_arg, help='JSON overrides of cbm.temporal_support, e.g. '
                   '\'{"tolerance_days": 8, "max_gap_at_peak_days": 25}\'')
    x.add_argument("--parcels", help="parcels.gpkg to re-evaluate (rules applied even when enabled is false)")
    x.add_argument("--routed", help="output .gpkg of the re-evaluated parcels layer (with --parcels)")
    x.add_argument("--id-field", default="parcel_id")
    x = cmd("evidence", "Per-parcel evidence packages (JSON + HTML + PNG) with every rule, data source and "
                        "model reference, plus a batch index")
    x.add_argument("--parcels", required=True, help="parcel decision table (.gpkg/.csv from parcels or "
                                                    "parcel-classify)")
    x.add_argument("--out", required=True, help="output directory")
    x.add_argument("--ids", nargs="*", help="parcel ids to package (default: all)")
    x.add_argument("--signatures", help="S2 signatures CSV/Parquet: series, observed/rejected acquisitions")
    x.add_argument("--start", help="season start (default: timeseries.start of --config)")
    x.add_argument("--end", help="season end (default: timeseries.end of --config)")
    x.add_argument("--markers", help="markers or event-markers CSV")
    x.add_argument("--temporal-support", help="per-parcel support CSV (default: computed from --signatures)")
    x.add_argument("--config", help="cbm.rules and cbm.temporal_support settings (applied when enabled)")
    x.add_argument("--marker-rules", type=_json_arg, help="JSON cbm.rules overrides")
    x.add_argument("--temporal-rules", type=_json_arg, help="JSON cbm.temporal_support overrides")
    x.add_argument("--model", help="model directory or model.joblib (sha256 recorded)")
    x.add_argument("--run-manifest", help="run_manifest.json to reference (geocrop manifest)")
    x.add_argument("--scenes", help="scene root: resolve acquisition dates to scene ids")
    x.add_argument("--summary", help="parcels summary JSON (default <parcels>.summary.json)")
    x.add_argument("--id-field", help="parcel id column (default parcels.id_field of --config or parcel_id)")
    x.add_argument("--no-png", action="store_true", help="skip the PNG chart (HTML keeps an inline SVG chart)")
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


def _ftw_boundaries(a):
    from . import ftw
    if a.list_models:
        _print(ftw.model_table())
        return
    try:
        checkpoint = a.checkpoint
        if checkpoint is None:
            model = ftw.check_licence(a.model, a.allow_noncommercial)
            if a.no_download:
                checkpoint = ftw.cached_checkpoint(a.model, a.cache_dir, a.allow_noncommercial)
            else:
                from .ftw_weights import download
                checkpoint = download(a.model, a.cache_dir, a.allow_noncommercial)   # prints the NC warning
            print(f"FTW model {model.name} ({model.licence}); cite: {ftw.CITATION}", file=sys.stderr)
        if a.download_only:
            _print({"model": a.model, "checkpoint": str(checkpoint)})
            return
        if not a.out or not (a.stack or (a.window_a and a.window_b)):
            raise SystemExit("ftw-boundaries needs --out and either --stack or both --window-a and --window-b")
        out = Path(a.out)
        raster = a.stack or a.grid or str(Path(a.window_a) / "B04.tif")
        info = ftw.segment(raster, out / "ftw_parcels.gpkg", checkpoint=checkpoint,
                           window_a=None if a.stack else a.window_a, window_b=None if a.stack else a.window_b,
                           allow_noncommercial=a.allow_noncommercial, patch=a.patch, overlap=a.overlap,
                           units=a.units, threads=a.threads, threshold=a.threshold,
                           min_area_pixels=a.min_area_pixels, seed_percentile=a.seed_percentile,
                           min_seed_distance=a.min_seed_distance, work_dir=out)
    except ftw.LicenceError as exc:
        raise SystemExit(str(exc)) from None
    if info["licence"] != ftw.CC_BY:
        print(ftw.NONCOMMERCIAL_WARNING.format(name=info["model"]) if info["licence"] in
              {m.licence for m in ftw.MODELS.values()} else f"Licence of checkpoint {info['model']} is unknown; "
              "you are responsible for checking its terms.", file=sys.stderr)
    if a.reference:
        from .segmentation_metrics import object_scores
        info["segment_metrics"] = object_scores(info["path"], a.reference, a.iou, out / "segment_metrics.json")
    _print(info)


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
    elif c == "preflight":
        from .config import load
        from .preflight import run as preflight_run
        from .preflight import to_markdown
        report = preflight_run(load(a.config), a.config, a.inventory)
        for target, text in ((a.out, lambda: json.dumps(report, indent=2, ensure_ascii=False, default=str)),
                             (a.markdown, lambda: to_markdown(report))):
            if target:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text(text(), encoding="utf8")
        print(to_markdown(report))
        if report["verdict"] == "blocked" or (a.strict and report["verdict"] != "ready"):
            sys.exit(2)
    elif c == "clean":
        from .cleanup import clean
        from .config import load
        _print(clean(load(a.config), dry_run=a.dry_run, all_intermediates=a.all_intermediates,
                     include_results=a.include_results, config_path=a.config))
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
    elif c == "acceptance":
        from .acceptance import run as acceptance_run
        report = acceptance_run(a.out, a.bbox, a.start, a.end, offline=a.offline, seed=a.seed, k=a.k,
                                coregister_s1=not a.no_s1_coregistration, log=lambda m: print(m, file=sys.stderr))
        _print({"verdict": report["verdict"], "checks": report["summary"], "runtime_s": report["runtime_s"],
                "error": report.get("error"), "report": str(Path(a.out) / "acceptance_report.md")})
        if report["verdict"] != "PASS":
            sys.exit(1)
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
        out = build_stack(a.composites, a.out, **opts)
        if a.config:
            from .config import embedding_options
            eo = embedding_options(cfg)
            if eo and eo["attach"]:
                from .embeddings import output_path
                from .features import attach_embeddings
                raster = output_path(eo["dir"], eo["source"], eo["year"])
                if not raster.exists():
                    raise FileNotFoundError(f"{raster} missing; run geocrop embeddings --config {a.config} first")
                attach_embeddings(out, raster)
        print(out)
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
        s = extract(a.features, a.polygons, a.label_field, a.id_field, a.max_per_polygon, a.inner_buffer_m,
                    region_field=a.region_field, year_field=a.year_field, year=a.year)
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
                  calibration_fraction=a.calibration_fraction, cv_strategy=a.cv_strategy)
        _print({k: r["cross_validation"][k] for k in ("overall_accuracy", "kappa", "macro_f1", "fold_macro_f1_std")})
        if "transfer_validation" in r:
            _print({"transfer_gap": r["transfer_validation"]["transfer_gap"],
                    "out_of_domain": {k: r["transfer_validation"]["out_of_domain"][k]
                                      for k in ("overall_accuracy", "kappa", "macro_f1")}})
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
        smoothing = None
        if a.config:
            from .config import load
            smoothing = dict(load(a.config)["predict"]["smoothing"])
        if a.smooth or a.smooth_window is not None or a.smoothness is not None:
            smoothing = {**(smoothing or {}), "enabled": True}
            if a.smooth_window is not None:
                smoothing["window"] = a.smooth_window
            if a.smoothness is not None:
                smoothing["smoothness"] = a.smoothness
        _print(predict(a.features, a.model, a.out, smoothing=smoothing))
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
                            calibration_fraction=m["calibration_fraction"],
                            cv_strategy=a.cv_strategy or cfg["cv"]["strategy"])
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
    elif c in ("parcel-classify", "parcel-series-demo"):
        _parcel_classify(a)
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
                                     min_observed_fraction=a.min_observed_fraction, max_gap_days=a.max_gap_days,
                                     temporal_support=a.temporal_support,
                                     support_tolerance_days=a.support_tolerance_days))
    elif c == "merge-stacks":
        from .tsfeatures import merge_stacks
        print(merge_stacks(a.stacks, a.out))
    elif c == "embeddings":
        from .config import ConfigError, embedding_options, load
        from .embedding_sources import extract
        cfg = load(a.config)
        eo = embedding_options(cfg, always=True)
        eo.update({k: v for k, v in (("source", a.source), ("year", a.year), ("dir", a.out), ("monthly", a.monthly),
                                     ("start", a.start), ("months", a.months), ("checkpoint", a.checkpoint))
                   if v is not None})
        if not eo["source"] or not eo["year"]:
            raise ConfigError("Set features.embeddings.source/year in the config or pass --source and --year")
        reference = a.reference or cfg["grid"].get("reference")
        if not reference:
            raise ConfigError("A study grid is needed: grid.reference in the config or --reference")
        monthly = eo["monthly"] or str(Path(eo["dir"]) / "monthly")
        if eo["source"] == "presto" and a.make_monthly:
            from .embeddings import monthly_composites
            comp = cfg["compositing"]
            monthly_composites(reference, monthly, eo["start"] or f"{eo['year']}-01", eo["months"],
                               cfg["paths"]["s2_scenes"], cfg["paths"]["s1_scenes"], statistic=comp["statistic"],
                               scl_invalid=comp["scl_invalid"], cloud_probability_max=comp["cloud_probability_max"],
                               mask_dilation_px=comp["mask_dilation_px"])
        parcels = a.parcels or (cfg["paths"]["parcels"] if eo["parcels"] else None)
        _print(extract(eo["source"], reference, eo["dir"], eo["year"], monthly=monthly, start=eo["start"],
                       months=eo["months"], parcels=parcels, id_field=a.id_field or cfg["parcels"]["id_field"],
                       checkpoint=eo["checkpoint"], mirror=eo["mirror"], index=eo["index"],
                       resampling=eo["resampling"]))
    elif c == "signatures":
        from datetime import date

        from .cbm import parcel_signatures
        t = parcel_signatures(a.scenes, a.parcels, a.reference, a.out, date.fromisoformat(a.start),
                              date.fromisoformat(a.end), a.indices, a.id_field, moments=a.moments)
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
        cands = {m: benchmark.candidate(m) for m in a.models}
        t = benchmark.models(s, a.out, feats, cands, a.grids, a.folds, a.block_m, repeats=a.repeats,
                             holdout_fraction=a.holdout, targets=a.targets)
        print(t.to_string(index=False))
    elif c == "benchmark-segmentation":
        from . import benchmark
        t = benchmark.segmentation(a.raster, a.reference, a.out, a.bands, a.unet_model, a.unet_bands, a.tile_size,
                                   a.segments_per_km2, methods=a.methods)
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
    elif c == "ftw-boundaries":
        _ftw_boundaries(a)
    elif c == "manifest":
        from .provenance import run_manifest
        cfg = None
        if a.config:
            from .config import load
            cfg = load(a.config)
        print(run_manifest(a.out, cfg, a.inputs, a.snap_gpt))
    elif c == "active-learning":
        from . import active_learning as al
        if a.al_cmd == "select":
            labelled, excluded = (), ()
            if a.training or a.exclude:
                import geopandas as gpd
                par = gpd.read_file(a.parcels)
                par[a.id_field] = par[a.id_field].astype(str)
                if a.training:
                    tr = gpd.read_file(a.training)
                    labelled = tuple(al.training_parcel_ids(tr, par, a.id_field).dropna())
                if a.exclude:
                    excluded = tuple(al.validation_parcel_ids(a.exclude, par, a.id_field, 0.0))
            _, summary = al.select(a.parcels, a.budget, a.out, features=a.features, feature_columns=a.feature_columns,
                                   id_field=a.id_field, candidate_factor=a.candidate_factor,
                                   uncertainty_weight=a.uncertainty_weight, min_spacing_m=a.min_spacing_m,
                                   stratify=a.stratify, trips=a.trips, labelled_ids=labelled, exclude_ids=excluded,
                                   round_id=a.round, seed=a.seed)
            _print(summary)
        elif a.al_cmd == "ingest":
            _print(al.ingest(a.labels, a.parcels, a.training, a.out, a.round, validation=a.validation,
                             provenance=a.provenance, id_field=a.id_field, label_field=a.label_field,
                             min_confidence=a.min_confidence, conflict_policy=a.conflict_policy,
                             existing_policy=a.existing_policy, validation_buffer_m=a.validation_buffer_m,
                             inner_buffer_m=a.inner_buffer_m))
        else:
            if a.samples:
                from .samples import Samples
                pool = al.pool_from_samples(Samples.load(a.samples))
            else:
                pool = al.synthetic_pool(a.synthetic_side, noise=a.synthetic_noise, seed=a.seed)
            r = al.simulate(pool, a.out, strategies=a.strategies, initial=a.initial, batch=a.batch, rounds=a.rounds,
                            repeats=a.repeats, seed=a.seed, params=a.params, block_m=a.block_m,
                            test_fraction=a.test_fraction, test_buffer_m=a.test_buffer_m,
                            candidate_factor=a.candidate_factor, uncertainty_weight=a.uncertainty_weight,
                            min_spacing_m=a.min_spacing_m, stratify=a.stratify, figure=not a.no_figure)
            _print({k: r[k] for k in ("mode", "test_units", "pool_units", "comparison", "outputs")})

    elif c == "s1-signatures":
        from datetime import date

        from .cbm import s1_signatures
        t = s1_signatures(a.scenes, a.parcels, a.reference, a.out, date.fromisoformat(a.start),
                          date.fromisoformat(a.end), a.id_field)
        _print({"rows": len(t), "parcels": int(t.parcel_id.nunique()), "acquisitions": int(t.date.nunique()),
                "out": a.out})
    elif c == "event-markers":
        from datetime import date

        import pandas as pd

        from .cbm_markers import event_markers

        def _table(path):
            return None if not path else pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if not (a.signatures or a.s1_signatures):
            raise SystemExit("event-markers needs --signatures and/or --s1-signatures")
        params = None
        if a.config:
            from .config import load
            params = load(a.config)["cbm"]["events"]
        m = event_markers(_table(a.signatures), date.fromisoformat(a.start), date.fromisoformat(a.end),
                          _table(a.s1_signatures), params)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(a.out, index=False)
        _print({"parcels": len(m), "harvest_detected": int(m.harvest_detected.sum()),
                "tillage_detected": int(m.tillage_detected.sum()), "mowing_events": int(m.mowing_events.sum()),
                "heterogeneous": int((m.homogeneity == "heterogeneous").sum()),
                "data_sufficiency": m.data_sufficiency.value_counts().to_dict(), "out": a.out})
    elif c == "traffic-light":
        from collections import Counter

        import geopandas as gpd
        import pandas as pd

        from .cbm import traffic_light_details
        from .cbm_markers import EVENT_COLUMNS
        rules = {}
        if a.config:
            from .config import load
            rules = load(a.config)["cbm"]["rules"]
        rules = {**rules, **(a.rules or {}), "enabled": True}
        gdf = gpd.read_file(a.parcels)
        mk = pd.read_csv(a.markers, dtype={"parcel_id": str}).set_index("parcel_id")
        mk = mk.reindex(gdf[a.id_field].astype(str))
        for col in EVENT_COLUMNS:
            if col in mk:
                gdf[col] = mk[col].to_numpy()
        details = traffic_light_details(gdf, rules)
        gdf["traffic_light"] = details["traffic_light"].to_numpy()
        gdf["traffic_light_reasons"] = details["traffic_light_reasons"].to_numpy()
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        gdf.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
        gdf.drop(columns=gdf.geometry.name).to_csv(out.with_suffix(".csv"), index=False)
        _print({"traffic_light_counts": gdf.traffic_light.value_counts().to_dict(),
                "reason_counts": dict(sorted(Counter(code for codes in gdf.traffic_light_reasons
                                                     for code in codes.split(";") if code).items())),
                "out": str(out)})

    elif c == "samples-merge":
        from .samples import Samples, combine
        manifests = None
        if a.stacks:
            from .features import load_manifest
            manifests = [load_manifest(st) for st in a.stacks]
        merged = combine([Samples.load(f) for f in a.inputs], years=a.years or None, manifests=manifests)
        merged.save(a.out)
        _print({"out": a.out, "samples": int(len(merged.y)), "years": sorted(set(merged.year)),
                "regions": None if merged.region is None else sorted(set(merged.region))})
    elif c == "transfer":
        from . import experiments
        from .config import load
        from .model import RF_KEYS
        from .samples import Samples
        cfg = load(a.config)
        m = cfg["model"]
        params = {k: m[k] for k in RF_KEYS if k in m and m[k] is not None}
        params.update(a.params or {})
        t = experiments.transfer(Samples.load(a.samples), a.out, params=params, strategies=a.strategies,
                                 n_splits=a.folds if a.folds is not None else m["cv_folds"],
                                 spatial_block_m=a.block_m if a.block_m is not None else m["spatial_block_m"],
                                 seed=a.seed if a.seed is not None else cfg["seed"],
                                 spatial_buffer_m=a.buffer_m if a.buffer_m is not None else m["spatial_buffer_m"])
        print(t.to_string(index=False))
    elif c == "smoothing-experiment":
        import pandas as pd

        from .experiments import smoothing_comparison
        r = smoothing_comparison(a.out, seed=a.seed, alpha=a.alpha)
        print(pd.DataFrame(r["rows"]).to_string(index=False))
    elif c in ("temporal-support", "evidence"):
        _evidence_cli(a)
    elif c in ("openeo-extract", "openeo-graph"):
        from . import openeo_adapter
        from .config import load
        cfg = load(a.config)
        recipe = openeo_adapter.build_recipe(cfg, a.products)
        graphs = (a.graphs if c == "openeo-extract" else a.out) or Path(cfg["openeo"]["work_dir"]) / "graphs"
        written = openeo_adapter.write_recipe(recipe, graphs)
        if c == "openeo-graph" and a.stdout:
            _print({i["name"]: i["process"] for i in recipe["graphs"]})
        elif c == "openeo-graph" or a.dry_run:
            _print({**written, "dry_run": c == "openeo-extract", "differences": recipe["differences"]})
        else:
            conn = openeo_adapter.connect(cfg["openeo"]["url"], a.auth)
            _print({**written, "results": openeo_adapter.execute(recipe, conn, cfg["openeo"]["work_dir"],
                                                                 cfg["openeo"]["job_options"])})


def _evidence_cli(a):
    """Dispatch of `temporal-support` and `evidence` (temporal quality, parcel audit trail)."""
    from datetime import date

    import pandas as pd

    cfg = None
    if a.config:
        from .config import load
        cfg = load(a.config)
    if a.cmd == "temporal-support":
        from collections import Counter

        from . import temporal_quality as tq
        params = {**(cfg["cbm"]["temporal_support"] if cfg else {}), **(a.params or {})}
        sig = pd.read_parquet(a.signatures) if a.signatures.endswith(".parquet") else \
            pd.read_csv(a.signatures, dtype={"parcel_id": str})
        markers = pd.read_csv(a.markers, dtype={"parcel_id": str}) if a.markers else None
        sup = tq.parcel_support(sig, date.fromisoformat(a.start), date.fromisoformat(a.end), params, markers)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        sup.to_csv(a.out, index=False)
        result = {"parcels": len(sup), "out": a.out,
                  "median_clear_observations": float(sup.tq_clear_observations.median()) if len(sup) else None,
                  "parcels_with_interpolated_phenology": int((sup.tq_interpolated_phenology > 0).sum())}
        if a.parcels:
            import geopandas as gpd

            from .cbm import traffic_light_details
            if not a.routed:
                raise SystemExit("--parcels needs --routed (output layer)")
            gdf = gpd.read_file(a.parcels)
            gdf = tq.attach(gdf, sup, a.id_field)
            mrules = cfg["cbm"]["rules"] if cfg and cfg["cbm"]["rules"]["enabled"] else None
            details = traffic_light_details(gdf, mrules, {**params, "enabled": True})
            gdf["traffic_light"] = details["traffic_light"].to_numpy()
            gdf["traffic_light_reasons"] = details["traffic_light_reasons"].to_numpy()
            out = Path(a.routed)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.unlink(missing_ok=True)
            gdf.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
            gdf.drop(columns=gdf.geometry.name).to_csv(out.with_suffix(".csv"), index=False)
            result.update({"routed": str(out), "traffic_light_counts": gdf.traffic_light.value_counts().to_dict(),
                           "reason_counts": dict(sorted(Counter(code for codes in gdf.traffic_light_reasons
                                                                for code in codes.split(";") if code).items()))})
        _print(result)
        return
    from . import evidence
    t = cfg["timeseries"] if cfg else {}
    start = a.start or (str(t["start"]) if t.get("start") else None)
    end = a.end or (str(t["end"]) if t.get("end") else None)
    mrules = {**(cfg["cbm"]["rules"] if cfg else {}), **(a.marker_rules or {})}
    trules = {**(cfg["cbm"]["temporal_support"] if cfg else {}), **(a.temporal_rules or {})}
    id_field = a.id_field or (cfg["parcels"]["id_field"] if cfg else "parcel_id")
    _print(evidence.build(
        a.parcels, a.out, a.ids, a.signatures, date.fromisoformat(start) if start else None,
        date.fromisoformat(end) if end else None, id_field, a.markers, a.temporal_support, mrules or None,
        trules or None, a.model, a.run_manifest, a.scenes, a.summary, png=not a.no_png))


def _parcel_classify(a):
    """Dispatch of `parcel-classify` and `parcel-series-demo` (gaps G4/G13)."""
    from datetime import date

    if a.cmd == "parcel-series-demo":
        from .demo import write_parcel_series
        _print({"mode": "SYNTHETIC ONLY - software verification, not scientific accuracy",
                **write_parcel_series(a.out, n_side=a.n_side, seed=a.seed, pixel_noise=a.pixel_noise,
                                      cloud_fraction=a.cloud_fraction)})
        return
    import geopandas as gpd
    import pandas as pd
    import yaml

    from . import parcel_classify as pc
    from .adapters import read_mapping

    if not a.model and not a.label_field:
        raise SystemExit("parcel-classify needs --label-field (training) or --model (prediction only)")
    out = Path(a.out)
    sig = pd.read_parquet(a.signatures) if a.signatures.endswith(".parquet") else pd.read_csv(a.signatures)
    parcels = gpd.read_file(a.parcels)
    frame = pc.parcel_features(sig, parcels, date.fromisoformat(a.start), date.fromisoformat(a.end), a.id_field,
                               a.indices, a.stats, a.groups, a.step_days, a.output_step_days, a.window_days, a.lam)
    params = {**pc.DEFAULT_PARAMS, "algorithm": a.algorithm} if a.algorithm in ("rf", "extra_trees") else \
        {"algorithm": a.algorithm}
    params.update(a.params or {})
    result = {}
    model_dir = Path(a.model) if a.model else out / "model"
    if not a.model:
        rep = pc.train(frame, parcels, a.label_field, model_dir, params, a.groups, a.id_field, a.folds, a.block_m,
                       a.seed, a.holdout, a.calibration_mode, a.calibration_fraction, a.buffer_m, a.bootstrap)
        result["cross_validation"] = {k: rep["cross_validation"][k] for k in ("overall_accuracy", "macro_f1", "kappa")}
        if "holdout" in rep:
            result["holdout"] = {k: rep["holdout"][k] for k in ("overall_accuracy", "macro_f1", "kappa")}
        result["parcels_trained"] = rep["samples"]
    result["prediction"] = pc.predict(
        model_dir, frame, parcels, out / "parcels.gpkg", a.id_field, a.declared_field, a.alpha, a.min_pixels,
        a.min_confidence, a.min_margin, a.min_aoa_fraction, read_mapping(a.crop_mapping) or None if a.crop_mapping
        else None, pd.read_csv(a.markers) if a.markers else None,
        yaml.safe_load(Path(a.calendar).read_text()) if a.calendar else None)
    if a.compare_pixels:
        from .samples import Samples
        if not a.label_field:
            raise SystemExit("--compare-pixels needs --label-field")
        cols = pc._resolve_columns(frame, params, a.groups)
        parcel_samples, _ = pc.to_samples(frame, parcels, a.label_field, cols, a.id_field)
        pixel_params = {**pc.DEFAULT_PARAMS, "algorithm": a.pixel_algorithm} \
            if a.pixel_algorithm in ("rf", "extra_trees") else {"algorithm": a.pixel_algorithm}
        pixel_params.update(a.pixel_params or {})
        cmp = pc.compare_pixel_aggregation(parcel_samples, Samples.load(a.compare_pixels), out / "comparison",
                                           params, pixel_params, a.folds, a.block_m, a.seed, a.holdout)
        result["comparison"] = {m: {"overall_accuracy": r["overall_accuracy"], "macro_f1": r["macro_f1"]}
                                for m, r in cmp["cross_validation"].items()}
        result["comparison"]["fold_fingerprint"] = cmp["fold_fingerprint"]
    _print(result)


if __name__ == "__main__":
    main()
