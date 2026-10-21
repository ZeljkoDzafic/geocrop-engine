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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geocrop", description="GeoCrop Engine: open Sentinel-1/2 crop classification "
                                "and parcel-level declaration verification.")
    p.add_argument("--version", action="version", version=f"geocrop-engine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def cmd(name, help_):
        return sub.add_parser(name, help=help_, description=help_)

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
    x.add_argument("--folds", type=int, default=5)
    x.add_argument("--params", type=_json_arg, help='JSON, e.g. {"n_estimators": 500}')
    x.add_argument("--tune", type=_json_arg, help='JSON grid, e.g. {"max_features": ["sqrt", 0.3]}')
    x.add_argument("--groups", nargs="*", help="restrict to feature groups")
    x.add_argument("--manifest", help="feature stack (for --groups)")

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
    x.add_argument("--block-m", type=float, default=5000)
    x.add_argument("--folds", type=int, default=5)

    x = cmd("reduce-features", "Recursive feature elimination curve under spatial CV")
    x.add_argument("--samples", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--block-m", type=float, default=5000)
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

    x = cmd("segment-unet-train", "Train the boundary U-Net (requires torch)")
    x.add_argument("--images", required=True)
    x.add_argument("--masks", required=True)
    x.add_argument("--model", required=True)
    x.add_argument("--epochs", type=int, default=30)
    x.add_argument("--batch-size", type=int, default=8)

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

    x = cmd("manifest", "Write a reproducibility manifest (versions, git, hashes)")
    x.add_argument("--out", required=True)
    x.add_argument("--config")
    x.add_argument("--inputs", nargs="*", default=[])
    x.add_argument("--snap-gpt")
    return p


def main(argv=None):  # noqa: C901 - flat dispatch is the clearest form here
    a = build_parser().parse_args(argv)
    c = a.cmd
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
        print(import_safe(a.safe, a.out))
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
        r = train(s, a.out, feats, a.params, a.folds, a.block_m, tune_grid=a.tune)
        _print({k: r["cross_validation"][k] for k in ("overall_accuracy", "kappa", "macro_f1", "fold_macro_f1_std")})
    elif c == "predict":
        from .model import predict
        _print(predict(a.features, a.model, a.out))
    elif c == "ablation":
        from . import experiments
        from .features import load_manifest
        from .samples import Samples
        exps, params = None, None
        if a.config:
            from .config import load
            cfg = load(a.config)
            exps = cfg["experiments"] or None
            m = cfg["model"]
            params = {k: m[k] for k in ("n_estimators", "max_features", "min_samples_leaf", "class_weight", "n_jobs")}
        t = experiments.run(Samples.load(a.samples), load_manifest(a.features), a.out, exps, params, a.folds,
                            a.block_m, targets=a.targets)
        print(t.to_string(index=False))
    elif c == "reduce-features":
        from .experiments import feature_reduction
        from .samples import Samples
        s = Samples.load(a.samples)
        t = feature_reduction(s, s.features, {"n_estimators": 300, "class_weight": "balanced_subsample", "n_jobs": -1},
                              a.out, a.block_m, step=a.step)
        print(t[["n_features", "macro_f1", "overall_accuracy"]].to_string(index=False))
    elif c == "parcels":
        from .adapters import read_mapping
        from .parcels import aggregate
        _print(aggregate(a.probabilities, a.parcels, a.out, a.id_field, a.declared_field, a.strategy, a.buffer_px,
                         a.min_pixels, a.min_confidence, a.min_majority_share, a.min_margin,
                         read_mapping(a.crop_mapping) or None))
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
        _print(obia(a.raster, a.out, a.bands, a.tile_size, a.segments_per_km2, a.compactness))
    elif c == "segment-unet-train":
        from .segmentation import unet_train
        r = unet_train(a.images, a.masks, a.model, a.epochs, a.batch_size)
        _print({"best_val_loss": r["best_val_loss"], "epochs": r["epochs"]})
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
    elif c == "manifest":
        from .provenance import run_manifest
        cfg = None
        if a.config:
            from .config import load
            cfg = load(a.config)
        print(run_manifest(a.out, cfg, a.inputs, a.snap_gpt))


if __name__ == "__main__":
    main()
