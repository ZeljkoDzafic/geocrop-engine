"""Read-only study input audit: schema, geometry, spatial leakage and stack compatibility.

Unlike the per-file checks in :mod:`geocrop.provenance`, this report checks relationships
between reference data and the feature stack. Counts describe the supplied layers, not
scientific accuracy. Raster checks inspect metadata and footprint, not every pixel.
"""
from __future__ import annotations

import math

import geopandas as gpd
import rasterio
from pyproj import CRS
from shapely.geometry import Polygon


def _metric(crs) -> bool:
    c = CRS.from_user_input(crs)
    return c.is_projected and bool(c.axis_info) and all(
        math.isclose(a.unit_conversion_factor, 1.0) for a in c.axis_info[:2])


def _missing(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def audit_inputs(training=None, parcels=None, observations=None, stack=None,
                 label_field: str = "class_name", id_field: str = "polygon_id",
                 parcel_id_field: str = "parcel_id", observation_label_field: str | None = None,
                 training_exclusion_m: float = 0.0, distance_crs=None) -> dict:
    """Audit supplied paths/GeoDataFrames without repairing or changing source data.

    Errors fail ``ok``; warnings identify limits that need review. Overlapping training
    polygons with different labels are errors; same-label overlaps are warnings. Points
    inside/on training polygons (or within the specified metric buffer) are flagged as
    validation leakage. With a positive buffer, a supplied metric ``distance_crs`` is
    preferred, then a metric input CRS, then a locally estimated UTM projection. For
    studies spanning many UTM zones supply a suitable projection explicitly.
    """
    checks, inputs, frames = [], {}, {}

    def check(code, layer, passed, message, severity="error", count=None):
        item = {"code": code, "layer": layer, "severity": severity,
                "status": "pass" if passed else "fail", "message": message}
        if count is not None:
            item["count"] = int(count)
        checks.append(item)

    if not any(v is not None for v in (training, parcels, observations, stack)):
        check("inputs.supplied", "study", False, "Supply at least one input layer or feature stack.")
    try:
        buffer_m = float(training_exclusion_m)
        if not math.isfinite(buffer_m) or buffer_m < 0:
            raise ValueError
    except (TypeError, ValueError):
        buffer_m = None
        check("leakage.buffer", "study", False, "training_exclusion_m must be finite and nonnegative.")

    specs = {"training": (training, ("Polygon", "MultiPolygon"), id_field, label_field),
             "parcels": (parcels, ("Polygon", "MultiPolygon"), parcel_id_field, None),
             "observations": (observations, ("Point",), None, observation_label_field or label_field)}
    for name, (source, types, ident, label) in specs.items():
        if source is None:
            continue
        try:
            frame = source.copy() if isinstance(source, gpd.GeoDataFrame) else gpd.read_file(source)
            geom = frame.geometry
        except Exception as exc:  # Reader backends raise several unrelated exception types.
            check("vector.readable", name, False, f"Cannot read vector input: {type(exc).__name__}: {exc}")
            continue
        inputs[name] = {"source": "in-memory" if isinstance(source, gpd.GeoDataFrame) else str(source),
                        "features": len(frame), "crs": str(frame.crs) if frame.crs else None}
        check("vector.nonempty", name, len(frame) > 0, "Layer must contain features.", count=len(frame))
        check("vector.crs", name, frame.crs is not None, "Layer needs an explicitly defined CRS.")
        absent = geom.isna() | geom.is_empty
        invalid = ~absent & ~geom.is_valid
        wrong_type = ~absent & ~geom.geom_type.isin(types)
        check("geometry.present", name, not absent.any(), "Null/empty geometries.", count=absent.sum())
        check("geometry.valid", name, not invalid.any(), "Invalid geometries; inspect and repair upstream.", count=invalid.sum())
        check("geometry.type", name, not wrong_type.any(), f"Expected geometry types: {', '.join(types)}.", count=wrong_type.sum())
        for field, unique in ((ident, True), (label, False)):
            if field is None:
                continue
            if field not in frame:
                check("schema.column", name, False, f"Missing required column {field!r}.")
                continue
            missing = _missing(frame[field])
            check("schema.values", name, not missing.any(), f"Null/blank values in {field!r}.", count=missing.sum())
            if unique:
                duplicates = frame.loc[~missing, field].astype(str).duplicated(keep=False)
                check("schema.unique", name, not duplicates.any(), f"IDs in {field!r} must be unique after string conversion.",
                      count=duplicates.sum())
        if label in frame:
            support = frame.loc[~_missing(frame[label]), label].astype(str).value_counts()
            inputs[name]["class_counts"] = {str(k): int(v) for k, v in support.items()}
            if name == "training":
                rare = support[support < 2]
                check("classes.support", name, not len(rare),
                      "Classes with fewer than two reference polygons cannot support polygon-disjoint train/test evaluation.",
                      severity="warning", count=len(rare))
        frames[name] = frame.loc[~(absent | invalid | wrong_type)].copy().reset_index(drop=True)

    train = frames.get("training")
    if train is not None and len(train) and label_field in train:
        labelled = train.loc[~_missing(train[label_field])].reset_index(drop=True)
        same, conflict = 0, 0
        index = labelled.sindex
        # Query one geometry at a time so pair enumeration does not allocate an n-by-n matrix.
        for i, row in labelled.iterrows():
            for j in index.query(row.geometry, predicate="intersects"):
                if j <= i:
                    continue
                if row.geometry.intersection(labelled.geometry.iloc[j]).area <= 0:
                    continue  # Touching boundaries are not contradictory sample interiors.
                if str(row[label_field]) == str(labelled[label_field].iloc[j]):
                    same += 1
                else:
                    conflict += 1
        check("training.conflicting_overlap", "training", conflict == 0,
              "Positive-area overlaps between training polygons with different labels.", count=conflict)
        check("training.duplicate_support", "training", same == 0,
              "Positive-area same-label overlaps may duplicate training pixels across polygon IDs.", "warning", same)

    obs = frames.get("observations")
    if (train is not None and obs is not None and len(train) and len(obs)
            and train.crs is not None and obs.crs is not None and buffer_m is not None):
        try:
            target_crs = train.crs
            if buffer_m:
                target_crs = distance_crs
                if target_crs is None:
                    target_crs = next((c for c in (train.crs, obs.crs) if _metric(c)), None)
                if target_crs is None:
                    target_crs = train.estimate_utm_crs()
                    check("leakage.local_projection", "study", False,
                          "Metric distances use estimated local UTM; review this choice for a large or multi-zone study.",
                          "warning")
                if target_crs is None or not _metric(target_crs):
                    raise ValueError("Buffered leakage checks require a projected CRS with metre units.")
            zones = train.to_crs(target_crs).geometry
            if buffer_m:
                zones = zones.buffer(buffer_m)
            points = obs.to_crs(target_crs).geometry
            index = zones.sindex
            exposed = sum(bool(len(index.query(point, predicate="intersects"))) for point in points)
            inputs["observations"]["training_leakage_count"] = int(exposed)
            inputs["observations"]["training_exclusion_m"] = buffer_m
            inputs["observations"]["leakage_crs"] = CRS.from_user_input(target_crs).to_string()
            check("validation.training_leakage", "observations", exposed == 0,
                  "Observations intersecting training polygons or their exclusion buffer must be excluded from independent validation.",
                  "warning", exposed)
        except Exception as exc:
            check("validation.leakage_check", "study", False, f"Cannot assess training/observation leakage: {exc}")

    if stack is not None:
        try:
            with rasterio.open(stack) as ds:
                inputs["stack"] = {"source": str(stack), "width": ds.width, "height": ds.height,
                                   "bands": ds.count, "crs": ds.crs.to_string() if ds.crs else None,
                                   "band_descriptions": list(ds.descriptions)}
                check("stack.crs", "stack", ds.crs is not None, "Feature stack needs a CRS.")
                check("stack.metric_crs", "stack", ds.crs is not None and _metric(ds.crs),
                      "Spatial blocks, buffers and areas require a projected CRS in metres.")
                names = [n for n in ds.descriptions if n and n.strip()]
                check("stack.band_names", "stack", len(names) == ds.count and len(set(names)) == ds.count,
                      "Feature band descriptions must be nonempty and unique.")
                check("stack.nodata", "stack", ds.nodata is not None,
                      "Feature stack should declare nodata explicitly.", "warning")
                t = ds.transform
                footprint = Polygon([(t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f)
                                     for x, y in ((0, 0), (ds.width, 0),
                                                  (ds.width, ds.height), (0, ds.height))])
                for name, frame in frames.items():
                    if frame.crs is None or ds.crs is None or not len(frame):
                        continue
                    try:
                        inside = frame.to_crs(ds.crs).geometry.intersects(footprint)
                        outside = int((~inside).sum())
                        inputs[name]["outside_stack_count"] = outside
                        check("stack.layer_intersection", name, inside.any(),
                              "Layer must intersect the feature-stack footprint.")
                        check("stack.outside_features", name, outside == 0,
                              "Features wholly outside the stack footprint cannot supply raster samples.", "warning", outside)
                    except Exception as exc:
                        check("stack.layer_transform", name, False, f"Cannot compare layer to stack: {exc}")
        except Exception as exc:
            check("stack.readable", "stack", False, f"Cannot inspect feature stack: {type(exc).__name__}: {exc}")

    failed = [c for c in checks if c["status"] == "fail"]
    errors = sum(c["severity"] == "error" for c in failed)
    return {"schema": "geocrop-input-audit/1", "ok": errors == 0,
            "summary": {"passed": len(checks) - len(failed), "errors": errors,
                        "warnings": sum(c["severity"] == "warning" for c in failed)},
            "inputs": inputs, "checks": checks,
            "scope": "Read-only schema, geometry, footprint and reference-overlap checks; no pixel-level data quality or accuracy validation."}
