"""Training-sample extraction from reference polygons.

Pixels are read only inside each polygon's bounding window, so extraction scales with the
reference data, not with the AOI. Results are cached as a compressed `.npz` that every
training run and ablation experiment reuses, guaranteeing identical samples across experiments.

Optional group metadata (``region`` and ``year`` per sample, copied from polygon attributes) supports
leave-one-region-out and leave-one-year-out validation; `combine` builds a multi-year sample store from
per-season sample files that share one feature manifest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from .raster import feature_names, open_input, read_block, valid_mask


@dataclass
class Samples:
    X: np.ndarray            # (n, features) float32
    y: np.ndarray            # (n,) class labels (str)
    polygon: np.ndarray      # (n,) source polygon id (str)
    x: np.ndarray            # (n,) pixel-centre easting in the stack CRS
    yy: np.ndarray           # (n,) pixel-centre northing
    features: list[str]
    crs: str
    region: np.ndarray | None = None   # (n,) region id (str) per sample, optional
    year: np.ndarray | None = None     # (n,) season/year id (str) per sample, optional
    #: every class of the reference polygons that intersect the stack, recorded by `extract`; lets
    #: `train`/`benchmark-models` refuse a sample set that silently lost a whole class (`check_classes`)
    reference_classes: np.ndarray | None = None

    def __post_init__(self):
        for name in ("region", "year"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value).astype(str)
                if value.shape != (len(self.y),):
                    raise ValueError(f"{name} metadata must have one value per sample")
                setattr(self, name, value)
        if self.reference_classes is not None:
            self.reference_classes = np.unique(np.asarray(self.reference_classes).astype(str))

    def missing_classes(self) -> list[str]:
        """Classes of the reference polygons without any sample (empty when unknown, e.g. older sample files)."""
        if self.reference_classes is None:
            return []
        return sorted(map(str, set(self.reference_classes) - set(np.asarray(self.y).astype(str))))

    def take(self, index) -> Samples:
        """Row subset (boolean mask or integer index) that keeps the group metadata."""
        meta = {k: None if getattr(self, k) is None else getattr(self, k)[index] for k in ("region", "year")}
        return Samples(self.X[index], self.y[index], self.polygon[index], self.x[index], self.yy[index],
                       self.features, self.crs, **meta, reference_classes=self.reference_classes)

    def subset(self, names: list[str]) -> np.ndarray:
        missing = [n for n in names if n not in self.features]
        if missing:
            raise KeyError(f"Features not in samples: {missing[:5]}")
        return self.X[:, [self.features.index(n) for n in names]]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(self, k) for k in ("region", "year", "reference_classes") if getattr(self, k) is not None}
        np.savez_compressed(path, X=self.X, y=self.y, polygon=self.polygon, x=self.x, yy=self.yy,
                            features=np.array(self.features), crs=np.array(self.crs), **meta)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Samples:
        z = np.load(path, allow_pickle=False)
        meta = {k: z[k].astype(str) if k in z.files else None for k in ("region", "year", "reference_classes")}
        return cls(z["X"], z["y"].astype(str), z["polygon"].astype(str), z["x"], z["yy"],
                   [str(f) for f in z["features"]], str(z["crs"]), **meta)


def _manifest_key(features) -> list[tuple]:
    """Comparable form of a feature manifest (list of `geocrop.features.Feature` or of dicts)."""
    out = []
    for f in features:
        d = f if isinstance(f, dict) else vars(f)
        out.append((str(d["name"]), str(d.get("group", "")), str(d.get("sensor", "")), str(d.get("period")),
                    tuple(d.get("periods", ()) or ())))
    return out


def combine(parts: list[Samples], years: list | None = None, manifests: list | None = None,
            prefix_polygon_ids: bool = True) -> Samples:
    """Concatenate per-season sample sets into one multi-year sample store.

    Every part must have exactly the same feature names in the same order and the same CRS; when
    `manifests` (one feature manifest per part, e.g. from `geocrop.features.load_manifest`) are given,
    group, sensor and period definitions must match as well. Mismatches are refused, never reconciled,
    because a column that means different things in different seasons would silently corrupt transfer
    validation. Window names must therefore be season-relative (e.g. ``early``/``peak``), not dated.

    `years` assigns a year id per part (overriding stored ids); otherwise each part must carry year metadata.
    Polygon ids are prefixed with the year (``2023:p17``) so that the same field observed in two seasons
    stays two reference objects; spatial blocks still place both in the same block.
    """
    if len(parts) < 2:
        raise ValueError("combine needs at least two sample sets")
    if years is not None and len(years) != len(parts):
        raise ValueError(f"Got {len(years)} year ids for {len(parts)} sample sets")
    if manifests is not None and len(manifests) != len(parts):
        raise ValueError(f"Got {len(manifests)} feature manifests for {len(parts)} sample sets")
    first = parts[0]
    for i, part in enumerate(parts[1:], 1):
        if part.crs != first.crs:
            raise ValueError(f"Sample set {i} has CRS {part.crs}, set 0 has {first.crs}; resample to one grid CRS")
        if part.features != first.features:
            missing = [f for f in first.features if f not in part.features]
            extra = [f for f in part.features if f not in first.features]
            detail = (f"missing {missing[:5]}, unexpected {extra[:5]}" if missing or extra
                      else "same names in a different order")
            raise ValueError(f"Feature manifest of sample set {i} does not match set 0 ({detail}). "
                             "Rebuild all seasons with the same feature configuration and season-relative "
                             "window names.")
        if manifests is not None and _manifest_key(manifests[i]) != _manifest_key(manifests[0]):
            diff = [a[0] for a, b in zip(_manifest_key(manifests[0]), _manifest_key(manifests[i])) if a != b]
            raise ValueError(f"Feature manifest {i} differs from manifest 0 in feature definitions "
                             f"{diff[:5] or '(length)'}; group/sensor/period must be identical across seasons")
    year_ids = []
    for i, part in enumerate(parts):
        if years is not None:
            year_ids.append(np.full(len(part.y), str(years[i])))
        elif part.year is not None:
            year_ids.append(part.year)
        else:
            raise ValueError(f"Sample set {i} has no year metadata; pass years=[...] (CLI: --years)")
    with_region = [p.region is not None for p in parts]
    if any(with_region) and not all(with_region):
        raise ValueError("Either all or none of the sample sets must carry region metadata")
    polygons = [np.char.add(np.char.add(yid.astype(str), ":"), p.polygon.astype(str)) if prefix_polygon_ids
                else p.polygon for p, yid in zip(parts, year_ids)]
    polygon = np.concatenate(polygons)
    if not prefix_polygon_ids:
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                shared = set(parts[a].polygon) & set(parts[b].polygon)
                if shared:
                    raise ValueError(f"Polygon ids {sorted(shared)[:5]} occur in sample sets {a} and {b}; "
                                     "use prefix_polygon_ids=True")
    return Samples(np.concatenate([p.X for p in parts]), np.concatenate([p.y for p in parts]), polygon,
                   np.concatenate([p.x for p in parts]), np.concatenate([p.yy for p in parts]),
                   list(first.features), first.crs,
                   region=np.concatenate([p.region for p in parts]) if all(with_region) else None,
                   year=np.concatenate(year_ids),
                   reference_classes=(np.concatenate([p.reference_classes for p in parts])
                                      if all(p.reference_classes is not None for p in parts) else None))


def polygon_window(geom, ds) -> Window | None:
    xmin, ymin, xmax, ymax = geom.bounds
    left, bottom, right, top = ds.bounds
    xmin, xmax = max(xmin, left), min(xmax, right)
    ymin, ymax = max(ymin, bottom), min(ymax, top)
    if xmin >= xmax or ymin >= ymax:
        return None
    raw = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
    c0, r0 = max(0, int(np.floor(raw.col_off))), max(0, int(np.floor(raw.row_off)))
    c1 = min(ds.width, int(np.ceil(raw.col_off + raw.width)))
    r1 = min(ds.height, int(np.ceil(raw.row_off + raw.height)))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


class SampleCoverageWarning(UserWarning):
    """Reference polygons were skipped because none of their pixels is valid for every feature."""


class MissingClassError(ValueError):
    """A class of the reference polygons has no sample at all (training/benchmarking would silently drop it)."""


def check_classes(samples: Samples, allow_missing: bool = False, where: str = "samples") -> list[str]:
    """Refuse `samples` that lost a whole reference class; returns the missing classes.

    With ``allow_missing=True`` the loss is only warned about (`SampleCoverageWarning`). Sample files written
    before `reference_classes` was recorded cannot be checked and pass.
    """
    missing = samples.missing_classes()
    if not missing:
        return []
    msg = (f"{where}: classes of the reference polygons without any sample: {missing} "
           f"(present: {sorted(set(map(str, samples.y)))}). Their polygons have no pixel with valid values for "
           "every stored feature. Extract samples only for the features you use "
           "(`geocrop samples --groups ts ...`, `extract(features=...)`, config `model.feature_groups`), fix the "
           "composites, or accept the reduced class set explicitly (--allow-missing-classes, config "
           "`model.allow_missing_classes: true`).")
    if not allow_missing:
        raise MissingClassError(msg)
    warnings.warn(msg, SampleCoverageWarning, stacklevel=2)
    return missing


def extract(stack: str | Path, polygons: str | Path | gpd.GeoDataFrame, label_field: str = "class_name",
            id_field: str = "polygon_id", max_per_polygon: int | None = 250, inner_buffer_m: float = 0.0,
            seed: int = 42, region_field: str | None = None, year_field: str | None = None,
            year: str | int | None = None, features: list[str] | None = None) -> Samples:
    """Sample valid pixels whose centres fall inside each reference polygon.

    max_per_polygon caps large polygons so they do not dominate a class; inner_buffer_m
    (e.g. 10-20 m) removes mixed boundary pixels from training. region_field / year_field copy a
    polygon attribute into per-sample group metadata; `year` sets one constant year for all samples
    (a single-season stack). Missing attribute values are refused.

    `features` (band names, e.g. ``features.select(manifest, groups=["ts"])``) reads, validates and stores
    only those bands: a pixel needs valid values for the selected features only, so NaN in unrelated
    groups (e.g. phenology metrics of flat series) no longer removes whole polygons. The classes of all
    polygons intersecting the stack are stored as `Samples.reference_classes` (see `check_classes`).
    """
    if year_field and year is not None:
        raise ValueError("Pass either year_field or a constant year, not both")
    rng = np.random.default_rng(seed)
    source = "reference polygons" if isinstance(polygons, gpd.GeoDataFrame) else f"reference polygons {polygons}"
    gdf = gpd.read_file(polygons) if not isinstance(polygons, gpd.GeoDataFrame) else polygons.copy()
    for col in (label_field, id_field, region_field, year_field):
        if col is None:
            continue
        if col not in gdf:
            raise ValueError(f"{source}: missing column {col!r} (columns: {[c for c in gdf.columns][:20]})")
    if gdf.crs is None:
        raise ValueError(f"{source}: no CRS defined; assign the layer's CRS before training")
    if gdf.empty or (gdf.geometry.isna() | gdf.geometry.is_empty).all():
        raise ValueError(f"{source}: contains no (non-empty) polygons")
    for col in (label_field, id_field):
        blank = gdf[col].isna() | (gdf[col].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(f"{source}: {int(blank.sum())} rows have an empty {col!r} "
                             f"(row positions {list(np.flatnonzero(blank.to_numpy())[:5])}); every polygon needs one")
    duplicated = gdf[id_field][gdf[id_field].duplicated()].astype(str)
    if len(duplicated):
        raise ValueError(f"{source}: {id_field} must be unique; duplicated ids {sorted(set(duplicated))[:5]}")
    for col in (region_field, year_field):
        if col and gdf[col].isna().any():
            raise ValueError(f"{source}: {col!r} is empty for {int(gdf[col].isna().sum())} polygons")
    X, y, pid, xs, ys, regions, years = [], [], [], [], [], [], []
    intersecting = 0
    emptied: dict[str, int] = {}          # class -> polygons inside the stack without any fully valid pixel
    with open_input(stack, "Feature stack") as ds:
        gdf = gdf.to_crs(ds.crs)
        if inner_buffer_m:
            gdf["geometry"] = gdf.geometry.buffer(-abs(inner_buffer_m))
        names = feature_names(ds)
        indexes = None
        if features is not None:
            features = list(dict.fromkeys(map(str, features)))
            unknown = [f for f in features if f not in names]
            if unknown or not features:
                raise ValueError(f"Features not in stack {ds.name}: {unknown[:5] or '(empty selection)'}")
            indexes = [names.index(f) + 1 for f in features]
            names = features
        intersecting_classes: set[str] = set()
        for row in gdf.itertuples():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            win = polygon_window(geom, ds)
            if win is None:
                continue
            intersecting += 1
            intersecting_classes.add(str(getattr(row, label_field)))
            cube = read_block(ds, indexes, window=win)
            transform = ds.window_transform(win)
            inside = geometry_mask([geom], out_shape=cube.shape[1:], transform=transform, invert=True)
            rr, cc = np.nonzero(inside & valid_mask(cube, ds.nodata))
            if len(rr) == 0:
                cls = str(getattr(row, label_field))
                emptied[cls] = emptied.get(cls, 0) + 1
                continue
            if max_per_polygon and len(rr) > max_per_polygon:
                keep = rng.choice(len(rr), size=max_per_polygon, replace=False)
                rr, cc = rr[keep], cc[keep]
            X.append(cube[:, rr, cc].T)
            label, ident = str(getattr(row, label_field)), str(getattr(row, id_field))
            y += [label] * len(rr)
            pid += [ident] * len(rr)
            if region_field:
                regions += [str(getattr(row, region_field))] * len(rr)
            if year_field:
                years += [_year_id(getattr(row, year_field))] * len(rr)
            px, py = rasterio.transform.xy(transform, rr, cc)
            xs.append(np.asarray(px))
            ys.append(np.asarray(py))
        crs = ds.crs.to_string()
        stack_name, bounds = ds.name, tuple(round(b, 1) for b in ds.bounds)
    if not X:
        if not intersecting:
            raise ValueError(f"No {source} intersect feature stack {stack_name} (bounds {bounds}, {crs}); "
                             "check the polygons' CRS and location")
        raise ValueError(f"{intersecting} {source} intersect feature stack {stack_name}, but none contains a pixel "
                         "with valid values for every feature (nodata/NaN); check the composites")
    if emptied:
        lost = sorted(set(emptied) - set(y))
        warnings.warn(
            f"{sum(emptied.values())} of {intersecting} {source} have no pixel with valid values for every feature "
            f"(nodata/NaN) and were skipped; per class: {dict(sorted(emptied.items()))}"
            + (f". Classes left WITHOUT any sample: {lost}" if lost else "")
            + ". Check the composites or extract only the feature groups you use (`--groups`/`features=`).",
            SampleCoverageWarning, stacklevel=2)
    if year is not None:
        years = [str(year)] * len(y)
    return Samples(np.concatenate(X).astype("float32"), np.array(y), np.array(pid),
                   np.concatenate(xs), np.concatenate(ys), names, crs,
                   region=np.array(regions) if region_field else None,
                   year=np.array(years) if years else None,
                   reference_classes=np.array(sorted(intersecting_classes)))


def _year_id(value) -> str:
    """Integer-valued years (2023 or 2023.0 from a float column) become ``"2023"``; other ids pass through."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
