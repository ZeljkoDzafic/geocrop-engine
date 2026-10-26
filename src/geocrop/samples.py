"""Training-sample extraction from reference polygons.

Pixels are read only inside each polygon's bounding window, so extraction scales with the
reference data, not with the AOI. Results are cached as a compressed `.npz` that every
training run and ablation experiment reuses, guaranteeing identical samples across experiments.
"""
from __future__ import annotations

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

    def subset(self, names: list[str]) -> np.ndarray:
        missing = [n for n in names if n not in self.features]
        if missing:
            raise KeyError(f"Features not in samples: {missing[:5]}")
        return self.X[:, [self.features.index(n) for n in names]]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, y=self.y, polygon=self.polygon, x=self.x, yy=self.yy,
                            features=np.array(self.features), crs=np.array(self.crs))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Samples:
        z = np.load(path, allow_pickle=False)
        return cls(z["X"], z["y"].astype(str), z["polygon"].astype(str), z["x"], z["yy"],
                   [str(f) for f in z["features"]], str(z["crs"]))


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


def extract(stack: str | Path, polygons: str | Path | gpd.GeoDataFrame, label_field: str = "class_name",
            id_field: str = "polygon_id", max_per_polygon: int | None = 250, inner_buffer_m: float = 0.0,
            seed: int = 42) -> Samples:
    """Sample valid pixels whose centres fall inside each reference polygon.

    max_per_polygon caps large polygons so they do not dominate a class; inner_buffer_m
    (e.g. 10-20 m) removes mixed boundary pixels from training.
    """
    rng = np.random.default_rng(seed)
    source = "reference polygons" if isinstance(polygons, gpd.GeoDataFrame) else f"reference polygons {polygons}"
    gdf = gpd.read_file(polygons) if not isinstance(polygons, gpd.GeoDataFrame) else polygons.copy()
    for col in (label_field, id_field):
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
    X, y, pid, xs, ys = [], [], [], [], []
    intersecting = 0
    with open_input(stack, "Feature stack") as ds:
        gdf = gdf.to_crs(ds.crs)
        if inner_buffer_m:
            gdf["geometry"] = gdf.geometry.buffer(-abs(inner_buffer_m))
        names = feature_names(ds)
        for row in gdf.itertuples():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            win = polygon_window(geom, ds)
            if win is None:
                continue
            intersecting += 1
            cube = read_block(ds, window=win)
            transform = ds.window_transform(win)
            inside = geometry_mask([geom], out_shape=cube.shape[1:], transform=transform, invert=True)
            rr, cc = np.nonzero(inside & valid_mask(cube, ds.nodata))
            if len(rr) == 0:
                continue
            if max_per_polygon and len(rr) > max_per_polygon:
                keep = rng.choice(len(rr), size=max_per_polygon, replace=False)
                rr, cc = rr[keep], cc[keep]
            X.append(cube[:, rr, cc].T)
            label, ident = str(getattr(row, label_field)), str(getattr(row, id_field))
            y += [label] * len(rr)
            pid += [ident] * len(rr)
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
    return Samples(np.concatenate(X).astype("float32"), np.array(y), np.array(pid),
                   np.concatenate(xs), np.concatenate(ys), names, crs)
