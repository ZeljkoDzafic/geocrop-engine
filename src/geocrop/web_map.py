"""Map layers for the web workspace: job artifacts -> GeoJSON and PNG overlays in WGS 84.

Rasters are warped to EPSG:4326, downsampled to at most `max_px` pixels on the longer side and encoded as
PNG data URLs (so the page's strict content-security policy, `img-src 'self' data:`, still holds). Vectors
are reprojected, simplified to about one output pixel and capped in feature count. Nothing is written to
disk and no data leave the machine.
"""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np

TRAFFIC = {"green": "#2e9d57", "yellow": "#e3b505", "red": "#d64545", "grey": "#9aa0a6"}
CLASS_PALETTE = ["#e6a817", "#3f9b3f", "#b5651d", "#7fb800", "#2f6f9f", "#8e44ad", "#c0392b", "#16a085",
                 "#d35400", "#7f8c8d", "#f1c40f", "#2c3e50", "#27ae60", "#e84393", "#00cec9"]
VECTOR_PROPS = ("parcel_id", "segment_id", "declared_class", "predicted_class", "declaration_status",
                "traffic_light", "decision", "confidence", "prediction_set", "class_name", "observed_at")


def _png(rgba: np.ndarray) -> str:
    """Minimal PNG encoder (RGBA, 8 bit) without extra dependencies."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return (len(data).to_bytes(4, "big") + tag + data
                + (zlib.crc32(tag + data) & 0xFFFFFFFF).to_bytes(4, "big"))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0]))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _hex(color: str):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _warp(path: Path, band: int, max_px: int, categorical: bool):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height,
                                                               *src.bounds)
        scale = max(width, height) / max_px if max(width, height) > max_px else 1.0
        width, height = max(1, int(width / scale)), max(1, int(height / scale))
        transform = transform @ transform.scale(scale, scale)
        nodata = src.nodata
        out = np.full((height, width), np.nan, dtype="float64")
        reproject(rasterio.band(src, band), out, src_transform=src.transform, src_crs=src.crs, src_nodata=nodata,
                  dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear)
        west, north = transform @ (0, 0)
        east, south = transform @ (width, height)
    return out, [[south, west], [north, east]]


def class_layer(path: Path, legend: dict, max_px: int = 1024) -> dict:
    data, bounds = _warp(path, 1, max_px, categorical=True)
    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    entries = []
    for key, name in sorted(legend.items(), key=lambda kv: int(kv[0])):
        color = CLASS_PALETTE[int(key) % len(CLASS_PALETTE)]
        mask = data == int(key)
        rgba[mask] = (*_hex(color), 200)
        entries.append({"label": name, "color": color})
    return {"name": "Crop classes", "type": "image", "url": _png(rgba), "bounds": bounds, "legend": entries,
            "visible": True}


def continuous_layer(path: Path, band: int, name: str, max_px: int = 1024, vmin=0.0, vmax=1.0,
                     low="#f7fbff", high="#08306b", visible=False) -> dict:
    data, bounds = _warp(path, band, max_px, categorical=False)
    valid = np.isfinite(data) & (data != -9999)
    t = np.clip((np.where(valid, data, vmin) - vmin) / max(vmax - vmin, 1e-12), 0, 1)[..., None]
    lo, hi = np.array(_hex(low)), np.array(_hex(high))
    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    rgba[..., :3] = (lo + (hi - lo) * t).astype("uint8")
    rgba[..., 3] = np.where(valid, 190, 0)
    return {"name": name, "type": "image", "url": _png(rgba), "bounds": bounds, "visible": visible,
            "legend": [{"label": f"{vmin:g}", "color": low}, {"label": f"{vmax:g}", "color": high}]}


def vector_layer(path: Path, name: str | None = None, max_features: int = 20000) -> dict:
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.empty:
        raise ValueError(f"{path.name}: empty or without CRS")
    truncated = len(gdf) > max_features
    gdf = gdf.head(max_features).to_crs("EPSG:4326")
    w, s, e, n = gdf.total_bounds
    gdf["geometry"] = gdf.geometry.simplify(max(e - w, n - s) / 2000, preserve_topology=True)
    keep = [c for c in VECTOR_PROPS if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]
    for c in keep:
        gdf[c] = gdf[c].astype(object).where(gdf[c].notna(), None)
        if gdf[c].dtype.kind == "f":
            gdf[c] = gdf[c].round(3)
    geojson = json.loads(gdf.to_json(drop_id=True))
    kind = "parcels" if "traffic_light" in keep else ("points" if gdf.geom_type.isin(["Point"]).all() else "polygons")
    legend = [{"label": k, "color": v} for k, v in TRAFFIC.items()] if kind == "parcels" else []
    label = {"parcels": "Parcels — traffic light", "points": f"Points ({name or path.stem})"}.get(kind,
                                                                                              name or path.stem)
    return {"name": label, "type": "geojson", "kind": kind, "data": geojson, "legend": legend,
            "truncated": truncated, "visible": kind == "parcels",
            "bounds": [[float(s), float(w)], [float(n), float(e)]]}


def job_layers(files: list[Path], max_px: int = 1024) -> dict:
    """Pick displayable outputs of one job; unknown or unreadable files are reported, not fatal."""
    layers, skipped = [], []

    def attempt(fn, *args, **kwargs):
        try:
            layers.append(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not hide the others
            skipped.append(f"{args[0].name if args else '?'}: {exc}")

    for f in files:
        if f.name == "classes.tif":
            legend_file = f.with_name("class_legend.json")
            legend = json.loads(legend_file.read_text()) if legend_file.exists() else {}
            attempt(class_layer, f, legend, max_px)
        elif f.name == "uncertainty.tif":
            attempt(continuous_layer, f, 1, "Max class probability", max_px, 0, 1, "#fff5eb", "#7f2704")
            attempt(continuous_layer, f, 3, "Normalised entropy", max_px, 0, 1, "#f7fcf5", "#00441b")
        elif f.name == "aoa.tif":
            attempt(continuous_layer, f, 2, "Inside area of applicability", max_px, 0, 1, "#d64545", "#2e9d57")
        elif f.name == "boundary_probability.tif" or f.name.endswith("_boundary.tif"):
            attempt(continuous_layer, f, 1, "Boundary probability", max_px, 0, 1, "#ffffff", "#000000")
        elif f.suffix.lower() in (".gpkg", ".geojson") and f.stat().st_size < 200 * 1024 * 1024:
            attempt(vector_layer, f, f"{f.parent.name}/{f.stem}")
    order = {"image": 0, "geojson": 1}
    layers.sort(key=lambda layer: order[layer["type"]])
    bounds = None
    for layer in layers:
        (s, w), (n, e) = layer["bounds"]
        bounds = [[s, w], [n, e]] if bounds is None else [[min(bounds[0][0], s), min(bounds[0][1], w)],
                                                          [max(bounds[1][0], n), max(bounds[1][1], e)]]
    return {"layers": layers, "bounds": bounds, "skipped": skipped,
            "note": "Previews are downsampled to the screen; download the files for analysis."}
