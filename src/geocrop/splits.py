"""Spatial fold assignment for reference polygons, exportable for visual inspection in a GIS."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def block_ids(frame: gpd.GeoDataFrame, block_m: float) -> np.ndarray:
    if frame.crs is None or not frame.crs.is_projected:
        raise ValueError("A projected metric CRS is required for spatial blocks")
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    c = frame.geometry.centroid
    return (np.floor(c.x / block_m).astype(int).astype(str) + ":" + np.floor(c.y / block_m).astype(int).astype(str)).to_numpy()


def assign_folds(polygons: str | Path | gpd.GeoDataFrame, output: str | Path | None = None, label_field: str = "class_name",
                 block_m: float = 5000, n_splits: int = 5, seed: int = 42) -> dict:
    """Add `block` and `fold` columns (class-stratified, block-grouped) and report class coverage per fold.

    Classes absent from a fold's test set (frequent for rare classes such as water) are listed so
    that the block size or number of folds can be adjusted before any model is trained.
    """
    frame = gpd.read_file(polygons) if not isinstance(polygons, gpd.GeoDataFrame) else polygons.copy()
    if label_field not in frame:
        raise ValueError(f"Missing {label_field!r}")
    blocks = block_ids(frame, block_m)
    k = min(n_splits, len(np.unique(blocks)))
    if k < 2:
        raise ValueError("Need at least two spatial blocks")
    fold = np.full(len(frame), -1)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(splitter.split(np.zeros(len(frame)), frame[label_field].astype(str), blocks)):
        fold[te] = i
    frame["block"], frame["fold"] = blocks, fold
    classes = sorted(frame[label_field].astype(str).unique())
    coverage = {int(f): sorted(set(classes) - set(frame.loc[frame.fold == f, label_field].astype(str)))
                for f in range(k)}
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        frame.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
    return {"n_folds": k, "blocks": int(len(np.unique(blocks))), "block_m": block_m,
            "polygons_per_fold": frame.fold.value_counts().sort_index().to_dict(),
            "classes_missing_from_fold": {f: m for f, m in coverage.items() if m}}
