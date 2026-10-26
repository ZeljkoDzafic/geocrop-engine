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


# --- Group strategies for transfer validation (leave-one-region-out / leave-one-year-out) ---------------------

#: Cross-validation strategies. ``spatial_blocks`` (default) is k-fold over square blocks; the others hold out
#: one whole region, one whole year, or one region-year cell per fold.
STRATEGIES = ("spatial_blocks", "region", "year", "region_year")


def check_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"cv strategy must be one of {', '.join(STRATEGIES)}; got {strategy!r}")
    return strategy


def strategy_groups(strategy: str, *, blocks=None, region=None, year=None, polygon=None) -> np.ndarray:
    """Group id per sample for a strategy.

    Region/year ids must be constant within each polygon (all pixels of a reference object move together).
    ``region_year`` ids are ``"<region>|<year>"`` cells; see `leave_one_group_out` for how they are held out.
    """
    check_strategy(strategy)
    if strategy == "spatial_blocks":
        if blocks is None:
            raise ValueError("spatial_blocks strategy needs block ids")
        return np.asarray(blocks)
    needed = {"region": ("region",), "year": ("year",), "region_year": ("region", "year")}[strategy]
    values = {"region": region, "year": year}
    for name in needed:
        if values[name] is None:
            raise ValueError(f"cv strategy {strategy!r} needs {name} metadata in the samples; extract with "
                             f"--{name}-field (or --year) or merge seasons with `geocrop samples-merge`")
        arr = np.asarray(values[name]).astype(str)
        if polygon is not None:
            per = (np.unique(np.column_stack([np.asarray(polygon).astype(str), arr]), axis=0)[:, 0])
            if len(per) != len(np.unique(np.asarray(polygon).astype(str))):
                raise ValueError(f"Some polygons have more than one {name} id")
        values[name] = arr
    if strategy == "region_year":
        return np.char.add(np.char.add(values["region"], "|"), values["year"])
    return values[strategy]


def leave_one_group_out(groups, strategy: str = "region", *, min_groups: int = 2):
    """Yield ``(group, train_idx, test_idx)``: each fold holds out one group entirely.

    For ``region_year`` the test set is one region-year cell and the training set excludes every sample that
    shares its region *or* its year, so the model has seen neither the place nor the season (strict
    spatio-temporal transfer). Train and test never share a group id (asserted).
    """
    check_strategy(strategy)
    groups = np.asarray(groups).astype(str)
    names = sorted(np.unique(groups))
    if len(names) < min_groups:
        raise ValueError(f"Leave-one-group-out ({strategy}) needs at least {min_groups} distinct groups; "
                         f"found {len(names)}: {names}")
    if strategy == "region_year":
        parts = np.array([g.split("|", 1) for g in groups])
        if len(np.unique(parts[:, 0])) < 2 or len(np.unique(parts[:, 1])) < 2:
            raise ValueError("region_year transfer needs at least two regions and two years")
    for name in names:
        test = groups == name
        if strategy == "region_year":
            region, year = name.split("|", 1)
            train = (parts[:, 0] != region) & (parts[:, 1] != year)
        else:
            train = ~test
        tr, te = np.flatnonzero(train), np.flatnonzero(test)
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Group leakage between train and test")
        if len(tr) == 0:
            continue   # a cell with no sample from another region *and* another year cannot be evaluated
        yield name, tr, te
