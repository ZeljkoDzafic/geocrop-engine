"""Field-campaign planning (active learning): which parcels should be visited next?

Parcels are ranked by how much a field visit would teach the model — high predictive uncertainty
(normalised entropy, ambiguous conformal sets) and extrapolation outside the area of applicability —
and then selected greedily with a minimum spacing so the campaign is spatially spread (spacing ≥ the
autocorrelation range from `geocrop.spatial.variogram` avoids redundant visits). Optionally stratified
by predicted class so rare classes are not starved (uncertainty sampling; Settles, 2009; Tuia et al., 2011).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np


def priority_scores(gdf) -> np.ndarray:
    """Visit priority: normalised entropy + 0.5 for ambiguous conformal sets + share of pixels outside the AOA."""
    score = gdf["entropy"].fillna(1.0).to_numpy(dtype=float)
    if "set_size" in gdf:
        score = score + 0.5 * (gdf["set_size"].fillna(1).to_numpy(dtype=float) > 1)
    if "aoa_fraction" in gdf:
        score = score + (1 - gdf["aoa_fraction"].fillna(1.0).to_numpy(dtype=float))
    return score


def rank(parcels: str | Path | gpd.GeoDataFrame, output: str | Path | None = None, n: int = 50,
         min_spacing_m: float = 1000.0, per_class: int | None = None, exclude_ids=()) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    gdf = gdf[gdf.get("n_valid", 0).fillna(0) > 0]
    if exclude_ids:
        gdf = gdf[~gdf.parcel_id.astype(str).isin({str(i) for i in exclude_ids})]
    gdf = gdf.assign(priority=priority_scores(gdf)).sort_values("priority", ascending=False)
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError("Parcels need a projected CRS")
    pts = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    chosen, taken = [], {}
    cls = gdf.get("predicted_class", gdf.get("mean_class")).astype(str).to_numpy()
    for i in range(len(gdf)):
        if len(chosen) >= n:
            break
        if per_class is not None and taken.get(cls[i], 0) >= per_class:
            continue
        if chosen and np.min(np.hypot(*(pts[chosen] - pts[i]).T)) < min_spacing_m:
            continue
        chosen.append(i)
        taken[cls[i]] = taken.get(cls[i], 0) + 1
    picked = gdf.iloc[chosen].assign(visit_rank=np.arange(1, len(chosen) + 1))
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        picked.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
    return picked
