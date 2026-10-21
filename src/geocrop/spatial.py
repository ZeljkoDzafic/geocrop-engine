"""Spatial autocorrelation diagnostics for choosing the cross-validation block size.

Spatial CV only removes optimism if test blocks are farther from training data than the range of
spatial autocorrelation (Roberts et al., 2017; Ploton et al., 2020). `variogram` estimates an empirical
semivariogram of any per-object variable (e.g. the first principal component of the features, or the
out-of-fold error indicator) and fits a spherical model; its range is a defensible lower bound for
`spatial_block_m`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


def spherical(h, nugget, sill, rng):
    h = np.asarray(h, dtype=float)
    inside = nugget + (sill - nugget) * (1.5 * h / rng - 0.5 * (h / rng) ** 3)
    return np.where(h < rng, inside, sill)


def variogram(x, y, values, max_distance: float | None = None, n_bins: int = 15) -> dict:
    x, y, v = (np.asarray(a, dtype=float) for a in (x, y, values))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(v) < 10:
        raise ValueError("Need at least 10 objects")
    pts = np.column_stack([x, y])
    if max_distance is None:
        span = np.hypot(np.ptp(x), np.ptp(y))
        max_distance = span / 2
    pairs = cKDTree(pts).query_pairs(max_distance, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("No pairs within max_distance")
    h = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    g = 0.5 * (v[pairs[:, 0]] - v[pairs[:, 1]]) ** 2
    edges = np.linspace(0, max_distance, n_bins + 1)
    which = np.digitize(h, edges) - 1
    centres, gamma, counts = [], [], []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() >= 10:
            centres.append(h[sel].mean())
            gamma.append(g[sel].mean())
            counts.append(int(sel.sum()))
    centres, gamma = np.array(centres), np.array(gamma)
    result = {"lag_m": centres.tolist(), "semivariance": gamma.tolist(), "pairs": counts,
              "variance": float(np.var(v)), "max_distance_m": float(max_distance)}
    try:
        p0 = [gamma.min(), gamma.max(), max_distance / 3]
        (nugget, sill, rng), _ = curve_fit(spherical, centres, gamma, p0=p0,
                                           bounds=([0, 0, 1e-6], [np.inf, np.inf, max_distance * 2]), maxfev=20000)
        result.update({"model": "spherical", "nugget": float(nugget), "sill": float(sill), "range_m": float(rng)})
    except (RuntimeError, ValueError):
        result.update({"model": None, "range_m": None})
    return result


def object_table(samples, features=None):
    """Per-polygon centroid and first principal component of standardised features (variogram input)."""
    import pandas as pd

    X = samples.X if features is None else samples.subset(features)
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
    _, _, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    pc1 = Z @ vt[0]
    frame = pd.DataFrame({"polygon": samples.polygon, "x": samples.x, "y": samples.yy, "pc1": pc1})
    return frame.groupby("polygon", as_index=False).mean()
