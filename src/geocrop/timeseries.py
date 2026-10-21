"""Per-pixel time-series analysis on a regular temporal grid.

Window composites summarise each phenological window by one value; crops whose development differs by a
few weeks (e.g. sunflower senescing before soybean, maize peaking later) are better separated by the
*timing* of their trajectories. This module provides, vectorised over many pixels at once:

* `bin_to_grid`     irregular, cloud-masked observations -> regular grid (e.g. 5 days) with weights
* `whittaker`       weighted Whittaker smoother / gap filler (Eilers, 2003), optional upper envelope
* `phenology`       TIMESAT-style season metrics: SOS, EOS, peak, LOS, slopes, integral (Jönsson & Eklundh, 2004)
* `harmonics`       weighted harmonic regression coefficients (amplitude/phase per order)

All functions take arrays shaped (n_pixels, n_times) and return NaN where data are insufficient.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 renamed trapz


def make_grid(start: date, end: date, step_days: int = 5) -> list[date]:
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    n = (end - start).days // step_days + 1
    return [start + timedelta(days=i * step_days) for i in range(n)]


def bin_to_grid(obs_dates: list[date], values: np.ndarray, valid: np.ndarray, grid: list[date]):
    """Average valid observations into the nearest grid step.

    values, valid: (n_pixels, n_obs). Returns (y, w) shaped (n_pixels, n_grid) where w is the number of
    valid observations in each bin (0 = gap).
    """
    values = np.asarray(values, dtype="float64")
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    origin = grid[0]
    step = (grid[1] - grid[0]).days if len(grid) > 1 else 1
    idx = np.array([round((d - origin).days / step) for d in obs_dates])
    keep = (idx >= 0) & (idx < len(grid))
    total = np.zeros((values.shape[0], len(grid)))
    count = np.zeros_like(total)
    for j in np.flatnonzero(keep):
        v = valid[:, j]
        total[v, idx[j]] += values[v, j]
        count[v, idx[j]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(count > 0, total / count, np.nan)
    return y, count


def _penalty(n: int, order: int = 2) -> np.ndarray:
    d = np.diff(np.eye(n), n=order, axis=0)
    return d.T @ d


def whittaker(y: np.ndarray, w: np.ndarray, lam: float = 100.0, order: int = 2, envelope_iterations: int = 0,
              chunk: int = 4096, min_valid: int = 4) -> np.ndarray:
    """Weighted Whittaker smoother: minimise sum w (y - z)^2 + lam * sum (Δ^order z)^2.

    Gaps (w = 0) are interpolated by the smoothness penalty. With envelope_iterations > 0 the weights of
    observations below the fit are reduced each iteration, fitting the upper envelope (useful for NDVI,
    whose residual clouds and shadows bias values downwards). Pixels with fewer than `min_valid`
    observations return NaN.
    """
    y = np.asarray(y, dtype="float64")
    w = np.where(np.isfinite(y), np.asarray(w, dtype="float64"), 0.0)
    y0 = np.where(w > 0, y, 0.0)
    n_pix, n = y.shape
    P = lam * _penalty(n, order)
    out = np.full((n_pix, n), np.nan)
    enough = (w > 0).sum(axis=1) >= min_valid
    eye = np.arange(n)
    for start in range(0, n_pix, chunk):
        sl = slice(start, start + chunk)
        ok = enough[sl]
        if not ok.any():
            continue
        ww = w[sl][ok].copy()
        yy = y0[sl][ok]
        for it in range(envelope_iterations + 1):
            A = np.broadcast_to(P, (len(ww), n, n)).copy()
            A[:, eye, eye] += ww
            z = np.linalg.solve(A, (ww * yy)[..., None])[..., 0]
            if it < envelope_iterations:
                below = (yy < z) & (ww > 0)
                ww = np.where(below, ww * 0.5, ww)
        block = out[sl]
        block[ok] = z
        out[sl] = block
    return out


def phenology(z: np.ndarray, days: np.ndarray, threshold: float = 0.5, min_amplitude: float = 0.1) -> dict:
    """Season metrics from smoothed series z (n_pixels, n_times) sampled at `days` (day of year).

    One main season per pixel (the global maximum). Base level = mean of the minima before and after the
    peak; SOS/EOS = first/last crossing of base + threshold * amplitude (linear interpolation).
    Returns arrays: peak_value, peak_doy, amplitude, sos, eos, los, greenup_rate, senescence_rate, integral.
    """
    z = np.asarray(z, dtype="float64")
    days = np.asarray(days, dtype="float64")
    n_pix, n = z.shape
    names = ("peak_value", "peak_doy", "amplitude", "sos", "eos", "los", "greenup_rate", "senescence_rate", "integral")
    res = {k: np.full(n_pix, np.nan) for k in names}
    finite = np.isfinite(z).all(axis=1)
    if not finite.any():
        return res
    zz = z[finite]
    pk = zz.argmax(axis=1)
    rows = np.arange(len(zz))
    peak = zz[rows, pk]
    idx = np.arange(n)[None, :]
    left_min = np.where(idx <= pk[:, None], zz, np.inf).min(axis=1)
    right_min = np.where(idx >= pk[:, None], zz, np.inf).min(axis=1)
    base = (left_min + right_min) / 2
    amp = peak - base
    level = base + threshold * amp
    above = zz >= level[:, None]
    # SOS: last index before the peak that is below the level, then interpolate to the next index
    below_left = (~above) & (idx < pk[:, None])
    i0 = np.where(below_left.any(axis=1), n - 1 - np.argmax(below_left[:, ::-1], axis=1), -1)
    below_right = (~above) & (idx > pk[:, None])
    i1 = np.where(below_right.any(axis=1), np.argmax(below_right, axis=1), -1)

    def cross(i_a, i_b):
        za, zb = zz[rows, i_a], zz[rows, i_b]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(zb != za, (level - za) / (zb - za), 0.0)
        return days[i_a] + frac * (days[i_b] - days[i_a])

    valid_season = (amp >= min_amplitude) & (i0 >= 0) & (i1 >= 0)
    sos = np.where(valid_season, cross(np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1)), np.nan)
    eos = np.where(valid_season, cross(np.clip(i1 - 1, 0, n - 1), np.clip(i1, 0, n - 1)), np.nan)
    dz = np.gradient(zz, days, axis=1)
    greenup = np.where(idx <= pk[:, None], dz, -np.inf).max(axis=1)
    senesc = np.where(idx >= pk[:, None], dz, np.inf).min(axis=1)
    inside = (days[None, :] >= sos[:, None]) & (days[None, :] <= eos[:, None])
    integral = _trapz(np.where(inside, np.clip(zz - base[:, None], 0, None), 0.0), days, axis=1)
    values = {"peak_value": peak, "peak_doy": days[pk], "amplitude": amp, "sos": sos, "eos": eos, "los": eos - sos,
              "greenup_rate": np.where(valid_season, greenup, np.nan),
              "senescence_rate": np.where(valid_season, senesc, np.nan),
              "integral": np.where(valid_season, integral, np.nan)}
    for k, v in values.items():
        res[k][finite] = v
    return res


def harmonics(y: np.ndarray, w: np.ndarray, days: np.ndarray, order: int = 2, period: float = 365.25) -> dict:
    """Weighted least-squares fit y ≈ a0 + Σ_k a_k cos(2πkt/T) + b_k sin(2πkt/T) per pixel.

    Returns a0, amp_k, phase_k (radians) for k = 1..order and rmse. Needs >= 2*order + 2 observations.
    """
    y = np.asarray(y, dtype="float64")
    w = np.where(np.isfinite(y), np.asarray(w, dtype="float64"), 0.0)
    y0 = np.where(w > 0, y, 0.0)
    t = 2 * np.pi * np.asarray(days, dtype="float64") / period
    cols = [np.ones_like(t)]
    for k in range(1, order + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    X = np.stack(cols, axis=1)                                   # (T, p)
    p = X.shape[1]
    XtWX = np.einsum("tp,nt,tq->npq", X, w, X)
    XtWy = np.einsum("tp,nt->np", X, w * y0)
    enough = (w > 0).sum(axis=1) >= 2 * order + 2
    beta = np.full((len(y), p), np.nan)
    if enough.any():
        beta[enough] = np.linalg.solve(XtWX[enough] + 1e-9 * np.eye(p), XtWy[enough][..., None])[..., 0]
    fit = beta @ X.T
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(np.nansum(w * (y0 - fit) ** 2, axis=1) / np.where(w.sum(axis=1) > 0, w.sum(axis=1), np.nan))
    out = {"a0": beta[:, 0], "rmse": np.where(enough, rmse, np.nan)}
    for k in range(1, order + 1):
        a, b = beta[:, 2 * k - 1], beta[:, 2 * k]
        out[f"amp{k}"] = np.hypot(a, b)
        out[f"phase{k}"] = np.arctan2(b, a)
    return out
