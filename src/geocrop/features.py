"""Multi-temporal optical + radar feature stack with a machine-readable feature manifest.

Input: a composites directory with one folder per phenological window, e.g.

    composites/P1/B02.tif ... B12.tif   (Sentinel-2 composite, reflectance)
    composites/P1/VV.tif, VH.tif         (Sentinel-1 composite, linear) or ASC_VV.tif, DESC_VH.tif, ...

Output: one float32 GeoTIFF (band descriptions = feature names) and `<stack>.features.json`
describing every feature (group, sensor, period, formula). Feature groups:

    s2_band    raw reflectance                    P2_B8A
    s2_index   spectral index                     P2_NDVI
    s1         radar feature (dB / ratio)         P2_VH_dB, P2_ASC_VHVV_dB
    delta      change between consecutive windows delta_NDVI_P1_P2
    seasonal   statistic across all windows       season_NDVI_max
"""
from __future__ import annotations

import json
import warnings
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio

from . import indices
from .raster import NODATA, blocks, ensure_parent, float_profile

GROUPS = ("s2_band", "s2_index", "s1", "delta", "seasonal", "ts", "phenology", "harmonic")
SEASON_GROUPS = ("seasonal", "ts", "phenology", "harmonic")   # span the whole season
SEASONAL_STATS = ("max", "min", "range", "argmax")


@dataclass
class Feature:
    name: str
    group: str
    sensor: str
    period: str | None = None
    periods: tuple[str, ...] = ()
    source: str = ""


def manifest_path(stack: str | Path) -> Path:
    stack = Path(stack)
    return stack.with_name(stack.stem + ".features.json")


def load_manifest(stack: str | Path) -> list[Feature]:
    path = manifest_path(stack)
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest {path} missing; rebuild the stack with geocrop features")
    body = json.loads(path.read_text(encoding="utf8"))
    return [Feature(**{**f, "periods": tuple(f.get("periods", ()))}) for f in body["features"]]


def select(features: list[Feature], groups=None, periods=None, names=None, exclude=None) -> list[str]:
    """Feature subset for ablation experiments.

    groups: subset of GROUPS; periods: windows to keep (deltas need both ends; seasonal
    features are dropped when periods are restricted, as they summarise all windows);
    names: explicit names (combined with AND); exclude: substrings to drop.
    """
    chosen = []
    for f in features:
        if groups and f.group not in groups:
            continue
        if periods:
            if f.group in SEASON_GROUPS:
                continue
            span = f.periods or ((f.period,) if f.period else ())
            if not set(span) <= set(periods):
                continue
        if names and f.name not in names:
            continue
        if exclude and any(x in f.name for x in exclude):
            continue
        chosen.append(f.name)
    if not chosen:
        raise ValueError("Feature selection is empty")
    return chosen


def _s1_prefixes(folder: Path) -> list[str]:
    prefixes = []
    for prefix in ("", "ASC_", "DESC_"):
        if all((folder / f"{prefix}{p}.tif").exists() for p in indices.S1_POLARISATIONS):
            prefixes.append(prefix)
    return prefixes


def plan(composites: str | Path, periods=None, s2_bands=indices.S2_BANDS, s2_indices=indices.DEFAULT_S2_INDICES,
         s1=indices.DEFAULT_S1_FEATURES, deltas=("NDVI", "NDRE", "NDMI"), seasonal=("NDVI",)):
    """Inspect the composite folders and decide which features can be produced."""
    root = Path(composites)
    periods = list(periods) if periods else sorted(p.name for p in root.iterdir() if p.is_dir())
    if not periods:
        raise ValueError(f"No window folders in {root}")
    has_s2 = [all((root / p / f"{b}.tif").exists() for b in indices.S2_BANDS) for p in periods]
    prefixes = [_s1_prefixes(root / p) for p in periods]
    use_s2 = any(has_s2)
    if use_s2 and not all(has_s2):
        raise ValueError("Sentinel-2 composites must exist for every window (or none)")
    use_s1 = bool(s1) and any(prefixes)
    if use_s1 and any(p != prefixes[0] for p in prefixes):
        raise ValueError("Sentinel-1 composites must have the same orbit layout in every window")
    if not use_s2 and not use_s1:
        raise ValueError("No Sentinel-2 or Sentinel-1 composites found")
    for name in list(s2_indices) + list(s1):
        indices.get(name)

    features: list[Feature] = []
    per_period: list[str] = []
    if use_s2:
        per_period += [b for b in s2_bands]
        per_period += [i for i in s2_indices]
    s1_names = [f"{pre}{n}" for pre in (prefixes[0] if use_s1 else []) for n in s1]
    per_period += s1_names
    for p in periods:
        if use_s2:
            features += [Feature(f"{p}_{b}", "s2_band", "S2", p, (p,), b) for b in s2_bands]
            features += [Feature(f"{p}_{i}", "s2_index", "S2", p, (p,), indices.get(i).formula) for i in s2_indices]
        for n in s1_names:
            base = n.split("_", 1)[1] if n.startswith(("ASC_", "DESC_")) else n
            features.append(Feature(f"{p}_{n}", "s1", "S1", p, (p,), indices.get(base).formula))
    delta_names = [n for n in per_period if n in deltas or (n.split("_", 1)[-1] in deltas and n.startswith(("ASC_", "DESC_")))]
    for n in delta_names:
        for a, b in zip(periods[:-1], periods[1:]):
            sensor = "S2" if n in per_period[: len(per_period) - len(s1_names)] else "S1"
            features.append(Feature(f"delta_{n}_{a}_{b}", "delta", sensor, None, (a, b), f"{b}_{n} - {a}_{n}"))
    if len(periods) > 1:
        for n in [n for n in per_period if n in seasonal]:
            for stat in SEASONAL_STATS:
                features.append(Feature(f"season_{n}_{stat}", "seasonal", "S2" if n in s2_indices else "S1",
                                        None, tuple(periods), f"{stat} of {n} over windows"))
    return periods, features, {"s2": use_s2, "s1_prefixes": prefixes[0] if use_s1 else []}


def build_stack(composites: str | Path, output: str | Path, block_size: int = 512, **options) -> Path:
    """Compute the feature stack window by window: memory is O(features x block²), not O(AOI)."""
    root = Path(composites)
    periods, features, layout = plan(root, **options)
    s2_bands = options.get("s2_bands", indices.S2_BANDS)
    s1_feats = options.get("s1", indices.DEFAULT_S1_FEATURES)
    s2_idx = options.get("s2_indices", indices.DEFAULT_S2_INDICES)
    names = [f.name for f in features]
    with ExitStack() as stack:
        sources = {}
        for p in periods:
            if layout["s2"]:
                for b in indices.S2_BANDS:
                    sources[p, b] = stack.enter_context(rasterio.open(root / p / f"{b}.tif"))
            for pre in layout["s1_prefixes"]:
                for pol in indices.S1_POLARISATIONS:
                    sources[p, pre + pol] = stack.enter_context(rasterio.open(root / p / f"{pre}{pol}.tif"))
        first = next(iter(sources.values()))
        if first.crs is None:
            raise ValueError("Composites need a CRS")
        for key, ds in sources.items():
            if (ds.shape, ds.crs) != (first.shape, first.crs) or not ds.transform.almost_equals(first.transform):
                raise ValueError(f"Grid mismatch for {key}; composite all windows on one reference grid")
        out = ensure_parent(output)
        with rasterio.open(out, "w", **float_profile(first, count=len(names))) as sink:
            for i, n in enumerate(names, 1):
                sink.set_band_description(i, n)
            for w in blocks(first.width, first.height, block_size):
                values: dict[str, np.ndarray] = {}
                for p in periods:
                    raw = {}
                    for key in [k for k in sources if k[0] == p]:
                        ds = sources[key]
                        v = ds.read(1, window=w).astype("float32")
                        if ds.nodata is not None:
                            v[v == ds.nodata] = np.nan
                        raw[key[1]] = v
                    if layout["s2"]:
                        for b in s2_bands:
                            values[f"{p}_{b}"] = raw[b]
                        for i in s2_idx:
                            values[f"{p}_{i}"] = indices.get(i).compute(raw)
                    for pre in layout["s1_prefixes"]:
                        pol = {q: raw[pre + q] for q in indices.S1_POLARISATIONS}
                        for n in s1_feats:
                            values[f"{p}_{pre}{n}"] = indices.get(n).compute(pol)
                for f in features:
                    if f.group == "delta":
                        a, b = f.periods
                        base = f.name[len("delta_"): -len(f"_{a}_{b}")]
                        values[f.name] = values[f"{b}_{base}"] - values[f"{a}_{base}"]
                    elif f.group == "seasonal":
                        base, stat = f.name[len("season_"):].rsplit("_", 1)
                        series = np.stack([values[f"{p}_{base}"] for p in periods])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            if stat == "max":
                                values[f.name] = np.nanmax(series, axis=0)
                            elif stat == "min":
                                values[f.name] = np.nanmin(series, axis=0)
                            elif stat == "range":
                                values[f.name] = np.nanmax(series, axis=0) - np.nanmin(series, axis=0)
                            else:
                                complete = np.isfinite(series).all(axis=0)
                                arg = np.argmax(np.where(np.isfinite(series), series, -np.inf), axis=0)
                                values[f.name] = np.where(complete, arg + 1, np.nan).astype("float32")
                for i, n in enumerate(names, 1):
                    v = values[n]
                    sink.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), i, window=w)
    manifest_path(out).write_text(json.dumps({
        "stack": out.name, "periods": periods, "nodata": NODATA,
        "features": [asdict(f) for f in features]}, indent=2), encoding="utf8")
    return out
