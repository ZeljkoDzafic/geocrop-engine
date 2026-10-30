"""Cloud-native Sentinel-1 RTC import (gamma0, linear power) from Microsoft Planetary Computer, without SNAP.

The `sentinel-1-rtc` collection on Planetary Computer holds radiometrically terrain-corrected Sentinel-1 IW GRD
backscatter (gamma0 terrain-flattened, Small 2011; processed by Catalyst with PlanetDEM) as Cloud-Optimised
GeoTIFFs on 10 m UTM grids. Only the window covering the study grid is read over HTTP and reprojected onto it,
so the output is the standard Sentinel-1 scene folder used by `geocrop.s1.composite`:

    VV.tif, VH.tif   gamma0, linear power, float32, nodata -9999 (non-positive values are no-data)
    scene.json       sensor S1, datetime, orbit_direction (from `sat:orbit_state`), relative/absolute orbit,
                     platform, backscatter "gamma0", source item ids, grid id, coverage

Slices of one datatake (same platform and absolute orbit) that each cover only part of the grid are mosaicked
into one acquisition, so an overpass is never counted twice in a temporal mean.

Access. Assets are Azure Blob URLs that must be signed with a short-lived SAS token. The optional
`planetary-computer` package is used when installed; otherwise the token is requested from the public
SAS endpoint with `requests` (anonymous tokens work at the time of writing; a subscription key from the
`PC_SDK_SUBSCRIPTION_KEY` environment variable is sent when set).

Licence. The RTC collection is distributed under CC BY 4.0 (Microsoft, processed by Catalyst) and is derived
from Copernicus Sentinel-1 data (ESA). Products must carry: "Contains modified Copernicus Sentinel data
<year>; Sentinel-1 RTC from Microsoft Planetary Computer (CC BY 4.0), processed by Catalyst."
"""
from __future__ import annotations

import os
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, transform_bounds

from . import attribution
from . import scenes as scene_meta
from .aoi import AoiMask, creation_options
from .indices import S1_POLARISATIONS
from .parallel import import_workers, ordered_map, with_backoff
from .raster import NODATA, float_profile
from .stac_import import (
    GDAL_ENV,
    _region_wgs84,
    footprint_intersects,
    grid_identity,
    read_window_aligned,
    read_window_aligned_active,
)

PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
ASSET_KEYS = {"VV": "vv", "VH": "vh"}
LICENCE = "CC-BY-4.0"
ATTRIBUTION = ("Contains modified Copernicus Sentinel data {year}; Sentinel-1 RTC from Microsoft Planetary "
               "Computer (CC BY 4.0), processed by Catalyst.")
S1_GDAL_ENV = {k: v for k, v in GDAL_ENV.items() if k != "AWS_NO_SIGN_REQUEST"}


class ServiceUnavailable(RuntimeError):
    """The catalogue or token service could not be reached (not an engine defect)."""


class Signer:
    """Appends a Planetary Computer SAS token to Azure Blob asset URLs; tokens are cached until shortly before expiry."""

    def __init__(self, collection: str = COLLECTION, token_url: str = TOKEN_URL, timeout: float = 30):
        self.collection, self.token_url, self.timeout = collection, token_url, timeout
        self._token: str | None = None
        self._expiry = 0.0
        self._lock = threading.Lock()      # one token request at a time when scenes are imported by threads

    def token(self) -> str:
        with self._lock:
            return self._token_locked()

    def _token_locked(self) -> str:
        if self._token and time.time() < self._expiry - 300:
            return self._token
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests ships with pystac-client
            raise RuntimeError("pip install requests (or planetary-computer) to sign Sentinel-1 RTC assets") from exc
        headers = {}
        if os.environ.get("PC_SDK_SUBSCRIPTION_KEY"):
            headers["Ocp-Apim-Subscription-Key"] = os.environ["PC_SDK_SUBSCRIPTION_KEY"]
        try:
            response = requests.get(self.token_url.format(collection=self.collection), headers=headers,
                                    timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            token = body["token"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise ServiceUnavailable(f"Planetary Computer token service unavailable: {exc}") from exc
        expiry = time.time() + 1800
        if body.get("msft:expiry"):
            try:
                expiry = datetime.fromisoformat(body["msft:expiry"].replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        self._token, self._expiry = token.lstrip("?"), expiry
        return self._token

    def sign(self, href: str) -> str:
        if ".blob.core.windows.net" not in href or "sig=" in href:
            return href
        try:
            import planetary_computer
            return planetary_computer.sign(href)
        except ImportError:
            pass
        return f"{href}{'&' if '?' in href else '?'}{self.token()}"


def cached_scene_valid(folder: Path, grid_id: str) -> bool:
    meta = folder / scene_meta.SIDE_CAR
    if not meta.exists():
        return False
    try:
        body = scene_meta.read(folder)
    except (ValueError, KeyError):
        return False
    return body.get("grid_id") == grid_id and all((folder / f"{p}.tif").is_file() for p in S1_POLARISATIONS)


def read_item(item, reference: str | Path, signer: Signer | None = None,
              asset_keys: dict | None = None, aoi: AoiMask | None = None) -> dict[str, np.ndarray]:
    """Linear gamma0 VV/VH of one STAC item on the reference grid (NaN where not covered or non-positive).

    With an AOI mask only the COG windows of the blocks that touch the AOI (plus halo) are read."""
    keys = asset_keys or ASSET_KEYS
    signer = signer or Signer()
    out = {}
    with rasterio.open(reference) as ref, rasterio.Env(**S1_GDAL_ENV):
        shape, transform, crs = ref.shape, ref.transform, ref.crs
        for pol in S1_POLARISATIONS:
            if keys[pol] not in item.assets:
                raise ValueError(f"{item.id}: no {pol} asset")
            with rasterio.open(signer.sign(item.assets[keys[pol]].href)) as src:
                if aoi is None:
                    v = read_window_aligned(src, transform, crs, shape, Resampling.bilinear, "float32", np.nan)
                else:
                    aoi.check_grid(ref, "the reference grid")
                    v = read_window_aligned_active(src, transform, crs, shape, Resampling.bilinear, "float32",
                                                   np.nan, aoi.windows())
            out[pol] = np.where(np.isfinite(v) & (v > 0), v, np.nan).astype("float32")
    return out


def _coverage(arrays: dict[str, np.ndarray], aoi: AoiMask | None) -> float:
    """Fraction of the grid (of the AOI pixels with a mask) with finite VV and VH."""
    covered = np.isfinite(arrays["VV"]) & np.isfinite(arrays["VH"])
    if aoi is None:
        return float(covered.mean())
    inside = aoi.dense()
    return float((covered & inside).sum() / max(1, inside.sum()))


def write_scene(arrays: dict[str, np.ndarray], items: list, reference: str | Path, out: str | Path,
                collection: str = COLLECTION, endpoint: str = PLANETARY_COMPUTER,
                aoi: AoiMask | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference) as ref:
        profile = float_profile(ref, **creation_options(aoi))
    atags = attribution.tags([items[0].properties.get("datetime")])
    for pol in S1_POLARISATIONS:
        with rasterio.open(out / f"{pol}.tif", "w", **profile) as dst:
            dst.write(np.where(np.isfinite(arrays[pol]), arrays[pol], NODATA).astype("float32"), 1)
            dst.set_band_description(1, f"gamma0_{pol}_linear")
            dst.update_tags(**atags, licence=LICENCE,
                            attribution=ATTRIBUTION.format(year=str(items[0].properties.get("datetime", ""))[:4]))
    p = items[0].properties
    coverage = _coverage(arrays, aoi)
    year = str(p.get("datetime", ""))[:4]
    scene_meta.write(out, sensor="S1", datetime=p.get("datetime"),
                     orbit_direction=str(p.get("sat:orbit_state", "")).upper() or None,
                     relative_orbit=p.get("sat:relative_orbit"), absolute_orbit=p.get("sat:absolute_orbit"),
                     platform=p.get("platform"), backscatter="gamma0", domain="linear",
                     orbit_source=p.get("s1:orbit_source"), product_timeliness=p.get("s1:product_timeliness"),
                     source=[i.id for i in items], collection=collection, endpoint=endpoint,
                     stac_self=[i.get_self_href() for i in items], coverage=round(coverage, 4),
                     grid_id=grid_identity(reference, aoi), licence=LICENCE,
                     attribution=ATTRIBUTION.format(year=year),
                     **({"aoi_mask": aoi.path.name, "aoi_halo_px": aoi.halo} if aoi is not None else {}),
                     note="Windowed COG import of Planetary Computer Sentinel-1 RTC (terrain-flattened gamma0) "
                          "on the reference grid; no SNAP processing")
    return {"path": str(out), "coverage": coverage}


def import_item(item, reference: str | Path, out_dir: str | Path, signer: Signer | None = None,
                min_coverage: float = 0.5, aoi: AoiMask | None = None) -> dict:
    arrays = read_item(item, reference, signer, aoi=aoi)
    coverage = _coverage(arrays, aoi)
    if coverage < min_coverage:
        return {"item": item.id, "status": f"skipped: coverage {coverage:.2f}"}
    return {"item": item.id, "status": "imported", **write_scene(arrays, [item], reference, out_dir, aoi=aoi)}


def _acquisition_key(item) -> str:
    p = item.properties
    if p.get("sat:absolute_orbit") is not None:
        return f"{p.get('platform', 'S1')}_{p['sat:absolute_orbit']}"
    return item.id


def _folder_name(items) -> str:
    p = items[0].properties
    stamp = str(p.get("datetime", ""))[:19].replace("-", "").replace(":", "")
    platform = str(p.get("platform", "S1")).replace("SENTINEL-", "S")
    return f"{platform}_{stamp}_{str(p.get('sat:orbit_state', 'x'))[:3].upper()}_R{p.get('sat:relative_orbit', 0):03d}"


def search(reference: str | Path, start: str, end: str, endpoint: str = PLANETARY_COMPUTER,
           collection: str = COLLECTION, max_items: int = 200, orbit_direction: str | None = None) -> list:
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError('pip install "geocrop-engine[stac]"') from exc
    with rasterio.open(reference) as ref:
        bbox = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
    query = {"sat:orbit_state": {"eq": orbit_direction.lower()}} if orbit_direction else None
    try:
        items = list(Client.open(endpoint).search(collections=[collection], bbox=list(bbox), datetime=f"{start}/{end}",
                                                  query=query, max_items=max_items).items())
    except Exception as exc:  # pystac-client raises several transport-specific exceptions
        raise ServiceUnavailable(f"Sentinel-1 STAC search failed at {endpoint}: {exc}") from exc
    return sorted(items, key=lambda i: i.properties.get("datetime", ""))


def import_season(reference: str | Path, out_root: str | Path, start: str, end: str,
                  endpoint: str = PLANETARY_COMPUTER, collection: str = COLLECTION, min_coverage: float = 0.9,
                  max_items: int = 200, orbit_direction: str | None = None, signer: Signer | None = None,
                  workers: int | None = None, aoi: AoiMask | None = None) -> list[dict]:
    """Search the RTC collection over the grid footprint and write one scene folder per overpass.

    Overpasses covering less than `min_coverage` of the grid (after mosaicking slices) are skipped. Per-item
    read failures are reported in the result list; a failing catalogue or token service raises ServiceUnavailable.

    With an AOI mask (`geocrop.aoi`) items that miss every block touching the AOI are dropped before any read,
    only the COG windows of those blocks are read, and coverage is measured over the AOI pixels.

    Overpasses are read by up to `workers` threads (default 4, at most 8; 1 = sequential); the result list and scene
    folders do not depend on `workers`, and throttling answers (HTTP 429/503) are retried with exponential backoff.
    """
    items = search(reference, start, end, endpoint, collection, max_items, orbit_direction)
    if aoi is not None:
        region = _region_wgs84(aoi)
        items = [i for i in items if footprint_intersects(i, region)]
    signer = signer or Signer(collection)
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(_acquisition_key(item), []).append(item)
    grid_id = grid_identity(reference, aoi)
    workers = import_workers(workers)
    results, todo = [], []
    for members in groups.values():
        ids = [i.id for i in members]
        target = Path(out_root) / _folder_name(members)
        if cached_scene_valid(target, grid_id):
            results.append({"item": ids, "status": "cached", "path": str(target),
                            "coverage": scene_meta.read(target).get("coverage")})
            continue
        (target / scene_meta.SIDE_CAR).unlink(missing_ok=True)
        results.append(members)
        todo.append(members)
    read = with_backoff(read_item) if workers > 1 else read_item

    def one(members):
        ids = [i.id for i in members]
        target = Path(out_root) / _folder_name(members)
        try:
            mosaic = None
            for item in members:
                arrays = read(item, reference, signer, aoi=aoi)
                if mosaic is None:
                    mosaic = arrays
                else:
                    for pol in S1_POLARISATIONS:
                        fill = ~np.isfinite(mosaic[pol])
                        mosaic[pol][fill] = arrays[pol][fill]
            coverage = _coverage(mosaic, aoi)
            if coverage < min_coverage:
                return {"item": ids, "status": f"skipped: coverage {coverage:.2f}"}
            written = write_scene(mosaic, members, reference, target, collection, endpoint, aoi=aoi)
            return {"item": ids, "status": "imported", **written}
        except ServiceUnavailable:
            raise
        except (ValueError, rasterio.errors.RasterioIOError) as exc:
            return {"item": ids, "status": f"failed: {exc}"}
    done = iter(list(ordered_map(one, todo, workers)))
    return [next(done) if not isinstance(r, dict) else r for r in results]


def plausibility(scene_dir: str | Path) -> dict:
    """Physical sanity statistics of one S1 scene folder (linear gamma0 VV/VH)."""
    folder = Path(scene_dir)
    with rasterio.open(folder / "VV.tif") as a, rasterio.open(folder / "VH.tif") as b:
        vv, vh = a.read(1), b.read(1)
    ok = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0) & (vv != NODATA) & (vh != NODATA)
    if not ok.any():
        return {"valid_fraction": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vv_db, vh_db = 10 * np.log10(vv[ok]), 10 * np.log10(vh[ok])
    lin = np.concatenate([vv[ok], vh[ok]])
    return {"valid_fraction": float(ok.mean()),
            "linear_in_1e-4_1": float(((lin >= 1e-4) & (lin <= 1)).mean()),
            "vv_db_p1_p50_p99": [round(float(x), 2) for x in np.percentile(vv_db, [1, 50, 99])],
            "vh_db_p1_p50_p99": [round(float(x), 2) for x in np.percentile(vh_db, [1, 50, 99])],
            "vv_gt_vh": float((vv[ok] > vh[ok]).mean())}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- co-registration to Sentinel-2

def nan_smooth(a: np.ndarray, size: int = 3) -> np.ndarray:
    """NaN-aware moving mean (pixels with < half valid neighbours stay NaN)."""
    from scipy.ndimage import uniform_filter
    ok = np.isfinite(a)
    num = uniform_filter(np.where(ok, a, 0).astype("float64"), size)
    den = uniform_filter(ok.astype("float64"), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, num / den, np.nan)


def shift_surface(reference: np.ndarray, moving: np.ndarray, max_shift: int) -> np.ndarray:
    """Pearson r between `reference[y, x]` and `moving[y + dy, x + dx]` for |dy|, |dx| <= max_shift."""
    h, w = reference.shape
    m = max_shift
    core = reference[m:h - m, m:w - m]
    surface = np.full((2 * m + 1, 2 * m + 1), np.nan)
    for i, dy in enumerate(range(-m, m + 1)):
        for j, dx in enumerate(range(-m, m + 1)):
            other = moving[m + dy:h - m + dy, m + dx:w - m + dx]
            ok = np.isfinite(core) & np.isfinite(other)
            if ok.sum() > 100:
                surface[i, j] = np.corrcoef(core[ok], other[ok])[0, 1]
    return surface


def surface_peak(surface: np.ndarray) -> dict:
    """Integer argmax of a correlation surface with parabolic sub-pixel refinement per axis."""
    m = surface.shape[0] // 2
    i, j = np.unravel_index(np.nanargmax(surface), surface.shape)

    def refine(lo, c, hi):
        den = lo - 2 * c + hi
        return 0.0 if not np.isfinite(den) or den >= 0 else float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))

    on_edge = i in (0, 2 * m) or j in (0, 2 * m)
    dy = i - m + (0.0 if on_edge else refine(surface[i - 1, j], surface[i, j], surface[i + 1, j]))
    dx = j - m + (0.0 if on_edge else refine(surface[i, j - 1], surface[i, j], surface[i, j + 1]))
    ring = np.concatenate([surface[0], surface[-1], surface[1:-1, 0], surface[1:-1, -1]])
    return {"dy": round(float(dy), 2), "dx": round(float(dx), 2), "r_peak": float(surface[i, j]),
            "r_zero": float(surface[m, m]), "r_ring": float(np.nanmean(ring)), "on_edge": bool(on_edge)}


def _vh_db(scene: Path) -> np.ndarray:
    with rasterio.open(Path(scene) / "VH.tif") as ds:
        v = ds.read(1).astype("float64")
    v[(v == NODATA) | ~np.isfinite(v) | (v <= 0)] = np.nan
    return 10 * np.log10(v)


def estimate_offsets(scene_dirs, reference_ndvi: np.ndarray, max_shift: int = 8, min_r: float = 0.2,
                     smooth: int = 5) -> dict:
    """Residual geolocation offset of Sentinel-1 per acquisition geometry relative to a Sentinel-2 NDVI field.

    Scenes are grouped by orbit direction and relative orbit (the offset of terrain-corrected products is
    geometry-dependent), the mean VH (dB) of each group is cross-correlated with NDVI (green vegetation such as
    crops and tree lines raises cross-polarised volume scattering) and the correlation peak is refined to
    sub-pixel precision. Returns {"<DIR>_R<orbit>": {"dy", "dx", "r_peak", "accepted", ...}} in pixels, with the
    convention reference[y, x] ~ sentinel1[y + dy, x + dx]. An estimate is accepted only for a clear interior
    peak (r >= min_r, not on the search border).
    """
    groups: dict[str, list[Path]] = {}
    for d in scene_dirs:
        meta = scene_meta.read(d)
        key = f"{str(meta.get('orbit_direction') or 'NA')[:3]}_R{meta.get('relative_orbit') or 0:03d}"
        groups.setdefault(key, []).append(Path(d))
    ref = nan_smooth(reference_ndvi, smooth)
    out = {}
    for key, members in sorted(groups.items()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vh = np.nanmean(np.stack([_vh_db(d) for d in members]), axis=0)
        peak = surface_peak(shift_surface(ref, nan_smooth(vh, smooth), max_shift))
        peak["scenes"] = len(members)
        peak["accepted"] = bool(peak["r_peak"] >= min_r and not peak["on_edge"])
        out[key] = peak
    return out


def apply_offsets(scene_dirs, offsets: dict, out_root: str | Path, min_offset_px: float = 0.5) -> list[Path]:
    """Write co-registered copies of S1 scene folders (bilinear shift in linear power; uncovered edges no-data).

    Scenes whose geometry has no accepted offset, or an offset below `min_offset_px` (estimation noise), are
    copied unchanged and flagged. The source folders are never modified, so cached imports stay raw.
    """
    from scipy.ndimage import shift as nd_shift

    written = []
    for d in scene_dirs:
        d = Path(d)
        meta = scene_meta.read(d)
        key = f"{str(meta.get('orbit_direction') or 'NA')[:3]}_R{meta.get('relative_orbit') or 0:03d}"
        est = offsets.get(key, {})
        use = bool(est.get("accepted")) and float(np.hypot(est["dy"], est["dx"])) >= min_offset_px
        dy, dx = (est["dy"], est["dx"]) if use else (0.0, 0.0)
        target = Path(out_root) / d.name
        target.mkdir(parents=True, exist_ok=True)
        for pol in S1_POLARISATIONS:
            with rasterio.open(d / f"{pol}.tif") as src:
                profile, desc = float_profile(src), src.descriptions[0]
                v = src.read(1).astype("float64")
            v[(v == NODATA) | ~np.isfinite(v) | (v <= 0)] = np.nan
            if dy or dx:
                v = nd_shift(v, (-dy, -dx), order=1, mode="constant", cval=np.nan)
            with rasterio.open(target / f"{pol}.tif", "w", **profile) as dst:
                dst.write(np.where(np.isfinite(v) & (v > 0), v, NODATA).astype("float32"), 1)
                attribution.propagate([d / f"{pol}.tif"], dst)
                if desc:
                    dst.set_band_description(1, desc)
        body = {k: v for k, v in meta.items() if k not in ("date", "path")}
        body["coregistration"] = {"applied": use, "dy_px": dy, "dx_px": dx, "geometry": key,
                                  "reference": "Sentinel-2 NDVI", "estimate": est, "raw_scene": d.name}
        scene_meta.write(target, **body)
        written.append(target)
    return written
