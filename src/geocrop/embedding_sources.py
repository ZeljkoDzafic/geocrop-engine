"""Remote embedding sources (gaps G2/G3): windowed reads of AlphaEarth / TESSERA tiles and the pinned Presto
checkpoint. Everything that touches the network lives here; dequantisation, regridding, the output contract and
Presto inference are in the core module `geocrop.embeddings`.

* AlphaEarth Foundations Satellite Embedding V1 - annual 64-d int8 COGs (8192 x 8192, UTM, *bottom-up*), public
  on Source Cooperative (``tge-labs/aef``, default) and Google Cloud Storage (``alphaearth_foundations``),
  CC-BY-4.0. Tiles are found through the official ``aef_index.parquet`` (about 78 MB, cached once; needs
  pyarrow). Only the COG windows covering the study grid are read (HTTP range requests through GDAL).
* TESSERA v1 - annual 128-d int8 ``.npy`` tiles on a 0.1 degree grid plus float32 per-pixel scales, public on
  Source Cooperative (``tessera/tessera``); the georeference of each tile comes from its landmask GeoTIFF. Only
  the needed rows of each ``.npy`` file are fetched (HTTP range requests on the uncompressed array).
* Presto - official ``default_model.pt`` from nasaharvest/presto pinned by commit URL and SHA-256.
"""
from __future__ import annotations

import io
import math
import os
import warnings
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window

from . import embeddings as emb
from .raster import blocks

USER_AGENT = "geocrop-engine (https://github.com/ZeljkoDzafic/geocrop-engine)"
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
            "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2", "GDAL_HTTP_MULTIRANGE": "YES",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES"}

AEF_MIRRORS = {
    "source.coop": "https://data.source.coop/tge-labs/aef/v1/annual",
    "gcs": "https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual",
}
AEF_INDEX = "aef_index.parquet"

TESSERA_URL = "https://data.source.coop/tessera/tessera"
TESSERA_DATASET = "v1"          # npy/<dataset>/<year>/grid_<lon>_<lat>/grid_<lon>_<lat>(_scales).npy
TESSERA_LANDMASKS = "v1"        # landmasks/<version>/grid_<lon>_<lat>.tiff

PRESTO_COMMIT = "11e207a668a34336ced1d8e492a1bd5849b96c4a"
PRESTO_CHECKPOINT = {
    "url": f"https://raw.githubusercontent.com/nasaharvest/presto/{PRESTO_COMMIT}/data/default_model.pt",
    "sha256": "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65",
    "licence": "MIT",
}


def cache_dir() -> Path:
    """Download cache: $GEOCROP_CACHE, else ~/.cache/geocrop."""
    return Path(os.environ.get("GEOCROP_CACHE") or Path.home() / ".cache" / "geocrop")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download(url: str, target: str | Path, sha256: str | None = None, session: requests.Session | None = None) -> Path:
    """Download once into ``target`` (atomic rename) and verify the SHA-256 when given."""
    from .hashing import sha256 as file_sha256

    target = Path(target)
    if target.exists() and (sha256 is None or file_sha256(target) == sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with (session or _session()).get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if sha256 is not None and file_sha256(tmp) != sha256:
        tmp.unlink()
        raise ValueError(f"SHA-256 mismatch for {url}; refusing to use it")
    tmp.replace(target)
    return target


def presto_checkpoint(directory: str | Path | None = None) -> Path:
    """The pinned official Presto checkpoint (downloaded at first use, verified by SHA-256)."""
    target = Path(directory or cache_dir() / "presto") / f"presto_{PRESTO_COMMIT[:12]}_default_model.pt"
    return download(PRESTO_CHECKPOINT["url"], target, PRESTO_CHECKPOINT["sha256"])


def _wgs84_bounds(ref) -> tuple[float, float, float, float]:
    return transform_bounds(ref.crs, "EPSG:4326", *ref.bounds, densify_pts=21)


def _source_window(src_transform, src_shape, dst_transform, dst_crs, dst_shape, src_crs, margin: int = 2):
    """Source pixel window (clipped) covering a destination window; None when they do not overlap."""
    h, w = dst_shape
    left, top = dst_transform * (0, 0)
    right, bottom = dst_transform * (w, h)
    xmin, ymin, xmax, ymax = transform_bounds(dst_crs, src_crs, min(left, right), min(top, bottom),
                                              max(left, right), max(top, bottom), densify_pts=21)
    inv = ~src_transform
    cols, rows = zip(*(inv * (x, y) for x in (xmin, xmax) for y in (ymin, ymax)))
    c0, c1 = math.floor(min(cols)) - margin, math.ceil(max(cols)) + margin
    r0, r1 = math.floor(min(rows)) - margin, math.ceil(max(rows)) + margin
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, src_shape[1]), min(r1, src_shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


# --------------------------------------------------------------------------- AlphaEarth


def alphaearth_index(index: str | Path | None = None, mirror: str = "source.coop") -> Path:
    """Local copy of the official AlphaEarth tile index (GeoParquet)."""
    if index is not None and Path(index).exists():
        return Path(index)
    url = str(index) if index else f"{AEF_MIRRORS[mirror]}/{AEF_INDEX}"
    return download(url, cache_dir() / "alphaearth" / AEF_INDEX)


def alphaearth_tiles(year: int, bbox: tuple[float, float, float, float], index: str | Path | None = None,
                     mirror: str = "source.coop") -> list[dict]:
    """Tiles of one year intersecting a WGS84 bbox: [{"url", "crs", "utm_zone"}]."""
    try:
        import pandas as pd
        table = pd.read_parquet(alphaearth_index(index, mirror),
                                columns=["path", "crs", "year", "utm_zone", "wgs84_west", "wgs84_south",
                                         "wgs84_east", "wgs84_north"],
                                filters=[("year", "==", int(year))])
    except ImportError as exc:    # pragma: no cover - depends on the environment
        raise RuntimeError('Reading the AlphaEarth index needs pyarrow: pip install -e ".[parquet]"') from exc
    w, s, e, n = bbox
    hit = table[(table.wgs84_west <= e) & (table.wgs84_east >= w) & (table.wgs84_south <= n) & (table.wgs84_north >= s)]
    base = AEF_MIRRORS[mirror]
    out = []
    for row in hit.itertuples():
        rel = str(row.path).split("/v1/annual/", 1)[1]
        out.append({"url": f"{base}/{rel}", "crs": row.crs, "utm_zone": row.utm_zone})
    return out


def fetch_alphaearth(reference: str | Path, output: str | Path, year: int, mirror: str = "source.coop",
                     index: str | Path | None = None, block_size: int = 1024,
                     resampling: Resampling = Resampling.nearest) -> Path:
    """AlphaEarth annual embeddings (64 bands, dequantised) on the study grid, reading only COG windows."""
    with rasterio.open(reference) as ref:
        tiles = alphaearth_tiles(year, _wgs84_bounds(ref), index, mirror)
        if not tiles:
            raise ValueError(f"No AlphaEarth tiles for {year} cover the grid")
        extra = {"tiles": ";".join(t["url"] for t in tiles), "mirror": mirror, "resampling": resampling.name}
        with rasterio.Env(**GDAL_ENV), emb.EmbeddingWriter(output, reference, "alphaearth", year, 64, extra) as sink:
            handles = [rasterio.open("/vsicurl/" + t["url"]) for t in tiles]
            handles.sort(key=lambda ds: ds.crs != ref.crs)      # same-zone tiles first (no reprojection)
            try:
                for w in blocks(ref.width, ref.height, block_size):
                    shape = (int(w.height), int(w.width))
                    dst_t = ref.window_transform(w)
                    acc = np.full((64, *shape), np.nan, dtype="float32")
                    for ds in handles:
                        win = _source_window(ds.transform, ds.shape, dst_t, ref.crs, shape, ds.crs)
                        if win is None:
                            continue
                        raw = ds.read(window=win)
                        part = emb.regrid(emb.dequantise_alphaearth(raw), ds.window_transform(win), ds.crs,
                                          dst_t, ref.crs, shape, resampling)
                        fill = ~np.isfinite(acc[0]) & np.isfinite(part).all(axis=0)
                        acc[:, fill] = part[:, fill]
                    sink.write(acc, window=w)
            finally:
                for ds in handles:
                    ds.close()
    return Path(output)


# --------------------------------------------------------------------------- TESSERA


def tessera_tile_centres(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """0.1 degree tile centres (x.x5) whose cells intersect a WGS84 bbox."""
    w, s, e, n = bbox
    lons = np.arange(math.floor(w * 10), math.ceil(e * 10)) / 10 + 0.05
    lats = np.arange(math.floor(s * 10), math.ceil(n * 10)) / 10 + 0.05
    return [(round(float(x), 2), round(float(y), 2)) for y in lats for x in lons]


def tessera_name(lon: float, lat: float) -> str:
    return f"grid_{lon:.2f}_{lat:.2f}"


class RemoteNpy:
    """Row-range reads of a remote C-ordered ``.npy`` array through HTTP range requests."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.session = url, session
        head = self._range(0, 4095)
        fp = io.BytesIO(head)
        major, _ = np.lib.format.read_magic(fp)
        reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        self.shape, fortran, self.dtype = reader(fp)
        if fortran:
            raise ValueError(f"{url}: Fortran-ordered arrays are not supported")
        self.offset = fp.tell()
        self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize

    def _range(self, a: int, b: int) -> bytes:
        r = self.session.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206 and a > 0:
            raise RuntimeError(f"{self.url}: server ignored the range request")
        return r.content if r.status_code == 206 else r.content[a:b + 1]

    def rows(self, r0: int, r1: int) -> np.ndarray:
        a = self.offset + r0 * self.row_bytes
        data = self._range(a, a + (r1 - r0) * self.row_bytes - 1)
        return np.frombuffer(data, dtype=self.dtype).reshape((r1 - r0, *self.shape[1:]))


def _tessera_tile(lon: float, lat: float, year: int, session: requests.Session) -> dict | None:
    name = tessera_name(lon, lat)
    base = f"{TESSERA_URL}/npy/{TESSERA_DATASET}/{int(year)}/{name}/{name}"
    if session.head(f"{base}.npy", timeout=60, allow_redirects=True).status_code == 404:
        return None
    mask = download(f"{TESSERA_URL}/landmasks/{TESSERA_LANDMASKS}/{name}.tiff",
                    cache_dir() / "tessera" / "landmasks" / TESSERA_LANDMASKS / f"{name}.tiff", session=session)
    with rasterio.open(mask) as m:
        crs, transform, shape = m.crs, m.transform, m.shape
    q, s = RemoteNpy(f"{base}.npy", session), RemoteNpy(f"{base}_scales.npy", session)
    if tuple(q.shape[:2]) != tuple(shape) or tuple(s.shape[:2]) != tuple(shape):
        raise ValueError(f"TESSERA {name}: array {q.shape} / scales {s.shape} do not match landmask {shape}")
    return {"name": name, "lon": lon, "lat": lat, "url": f"{base}.npy", "crs": crs, "transform": transform, "shape": shape, "q": q, "s": s}


def fetch_tessera(reference: str | Path, output: str | Path, year: int, block_size: int = 1024,
                  resampling: Resampling = Resampling.nearest) -> Path:
    """TESSERA annual embeddings (128 bands, dequantised) on the study grid, fetching only needed tile rows."""
    session = _session()
    with rasterio.open(reference) as ref:
        tiles, missing = [], []
        for lon, lat in tessera_tile_centres(_wgs84_bounds(ref)):
            t = _tessera_tile(lon, lat, year, session)
            (tiles.append(t) if t else missing.append(tessera_name(lon, lat)))
        if not tiles:
            raise ValueError(f"No TESSERA {TESSERA_DATASET} tiles for {year} cover the grid")
        if missing:
            warnings.warn(f"TESSERA tiles without coverage for {year}: {', '.join(missing)}", stacklevel=2)
        extra = {"tiles": ";".join(t["url"] for t in tiles), "missing_tiles": ";".join(missing),
                 "dataset": TESSERA_DATASET, "resampling": resampling.name}
        with emb.EmbeddingWriter(output, reference, "tessera", year, 128, extra) as sink:
            for w in blocks(ref.width, ref.height, block_size):
                shape = (int(w.height), int(w.width))
                dst_t = ref.window_transform(w)
                acc = np.full((128, *shape), np.nan, dtype="float32")
                lat, lon = _pixel_lonlat(ref, w)
                # Neighbouring tiles overlap in UTM; a pixel is taken from the tile whose 0.1 degree cell
                # contains its centre, other tiles only fill what is still missing.
                parts = []
                for t in tiles:
                    win = _source_window(t["transform"], t["shape"], dst_t, ref.crs, shape, t["crs"])
                    if win is None:
                        continue
                    r0, r1 = int(win.row_off), int(win.row_off + win.height)
                    c0, c1 = int(win.col_off), int(win.col_off + win.width)
                    data = np.moveaxis(emb.dequantise_tessera(t["q"].rows(r0, r1)[:, c0:c1],
                                                              t["s"].rows(r0, r1)[:, c0:c1]), -1, 0)
                    part = emb.regrid(data, rasterio.windows.transform(win, t["transform"]), t["crs"], dst_t,
                                      ref.crs, shape, resampling)
                    parts.append((part, (abs(lon - t["lon"]) <= 0.05) & (abs(lat - t["lat"]) <= 0.05)))
                for part, own in [*parts, *((p, None) for p, _ in parts)]:
                    fill = ~np.isfinite(acc[0]) & np.isfinite(part).all(axis=0)
                    if own is not None:
                        fill &= own
                    acc[:, fill] = part[:, fill]
                sink.write(acc, window=w)
    return Path(output)


def _pixel_lonlat(ref, window) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    rows, cols = np.mgrid[int(window.row_off):int(window.row_off + window.height),
                          int(window.col_off):int(window.col_off + window.width)]
    xs, ys = rasterio.transform.xy(ref.transform, rows.ravel(), cols.ravel(), offset="center")
    lon, lat = Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    return np.asarray(lat).reshape(rows.shape), np.asarray(lon).reshape(rows.shape)


# --------------------------------------------------------------------------- one entry point


def extract(source: str, reference: str | Path, out_dir: str | Path, year: int, monthly: str | Path | None = None,
            start: str | None = None, months: int = 12, parcels: str | Path | None = None,
            id_field: str = "parcel_id", checkpoint: str | Path | None = None, mirror: str = "source.coop",
            index: str | Path | None = None, resampling: str = "nearest", block_size: int | None = None) -> dict:
    """Produce ``out_dir/<source>_<year>.tif`` (and, for Presto with parcels, ``<...>_parcels.csv``)."""
    rs = Resampling[resampling]
    target = emb.output_path(out_dir, source, year)
    result: dict = {"source": source, "year": int(year)}
    if source == "alphaearth":
        fetch_alphaearth(reference, target, year, mirror, index, block_size or 1024, rs)
    elif source == "tessera":
        fetch_tessera(reference, target, year, block_size or 1024, rs)
    else:
        if monthly is None:
            raise ValueError("Presto needs monthly composites (embeddings.monthly / --monthly)")
        start = start or f"{int(year)}-01"
        ckpt = Path(checkpoint) if checkpoint else presto_checkpoint()
        model = emb.load_presto(ckpt)
        extra = {"checkpoint_url": PRESTO_CHECKPOINT["url"], "checkpoint_sha256": PRESTO_CHECKPOINT["sha256"]} \
            if not checkpoint else {"checkpoint": str(checkpoint)}
        emb.presto_pixels(monthly, reference, target, start, months, model=model, block_size=block_size or 256,
                          extra=extra)
        if parcels:
            table = target.with_name(target.stem + "_parcels.csv")
            emb.presto_parcels(monthly, reference, parcels, table, start, months, id_field, model=model)
            result["parcels"] = str(table)
    result["raster"] = str(target)
    result["summary"] = emb.summary(target)
    return result
