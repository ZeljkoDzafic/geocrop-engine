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
* Weights of every monthly feature provider (Presto, Galileo, AnySat, plugins) - `fetch_weights` downloads the
  provider's pinned files (URL + SHA-256) into ``$GEOCROP_CACHE/<provider>/``.

The pre-computed sources are also exposed as feature providers (`ALPHAEARTH`, `TESSERA`), registered by name
in `geocrop.feature_providers`.
"""
from __future__ import annotations

import io
import math
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
    """Download cache root (`geocrop.cachedirs.cache_root`: $GEOCROP_CACHE, else ~/.cache/geocrop)."""
    from .cachedirs import cache_root
    return cache_root()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download(url: str, target: str | Path, sha256: str | None = None, session: requests.Session | None = None,
             size: int | None = None) -> Path:
    """Download once into ``target`` (`geocrop.downloads.fetch`: retry/backoff, resume, atomic rename) and
    verify the SHA-256 (and size) when given."""
    from .downloads import fetch
    return fetch(url, target, f"sha256:{sha256}" if sha256 else None, size=size, timeout=120,
                 session=session or _session())


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
                     resampling: Resampling = Resampling.nearest, names: list[str] | None = None) -> Path:
    """AlphaEarth annual embeddings (64 bands, dequantised) on the study grid, reading only COG windows."""
    with rasterio.open(reference) as ref:
        tiles = alphaearth_tiles(year, _wgs84_bounds(ref), index, mirror)
        if not tiles:
            raise ValueError(f"No AlphaEarth tiles for {year} cover the grid")
        extra = {"tiles": ";".join(t["url"] for t in tiles), "mirror": mirror, "resampling": resampling.name}
        with rasterio.Env(**GDAL_ENV), emb.EmbeddingWriter(output, reference, "alphaearth", year, 64, extra,
                                                                 names) as sink:
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
                  resampling: Resampling = Resampling.nearest, names: list[str] | None = None) -> Path:
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
        with emb.EmbeddingWriter(output, reference, "tessera", year, 128, extra, names) as sink:
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


# --------------------------------------------------------------------------- feature providers


def fetch_weights(provider, directory: str | Path | None = None) -> Path | list[Path]:
    """Download a provider's pinned weight files once (verified by SHA-256); one path, or a list for several."""
    if not provider.weights:
        raise ValueError(f"Feature provider {provider.name!r} has no weights to download")
    root = Path(directory or cache_dir() / provider.name)
    paths = [download(w.url, root / w.name, w.sha256, size=w.size) for w in provider.weights]
    return paths[0] if len(paths) == 1 else paths


class _Precomputed:
    """Feature provider for published annual embeddings read on the study grid (no model, no weights)."""
    weights: tuple = ()
    requires_licence_ack = False
    legacy_names = True

    def __init__(self, name: str, fetch, contract):
        info = emb.SOURCE_INFO[name]
        self.name, self._fetch, self.contract = name, fetch, contract
        self.dims, self.sensor, self.product = info["dims"], info["sensor"], info["product"]
        self.licence, self.attribution, self.dequantisation = info["licence"].split(" ")[0], info["attribution"], \
            info["dequantisation"]
        self.code_licence = "n/a (published data)"

    def load_model(self, weights=None, **_):
        return None

    def model_dims(self, model) -> int:
        return self.dims

    def embed_pixels(self, reference, output, *, year: int, block_size: int | None = None, resampling: str = "nearest",
                     names: list[str] | None = None, **options) -> Path:
        kw = {k: options[k] for k in ("mirror", "index") if k in options and options[k] is not None}
        return self._fetch(reference, output, year, block_size=block_size or 1024, resampling=Resampling[resampling],
                           names=names, **kw)

    def embed_parcels(self, reference, parcels, output, *, year: int, raster: str | Path | None = None,
                      id_field: str = "parcel_id", min_pixels: int = 1, **options):
        """Parcel means of the pixel embeddings (``raster``, or a fresh `embed_pixels` next to ``output``)."""
        if raster is None:
            raster = Path(output).with_suffix(".tif")
            self.embed_pixels(reference, raster, year=year, **options)
        return emb.raster_parcel_means(raster, parcels, output, id_field, min_pixels=min_pixels)

    def __repr__(self) -> str:
        return f"<FeatureProvider {self.name}>"


def _contract(sensors, bands, temporal, normalisation):
    from .feature_providers import InputContract
    return InputContract(kind="precomputed", sensors=sensors, bands=bands, temporal=temporal,
                         normalisation=normalisation)


ALPHAEARTH = _Precomputed("alphaearth", fetch_alphaearth, _contract(
    ("AlphaEarth Foundations (Sentinel-1/2, Landsat and other inputs, producer side)",),
    tuple(f"A{i:02d}" for i in range(64)), "annual (2017-2025)",
    "int8 dequantised ((q / 127.5) ** 2) * sign(q); unit-length vectors"))
TESSERA = _Precomputed("tessera", fetch_tessera, _contract(
    ("TESSERA (Sentinel-1/2, producer side)",), tuple(f"D{i:03d}" for i in range(128)),
    "annual (coverage varies by year and tile)", "int8 * float32 per-pixel scales"))


# --------------------------------------------------------------------------- one entry point


def extract(source: str, reference: str | Path, out_dir: str | Path, year: int, monthly: str | Path | None = None,
            start: str | None = None, months: int = 12, parcels: str | Path | None = None,
            id_field: str = "parcel_id", checkpoint: str | Path | None = None, mirror: str = "source.coop",
            index: str | Path | None = None, resampling: str = "nearest", block_size: int | None = None,
            namespace: bool | None = None, allow_license: list[str] | tuple[str, ...] | None = None) -> dict:
    """Produce ``out_dir/<source>_<year>.tif`` with any registered feature provider (and, with parcels,
    ``<source>_<year>_parcels.csv``). `namespace` forces ``emb_<source>_NN`` (True) or legacy ``emb_NN`` (False)
    feature names; by default only new providers are namespaced. `allow_license` acknowledges licence ids of
    providers that are not permissive."""
    from . import feature_providers as fp

    provider = fp.get(source)
    fp.check_licence(provider, allow_license)
    target = emb.output_path(out_dir, source, year)
    names = fp.feature_names(provider, namespace=namespace)
    result: dict = {"source": source, "year": int(year)}
    if provider.contract.kind == "precomputed":
        rs = Resampling[resampling].name
        provider.embed_pixels(reference, target, year=year, block_size=block_size, resampling=rs, mirror=mirror,
                              index=index, names=names)
        if parcels:
            table = target.with_name(target.stem + "_parcels.csv")
            provider.embed_parcels(reference, parcels, table, year=year, raster=target, id_field=id_field)
            result["parcels"] = str(table)
    else:
        if monthly is None:
            raise ValueError(f"{source} needs monthly composites (embeddings.monthly / --monthly)")
        start = start or f"{int(year)}-01"
        if checkpoint:
            weights = [Path(p) for p in checkpoint] if isinstance(checkpoint, (list, tuple)) else Path(checkpoint)
            extra = {"checkpoint": str(checkpoint)}
        else:
            weights = presto_checkpoint() if source == "presto" else fetch_weights(provider)
            extra = {"checkpoint_url": PRESTO_CHECKPOINT["url"], "checkpoint_sha256": PRESTO_CHECKPOINT["sha256"]} \
                if source == "presto" else {"checkpoint_url": ";".join(w.url for w in provider.weights),
                                            "checkpoint_sha256": ";".join(w.sha256 for w in provider.weights)}
        model = provider.load_model(weights)
        names = fp.feature_names(provider, provider.model_dims(model), namespace)
        common = {"monthly": monthly, "start": start, "months": months, "model": model, "names": names}
        if block_size:
            common["block_size"] = block_size
        provider.embed_pixels(reference, target, extra=extra, **common)
        if parcels:
            table = target.with_name(target.stem + "_parcels.csv")
            common.pop("block_size", None)
            provider.embed_parcels(reference, parcels, table, id_field=id_field, **common)
            result["parcels"] = str(table)
    result["raster"] = str(target)
    result["feature_names"] = f"{names[0]} ... {names[-1]}"
    result["summary"] = emb.summary(target)
    return result
