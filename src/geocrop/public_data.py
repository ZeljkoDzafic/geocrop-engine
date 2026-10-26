"""Run-time access to public reference datasets (never bundled with the package).

* **EuroCrops** (Schneider et al., 2023; Zenodo record 14094196, CC-BY-SA-4.0): harmonised IACS/LPIS parcel
  declarations with HCAT codes, one zipped shapefile per country/region and year (e.g. ``SI_2021.zip``).
* **WorldCereal crop calendars** (Franch et al., 2022; PANGAEA doi:10.1594/PANGAEA.946550, CC-BY-SA-4.0): global
  0.5° GeoTIFFs of start (SOS) and end (EOS) of season, day of year, for maize (``M1_*``) and winter wheat
  (``WW_*``). PANGAEA stores the files on tape: the first request answers ``503`` with ``Retry-After`` while the
  file is staged, so `fetch` polls until it is available.

Files are downloaded into a cache directory (``$GEOCROP_DATA_CACHE`` or ``~/.cache/geocrop/public``) and
verified against a pinned checksum when one is given. Derived labels and calendars inherit the ShareAlike terms
of their sources; do not commit or redistribute them without attribution and the same licence.
"""
from __future__ import annotations

import hashlib
import os
import time
import zipfile
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api/records"
EUROCROPS_RECORD = "14094196"
EUROCROPS_CITATION = ("Schneider, M., Schelte, T., Schmitz, F., Körner, M. (2023). EuroCrops: The Largest Harmonized "
                      "Open Crop Dataset Across the European Union. Scientific Data 10, 612. "
                      "doi:10.1038/s41597-023-02517-0. Data: Zenodo doi:10.5281/zenodo.14094196, CC-BY-SA-4.0.")
PANGAEA_FILES = "https://download.pangaea.de/dataset/946550/files"
WORLDCEREAL_CALENDARS = {
    "maize": {"sos": "M1_SOS_WGS84.tif", "eos": "M1_EOS_WGS84.tif"},
    "winter_wheat": {"sos": "WW_SOS_WGS84.tif", "eos": "WW_EOS_WGS84.tif"},
}
#: SHA-256 of the PANGAEA files, verified on download (retrieved 2026-10-26).
WORLDCEREAL_CHECKSUMS = {
    "M1_SOS_WGS84.tif": "sha256:e5a2e4b13a7ee064a3e26bfb0990266e59297575d781ba1390360080e832ee0e",
    "M1_EOS_WGS84.tif": "sha256:6947a06e5ac3c2a366eb41db324d654b0683c46dc5b8f4144c6355adfbb1986a",
    "WW_SOS_WGS84.tif": "sha256:56a96b41e6116a8909743dd8e6dd0b40e13f6d6ed5b59cf46ebe0432e09ebf50",
    "WW_EOS_WGS84.tif": "sha256:6765b4e229e6a19fd9c63b36d9451cc18ed66706dd942a2c0479bc58eb288f07",
}
WORLDCEREAL_CITATION = ("Cintas, J., Franch, B., et al. (2022). Global crop calendars of maize and wheat in the framework "
                        "of the WorldCereal project [dataset]. PANGAEA, doi:10.1594/PANGAEA.946550, CC-BY-SA-4.0; "
                        "Franch et al. (2022), GIScience & Remote Sensing 59(1), 885-913.")
USER_AGENT = "geocrop-engine (https://github.com/ZeljkoDzafic/geocrop-engine)"


def cache_dir(path: str | Path | None = None) -> Path:
    root = Path(path or os.environ.get("GEOCROP_DATA_CACHE") or Path.home() / ".cache" / "geocrop" / "public")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str | Path, checksum: str | None) -> bool:
    """True when `checksum` ('md5:<hex>' or 'sha256:<hex>') matches, or when no checksum is pinned."""
    if not checksum:
        return Path(path).is_file()
    algorithm, _, expected = checksum.partition(":")
    if algorithm not in ("md5", "sha256") or not expected:
        raise ValueError(f"Checksum must be 'md5:<hex>' or 'sha256:<hex>', got {checksum!r}")
    return Path(path).is_file() and _digest(Path(path), algorithm) == expected.lower()


def fetch(url: str, dest: str | Path, checksum: str | None = None, *, max_wait_s: float = 900,
          timeout: float = 60, session: requests.Session | None = None) -> Path:
    """Download `url` to `dest` (atomic rename), reuse a verified cached copy, poll on 503/Retry-After."""
    dest = Path(dest)
    if dest.is_file() and verify(dest, checksum):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    deadline = time.monotonic() + max_wait_s
    part = dest.with_name(dest.name + ".part")
    while True:
        with http.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as r:
            staged = r.status_code in (429, 503) or "text/html" in r.headers.get("content-type", "") and \
                not url.endswith((".html", ".htm"))
            if staged:
                wait = float(r.headers.get("retry-after") or 30)
                if time.monotonic() + wait > deadline:
                    raise TimeoutError(f"{url} not available after {max_wait_s:.0f} s (HTTP {r.status_code}); "
                                       "the provider may still be staging it, retry later")
                time.sleep(max(wait, 5))
                continue
            r.raise_for_status()
            with part.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        break
    if not verify(part, checksum):
        part.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}; expected {checksum}")
    part.replace(dest)
    return dest


def zenodo_files(record: str = EUROCROPS_RECORD, session: requests.Session | None = None) -> dict[str, dict]:
    """``{file name: {"url", "checksum", "size"}}`` of a Zenodo record (to pin checksums in a protocol)."""
    http = session or requests.Session()
    r = http.get(f"{ZENODO_API}/{record}", timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return {f["key"]: {"url": f["links"]["self"], "checksum": f["checksum"], "size": f["size"]}
            for f in r.json()["files"]}


def zenodo_url(record: str, name: str) -> str:
    return f"{ZENODO_API}/{record}/files/{name}/content"


def eurocrops_file(name: str, checksum: str | None = None, record: str = EUROCROPS_RECORD,
                   cache: str | Path | None = None) -> Path:
    """Download one EuroCrops file (e.g. ``SI_2021.zip``, ``si_2021.csv``, ``HCAT3.csv``) into the cache."""
    return fetch(zenodo_url(record, name), cache_dir(cache) / "eurocrops" / record / name, checksum)


def eurocrops_layer(archive: str, checksum: str | None = None, record: str = EUROCROPS_RECORD,
                    cache: str | Path | None = None) -> Path:
    """Path of the vector layer (``.shp``/``.gpkg``) inside a downloaded and extracted EuroCrops archive."""
    return extract_layer(eurocrops_file(archive, checksum, record, cache), checksum)


def url_layer(url: str, checksum: str, cache: str | Path | None = None, subdir: str = "url") -> Path:
    """Download a zipped single-layer vector dataset from `url` (checksum-pinned) and return its layer path.

    Used for national open data published outside Zenodo (e.g. the Slovenian KMRS crop declarations)."""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return extract_layer(fetch(url, cache_dir(cache) / subdir / name, checksum), checksum)


def extract_layer(zpath: str | Path, checksum: str | None = None) -> Path:
    """Extract a zip archive next to itself once (path traversal refused) and return its only vector layer."""
    zpath = Path(zpath)
    archive = zpath.name
    target = zpath.with_suffix("")
    marker = target / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():      # refuse path traversal
                if member.startswith("/") or ".." in Path(member).parts:
                    raise ValueError(f"Unsafe path in archive: {member}")
            zf.extractall(target)
        marker.write_text(checksum or "unverified", encoding="utf8")
    layers = sorted(p for p in target.rglob("*") if p.suffix.lower() in (".shp", ".gpkg"))
    if len(layers) != 1:
        raise ValueError(f"Expected one vector layer in {archive}, found {[p.name for p in layers]}")
    return layers[0]


def read_parcels(layer: str | Path, bbox, bbox_crs: str = "EPSG:4326", columns: list[str] | None = None):
    """Parcels whose geometry intersects `bbox` (W, S, E, N in `bbox_crs`), read without loading the country."""
    import geopandas as gpd
    import pyogrio
    from pyproj import Transformer

    layer_crs = pyogrio.read_info(str(layer))["crs"]
    tf = Transformer.from_crs(bbox_crs, layer_crs, always_xy=True)
    w, s, e, n = tf.transform_bounds(*bbox)
    return gpd.read_file(layer, bbox=(w, s, e, n), columns=columns)


def search_items(endpoint: str, collection: str, bbox, start: str, end: str, max_cloud: float | None = None,
                 max_items: int = 2000) -> list:
    """STAC items (pystac) over `bbox` (WGS84) and a date range, optionally filtered by ``eo:cloud_cover``."""
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError('Install optional STAC support: pip install "geocrop-engine[stac]"') from exc
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    search = Client.open(endpoint).search(collections=[collection], bbox=list(bbox), datetime=f"{start}/{end}",
                                          query=query, max_items=max_items)
    return list(search.items())


def worldcereal_calendar_files(crops=("maize", "winter_wheat"), cache: str | Path | None = None,
                               checksums: dict[str, str] | None = None, max_wait_s: float = 900) -> dict:
    """``{crop: {"sos": Path, "eos": Path}}`` of the WorldCereal crop-calendar rasters, downloaded at run time."""
    out = {}
    root = cache_dir(cache) / "worldcereal_calendars"
    for crop in crops:
        if crop not in WORLDCEREAL_CALENDARS:
            raise ValueError(f"Unknown WorldCereal calendar {crop!r}; choose from {sorted(WORLDCEREAL_CALENDARS)}")
        pinned = {**WORLDCEREAL_CHECKSUMS, **(checksums or {})}
        out[crop] = {stage: fetch(f"{PANGAEA_FILES}/{name}", root / name, pinned.get(name),
                                  max_wait_s=max_wait_s)
                     for stage, name in WORLDCEREAL_CALENDARS[crop].items()}
    return out
