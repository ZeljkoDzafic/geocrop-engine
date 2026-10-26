"""Error and edge paths of the network adapters with fake HTTP sessions, STAC clients and openEO connections.

No test here touches the network: STAC items point at local GeoTIFFs, ``pystac_client.Client.open``,
``requests`` sessions and the ``openeo`` module are replaced by in-process fakes, so the default tier covers the
retry, staging, signing, caching and contract-conversion branches of `stac_import`, `s1_stac`, `downloads`,
`public_data`, `embedding_sources` and `openeo_adapter` in the minimal CI install as well.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import requests
from rasterio.transform import from_origin
from requests.structures import CaseInsensitiveDict
from shapely.geometry import box

from geocrop import downloads, embedding_sources, public_data, s1_stac, stac_import
from geocrop import openeo_adapter as oa
from geocrop import scenes as scene_meta
from geocrop.indices import S2_BANDS
from geocrop.raster import NODATA

CRS = "EPSG:32634"
ORIGIN = (500000.0, 5000060.0)
SHAPE = (6, 8)


def _reference(path: Path, crs=CRS, res=(10, 10)) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=1, dtype="uint8",
                       crs=crs, transform=from_origin(*ORIGIN, *res), nodata=0) as ds:
        ds.write(np.ones((1, *SHAPE), "uint8"))
    return path


@pytest.fixture()
def ref(tmp_path):
    return _reference(tmp_path / "grid.tif")


def _tif(path: Path, data: np.ndarray, dtype="uint16", origin=ORIGIN, res=10.0, crs=CRS, nodata=None) -> Path:
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                       dtype=dtype, crs=crs, transform=from_origin(*origin, res, res), nodata=nodata) as ds:
        ds.write(data.astype(dtype))
    return path


def _items_client(monkeypatch, items=None, error=None):
    """Replace pystac_client.Client.open by a fake whose search returns `items` (or raises `error`)."""
    import pystac_client
    calls = []

    class Search:
        def __init__(self, kw):
            self.kw = kw

        def items(self):
            if error is not None:
                raise error
            return iter(items or [])

    class Fake:
        def search(self, **kw):
            calls.append(kw)
            return Search(kw)

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda endpoint, *a, **k: Fake()))
    return calls


def _block_import(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, None)


# --------------------------------------------------------------------------- stac_import

def _s2_item(folder: Path, item_id: str, scl=4, dn=2000, red=None, offset=-0.1, boa_applied=True,
             when="2023-07-03T09:48:18", scale=True) -> pystac.Item:
    folder.mkdir(parents=True, exist_ok=True)
    item = pystac.Item(id=item_id, geometry={"type": "Polygon", "coordinates": [[
        [20.9, 45.0], [21.1, 45.0], [21.1, 45.2], [20.9, 45.2], [20.9, 45.0]]]},
        bbox=[20.9, 45.0, 21.1, 45.2], datetime=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        properties={"earthsearch:boa_offset_applied": boa_applied, "s2:mgrs_tile": "34TDQ", "platform": "S2A",
                    "datetime": when + "Z"})
    for band in (*S2_BANDS, "SCL"):
        value = scl if band == "SCL" else (red if band == "B04" and red is not None else dn)
        path = _tif(folder / f"{band}.tif", np.full(SHAPE, value), "uint8" if band == "SCL" else "uint16")
        asset = pystac.Asset(href=str(path), media_type="image/tiff")
        if band != "SCL" and scale:
            asset.extra_fields["raster:bands"] = [{"scale": 1e-4, "offset": offset}]
        item.add_asset(stac_import.ASSET_KEYS[band], asset)
    return item


def test_stac_cached_scene_rejects_corrupt_or_non_object_json(tmp_path):
    folder = tmp_path / "scene"
    folder.mkdir()
    assert not stac_import.cached_scene_valid(folder, "g")
    (folder / "scene.json").write_text("{not json")
    assert not stac_import.cached_scene_valid(folder, "g")
    (folder / "scene.json").write_text("[1, 2]")          # valid JSON, but not an object
    assert not stac_import.cached_scene_valid(folder, "g")


def test_stac_radiometry_rules(tmp_path):
    item = _s2_item(tmp_path / "a", "A", boa_applied=False)
    asset = item.assets["red"]
    assert stac_import.radiometry(item, asset) == (pytest.approx(1e-4), pytest.approx(-0.1))
    item.properties["earthsearch:boa_offset_applied"] = True
    assert stac_import.radiometry(item, asset)[1] == 0.0
    asset.extra_fields["raster:bands"] = [{"scale": 2e-4, "offset": None}]
    item.properties["earthsearch:boa_offset_applied"] = False
    assert stac_import.radiometry(item, asset) == (pytest.approx(2e-4), 0.0)
    asset.extra_fields.pop("raster:bands")
    with pytest.raises(ValueError, match="scale"):
        stac_import.radiometry(item, asset)


def test_stac_import_item_skips_and_rejects(ref, tmp_path):
    empty = _s2_item(tmp_path / "e", "E", scl=0)
    assert stac_import.import_item(empty, ref, tmp_path / "out" / "E")["status"] == "skipped: no coverage"
    cloudy = _s2_item(tmp_path / "c", "C", scl=9)
    res = stac_import.import_item(cloudy, ref, tmp_path / "out" / "C", min_clear_fraction=0.5)
    assert res["status"].startswith("skipped: clear fraction 0.00")
    assert not (tmp_path / "out" / "C").exists()
    negative = _s2_item(tmp_path / "n", "N", red=500, boa_applied=False)      # 500 * 1e-4 - 0.1 < 0
    with pytest.raises(ValueError, match="negative red"):
        stac_import.import_item(negative, ref, tmp_path / "out" / "N")


def test_stac_import_season_cached_failed_and_imported(ref, tmp_path, monkeypatch):
    good = _s2_item(tmp_path / "g", "G", when="2023-07-08T09:48:18")
    bad = _s2_item(tmp_path / "b", "B", red=500, boa_applied=False, when="2023-07-03T09:48:18")
    calls = _items_client(monkeypatch, [good, bad])
    out = tmp_path / "scenes"
    first = stac_import.import_season(ref, out, "2023-07-01", "2023-07-31", max_cloud=None)
    assert [r["item"] for r in first] == ["B", "G"]                      # sorted by acquisition time
    assert first[0]["status"].startswith("failed:") and first[1]["status"] == "imported"
    assert calls[0]["query"] is None
    second = stac_import.import_season(ref, out, "2023-07-01", "2023-07-31", max_cloud=30)
    assert [r["status"] for r in second][1] == "cached"
    assert calls[1]["query"] == {"eo:cloud_cover": {"lt": 30}}


def test_stac_import_season_needs_pystac_client(ref, tmp_path, monkeypatch):
    _block_import(monkeypatch, "pystac_client")
    with pytest.raises(RuntimeError, match="stac"):
        stac_import.import_season(ref, tmp_path, "2023-07-01", "2023-07-31")


# --------------------------------------------------------------------------- s1_stac

class _TokenResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def test_signer_token_caching_expiry_and_subscription_key(monkeypatch):
    seen = []

    def get(url, headers=None, timeout=None):
        seen.append((url, headers))
        return _TokenResponse({"token": "?sv=1&sig=abc", "msft:expiry": "2999-01-01T00:00:00Z"})
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setenv("PC_SDK_SUBSCRIPTION_KEY", "secret")
    signer = s1_stac.Signer("sentinel-1-rtc")
    assert signer.token() == "sv=1&sig=abc"
    assert signer.token() == "sv=1&sig=abc"                 # cached until shortly before expiry
    assert len(seen) == 1 and seen[0][1] == {"Ocp-Apim-Subscription-Key": "secret"}
    assert seen[0][0].endswith("/sentinel-1-rtc")


def test_signer_unparseable_expiry_falls_back(monkeypatch):
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    sent = []
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: (
        sent.append(headers) or _TokenResponse({"token": "tok", "msft:expiry": "tomorrow"})))
    signer = s1_stac.Signer()
    assert signer.token() == "tok"
    assert sent == [{}]
    assert signer._expiry > 0


@pytest.mark.parametrize("body", [{"no_token": 1}, ValueError("not json"), "http"])
def test_signer_service_failures(monkeypatch, body):
    if body == "http":
        monkeypatch.setattr(requests, "get", lambda *a, **k: _TokenResponse({}, status=503))
    else:
        monkeypatch.setattr(requests, "get", lambda *a, **k: _TokenResponse(body))
    with pytest.raises(s1_stac.ServiceUnavailable, match="token service"):
        s1_stac.Signer().token()


def test_signer_sign_only_unsigned_blob_urls(monkeypatch):
    _block_import(monkeypatch, "planetary_computer")
    signer = s1_stac.Signer()
    signer._token, signer._expiry = "sig=x", 4e9
    assert signer.sign("/local/VV.tif") == "/local/VV.tif"
    blob = "https://a.blob.core.windows.net/c/VV.tif"
    assert signer.sign(blob + "?sig=y") == blob + "?sig=y"
    assert signer.sign(blob) == blob + "?sig=x"
    assert signer.sign(blob + "?v=1") == blob + "?v=1&sig=x"


def _s1_item(folder: Path, item_id: str, vv=0.1, vh=0.02, cover=None, when="2023-07-03T16:30:00",
             absolute_orbit=48000, pols=("VV", "VH")) -> pystac.Item:
    folder.mkdir(parents=True, exist_ok=True)
    item = pystac.Item(id=item_id, geometry=None, bbox=None,
                       datetime=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
                       properties={"platform": "SENTINEL-1A", "sat:orbit_state": "ascending", "datetime": when + "Z",
                                   "sat:relative_orbit": 73, "sat:absolute_orbit": absolute_orbit})
    for pol, value in (("VV", vv), ("VH", vh)):
        if pol not in pols:
            continue
        data = np.full(SHAPE, value, "float32")
        if cover is not None:
            data[~cover] = 0.0                                # non-positive = not covered
        path = _tif(folder / f"{pol}.tif", data, "float32")
        item.add_asset(s1_stac.ASSET_KEYS[pol], pystac.Asset(href=str(path)))
    return item


class _PassSigner:
    def sign(self, href):
        return href


def test_s1_cached_scene_valid(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    assert not s1_stac.cached_scene_valid(folder, "g")
    (folder / "scene.json").write_text("{broken")
    assert not s1_stac.cached_scene_valid(folder, "g")
    scene_meta.write(folder, sensor="S1", datetime="2023-07-03T16:30:00", grid_id="g")
    assert not s1_stac.cached_scene_valid(folder, "g")         # bands missing
    for pol in ("VV", "VH"):
        _tif(folder / f"{pol}.tif", np.ones(SHAPE), "float32")
    assert s1_stac.cached_scene_valid(folder, "g") and not s1_stac.cached_scene_valid(folder, "other")


def test_s1_import_item_coverage_and_missing_asset(ref, tmp_path):
    half = np.zeros(SHAPE, bool)
    half[:, :3] = True
    low = _s1_item(tmp_path / "low", "LOW", cover=half)
    assert s1_stac.import_item(low, ref, tmp_path / "out", _PassSigner())["status"] == "skipped: coverage 0.38"
    full = _s1_item(tmp_path / "full", "FULL")
    res = s1_stac.import_item(full, ref, tmp_path / "out", _PassSigner(), min_coverage=0.9)
    assert res["status"] == "imported" and res["coverage"] == 1.0
    meta = scene_meta.read(tmp_path / "out")
    assert meta["orbit_direction"] == "ASCENDING" and meta["licence"] == s1_stac.LICENCE
    with pytest.raises(ValueError, match="no VH asset"):
        s1_stac.read_item(_s1_item(tmp_path / "vv", "VVONLY", pols=("VV",)), ref, _PassSigner())


def test_s1_acquisition_key_and_folder_name(tmp_path):
    item = _s1_item(tmp_path / "a", "A")
    assert s1_stac._acquisition_key(item) == "SENTINEL-1A_48000"
    assert s1_stac._folder_name([item]) == "S1A_20230703T163000_ASC_R073"
    item.properties.pop("sat:absolute_orbit")
    assert s1_stac._acquisition_key(item) == "A"


def test_s1_search_query_errors_and_missing_client(ref, monkeypatch):
    b = pystac.Item(id="b", geometry=None, bbox=None, datetime=datetime(2023, 7, 9, tzinfo=timezone.utc),
                    properties={"datetime": "2023-07-09T00:00:00Z"})
    a = pystac.Item(id="a", geometry=None, bbox=None, datetime=datetime(2023, 7, 1, tzinfo=timezone.utc),
                    properties={"datetime": "2023-07-01T00:00:00Z"})
    calls = _items_client(monkeypatch, [b, a])
    assert [i.id for i in s1_stac.search(ref, "2023-07-01", "2023-07-31", orbit_direction="DESC")] == ["a", "b"]
    assert calls[0]["query"] == {"sat:orbit_state": {"eq": "desc"}}
    _items_client(monkeypatch, error=OSError("connection reset"))
    with pytest.raises(s1_stac.ServiceUnavailable, match="connection reset"):
        s1_stac.search(ref, "2023-07-01", "2023-07-31")
    _block_import(monkeypatch, "pystac_client")
    with pytest.raises(RuntimeError, match="stac"):
        s1_stac.search(ref, "2023-07-01", "2023-07-31")


def test_s1_import_season_mosaics_slices_skips_fails_and_caches(ref, tmp_path, monkeypatch):
    left, right = np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)
    left[:, :4], right[:, 4:] = True, True
    sparse = np.zeros(SHAPE, bool)
    sparse[0, 0] = True
    items = [_s1_item(tmp_path / "l", "L", cover=left, vv=0.1),
             _s1_item(tmp_path / "r", "R", cover=right, vv=0.2, when="2023-07-03T16:30:25"),
             _s1_item(tmp_path / "s", "S", cover=sparse, when="2023-07-09T16:30:00", absolute_orbit=48100),
             _s1_item(tmp_path / "m", "M", when="2023-07-15T16:30:00", absolute_orbit=48200, pols=("VV",))]
    monkeypatch.setattr(s1_stac, "search", lambda *a, **k: items)
    out = tmp_path / "s1"
    res = s1_stac.import_season(ref, out, "2023-07-01", "2023-07-31", signer=_PassSigner())
    status = {tuple(r["item"]): r["status"] for r in res}
    assert status[("L", "R")] == "imported"
    assert status[("S",)].startswith("skipped: coverage")
    assert status[("M",)].startswith("failed:") and "no VH asset" in status[("M",)]
    with rasterio.open(Path(res[0]["path"]) / "VV.tif") as ds:     # slices mosaicked into one acquisition
        vv = ds.read(1)
    np.testing.assert_allclose(vv[:, :4], 0.1, rtol=1e-6)
    np.testing.assert_allclose(vv[:, 4:], 0.2, rtol=1e-6)
    again = s1_stac.import_season(ref, out, "2023-07-01", "2023-07-31", signer=_PassSigner())
    assert again[0]["status"] == "cached" and again[0]["coverage"] == 1.0


def test_s1_import_season_propagates_service_outage(ref, tmp_path, monkeypatch):
    class Down:
        def sign(self, href):
            raise s1_stac.ServiceUnavailable("token service down")
    monkeypatch.setattr(s1_stac, "search", lambda *a, **k: [_s1_item(tmp_path / "x", "X")])
    with pytest.raises(s1_stac.ServiceUnavailable):
        s1_stac.import_season(ref, tmp_path / "s1", "2023-07-01", "2023-07-31", signer=Down())


def test_s1_plausibility_without_valid_pixels(tmp_path):
    for pol in ("VV", "VH"):
        _tif(tmp_path / f"{pol}.tif", np.full(SHAPE, NODATA), "float32")
    assert s1_stac.plausibility(tmp_path) == {"valid_fraction": 0.0}


def test_s1_shift_surface_needs_overlap_and_offsets_keep_descriptions(tmp_path):
    small = np.random.default_rng(0).normal(size=(8, 8))
    assert np.isnan(s1_stac.shift_surface(small, small, 1)).all()      # < 100 overlapping pixels per shift
    scene = tmp_path / "raw" / "S1A_x"
    scene.mkdir(parents=True)
    for pol in ("VV", "VH"):
        _tif(scene / f"{pol}.tif", np.full(SHAPE, 0.05), "float32")       # no band description
    scene_meta.write(scene, sensor="S1", datetime="2023-07-03T16:30:00", orbit_direction="DESCENDING",
                     relative_orbit=None)
    written = s1_stac.apply_offsets([scene], {"DES_R000": {"accepted": True, "dy": 0.1, "dx": 0.1}},
                                    tmp_path / "coreg")
    meta = scene_meta.read(written[0])
    assert meta["coregistration"]["applied"] is False and meta["coregistration"]["geometry"] == "DES_R000"
    with rasterio.open(written[0] / "VV.tif") as ds:
        assert ds.descriptions[0] is None


def test_signer_token_without_expiry(monkeypatch):
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _TokenResponse({"token": "t"}))
    signer = s1_stac.Signer()
    assert signer.token() == "t" and 0 < signer._expiry - s1_stac.time.time() <= 1800
    assert s1_stac.utc_now().endswith("+00:00")


def test_s1_coregistration_recovers_a_known_shift(tmp_path):
    rng = np.random.default_rng(3)
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift
    field = gaussian_filter(rng.normal(size=(48, 48)), 2)
    ndvi = (field - field.min()) / np.ptp(field)
    vh_db = -25 + 10 * nd_shift(ndvi, (2, -1), order=1, mode="nearest")      # S1 displaced by (+2, -1) px
    scene = tmp_path / "raw" / "S1A_shifted"
    scene.mkdir(parents=True)
    profile = {"driver": "GTiff", "width": 48, "height": 48, "count": 1, "dtype": "float32", "crs": CRS,
               "transform": from_origin(*ORIGIN, 10, 10), "nodata": NODATA}
    for pol, lin in (("VV", 10 ** ((vh_db + 7) / 10)), ("VH", 10 ** (vh_db / 10))):
        with rasterio.open(scene / f"{pol}.tif", "w", **profile) as ds:
            ds.write(lin.astype("float32"), 1)
            ds.set_band_description(1, f"gamma0_{pol}_linear")
    scene_meta.write(scene, sensor="S1", datetime="2023-07-03T16:30:00", orbit_direction="ASCENDING",
                     relative_orbit=73)
    plaus = s1_stac.plausibility(scene)
    assert plaus["valid_fraction"] == 1.0 and plaus["vv_gt_vh"] == 1.0
    est = s1_stac.estimate_offsets([scene], ndvi, max_shift=4, smooth=3)["ASC_R073"]
    assert est["accepted"] and est["dy"] == pytest.approx(2, abs=0.3) and est["dx"] == pytest.approx(-1, abs=0.3)
    out = s1_stac.apply_offsets([scene], {"ASC_R073": est}, tmp_path / "coreg")[0]
    meta = scene_meta.read(out)
    assert meta["coregistration"]["applied"] and meta["coregistration"]["raw_scene"] == "S1A_shifted"
    with rasterio.open(out / "VH.tif") as ds:
        assert ds.descriptions[0] == "gamma0_VH_linear"
        moved = 10 * np.log10(ds.read(1)[8:-8, 8:-8])
    np.testing.assert_allclose(moved, -25 + 10 * ndvi[8:-8, 8:-8], atol=0.5)


# --------------------------------------------------------------------------- downloads (fake session)

BODY = bytes(range(256)) * 64          # 16 KiB
SHA = hashlib.sha256(BODY).hexdigest()


class _Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.body = status, body
        self.headers = CaseInsensitiveDict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk):
        for i in range(0, len(self.body), chunk):
            yield self.body[i:i + chunk]


class _Session:
    """Scripted session: each entry is a response or a callable(headers) -> response."""

    def __init__(self, *script):
        self.script, self.headers = list(script), []

    def get(self, url, stream=None, timeout=None, headers=None, allow_redirects=None):
        self.headers.append(dict(headers or {}))
        nxt = self.script.pop(0)
        return nxt(headers or {}) if callable(nxt) else nxt


@pytest.fixture()
def waits(monkeypatch):
    out = []
    monkeypatch.setattr(downloads, "_sleep", out.append)
    return out


def _full(headers=None):
    return _Resp(200, BODY, {"content-length": str(len(BODY)), **(headers or {})})


def test_download_verify_and_checksum_parsing(tmp_path):
    f = tmp_path / "x.bin"
    assert not downloads.verify(f)
    f.write_bytes(BODY)
    assert not downloads.verify(f, size=1)
    assert downloads.verify(f, SHA, size=len(BODY))
    assert downloads.verify(f, "md5:" + hashlib.md5(BODY).hexdigest())
    assert downloads.parse_checksum(None) is None
    with pytest.raises(ValueError, match="Checksum"):
        downloads.parse_checksum("crc32:abc")


def test_download_http_date_retry_after_uses_backoff(tmp_path, waits):
    session = _Session(_Resp(503, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), _full())
    # 503 with a Retry-After header counts as staging; an unparseable value waits the default 30 s
    out = downloads.fetch("https://x/data.bin", tmp_path / "data.bin", SHA, session=session, max_wait_s=600)
    assert out.read_bytes() == BODY and waits == [30.0]
    session = _Session(_Resp(429, headers={"retry-after": "soon"}), _full())
    downloads.fetch("https://x/d2.bin", tmp_path / "d2.bin", SHA, session=session, backoff=1.0)
    assert 0.8 <= waits[-1] <= 1.2                        # backoff schedule, not the unparseable header


def test_download_staging_beyond_deadline(tmp_path, waits):
    session = _Session(_Resp(503, headers={"retry-after": "3600"}))
    with pytest.raises(downloads.DownloadError, match="not available"):
        downloads.fetch("https://x/f.tif", tmp_path / "f.tif", session=session, max_wait_s=10)


def test_download_stale_partial_larger_than_size_restarts(tmp_path, waits):
    target = tmp_path / "w.bin"
    (tmp_path / "w.bin.part").write_bytes(b"z" * (len(BODY) + 10))
    session = _Session(_full())
    downloads.fetch("https://x/w.bin", target, SHA, size=len(BODY), session=session)
    assert target.read_bytes() == BODY and "Range" not in session.headers[0]


def test_download_416_on_complete_partial_and_on_corrupt_partial(tmp_path, waits):
    target = tmp_path / "a.bin"
    (tmp_path / "a.bin.part").write_bytes(BODY)
    session = _Session(_Resp(416))
    downloads.fetch("https://x/a.bin", target, SHA, session=session)
    assert target.read_bytes() == BODY and session.headers[0]["Range"] == f"bytes={len(BODY)}-"
    target2 = tmp_path / "b.bin"
    (tmp_path / "b.bin.part").write_bytes(b"garbage")
    session = _Session(_Resp(416), _full())
    downloads.fetch("https://x/b.bin", target2, SHA, session=session)
    assert target2.read_bytes() == BODY and "Range" not in session.headers[1]


def test_download_unexpected_content_range_and_progress(tmp_path, waits):
    target = tmp_path / "c.bin"
    (tmp_path / "c.bin.part").write_bytes(BODY[:100])
    seen = []
    session = _Session(_Resp(206, BODY[50:], {"content-range": f"bytes 50-{len(BODY) - 1}/{len(BODY)}"}), _full())
    downloads.fetch("https://x/c.bin", target, SHA, session=session, progress=lambda d, t: seen.append((d, t)),
                    chunk_size=4096)
    assert target.read_bytes() == BODY
    assert seen[-1] == (len(BODY), len(BODY)) and len(seen) == len(BODY) // 4096


def test_download_resume_without_total_in_content_range(tmp_path, waits):
    target = tmp_path / "d.bin"
    (tmp_path / "d.bin.part").write_bytes(BODY[:100])
    session = _Session(_Resp(206, BODY[100:], {"content-range": "bytes 100-*/*"}))
    downloads.fetch("https://x/d.bin", target, SHA, session=session)
    assert target.read_bytes() == BODY


def test_download_truncated_body_is_retried_then_fails(tmp_path, waits):
    short = _Resp(200, BODY[:10], {"content-length": str(len(BODY))})
    session = _Session(short, _Resp(200, BODY[:10], {"content-length": str(len(BODY))}))
    with pytest.raises(downloads.DownloadError, match="after 2 attempts.*truncated"):
        downloads.fetch("https://x/t.bin", tmp_path / "t.bin", SHA, session=session, retries=1)
    assert len(waits) == 1


def test_download_client_error_is_not_retried(tmp_path, waits):
    with pytest.raises(downloads.DownloadError, match="HTTP 404"):
        downloads.fetch("https://x/missing.bin", tmp_path / "m.bin", session=_Session(_Resp(404)))
    assert waits == []


def test_download_checksum_mismatch_without_resume(tmp_path, waits):
    with pytest.raises(downloads.ChecksumError, match="refusing"):
        downloads.fetch("https://x/e.bin", tmp_path / "e.bin", "0" * 64, session=_Session(_full()))
    assert not (tmp_path / "e.bin").exists() and not (tmp_path / "e.bin.part").exists()


# --------------------------------------------------------------------------- public_data

def test_public_verify_wrapper_and_zenodo_listing(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(BODY)
    assert public_data.verify(f, "sha256:" + SHA) and not public_data.verify(f, "sha256:" + "0" * 64)

    class Http:
        def get(self, url, timeout=None, headers=None):
            assert url.endswith("/records/123") and "geocrop" in headers["User-Agent"]
            r = _TokenResponse({"files": [{"key": "SI_2021.zip", "links": {"self": "https://z/SI"},
                                           "checksum": "md5:ab", "size": 3}]})
            return r
    assert public_data.zenodo_files("123", session=Http()) == {
        "SI_2021.zip": {"url": "https://z/SI", "checksum": "md5:ab", "size": 3}}


def _zip(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_layer_refuses_traversal_and_ambiguous_archives(tmp_path):
    with pytest.raises(ValueError, match="Unsafe path"):
        public_data.extract_layer(_zip(tmp_path / "evil.zip", {"../escape.shp": b"x"}))
    with pytest.raises(ValueError, match="Unsafe path"):
        public_data.extract_layer(_zip(tmp_path / "abs.zip", {"/etc/x.shp": b"x"}))
    with pytest.raises(ValueError, match="Expected one vector layer"):
        public_data.extract_layer(_zip(tmp_path / "two.zip", {"a.shp": b"", "b.gpkg": b""}))
    one = _zip(tmp_path / "one.zip", {"dir/a.shp": b"", "dir/a.dbf": b""})
    first = public_data.extract_layer(one, "sha256:pinned")
    assert first.name == "a.shp" and (tmp_path / "one" / ".extracted").read_text() == "sha256:pinned"
    one.unlink()                                          # extracted once: the archive is not reopened
    assert public_data.extract_layer(one) == first


def test_public_search_items_and_calendar_files(tmp_path, monkeypatch):
    item = pystac.Item(id="i", geometry=None, bbox=None, datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
                       properties={})
    calls = _items_client(monkeypatch, [item])
    assert public_data.search_items("https://stac", "c", (1, 2, 3, 4), "2023-01-01", "2023-02-01", 20) == [item]
    assert calls[0]["query"] == {"eo:cloud_cover": {"lt": 20}} and calls[0]["bbox"] == [1, 2, 3, 4]
    public_data.search_items("https://stac", "c", (1, 2, 3, 4), "2023-01-01", "2023-02-01")
    assert calls[1]["query"] is None
    fetched = []
    monkeypatch.setattr(public_data, "fetch", lambda url, dest, checksum=None, **kw: fetched.append(
        (url, Path(dest).name, checksum)) or Path(dest))
    out = public_data.worldcereal_calendar_files(("maize",), cache=tmp_path, checksums={"M1_SOS_WGS84.tif": "md5:1"})
    assert set(out["maize"]) == {"sos", "eos"}
    assert ("md5:1" in [c for _, n, c in fetched if n == "M1_SOS_WGS84.tif"])
    assert [c for _, n, c in fetched if n == "M1_EOS_WGS84.tif"] == [public_data.WORLDCEREAL_CHECKSUMS["M1_EOS_WGS84.tif"]]
    with pytest.raises(ValueError, match="Unknown WorldCereal calendar"):
        public_data.worldcereal_calendar_files(("rice",), cache=tmp_path)
    _block_import(monkeypatch, "pystac_client")
    with pytest.raises(RuntimeError, match="stac"):
        public_data.search_items("https://stac", "c", (1, 2, 3, 4), "2023-01-01", "2023-02-01")


# --------------------------------------------------------------------------- embedding_sources

def test_source_window_none_when_disjoint():
    t = from_origin(0, 1000, 10, 10)
    far = from_origin(100000, 1000, 10, 10)
    assert embedding_sources._source_window(t, (10, 10), far, CRS, (5, 5), CRS) is None


def test_alphaearth_index_local_and_download(tmp_path, monkeypatch):
    local = tmp_path / "idx.parquet"
    local.write_bytes(b"x")
    assert embedding_sources.alphaearth_index(local) == local
    got = []
    monkeypatch.setattr(embedding_sources, "download", lambda url, target, *a, **k: got.append((url, target)) or target)
    monkeypatch.setenv("GEOCROP_CACHE", str(tmp_path / "cache"))
    embedding_sources.alphaearth_index(None, "gcs")
    assert got[-1][0] == f"{embedding_sources.AEF_MIRRORS['gcs']}/{embedding_sources.AEF_INDEX}"
    embedding_sources.alphaearth_index("https://mirror/aef_index.parquet")
    assert got[-1][0] == "https://mirror/aef_index.parquet"
    path = embedding_sources.presto_checkpoint(tmp_path / "presto")
    assert got[-1][0] == embedding_sources.PRESTO_CHECKPOINT["url"] and path.parent == tmp_path / "presto"


def test_alphaearth_tiles_filters_bbox(tmp_path, monkeypatch):
    table = pd.DataFrame({"path": ["s3://x/v1/annual/2023/34N/a.tiff", "s3://x/v1/annual/2023/30N/b.tiff"],
                          "crs": ["EPSG:32634", "EPSG:32630"], "year": [2023, 2023], "utm_zone": ["34N", "30N"],
                          "wgs84_west": [20.0, -5.0], "wgs84_south": [44.0, 40.0],
                          "wgs84_east": [22.0, -3.0], "wgs84_north": [46.0, 42.0]})
    monkeypatch.setattr(embedding_sources, "alphaearth_index", lambda index, mirror: tmp_path / "i.parquet")
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns=None, filters=None: table)
    tiles = embedding_sources.alphaearth_tiles(2023, (20.9, 45.0, 21.1, 45.2))
    assert tiles == [{"url": f"{embedding_sources.AEF_MIRRORS['source.coop']}/2023/34N/a.tiff",
                      "crs": "EPSG:32634", "utm_zone": "34N"}]


def _local_vsicurl(monkeypatch):
    real = rasterio.open

    def opener(path, *a, **k):
        if isinstance(path, str) and path.startswith("/vsicurl/"):
            path = path[len("/vsicurl/"):]
        return real(path, *a, **k)
    monkeypatch.setattr(rasterio, "open", opener)


def test_fetch_alphaearth_reads_overlapping_tiles_only(ref, tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    q = rng.integers(-127, 128, (64, 10, 12)).astype("int8")
    tile = _tif(tmp_path / "tile.tif", q, "int8", origin=(ORIGIN[0] - 20, ORIGIN[1] + 20))
    far = _tif(tmp_path / "far.tif", q, "int8", origin=(ORIGIN[0] + 1e5, ORIGIN[1]))
    _local_vsicurl(monkeypatch)
    monkeypatch.setattr(embedding_sources, "alphaearth_tiles",
                        lambda year, bbox, index, mirror: [{"url": str(far)}, {"url": str(tile)}])
    out = embedding_sources.fetch_alphaearth(ref, tmp_path / "alphaearth_2023.tif", 2023, block_size=4)
    with rasterio.open(out) as ds:
        data = ds.read()
        assert ds.count == 64 and ds.tags()["embedding_source"] == "alphaearth"
    from geocrop.embeddings import dequantise_alphaearth
    np.testing.assert_allclose(data, dequantise_alphaearth(q)[:, 2:8, 2:10], rtol=1e-6)
    monkeypatch.setattr(embedding_sources, "alphaearth_tiles", lambda *a: [])
    with pytest.raises(ValueError, match="No AlphaEarth tiles"):
        embedding_sources.fetch_alphaearth(ref, tmp_path / "none.tif", 2023)


def _npy(arr: np.ndarray, fortran=False) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asfortranarray(arr) if fortran else arr)
    return buf.getvalue()


class _NpySession:
    """Serves in-memory files with HTTP range semantics; HEAD answers 404 for unknown URLs."""

    def __init__(self, files: dict, ignore_range=False):
        self.files, self.ignore_range = files, ignore_range

    def head(self, url, timeout=None, allow_redirects=None):
        return types.SimpleNamespace(status_code=200 if url in self.files else 404)

    def get(self, url, headers=None, timeout=None):
        data = self.files[url]
        a, b = map(int, headers["Range"].split("=")[1].split("-"))
        if self.ignore_range:
            return types.SimpleNamespace(status_code=200, content=data, raise_for_status=lambda: None)
        return types.SimpleNamespace(status_code=206, content=data[a:b + 1], raise_for_status=lambda: None)


def test_remote_npy_rejects_fortran_order_and_ignored_ranges():
    arr = np.arange(24, dtype="int8").reshape(4, 6)
    with pytest.raises(ValueError, match="Fortran"):
        embedding_sources.RemoteNpy("u", _NpySession({"u": _npy(arr, fortran=True)}))
    npy = embedding_sources.RemoteNpy("u", _NpySession({"u": _npy(arr)}, ignore_range=True))  # header read: ok
    with pytest.raises(RuntimeError, match="ignored the range"):
        npy.rows(1, 2)


def _tessera_files(tmp_path, lon, lat, rows=12, cols=14, year=2023, mismatch=False):
    name = embedding_sources.tessera_name(lon, lat)
    base = f"{embedding_sources.TESSERA_URL}/npy/{embedding_sources.TESSERA_DATASET}/{year}/{name}/{name}"
    rng = np.random.default_rng(int(lon * 100))
    q = rng.integers(-100, 100, (rows, cols + (1 if mismatch else 0), 128)).astype("int8")
    s = rng.uniform(0.01, 0.02, (rows, cols)).astype("float32")
    mask = _tif(tmp_path / f"{name}.tiff", np.ones((rows, cols)), "uint8", origin=(ORIGIN[0] - 20, ORIGIN[1] + 20),
                res=10.0)
    return name, {f"{base}.npy": _npy(q), f"{base}_scales.npy": _npy(s)}, mask, q, s


def test_fetch_tessera_with_missing_tile_warning(ref, tmp_path, monkeypatch):
    name, files, mask, q, s = _tessera_files(tmp_path, 21.05, 45.15)
    monkeypatch.setattr(embedding_sources, "tessera_tile_centres", lambda bbox: [(21.05, 45.15), (21.15, 45.15)])
    monkeypatch.setattr(embedding_sources, "_session", lambda: _NpySession(files))
    monkeypatch.setattr(embedding_sources, "download", lambda url, target, **k: mask)
    with pytest.warns(UserWarning, match="grid_21.15_45.15"):
        out = embedding_sources.fetch_tessera(ref, tmp_path / "tessera_2023.tif", 2023, block_size=4)
    from geocrop.embeddings import dequantise_tessera
    expected = np.moveaxis(dequantise_tessera(q, s), -1, 0)[:, 2:8, 2:10]
    with rasterio.open(out) as ds:
        np.testing.assert_allclose(ds.read(), expected, rtol=1e-6)
        assert "grid_21.15_45.15" in ds.tags()["missing_tiles"]


def test_fetch_tessera_skips_disjoint_tiles(ref, tmp_path, monkeypatch):
    _, files, mask, *_ = _tessera_files(tmp_path, 21.05, 45.15)
    far = tmp_path / "far"
    far.mkdir()
    far_name, far_files, *_ = _tessera_files(far, 21.25, 45.15)
    far_mask = _tif(far / "mask.tiff", np.ones((12, 14)), "uint8", origin=(ORIGIN[0] + 1e5, ORIGIN[1]))
    monkeypatch.setattr(embedding_sources, "tessera_tile_centres", lambda bbox: [(21.05, 45.15), (21.25, 45.15)])
    monkeypatch.setattr(embedding_sources, "_session", lambda: _NpySession({**files, **far_files}))
    monkeypatch.setattr(embedding_sources, "download", lambda url, target, **k: far_mask if far_name in url else mask)
    out = embedding_sources.fetch_tessera(ref, tmp_path / "tessera_2023.tif", 2023)
    with rasterio.open(out) as ds:
        assert not ds.tags().get("missing_tiles") and np.isfinite(ds.read()).all()


def test_fetch_tessera_without_tiles_or_with_mismatched_arrays(ref, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_sources, "tessera_tile_centres", lambda bbox: [(21.05, 45.15)])
    monkeypatch.setattr(embedding_sources, "_session", lambda: _NpySession({}))
    with pytest.raises(ValueError, match="No TESSERA"):
        embedding_sources.fetch_tessera(ref, tmp_path / "t.tif", 2023)
    _, files, mask, *_ = _tessera_files(tmp_path, 21.05, 45.15, mismatch=True)
    monkeypatch.setattr(embedding_sources, "_session", lambda: _NpySession(files))
    monkeypatch.setattr(embedding_sources, "download", lambda url, target, **k: mask)
    with pytest.raises(ValueError, match="do not match landmask"):
        embedding_sources.fetch_tessera(ref, tmp_path / "t.tif", 2023)


def test_fetch_weights_one_or_several_files(tmp_path, monkeypatch):
    from geocrop.feature_providers import WeightFile
    monkeypatch.setattr(embedding_sources, "download",
                        lambda url, target, sha256=None, size=None: Path(target))
    one = types.SimpleNamespace(name="p", weights=(WeightFile("a.pt", "https://w/a", "0" * 64),))
    assert embedding_sources.fetch_weights(one, tmp_path) == tmp_path / "a.pt"
    two = types.SimpleNamespace(name="p", weights=(WeightFile("a.pt", "https://w/a", "0" * 64),
                                                   WeightFile("b.pt", "https://w/b", "1" * 64, 10)))
    assert embedding_sources.fetch_weights(two, tmp_path) == [tmp_path / "a.pt", tmp_path / "b.pt"]
    with pytest.raises(ValueError, match="no weights"):
        embedding_sources.fetch_weights(types.SimpleNamespace(name="none", weights=()))


def _parcels_file(path: Path) -> Path:
    x0, y1 = ORIGIN
    gpd.GeoDataFrame({"parcel_id": ["P1", "P2"]},
                     geometry=[box(x0, y1 - 60, x0 + 40, y1), box(x0 + 40, y1 - 60, x0 + 80, y1)],
                     crs=CRS).to_file(path, driver="GPKG")
    return path


def test_extract_precomputed_with_parcels(ref, tmp_path, monkeypatch):
    def fake_fetch(reference, output, year, block_size=1024, resampling=None, names=None, **kw):
        from geocrop.embeddings import EmbeddingWriter
        with EmbeddingWriter(output, reference, "alphaearth", year, 64, {"mirror": kw.get("mirror")}, names) as sink:
            sink.write(np.full((64, *SHAPE), 0.25, "float32"))
        return Path(output)
    monkeypatch.setattr(embedding_sources.ALPHAEARTH, "_fetch", fake_fetch)
    res = embedding_sources.extract("alphaearth", ref, tmp_path / "emb", 2023, parcels=_parcels_file(
        tmp_path / "parcels.gpkg"), mirror="gcs")
    table = pd.read_csv(res["parcels"])
    assert len(table) == 2 and table.filter(like="emb_").to_numpy() == pytest.approx(0.25)
    assert res["feature_names"] == "emb_00 ... emb_63"
    # embed_parcels without a raster first computes the pixels next to the table
    out = embedding_sources.ALPHAEARTH.embed_parcels(ref, _parcels_file(tmp_path / "p2.gpkg"),
                                                     tmp_path / "again" / "t.csv", year=2023)
    assert (tmp_path / "again" / "t.tif").is_file() and len(out) == 2
    assert len(pd.read_csv(tmp_path / "again" / "t.csv")) == 2
    assert embedding_sources.ALPHAEARTH.load_model() is None
    assert embedding_sources.ALPHAEARTH.model_dims(None) == 64 and "alphaearth" in repr(embedding_sources.ALPHAEARTH)


class _MonthlyFake:
    """A minimal monthly feature provider registered for the duration of one test."""
    name, dims, sensor, product = "covfake", 3, "COVFAKE", "fake encoder"
    licence, code_licence, attribution, dequantisation = "MIT", "MIT", "test", "none"
    requires_licence_ack, legacy_names = False, False

    def __init__(self):
        from geocrop.feature_providers import InputContract, WeightFile
        self.contract = InputContract("monthly", ("S2",), ("B04",), "monthly", "none")
        self.weights = (WeightFile("w.pt", "https://w/w.pt", "a" * 64),)
        self.calls = []

    def load_model(self, weights=None, **_):
        self.calls.append(("load", weights))
        return types.SimpleNamespace(embedding_size=3)

    def model_dims(self, model):
        return model.embedding_size

    def embed_pixels(self, reference, output, **kw):
        from geocrop.embeddings import EmbeddingWriter
        self.calls.append(("pixels", kw))
        with EmbeddingWriter(output, reference, self.name, 2023, 3, kw.get("extra"), kw["names"]) as sink:
            sink.write(np.ones((3, *SHAPE), "float32"))
        return Path(output)

    def embed_parcels(self, reference, parcels, output, **kw):
        self.calls.append(("parcels", kw))
        Path(output).write_text("parcel_id\n")
        return Path(output)


def test_extract_monthly_provider_paths(ref, tmp_path, monkeypatch):
    from geocrop import plugins
    fake = _MonthlyFake()
    plugins.feature_providers.register(fake.name, fake)
    try:
        with pytest.raises(ValueError, match="monthly composites"):
            embedding_sources.extract(fake.name, ref, tmp_path, 2023)
        res = embedding_sources.extract(fake.name, ref, tmp_path / "a", 2023, monthly=tmp_path, checkpoint=["x.pt"],
                                        block_size=128, parcels=tmp_path / "parcels.gpkg")
        assert fake.calls[0] == ("load", [Path("x.pt")])
        pixels = dict(fake.calls)["pixels"]
        assert pixels["block_size"] == 128 and pixels["extra"] == {"checkpoint": "['x.pt']"}
        assert "block_size" not in dict(fake.calls)["parcels"] and res["parcels"].endswith("_parcels.csv")
        assert res["feature_names"] == "emb_covfake_00 ... emb_covfake_02"
        fake.calls.clear()
        monkeypatch.setattr(embedding_sources, "fetch_weights", lambda provider: Path("cached.pt"))
        embedding_sources.extract(fake.name, ref, tmp_path / "b", 2023, monthly=tmp_path)
        assert fake.calls[0] == ("load", Path("cached.pt"))
        assert dict(fake.calls)["pixels"]["extra"] == {"checkpoint_url": "https://w/w.pt", "checkpoint_sha256": "a" * 64}
        assert "block_size" not in dict(fake.calls)["pixels"]
    finally:
        plugins.feature_providers.unregister(fake.name)


# --------------------------------------------------------------------------- openeo_adapter

def test_openeo_reference_grid_requirements(tmp_path):
    custom = "+proj=tmerc +lat_0=0 +lon_0=17.3 +k=0.99991 +x_0=512345 +y_0=-1234 +ellps=bessel +units=m +no_defs"
    with pytest.raises(ValueError, match="EPSG"):
        oa.reference_grid(_reference(tmp_path / "custom.tif", crs=custom))
    with pytest.raises(ValueError, match="square"):
        oa.reference_grid(_reference(tmp_path / "rect.tif", res=(10, 20)))


def test_openeo_graph_argument_errors(ref):
    grid = oa.reference_grid(ref)
    g = oa.ProcessGraph()
    assert g.nodes[oa._invalid_expression(g, [])["from_node"]] == {"process_id": "constant", "arguments": {"x": False}}
    with pytest.raises(ValueError, match="Unsupported statistic"):
        oa._statistic(oa.ProcessGraph(), "mode")
    d = (date(2023, 4, 1), date(2023, 4, 30))
    with pytest.raises(ValueError, match="coefficient"):
        oa.s1_composite_graph(grid, *d, coefficient="beta0")
    with pytest.raises(ValueError, match="mean or median"):
        oa.s1_composite_graph(grid, *d, statistic="p90")
    card4l = oa.s1_composite_graph(grid, *d, card4l=True)
    assert {n["process_id"] for n in card4l.values()} >= {"ard_normalized_radar_backscatter", "filter_bands"}


def test_openeo_parcel_geometries_need_id_field(ref, tmp_path):
    with pytest.raises(ValueError, match="id field"):
        oa.parcel_geometries(_parcels_file(tmp_path / "p.gpkg"), oa.reference_grid(ref), id_field="fid_x")


def _cfg(ref, tmp_path, **openeo):
    from geocrop.config import load
    return load(None, {"grid": {"reference": str(ref)},
                       "paths": {"composites": str(tmp_path / "c"), "s2_scenes": str(tmp_path / "s2"),
                                 "parcels": str(_parcels_file(tmp_path / "parcels.gpkg"))},
                       "openeo": {"work_dir": str(tmp_path / "openeo"), **openeo}})


def test_openeo_recipe_errors_and_period_fallback(ref, tmp_path):
    cfg = _cfg(ref, tmp_path)
    with pytest.raises(ValueError, match="Unknown openEO products"):
        oa.build_recipe(cfg, ["s2_composites", "ndvi_movie"])
    with pytest.raises(ValueError, match="need windows"):
        oa.build_recipe(cfg, ["s1_composites"])
    recipe = oa.build_recipe(cfg, ["s2_scenes"])            # no windows and no period: placeholder dates
    assert recipe["graphs"][0]["start"] == "1970-01-01" == recipe["graphs"][0]["end"]
    files = oa.write_recipe(recipe, tmp_path / "recipe")
    meta = json.loads(Path(files["recipe"]).read_text())
    assert meta["graphs"][0]["file"] == "s2_scenes.json" and "process" not in meta["graphs"][0]
    assert json.loads(Path(files["graphs"][0]).read_text())["id"] == "geocrop_s2_scenes"


class _Conn:
    def __init__(self):
        self.auth = None

    def authenticate_oidc_client_credentials(self, client_id, client_secret, provider_id):
        self.auth = ("client", client_id, provider_id)
        return self

    def authenticate_oidc_device(self, provider_id, store_refresh_token):
        self.auth = ("device", provider_id, store_refresh_token)
        return self


@pytest.fixture()
def fake_openeo(monkeypatch):
    mod = types.ModuleType("openeo")
    mod.__version__ = "0.0-test"
    mod.urls = []

    def connect(url):
        mod.urls.append(url)
        return _Conn()
    mod.connect = connect
    monkeypatch.setitem(sys.modules, "openeo", mod)
    for var in (oa.ENV_URL, oa.ENV_CLIENT_ID, oa.ENV_CLIENT_SECRET, oa.ENV_PROVIDER):
        monkeypatch.delenv(var, raising=False)
    return mod


def test_openeo_connect_authentication_modes(fake_openeo, monkeypatch):
    with pytest.raises(ValueError, match="auth must be"):
        oa.connect(auth="password")
    assert oa.connect().auth == ("device", None, False)
    assert fake_openeo.urls == [oa.CDSE_URL]
    with pytest.raises(RuntimeError, match=oa.ENV_CLIENT_ID):
        oa.connect(auth="client-credentials")
    monkeypatch.setenv(oa.ENV_CLIENT_ID, "id")
    monkeypatch.setenv(oa.ENV_CLIENT_SECRET, "secret")
    monkeypatch.setenv(oa.ENV_PROVIDER, "CDSE")
    monkeypatch.setenv(oa.ENV_URL, "https://other")
    assert oa.connect("https://ignored").auth == ("client", "id", "CDSE")
    assert fake_openeo.urls[-1] == "https://other"
    assert oa.connect(auth="device").auth == ("device", "CDSE", False)       # device flow even with credentials


def test_openeo_execute_runs_jobs_and_converts(fake_openeo, ref, tmp_path):
    labels = oa.signature_labels(["NDVI"])
    log = []

    class Job:
        job_id = "j-1"

        def start_and_wait(self):
            log.append("waited")

        def get_results(self):
            return self

        def download_files(self, target):
            target = Path(target)
            target.mkdir(parents=True)
            (target / "job-results.json").write_text("{}")
            (target / "timeseries.json").write_text(json.dumps({"2023-04-05T00:00:00Z": [[1.0, 0.5, 2, 4]]}))
            return [target / "job-results.json", target / "timeseries.json"]

    class Cube:
        def create_job(self, title, job_options):
            log.append((title, job_options))
            return Job()

    class Connection:
        def datacube_from_flat_graph(self, graph):
            return Cube()

    item = {"name": "signatures", "product": "signatures", "process": {"process_graph": {}},
            "process_graph_sha256": "abc", "parcel_ids": ["P1"], "indices": ["NDVI"], "band_order": labels,
            "target": str(tmp_path / "out" / "signatures.csv")}
    recipe = {"reference_grid": {"path": str(ref)}, "backend_url": "https://b", "differences": [], "graphs": [item]}
    res = oa.execute(recipe, Connection(), tmp_path / "work", {"memory": "2G"}, log=log.append)
    assert res == [{"product": "signatures", "rows": 1, "path": item["target"]}]
    assert ("geocrop signatures", {"memory": "2G"}) in log and "[openeo] signatures: job j-1" in log
    row = pd.read_csv(item["target"]).iloc[0]
    assert row["mean"] == 0.5 and row["n_valid"] == 2 and row["n_pixels"] == 4


def test_openeo_convert_rejects_unexpected_file_sets(ref, tmp_path):
    sig = {"name": "signatures", "product": "signatures", "parcel_ids": ["P1"], "indices": ["NDVI"]}
    with pytest.raises(ValueError, match="one signature result"):
        oa.convert(sig, [tmp_path / "a.json", tmp_path / "b.csv"], ref, {}, {})
    comp = {"name": "s2_composite_spring", "product": "s2_composites", "start": "2023-04-01", "end": "2023-04-30"}
    with pytest.raises(ValueError, match="expected one GeoTIFF"):
        oa.convert(comp, [tmp_path / "a.tif", tmp_path / "b.tif"], ref, {}, {})


def test_openeo_alignment_method_cases(ref, tmp_path):
    with rasterio.open(ref) as r:
        same = _tif(tmp_path / "same.tif", np.ones(SHAPE), "float32", origin=(ORIGIN[0] - 20, ORIGIN[1] + 10))
        half = _tif(tmp_path / "half.tif", np.ones(SHAPE), "float32", origin=(ORIGIN[0] - 5, ORIGIN[1]))
        coarse = _tif(tmp_path / "coarse.tif", np.ones(SHAPE), "float32", res=20.0)
        other = _tif(tmp_path / "other.tif", np.ones(SHAPE), "float32", crs="EPSG:32633",
                     origin=(-10000.0, 5000000.0))
        methods = {}
        for p in (same, half, coarse, other):
            with rasterio.open(p) as src:
                methods[p.stem] = oa.alignment_method(src, r)
    assert methods == {"same": "exact", "half": "resampled", "coarse": "resampled", "other": "resampled"}


def test_openeo_result_without_crs_and_scene_edge_cases(ref, tmp_path):
    nocrs = tmp_path / "nocrs.tif"
    with rasterio.open(nocrs, "w", driver="GTiff", width=8, height=6, count=2, dtype="float32") as ds:
        ds.write(np.ones((2, 6, 8), "float32"))
    with rasterio.open(ref) as r, pytest.raises(ValueError, match="no CRS"):
        oa._open_result(nocrs, r, 2)
    order = [*S2_BANDS, "SCL"]
    data = np.full((len(order), 10, 12), 2000.0)
    with pytest.raises(ValueError, match="no acquisition date"):
        oa.write_scenes([_tif(tmp_path / "result.tif", data, "float32", origin=(ORIGIN[0] - 20, ORIGIN[1] + 20))],
                        tmp_path / "scenes", ref, order, {})
    data[-1] = 0                                           # SCL 0 everywhere: no coverage on that day
    empty = _tif(tmp_path / "openEO_2023-04-05Z.tif", data, "float32", origin=(ORIGIN[0] - 20, ORIGIN[1] + 20))
    assert oa.write_scenes([empty], tmp_path / "scenes", ref, order, {}) == []


def test_openeo_signature_reader_validation(tmp_path):
    labels = oa.signature_labels(["NDVI"])
    csv = tmp_path / "s.csv"
    csv.write_text("date,feature_index,a,b\n2023-04-05,0,1,2\n")
    with pytest.raises(ValueError, match="2 value columns, expected 4"):
        oa.read_signatures(csv, ["P1"], ["NDVI"])
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"2023-04-05": [[1, 2, 3]]}))
    with pytest.raises(ValueError, match="3 values per feature"):
        oa.read_signatures(short, ["P1"], ["NDVI"])
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"2023-04-05": [[0] * len(labels), [1, 1, 1, 1]]}))
    with pytest.raises(ValueError, match="feature index 1 outside"):
        oa.read_signatures(outside, ["P1"], ["NDVI"])


def test_openeo_write_table_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    table = pd.DataFrame({"parcel_id": ["P1"], "mean": [0.5]})
    out = oa.write_table(table, tmp_path / "sub" / "sig.parquet")
    assert pd.read_parquet(out).equals(table)


# --------------------------------------------------------------------------- datacube without xarray

def test_datacube_names_missing_xarray(monkeypatch):
    from geocrop import datacube
    _block_import(monkeypatch, "xarray")
    with pytest.raises(RuntimeError, match="xarray"):
        datacube._xr()
