"""Acceptance run: offline (synthetic, default tier) and network (real Sentinel-1/2, `pytest -m network`)."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import acceptance


@pytest.fixture(scope="module")
def offline_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("acceptance")
    return out, acceptance.run(out, offline=True, log=lambda m: None)


def test_offline_acceptance_passes_with_all_checks(offline_report):
    out, report = offline_report
    assert report["verdict"] == "PASS", report["summary"]
    ids = {c["id"] for c in report["checks"]}
    assert {"grid_identity", "s2_negative_reflectance", "s1_vv_gt_vh", "s1_s2_coregistration", "valid_coverage",
            "nodata_leakage", "index_ranges", "probabilities", "reproducibility", "parcel_schema",
            "holdout_recovery", "export_cog", "export_stac", "run_manifest"} <= ids
    for c in report["checks"]:
        assert c["threshold"] and "value" in c
    stages = {s["stage"] for s in report["stages"]}
    assert {"s2_composites", "s1_coregistration", "s1_composites", "coregistration_check", "features", "segmentation", "training", "prediction", "parcels", "validation",
            "export"} <= stages
    assert all(s["seconds"] >= 0 and s["process_peak_rss_mb"] > 0 for s in report["stages"])


def test_offline_report_files_are_schema_versioned(offline_report):
    out, report = offline_report
    body = json.loads((out / "acceptance_report.json").read_text())
    assert body["schema"] == acceptance.SCHEMA and body["engine"]["version"]
    assert body["pseudo_reference"]["note"] == acceptance.PSEUDO_NOTE
    md = (out / "acceptance_report.md").read_text()
    assert "PASS" in md and "not crop types" in md and "does NOT prove" in md
    assert str(out) not in md          # report is portable: paths are relative to the output directory


def test_failed_check_and_engine_error_give_fail_verdict(tmp_path, monkeypatch):
    # a sentinel leak must fail its check; a later engine exception must be recorded, not crash the report
    monkeypatch.setattr(acceptance, "nodata_leakage", lambda *a: {"declared_nodata": -9999, "window_nodata_leaks": 7,
                                                                  "near_sentinel_values": 0, "nonfinite_values": 0})

    def broken(*a, **k):
        raise ValueError("engine defect")
    monkeypatch.setattr(acceptance, "pseudo_reference", broken)
    report = acceptance.run(tmp_path, offline=True, log=lambda m: None)
    assert report["verdict"] == "FAIL" and "nodata_leakage" in report["summary"]["failed"]
    assert "engine defect" in report["error"]
    assert report["stages"][-1] == {**report["stages"][-1], "stage": "pseudo_reference", "status": "error"}


def test_unavailable_service_is_incomplete_not_fail(tmp_path, monkeypatch):
    def outage(*a, **k):
        raise acceptance.DataUnavailable("Sentinel-2 STAC service unavailable: timeout")
    monkeypatch.setattr(acceptance, "_real_scenes", outage)
    report = acceptance.run(tmp_path, log=lambda m: None)
    assert report["verdict"] == "INCOMPLETE" and "unavailable" in report["error"]
    assert (tmp_path / "acceptance_report.md").exists()


def test_coregistration_detects_shift(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :120]
    field = (np.sin(xx / 7.0) + np.cos(yy / 5.0) + rng.normal(0, 0.1, (120, 120))).astype("float32")
    ndvi = 0.5 + 0.2 * field
    prof = {"driver": "GTiff", "height": 120, "width": 120, "count": 1, "dtype": "float32", "crs": "EPSG:32634",
            "transform": from_origin(500000, 5000000, 10, 10), "nodata": -9999}
    folder = tmp_path / "P1"
    folder.mkdir()
    red = np.full_like(ndvi, 0.05)
    nir = red * (1 + ndvi) / (1 - ndvi)
    ratio_db = -8 + 4 * np.roll(field, (2, 0), axis=(0, 1))      # radar shifted by two rows
    vv = np.full_like(ndvi, 0.1)
    vh = vv * 10 ** (ratio_db / 10)
    for name, arr in (("B04", red), ("B08", nir), ("VV", vv), ("VH", vh)):
        with rasterio.open(folder / f"{name}.tif", "w", **prof) as dst:
            dst.write(arr.astype("float32"), 1)
    co = acceptance.coregistration(tmp_path, ["P1"])
    assert co["best_shift_px"] == pytest.approx([2, 0], abs=0.3) and co["r_at_best_shift"] > co["r_at_zero_shift"]


def test_s1_offsets_are_estimated_and_removed(tmp_path):
    from scipy.ndimage import gaussian_filter, shift

    from geocrop import s1_stac
    from geocrop.scenes import write
    rng = np.random.default_rng(1)
    field = gaussian_filter(rng.normal(0, 1, (150, 150)), 3)
    ndvi = 0.5 + 0.3 * field / np.abs(field).max()
    prof = {"driver": "GTiff", "height": 150, "width": 150, "count": 1, "dtype": "float32", "crs": "EPSG:32634",
            "transform": from_origin(500000, 5000000, 10, 10), "nodata": -9999}
    true = {"ASCENDING": (1.5, -3.0), "DESCENDING": (0.0, 0.0)}
    dirs = []
    for k, (orbit, (dy, dx)) in enumerate(true.items()):
        for rep in range(2):
            folder = tmp_path / "raw" / f"{orbit}{rep}"
            folder.mkdir(parents=True)
            vh_db = -20 + 10 * ndvi + rng.normal(0, 0.5, ndvi.shape)
            vh_db = shift(vh_db, (dy, dx), order=1, mode="nearest")      # S1[y + dy, x + dx] ~ NDVI[y, x]
            for pol, db in (("VH", vh_db), ("VV", vh_db + 6)):
                with rasterio.open(folder / f"{pol}.tif", "w", **prof) as dst:
                    dst.write((10 ** (db / 10)).astype("float32"), 1)
            write(folder, sensor="S1", datetime=f"2023-07-0{k * 2 + rep + 1}T05:00:00", orbit_direction=orbit,
                  relative_orbit=51 + k)
            dirs.append(folder)
    est = s1_stac.estimate_offsets(dirs, ndvi)
    asc, desc = est["ASC_R051"], est["DES_R052"]
    assert asc["accepted"] and asc["dy"] == pytest.approx(1.5, abs=0.3) and asc["dx"] == pytest.approx(-3, abs=0.3)
    assert abs(desc["dy"]) < 0.3 and abs(desc["dx"]) < 0.3
    fixed = s1_stac.apply_offsets(dirs, est, tmp_path / "fixed")
    again = s1_stac.estimate_offsets(fixed, ndvi)
    assert abs(again["ASC_R051"]["dy"]) < 0.3 and abs(again["ASC_R051"]["dx"]) < 0.3
    meta = json.loads((fixed[0] / "scene.json").read_text())
    assert meta["coregistration"]["applied"] and meta["coregistration"]["raw_scene"] == dirs[0].name
    with rasterio.open(dirs[0] / "VH.tif") as raw:           # raw cache untouched
        assert raw.read(1)[75, 75] > 0


def test_cli_acceptance_registered():
    from geocrop.cli import build_parser
    a = build_parser().parse_args(["acceptance", "--out", "x", "--offline"])
    assert a.offline and a.bbox == list(acceptance.DEFAULT_BBOX) and a.start == acceptance.DEFAULT_START


def test_s1_signer_appends_token_once(monkeypatch):
    from geocrop import s1_stac
    s = s1_stac.Signer()
    monkeypatch.setattr(s, "token", lambda: "st=1&sig=abc")
    monkeypatch.setitem(__import__("sys").modules, "planetary_computer", None)
    url = "https://x.blob.core.windows.net/c/a.tif"
    assert s.sign(url) == url + "?st=1&sig=abc"
    assert s.sign(url + "?st=1&sig=abc") == url + "?st=1&sig=abc"
    assert s.sign("https://example.org/a.tif") == "https://example.org/a.tif"


def test_s1_token_outage_is_service_unavailable(monkeypatch):
    import requests

    from geocrop import s1_stac

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(s1_stac.ServiceUnavailable):
        s1_stac.Signer().token()


# --------------------------------------------------------------------------- real data

@pytest.mark.network
def test_s1_rtc_import_is_physically_plausible(tmp_path):
    pytest.importorskip("pystac_client")
    from geocrop import s1_stac
    grid = tmp_path / "grid.tif"
    prof = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "uint8", "crs": "EPSG:32634",
            "transform": from_origin(390000.0, 4960000.0, 10, 10), "nodata": 0}
    with rasterio.open(grid, "w", **prof) as dst:
        dst.write(np.ones((100, 100), "uint8"), 1)
    try:
        res = s1_stac.import_season(grid, tmp_path / "s1", "2023-07-01", "2023-07-10", max_items=4)
    except s1_stac.ServiceUnavailable as exc:
        pytest.skip(str(exc))
    done = [r for r in res if r["status"] == "imported"]
    if not done:
        pytest.skip(f"no Sentinel-1 RTC acquisitions: {res}")
    meta = json.loads((tmp_path / "s1" / done[0]["path"].split("/")[-1] / "scene.json").read_text())
    assert meta["sensor"] == "S1" and meta["backscatter"] == "gamma0"
    assert meta["orbit_direction"] in ("ASCENDING", "DESCENDING") and isinstance(meta["relative_orbit"], int)
    stats = s1_stac.plausibility(done[0]["path"])
    assert stats["linear_in_1e-4_1"] > 0.95 and stats["vv_gt_vh"] > 0.9
    assert -20 < stats["vv_db_p1_p50_p99"][1] < -3 and -28 < stats["vh_db_p1_p50_p99"][1] < -10


@pytest.mark.network
def test_real_acceptance_semberija(tmp_path):
    pytest.importorskip("pystac_client")
    report = acceptance.run(tmp_path, log=lambda m: None)
    if report["verdict"] == "INCOMPLETE":
        pytest.skip(f"data services unavailable: {report['error']}")
    assert report["verdict"] == "PASS", report["summary"]
    assert report["stac_items"]["sentinel-2"] and report["stac_items"]["sentinel-1"]
