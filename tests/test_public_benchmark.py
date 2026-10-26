"""Public benchmark (G1), HCAT harmonisation (G12) and crop calendars (G11) on synthetic data.

The real EuroCrops / Earth Search run is documented in docs/benchmarks/; only the network smoke test below
touches the internet (``pytest -m network``).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from geocrop import calendars, cli, hcat, public_benchmark, public_data
from geocrop.config import parse_windows
from geocrop.export import read_hcat

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "benchmarks" / "eurocrops_si_2021_pomurje.v1.yaml"
HARMONISATION = ROOT / "configs" / "hcat" / "rs_bih_hcat_harmonisation.csv"
EXPORT_MAPPING = ROOT / "configs" / "hcat" / "hcat_mapping_rs_bih.csv"


# --------------------------------------------------------------------------- HCAT (G12)

def test_significant_prefix_and_code_normalisation():
    assert hcat.significant_prefix("3302000000") == "3302"
    assert hcat.significant_prefix("3301010101") == "3301010101"
    assert hcat.normalise_code(3301010600.0) == "3301010600"
    assert hcat.normalise_code("") is None and hcat.normalise_code(float("nan")) is None
    with pytest.raises(ValueError):
        hcat.normalise_code("33010")


def test_exact_rules_win_over_subtree_and_longest_prefix_wins():
    rules = hcat.rules_from_spec({"cereals": {"subtree": ["3301010000"]}, "wheat": {"subtree": ["3301010100"]},
                                  "winter_barley": {"exact": ["3301010401"]}})
    assert hcat.classify("3301010101", rules) == "wheat"          # longer prefix than cereals
    assert hcat.classify("3301010402", rules) == "cereals"
    assert hcat.classify("3301010401", rules) == "winter_barley"  # exact beats subtree
    assert hcat.classify("3302000000", rules) is None
    mapped = hcat.harmonise(["3301010101", None, "bad", "3302000000"], rules)
    assert mapped.tolist() == ["wheat", None, None, None]


def test_conflicting_rules_are_rejected():
    with pytest.raises(ValueError, match="maps to both"):
        hcat.rules_from_spec({"a": ["3301010600"], "b": ["3301010600"]})
    with pytest.raises(ValueError):
        hcat.rules_from_spec({"a": {}})


def test_committed_harmonisation_covers_engine_classes():
    rules = hcat.read_rules(HARMONISATION)
    with (ROOT / "configs" / "hcat_mapping.template.csv").open(encoding="utf8") as fh:
        engine_classes = {r["class_name"] for r in csv.DictReader(fh)}
    assert engine_classes == {r.class_name for r in rules}
    assert hcat.classify("3301010600", rules) == "maize"
    assert hcat.classify("3301090400", rules) == "maize"
    assert hcat.classify("3301010101", rules) == "winter_crops"
    assert hcat.classify("3301010102", rules) is None            # spring wheat is not a winter crop
    assert hcat.classify("3302000000", rules) == "grassland"
    assert hcat.classify("3306010000", rules) == "forest"        # subtree: afforestation
    assert hcat.classify("3301220400", rules) == "vegetables"    # onions under alliums
    assert hcat.classify("3301060401", rules) is None            # winter rapeseed deliberately unmapped
    report = hcat.coverage_report(["3301010600", "3301060401", "3301060401"], rules)
    assert report["mapped"] == 1 and report["top_unmapped_codes"] == {"3301060401": 2}


def test_export_mapping_file_is_complete_and_licensed():
    mapping = read_hcat(EXPORT_MAPPING)
    rules = hcat.read_rules(HARMONISATION)
    assert set(mapping) == {r.class_name for r in rules}
    assert all(v["code"] and len(v["code"]) == 10 for v in mapping.values())
    licence = (ROOT / "configs" / "hcat" / "LICENSE-HCAT.md").read_text(encoding="utf8")
    assert "CC-BY-SA-4.0" in licence and "10.5281/zenodo.14094196" in licence
    assert hcat.export_mapping(rules)["maize"]["code"] == "3301010600"


# --------------------------------------------------------------------------- calendars (G11)

def _markers(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    specs = {"winter_crops": (60, 125, 180, 190), "maize": (150, 215, 265, np.nan), "soybean": (160, 220, 260, np.nan)}
    for cls, (sos, peak, eos, harvest) in specs.items():
        for k in range(30):
            rows.append({"parcel_id": f"{cls}-{k}", "sos_doy": sos + rng.normal(0, 5), "peak_doy": peak + rng.normal(0, 5),
                         "eos_doy": eos + rng.normal(0, 5),
                         "harvest_doy": harvest + rng.normal(0, 5) if np.isfinite(harvest) else np.nan,
                         "class_name": cls})
    frame = pd.DataFrame(rows)
    return frame.drop(columns="class_name"), frame[["parcel_id", "class_name"]]


def test_derive_calendar_percentiles_and_consistency_format():
    markers, labels = _markers()
    derived = calendars.derive(markers, labels)
    maize = derived["classes"]["maize"]
    assert maize["n_parcels"] == 30 and maize["harvest"] is None
    assert 205 < maize["peak"]["p50"] < 225
    assert maize["peak"]["p10"] < maize["peak"]["p50"] < maize["peak"]["p90"]
    cal = calendars.to_calendar(derived)
    assert set(cal["winter_crops"]) == {"sos", "peak", "eos", "harvest"} and "harvest" not in cal["maize"]
    lo, hi = cal["maize"]["peak"]
    assert isinstance(lo, int) and lo <= maize["peak"]["p10"] and hi >= maize["peak"]["p90"]
    from geocrop.cbm import calendar_consistency
    ok = calendar_consistency(markers.iloc[[0, 40]].assign(harvest_doy=np.nan), pd.Series(["winter_crops", "winter_crops"]), cal)
    assert ok.tolist() == [True, False]
    assert yaml.safe_load(calendars.calendar_yaml(cal, derived))["maize"]["peak"] == [lo, hi]


def test_min_parcels_hides_unsupported_events():
    markers, labels = _markers()
    derived = calendars.derive(markers.iloc[:3], labels, min_parcels=5)
    assert derived["classes"]["winter_crops"]["peak"] is None
    with pytest.raises(ValueError):
        calendars.derive(markers, pd.DataFrame({"parcel_id": ["x"], "class_name": ["y"]}))


def test_suggested_windows_are_ordered_long_enough_and_valid_config():
    markers, labels = _markers()
    derived = calendars.derive(markers, labels)
    windows = calendars.suggest_windows(derived, 2023, min_days=30, max_windows=5)
    parsed = parse_windows(windows)            # raises on overlap or reversed dates
    assert 2 <= len(parsed) <= 5
    for start, end in parsed.values():
        assert (end - start).days + 1 >= 29
    first = min(s for s, _ in parsed.values())
    assert first <= date(2023, 3, 1)           # earliest SOS (~DOY 60) minus margin
    snippet = yaml.safe_load(calendars.windows_yaml(windows, derived, "synthetic"))
    assert snippet["windows"] == windows


def test_worldcereal_priors_from_synthetic_raster(tmp_path):
    profile = {"driver": "GTiff", "width": 4, "height": 4, "count": 1, "dtype": "float32", "crs": "EPSG:4326",
               "transform": from_origin(15, 48, 0.5, 0.5), "nodata": 0.0}
    paths = {}
    for stage, value in (("sos", 113.0), ("eos", 0.0)):
        p = tmp_path / f"{stage}.tif"
        with rasterio.open(p, "w", **profile) as ds:
            ds.write(np.full((1, 4, 4), value, dtype="float32"))
        paths[stage] = p
    priors = calendars.worldcereal_priors({"maize": paths}, 16.2, 46.6)
    assert priors == {"maize": {"sos": 113, "eos": None}}


def test_cli_calendar_derive(tmp_path, capsys):
    markers, labels = _markers()
    markers.to_csv(tmp_path / "markers.csv", index=False)
    labels.rename(columns={"class_name": "crop"}).to_csv(tmp_path / "labels.csv", index=False)
    cli.main(["calendar", "derive", "--markers", str(tmp_path / "markers.csv"), "--labels", str(tmp_path / "labels.csv"),
              "--label-field", "crop", "--year", "2023", "--out", str(tmp_path / "cal.yaml"),
              "--windows-out", str(tmp_path / "win.yaml")])
    out = json.loads(capsys.readouterr().out)
    assert out["classes"] == {"maize": 30, "soybean": 30, "winter_crops": 30}
    assert set(yaml.safe_load((tmp_path / "cal.yaml").read_text())) == {"maize", "soybean", "winter_crops"}
    parse_windows(yaml.safe_load((tmp_path / "win.yaml").read_text())["windows"])


# --------------------------------------------------------------------------- public data adapter

class _Response:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, size):
        yield self.body


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, **kw):
        self.calls += 1
        return self.responses.pop(0)


def test_fetch_waits_for_staged_file_and_verifies_checksum(tmp_path, monkeypatch):
    from geocrop import downloads
    monkeypatch.setattr(downloads, "_sleep", lambda s: None)
    body = b"tiff bytes"
    md5 = "md5:" + hashlib.md5(body).hexdigest()
    session = _Session([_Response(503, b"<html>loading from tape</html>", {"retry-after": "1",
                                                                         "content-type": "text/html"}),
                        _Response(200, body, {"content-type": "image/tiff"})])
    dest = public_data.fetch("https://example.org/a.tif", tmp_path / "a.tif", md5, session=session)
    assert dest.read_bytes() == body and session.calls == 2
    # cached and verified: no further request
    assert public_data.fetch("https://example.org/a.tif", dest, md5, session=_Session([])) == dest
    bad = _Session([_Response(200, b"other", {"content-type": "image/tiff"})])
    with pytest.raises(ValueError, match="Checksum mismatch"):
        public_data.fetch("https://example.org/b.tif", tmp_path / "b.tif", md5, session=bad)
    assert not (tmp_path / "b.tif").exists()


def test_eurocrops_layer_extracts_cached_archive(tmp_path):
    parcels = gpd.GeoDataFrame({"ID": [1], "EC_hcat_c": ["3301010600"]}, geometry=[box(0, 0, 100, 100)],
                               crs="EPSG:3794")
    src = tmp_path / "src"
    src.mkdir()
    parcels.to_file(src / "XX_2021.shp")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"XX_2021/{f.name}")
    cache = tmp_path / "cache"
    target = cache / "eurocrops" / "1" / "XX_2021.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(buf.getvalue())
    checksum = "md5:" + hashlib.md5(buf.getvalue()).hexdigest()
    layer = public_data.eurocrops_layer("XX_2021.zip", checksum, "1", cache)
    assert layer.name == "XX_2021.shp"
    got = public_data.read_parcels(layer, (-1, -1, 1, 1), "EPSG:3794")
    assert got["EC_hcat_c"].tolist() == ["3301010600"]


# --------------------------------------------------------------------------- benchmark protocol (G1)

def test_committed_protocol_is_valid_and_hashed():
    p, digest = public_benchmark.load_protocol(PROTOCOL)
    assert digest == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert p["dataset"]["checksum"].startswith("md5:") and p["dataset"]["licence"] == "CC-BY-SA-4.0"
    assert "rf" in p["models"] and 5 <= len(p["classes"]) <= 8
    boxes = public_benchmark.tile_boxes(p)
    assert all(abs(b.area - 1e8) < 1 for b in boxes.values())


def test_protocol_validation_rejects_inconsistent_windows():
    p = yaml.safe_load(PROTOCOL.read_text())
    p["windows"]["P9"] = {"start": "2022-01-01", "end": "2022-01-31"}
    with pytest.raises(ValueError, match="protocol year"):
        public_benchmark.validate_protocol(p)


def _synthetic_parcels(n_per_class=40, classes=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    geoms, cls, ids = [], [], []
    for c in classes:
        for i in range(n_per_class):
            x, y = rng.uniform(500000, 520000), rng.uniform(5000000, 5020000)
            geoms.append(box(x, y, x + 80, y + 80))
            cls.append(c)
            ids.append(f"{c}{i}")
    return gpd.GeoDataFrame({"parcel_id": ids, "class_name": cls}, geometry=geoms, crs="EPSG:32633")


def test_label_parcels_and_stratified_sampling():
    p = yaml.safe_load(PROTOCOL.read_text())
    tile = box(500000, 5000000, 510000, 5010000)
    raw = gpd.GeoDataFrame({"ID": [1, 2, 3, 4, 5],
                            "EC_hcat_c": ["3301010600", "3301010101", "3301010600", "3399000000", "3301160000"]},
                           geometry=[box(500100, 5000100, 500200, 5000200), box(500300, 5000300, 500400, 5000400),
                                     box(509950, 5000100, 510050, 5000200),           # crosses the tile edge
                                     box(500500, 5000500, 500600, 5000600), box(500700, 5000700, 500710, 5000710)],
                           crs="EPSG:32633")
    got = public_benchmark.label_parcels(raw, p, "T", tile)
    assert got[["parcel_id", "class_name"]].values.tolist() == [["1", "maize"], ["2", "winter_cereals"]]
    pop = _synthetic_parcels()
    s1 = public_benchmark.sample_parcels(pop, 10, 3)
    s2 = public_benchmark.sample_parcels(pop.sample(frac=1, random_state=1), 10, 3)
    assert s1["parcel_id"].tolist() == s2["parcel_id"].tolist()           # order of the population is irrelevant
    assert s1["class_name"].value_counts().to_dict() == {"a": 10, "b": 10, "c": 10}


def test_split_parcels_separates_blocks_and_buffers():
    parcels = _synthetic_parcels(60)
    split = {"spatial_block_m": 2000, "test_fraction": 0.3, "calibration_fraction": 0.25, "buffer_m": 300, "seed": 1}
    roles = public_benchmark.split_parcels(parcels, split)
    assert set(roles) <= {"fit", "calibration", "test", "buffer_excluded"}
    blocks = public_benchmark.spatial_block(parcels.geometry.centroid, 2000)
    frame = pd.DataFrame({"role": roles.to_numpy(), "block": blocks})
    test_blocks = set(frame.loc[frame.role == "test", "block"])
    assert not test_blocks & set(frame.loc[frame.role.isin(["fit", "calibration"]), "block"])
    cent = parcels.geometry.centroid
    test_xy = np.column_stack([cent.x, cent.y])[roles.to_numpy() == "test"]
    for i in np.flatnonzero(roles.isin(["fit", "calibration"]).to_numpy()):
        assert np.min(np.hypot(*(test_xy - [cent.x.iloc[i], cent.y.iloc[i]]).T)) > 300
    assert public_benchmark.split_parcels(parcels, split).equals(roles)


def test_review_threshold_and_block_bootstrap():
    conf = np.array([0.99, 0.95, 0.9, 0.9, 0.6, 0.4])
    correct = np.array([True, True, True, False, True, False])
    assert public_benchmark.review_threshold(conf, correct, 1.0) == 0.95
    assert public_benchmark.review_threshold(conf, correct, 0.75) == 0.6
    assert public_benchmark.review_threshold(conf, ~np.ones(6, bool), 0.5) == float("inf")
    y = np.array(["a"] * 20 + ["b"] * 20)
    boot = public_benchmark.block_bootstrap(y, y, np.arange(40) // 4, ["a", "b"], 50, 0, 0.95)
    assert boot["overall_accuracy_ci"] == [1.0, 1.0] and boot["per_class_f1_ci"]["a"] == [1.0, 1.0]


def test_window_candidates_dedupe_versions_and_tiles():
    aoi = box(16.0, 46.5, 16.1, 46.6)

    def item(iid, day, cloud, baseline, geom):
        return SimpleNamespace(id=iid, geometry=mapping(geom), properties={
            "datetime": f"{day}T10:00:00Z", "eo:cloud_cover": cloud, "s2:processing_baseline": baseline})
    full, partial = box(15, 46, 17, 47), box(16.05, 46.5, 17, 47)
    items = [item("A_0", "2021-05-03", 10, "02.14", full), item("A_1", "2021-05-03", 10, "05.00", full),
             item("B_0", "2021-05-03", 5, "05.00", partial), item("C_0", "2021-05-10", 1, "05.00", full),
             item("D_0", "2021-06-10", 0, "05.00", full)]
    got = public_benchmark.window_candidates(items, date(2021, 5, 1), date(2021, 5, 31), aoi)
    assert [i.id for i in got] == ["C_0", "A_1"]


def _synthetic_data(classes=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    parts = {}
    truth, block = {}, {}
    for role, n in (("fit", 30), ("cal", 15), ("test", 15)):
        X, y, poly = [], [], []
        for k, c in enumerate(classes):
            for i in range(n):
                pid = f"{role}-{c}-{i}"
                truth[pid], block[pid] = c, f"{role}{i % 5}"
                X.append(rng.normal(k, 0.6, size=(8, 4)))
                y += [c] * 8
                poly += [pid] * 8
        parts[role] = (np.vstack(X).astype("float32"), np.array(y), np.array(poly))
    return {"X_fit": parts["fit"][0], "y_fit": parts["fit"][1], "X_cal": parts["cal"][0], "poly_cal": parts["cal"][2],
            "X_test": parts["test"][0], "poly_test": parts["test"][2], "y_test": parts["test"][1],
            "truth": pd.Series(truth), "block": pd.Series(block)}


def test_evaluate_reports_parcel_metrics_conformal_and_review():
    p = yaml.safe_load(PROTOCOL.read_text())
    p["evaluation"]["bootstrap"] = 50
    data = _synthetic_data()
    res = public_benchmark.evaluate("rf", {"algorithm": "rf", "n_estimators": 30, "n_jobs": 1}, data, p,
                                    ["a", "b", "c"], 0)
    assert res["status"] == "ok"
    assert res["parcel"]["n"] == 45 and res["parcel"]["overall_accuracy"] > 0.8
    assert res["conformal"]["target"] == pytest.approx(0.9) and res["conformal"]["mean_set_size"] > 0
    assert 0 <= res["review"]["review_share"] <= 1
    lo, hi = res["bootstrap"]["macro_f1_ci"]
    assert lo <= res["parcel"]["macro_f1"] + 1e-9 and hi >= lo
    assert res["time"]["fit_pixels"] == len(data["y_fit"])
    skipped = public_benchmark.evaluate("x", {"algorithm": "catboost"}, data, p, ["a", "b", "c"], 0) \
        if not _has("catboost") else {"status": "skipped"}
    assert skipped["status"] == "skipped"


def _has(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


def test_markdown_report_renders_all_sections():
    p = yaml.safe_load(PROTOCOL.read_text())
    p["evaluation"]["bootstrap"] = 20
    data = _synthetic_data()
    ok = public_benchmark.evaluate("rf", {"algorithm": "rf", "n_estimators": 20, "n_jobs": 1}, data, p, ["a", "b", "c"], 0)
    report = {"protocol": {"id": "synthetic", "version": 1, "file": "x.yaml", "sha256": "0" * 64},
              "command": "geocrop benchmark-public --protocol x.yaml", "engine": {"version": "0", "git_commit": "abc",
                                                                                  "git_dirty": False},
              "started_utc": "t0", "finished_utc": "t1", "runtime": {"platform": "test", "cpu_count": 1},
              "peak_rss_mb": 1.0, "dataset": {"name": "Synthetic", "archive": "none", "zenodo_record": "0",
                                              "checksum": "md5:0", "licence": "none", "citation": "none"},
              "samples": {"pixels": 1, "features": 4, "fingerprint": "f" * 64}, "classes": ["a", "b", "c"],
              "models": {"rf": ok, "lightgbm": {"status": "skipped", "reason": "pip install lightgbm"}},
              "labels": {"population_per_class": {"a": 1}, "sampled_per_class": {"a": 1}, "with_pixels_per_class": {"a": 1}},
              "split": {"parcels_per_class_and_role": {"a": {"test": 1}}, "blocks_per_role": {}, "pixels_per_role": {}},
              "imagery": {"T": {"windows": {"P1": {"start": "s", "end": "e", "used": ["S2A_X"]}}}},
              "stages": {"labels": {"seconds": 1, "peak_rss_mb": 1}}, "note": "synthetic"}
    md = public_benchmark.markdown(report)
    for text in ("## Models", "## Per-class F1", "skipped: pip install lightgbm", "S2A_X", "0" * 64):
        assert text in md


@pytest.mark.network
def test_eurocrops_mapping_table_is_reachable(tmp_path):
    """The Slovenian EuroCrops mapping table on Zenodo still has the HCAT columns the protocol relies on."""
    path = public_data.eurocrops_file("si_2021.csv", "md5:c6554c1e9408da7e6a867c17c6b48f9e", cache=tmp_path)
    header = path.read_text(encoding="utf8", errors="replace").splitlines()[0]
    assert "HCAT3_code" in header
