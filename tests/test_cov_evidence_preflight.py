"""Edge and error paths of evidence packages and preflight checks (synthetic, fast)."""
import json
import os
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import test_preflight
from rasterio.transform import from_origin
from shapely.geometry import box
from test_preflight import _cfg, codes
from test_temporal_quality import END, START, signatures

from geocrop import evidence, preflight, scenes

# the synthetic study of the preflight tests (module-scoped scenes, fresh copy per test)
study_src, study = test_preflight.study_src, test_preflight.study

# --------------------------------------------------------------------------- evidence helpers


def _table(tmp_path, n=2, **extra):
    ids = ["A", "B", "C"][:n]
    frame = pd.DataFrame({"parcel_id": ids, "declared_class": ["maize"] * n, "predicted_class": ["maize"] * n,
                          "decision": ["accepted"] * n, "declaration_status": ["agreement"] * n,
                          "p_maize": [0.9] * n, "p_soybean": [0.1] * n, **extra})
    path = tmp_path / "parcels.csv"
    frame.to_csv(path, index=False)
    return path


def test_clean_converts_every_value_to_json_safe_form():
    assert evidence._clean({"a": (np.int64(2), np.float32(np.nan))}) == {"a": [2, None]}
    assert evidence._clean(np.bool_(True)) is True
    assert evidence._clean(float("inf")) is None
    assert evidence._clean(pd.NA) is None
    assert evidence._clean(pd.NaT) is None
    assert evidence._clean(date(2023, 1, 2)) == "2023-01-02"
    assert evidence._clean("x") == "x" and evidence._clean(None) is None
    assert evidence._file_ref(None) is None
    assert evidence._file_ref("/no/such/file")["sha256"] is None


def test_numeric_restores_booleans_and_numbers_but_keeps_ids_and_text():
    frame = pd.DataFrame({"parcel_id": ["001", "002"], "flag": ["True", "False"], "n": ["1", "2.5"],
                          "text": ["a", "1"], "empty": [None, None]})
    out = evidence._numeric(frame, "parcel_id")
    assert out.parcel_id.tolist() == ["001", "002"]
    assert out.flag.tolist() == [True, False]
    assert out.n.tolist() == [1.0, 2.5]
    assert out.text.tolist() == ["a", "1"]          # mixed column stays text
    assert out["empty"].isna().all()


def test_parquet_parcel_table(tmp_path):
    pytest.importorskip("pyarrow")
    pd.read_csv(_table(tmp_path), dtype={"parcel_id": str}).to_parquet(tmp_path / "p.parquet")
    table, gdf = evidence._read_table(tmp_path / "p.parquet", "parcel_id")
    assert gdf is None and list(table.parcel_id) == ["A", "B"]


def test_decision_rules_for_combined_strategy_and_calendar():
    row = {"decision_reasons": "strategies_disagree;missing_margin", "majority_class": "a", "mean_class": "b",
           "declaration_status": "agreement", "calendar_consistent": False, "peak_doy": 200.0}
    rules = {r["rule"]: r for r in evidence.decision_rules(row, {"strategy": "combined", "thresholds": None})}
    assert rules["decision.strategies_agree"]["triggered"]
    assert rules["decision.min_margin"]["reason_code"] == "missing_margin"
    assert rules["calendar.consistent"]["outcome"] == "inconsistent"
    assert "decision.min_valid_pixels" not in rules and "applicability.min_aoa_fraction" not in rules
    assert "conformal.declared_in_set" not in rules
    for cal, outcome in ((True, "consistent"), (None, "not_applicable"), ("True", "consistent")):
        r = {x["rule"]: x for x in evidence.decision_rules({"calendar_consistent": cal}, {})}
        assert r["calendar.consistent"]["outcome"] == outcome and not r["calendar.consistent"]["triggered"]


def test_marker_rules_cover_every_family_and_not_applicable():
    rules = evidence.cbm_markers.resolve_rules({"enabled": True, "harvest_expected_classes": ["maize"],
                                                "mowing_expected_classes": ["grass"], "mixed_class_share": 0.3})
    row = {"declared_class": "maize", "harvest_evidence": "insufficient_data", "homogeneity": "heterogeneous",
           "homogeneity_confidence": 0.9}
    frame = pd.DataFrame([{"f_maize": 0.6, "f_soy": 0.4}])
    out = {r["rule"]: r for r in evidence.marker_rules(row, frame, rules)}
    assert [r for r in out] == [f"markers.{f[0]}" for f in evidence._MARKER_FAMILIES]
    assert out["markers.harvest"]["reason_code"] == "insufficient_marker_data_harvest"
    assert out["markers.harvest"]["outcome"] == rules["insufficient_data_outcome"]
    assert out["markers.mowing"]["outcome"] == "not_applicable"
    assert out["markers.heterogeneity"]["triggered"]
    assert out["markers.mixed_class"]["triggered"] and out["markers.mixed_class"]["inputs"]["second_class_share"] == 0.4
    no_mix = evidence.cbm_markers.resolve_rules({"enabled": True})
    out = {r["rule"]: r for r in evidence.marker_rules({"declared_class": None}, pd.DataFrame([{}]), no_mix)}
    assert out["markers.mixed_class"]["outcome"] == "not_applicable"
    assert out["markers.mixed_class"]["inputs"]["second_class_share"] is None
    assert all(not r["triggered"] for r in out.values())


def test_validate_lists_every_problem():
    body = {"schema": "other", "schema_version": True, "rules": [{"rule": "x"}, {k: 1 for k in evidence.RULE_KEYS}],
            "outcome": {"traffic_light": "blue"}, "data_sources": {"observed_acquisitions": "x"}}
    problems = evidence.validate(body)
    text = "\n".join(problems)
    assert "missing parcel_id" in text and "schema_version has type bool" in text
    assert "schema must be" in text and "rules[0] lacks" in text and "rules[1].triggered must be boolean" in text
    assert "traffic_light must be" in text and "data_sources.interpolated_grid must be a list" in text
    assert "rules has type" not in text
    assert any("schema_version" in p and "Invalid" in p for p in problems)   # the contract check also reports it


def test_load_refuses_corrupt_and_invalid_packages(tmp_path):
    bad = tmp_path / "evidence.json"
    bad.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        evidence.load(bad)
    bad.write_text(json.dumps({"schema": evidence.SCHEMA}))
    with pytest.raises(ValueError, match="invalid evidence package"):
        evidence.load(bad)


def test_build_input_errors(tmp_path):
    path = _table(tmp_path)
    with pytest.raises(ValueError, match="lacks 'pid'"):
        evidence.build(path, tmp_path / "ev", id_field="pid")
    dup = tmp_path / "dup.csv"
    pd.DataFrame({"parcel_id": ["A", "A"]}).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="unique"):
        evidence.build(dup, tmp_path / "ev")
    with pytest.raises(ValueError, match="Unknown parcel"):
        evidence.build(path, tmp_path / "ev", ids=["Z"])
    with pytest.raises(ValueError, match="both start and end"):
        evidence.build(path, tmp_path / "ev", start=START)


def test_build_with_markers_rules_scenes_and_informational_support(tmp_path):
    path = _table(tmp_path, n=3, calendar_consistent=[True, False, None], f_maize=[0.9, 0.5, 0.6],
                  f_soybean=[0.1, 0.5, 0.4])
    sig = signatures()
    markers = pd.DataFrame({"parcel_id": ["A", "B", "C"], "harvest_doy": [250.0, np.nan, 260.0],
                            "harvest_evidence": ["s2", "insufficient_data", "s2"], "harvest_detected": [True, False, True],
                            "harvest_confidence": [0.9, 0.0, 0.9], "data_sufficiency": ["sufficient"] * 3,
                            "calendar_consistent": [False, False, False]})
    support = evidence.tq.parcel_support(sig, START, END, evidence.tq.resolve(None))
    support["tq_observed_dates"] = "2023-04-01;2023-05-01"
    # scene folders resolve acquisition dates to scene ids
    for k, d in enumerate(("2023-04-01", "2023-04-06")):
        folder = tmp_path / "scenes" / f"s{k}"
        folder.mkdir(parents=True)
        scenes.write(folder, sensor="S2", datetime=f"{d}T10:00:00", tile="34TDR", relative_orbit=36)
    result = evidence.build(path, tmp_path / "ev", signatures=sig, start=START, end=END, markers=markers,
                            temporal_support=support, scenes=tmp_path / "scenes", png=False,
                            marker_rule_config={"enabled": True, "harvest_expected_classes": ["maize"],
                                                "mixed_class_share": 0.45})
    assert result["packages"] == 3 and result["with_series"] and result["with_temporal_support"]
    assert "index_gpkg" not in result      # CSV input has no geometry
    body = evidence.load(tmp_path / "ev" / "B" / "evidence.json")
    rules = {r["rule"]: r for r in body["rules"]}
    assert {"markers.harvest", "markers.mixed_class", "calendar.consistent"} <= set(rules)
    assert rules["markers.mixed_class"]["triggered"]
    # the table's own calendar column wins over the marker table
    assert body["markers"]["calendar_consistent"] is False
    assert any("information only" in n for n in body["notes"])
    assert body["temporal_support"]["observed_dates"] == ["2023-04-01", "2023-05-01"]
    observed = body["data_sources"]["observed_acquisitions"]
    first = next(o for o in observed if o["date"] == "2023-04-01")
    assert first["scenes"] and first["scenes"][0]["tile"] == "34TDR"
    a = evidence.load(tmp_path / "ev" / "A" / "evidence.json")
    svg = evidence.chart_svg(a)
    assert svg.startswith("<svg") and "Harvest" in svg
    index = pd.read_csv(tmp_path / "ev" / "index.csv")
    assert index.chart_png.isna().all()


def test_chart_edge_cases(tmp_path):
    body = {"data_sources": {"interpolated_grid": [], "observed_acquisitions": [], "rejected_acquisitions": []},
            "temporal_support": {"index": "NDVI"}, "markers": {"harvest_doy": 200}}
    assert evidence.chart_svg(body).startswith("<p>No time series")
    grid = [{"date": "2023-05-01", "value": 0.3, "interpolated": False},
            {"date": "2023-05-11", "value": None, "interpolated": True}]
    body = {"data_sources": {"interpolated_grid": grid, "observed_acquisitions": [],
                             "rejected_acquisitions": [{"date": "2023-05-06", "index": "NDVI"}]},
            "temporal_support": {"index": "NDVI", "sos_date": "2024-01-01", "peak_date": None},
            "markers": {"harvest_doy": None}, "outcome": {"traffic_light": "grey", "traffic_light_reasons": ""},
            "parcel_id": "X"}
    svg = evidence.chart_svg(body)
    assert "SOS" not in svg and "rejected" in svg      # event outside the plotted range is not drawn
    pytest.importorskip("matplotlib")
    body["temporal_support"]["eos_date"] = "2023-05-03"
    out = evidence.chart_png(body, tmp_path / "c.png")
    assert out.is_file() and out.stat().st_size > 0


# --------------------------------------------------------------------------- preflight helpers


class _FakeOs:
    """`os` stand-in for preflight only: the named functions raise, everything else is the real module."""

    def __init__(self, **failing):
        self._failing = failing

    def __getattr__(self, name):
        if name in self._failing:
            def fail(*a, **k):
                raise self._failing[name]
            return fail
        return getattr(os, name)


def test_volume_and_ancestor_of_missing_paths(tmp_path, monkeypatch):
    missing = tmp_path / "a" / "b" / "c.tif"
    assert preflight._existing_ancestor(missing) == tmp_path
    assert preflight.free_bytes(missing) > 0
    monkeypatch.setattr(preflight, "os", _FakeOs(stat=OSError("denied")))
    assert preflight._volume(missing) == str(tmp_path)


def test_discover_reports_corrupt_sidecars(tmp_path):
    assert preflight._discover(tmp_path / "none") == ([], [])
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "scene.json").write_text("{corrupt")
    found, bad = preflight._discover(tmp_path)
    assert found == [] and len(bad) == 1 and "not valid JSON" in bad[0]


def test_inventory_rows_skip_unknown_and_report_unreadable(tmp_path):
    inv = tmp_path / "inv.csv"
    inv.write_text("id,collection,datetime,orbit_state\n"
                   "a,SENTINEL-2,2023-05-01T10:00:00Z,\n"
                   "a,SENTINEL-2,2023-05-01T10:00:00Z,\n"     # duplicate id
                   "b,LANDSAT,2023-05-02T10:00:00Z,\n"        # unknown sensor
                   "c,S1_GRD,,\n"                              # no date
                   "d,S1_GRD,2023-05-03T05:00:00Z,descending\n")
    bad = tmp_path / "bad.csv"
    bad.write_text("id,collection,datetime\nx,S2,not-a-date\n")
    rows, problems = preflight._inventory_rows([inv, bad, tmp_path / "missing.csv"])
    assert [(r["sensor"], r["orbit_direction"]) for r in rows] == [("S2", ""), ("S1", "DESCENDING")]
    assert len(problems) == 2


def test_check_disk_skips_and_records_estimate_errors(study, monkeypatch):
    cfg = _cfg(study)
    assert preflight.check_disk(cfg, "train") is None
    off = _cfg(study, resources={"disk_guard": False})
    assert preflight.check_disk(off, "features") is None
    monkeypatch.setattr(preflight, "plan", lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    info = preflight.check_disk(cfg, "composite")
    assert info["estimate_gb"] == 0 and "RuntimeError: boom" in info["estimate_error"]


def test_predict_stage_bytes_uses_feature_stack_and_class_count(study, tmp_path):
    cfg = _cfg(study)
    stack = tmp_path / "stack.tif"
    with rasterio.open(stack, "w", driver="GTiff", width=10, height=5, count=1, dtype="uint8",
                       crs="EPSG:32634", transform=from_origin(0, 50, 10, 10)) as ds:
        ds.write(np.zeros((1, 5, 10), "uint8"))
    cfg["paths"]["features"] = str(stack)
    info = preflight.check_disk(cfg, "predict")
    assert info is not None and info["estimate_gb"] >= 0
    layout = {"pixels": 1000}
    assert preflight.stage_bytes(cfg, layout, "predict", n_classes=3) == 1000 * ((3 + 5) * 4 + 1)
    assert preflight.stage_bytes(cfg, layout, "other") == 0


def test_n_classes_fallbacks(tmp_path):
    cfg = {"paths": {"output": str(tmp_path), "training": str(tmp_path / "none.gpkg")},
           "classes": {"label_field": "class_name"}}
    assert preflight._n_classes(cfg) == 10
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "train_report.json").write_text(json.dumps({"class_counts": {"a": 1, "b": 2, "c": 3}}))
    assert preflight._n_classes(cfg) == 3


def test_grid_and_layers_problems_are_reported(study, tmp_path):
    corrupt = tmp_path / "grid.tif"
    corrupt.write_bytes(b"not a tiff")
    for p in ("parcels", "observations"):
        (study / "reference" / f"{p}.gpkg").unlink()
    (study / "reference" / "training.gpkg").write_text("garbage")
    report = preflight.run(_cfg(study, grid={"reference": str(corrupt), "resolution_m": 20}))
    assert {"grid.readable", "grid.available"} & codes(report, "error") and report["verdict"] == "blocked"
    report = preflight.run(_cfg(study))
    errors, warns = codes(report, "error"), codes(report, "warn")
    assert "training.readable" in errors
    assert {"parcels.present", "observations.present"} <= warns


def test_missing_training_declared_crs_and_resolution(study):
    (study / "reference" / "training.gpkg").unlink()
    report = preflight.run(_cfg(study, grid={"crs": "EPSG:32633", "resolution_m": 20}))
    assert {"training.present", "grid.declared_crs"} <= codes(report, "error")
    assert "grid.resolution" in codes(report, "warn")
    assert report["classes"] == {}


def test_partial_layer_overlap_and_geographic_training(study):
    path = study / "reference" / "parcels.gpkg"
    gdf = gpd.read_file(path)
    far = gdf.iloc[[0]].copy()
    far["geometry"] = [box(0, 0, 10, 10)]
    pd.concat([gdf, far]).pipe(gpd.GeoDataFrame, crs=gdf.crs).to_file(path, driver="GPKG")
    report = preflight.run(_cfg(study))
    assert any(c["code"] == "parcels.overlap" and c["status"] == "warn" for c in report["checks"])


def test_no_windows_no_grid_and_timeseries_without_season(study, tmp_path):
    cfg = _cfg(study, timeseries={"enabled": True})
    cfg["windows"] = {}
    cfg["grid"]["reference"] = None
    cfg["paths"]["s2_scenes"] = cfg["paths"]["s1_scenes"] = str(tmp_path / "empty")
    report = preflight.run(cfg)
    errors = codes(report, "error")
    assert {"windows.defined", "grid.available", "timeseries.season"} <= errors
    assert "resources.estimate" in codes(report, "warn")
    assert report["resources"]["note"].startswith("Grid unknown")
    md = preflight.to_markdown(report)
    assert "BLOCKED" in md and "_none_" in md


def test_unreadable_sidecar_inventory_and_scene_files_in_report(study, tmp_path):
    bad = study / "scenes" / "s2" / "zz_bad"
    bad.mkdir()
    (bad / "scene.json").write_text("[]")
    first = sorted((study / "scenes" / "s2").glob("*/B02.tif"))[0]
    first.unlink()
    report = preflight.run(_cfg(study), inventories=[tmp_path / "missing.csv"])
    errors = codes(report, "error")
    assert {"scenes.sidecar", "inventory.readable", "scenes.files"} <= errors


def test_coverage_with_aoi_mask_and_missing_scl(study, tmp_path):
    cfg = _cfg(study)
    with rasterio.open(cfg["grid"]["reference"]) as src:
        profile = {**src.profile, "dtype": "uint8", "nodata": 0, "count": 1}
        mask = np.zeros((src.height, src.width), "uint8")
        mask[:, : src.width // 2] = 1
    aoi = tmp_path / "aoi.tif"
    with rasterio.open(aoi, "w", **profile) as dst:
        dst.write(mask, 1)
    for scl in list((study / "scenes" / "s2").glob("*/SCL.tif"))[:2]:
        scl.unlink()
    cfg = _cfg(study, grid={"reference": str(aoi)}, compositing={"cloud_probability_max": 50})
    layout = preflight.plan(cfg)
    assert layout["grid"]["aoi_mask"]
    rep = preflight._Report()
    rows = preflight._check_coverage(cfg, layout, rep)
    assert rows and all(0 <= r["aoi_fraction_below_min_valid"] <= 1 for r in rows)
    strict = _cfg(study, grid={"reference": str(aoi)}, compositing={"min_valid_observations": 3},
                  preflight={"max_unobserved_fraction": 0.0})
    rep = preflight._Report()
    preflight._check_coverage(strict, preflight.plan(strict), rep)
    assert {c["status"] for c in rep.checks} <= {"warn", "error", "pass"}


@pytest.mark.parametrize("free_gb, status", [(0.5, "error"), (2.0, "warn"), (5.0, "pass")])
def test_resources_disk_levels_and_unknown_ram(study, monkeypatch, free_gb, status):
    cfg = _cfg(study, resources={"safety_factor": 1.0, "min_free_gb": 0.0})
    est = {"disk": {"composite": {"target": str(study), "expected_gb": 1.0, "uncompressed_gb": 4.0}},
           "peak_ram_gb": {"max": 1e9}}
    monkeypatch.setattr(preflight, "free_bytes", lambda p: free_gb * preflight.GB)
    rep = preflight._Report()
    rows = preflight._check_resources(cfg, est, rep)
    assert rows[0]["need_gb"] == 1.0 and rows[0]["need_uncompressed_gb"] == 4.0
    assert [c["status"] for c in rep.checks if c["code"] == "resources.disk"] == [status]
    ram = [c for c in rep.checks if c["code"] == "resources.ram"]
    assert not ram or ram[0]["status"] == "error"       # 1e9 GB of RAM is never installed
    monkeypatch.setattr(preflight, "os", _FakeOs(sysconf=ValueError("no sysconf")))
    assert preflight.memory_limit_bytes(cgroup_files=[]) == (None, "installed")
    # Unknown installed RAM alone is no longer unknown memory: containers can still expose a cgroup limit.
    monkeypatch.setattr(preflight, "memory_limit_bytes", lambda: (None, "installed"))
    rep = preflight._Report()
    preflight._check_resources(cfg, est, rep)
    assert not any(c["code"] == "resources.ram" for c in rep.checks)


def test_timeseries_layout_counts_series_per_sensor(study):
    cfg = _cfg(study, timeseries={"enabled": True, "start": "2023-04-01", "end": "2023-10-31"})
    out = preflight.timeseries_layout(cfg, [date(2023, 5, 1)], [])
    assert out["scenes"] == 1 and out["series"] == len(cfg["timeseries"]["s2_indices"])
    assert out["features"] % out["series"] == 0 and out["features"] > 0
    assert preflight.timeseries_layout(cfg, [], [])["series"] == 0
    layout = {"pixels": 10, "window_features": 3, "timeseries": {"features": 5, "series": 1}}
    assert preflight.stage_bytes(cfg, layout, "features") == 10 * (3 + 5 + 2 + 3 + 5) * 4


def test_classes_checks_edge_cases(study, tmp_path):
    cfg = _cfg(study)
    layout = preflight.plan(cfg)
    train = gpd.read_file(cfg["paths"]["training"])
    # geographic training layer and no grid -> no metric CRS
    geo = tmp_path / "geo.gpkg"
    train.to_crs("EPSG:4326").to_file(geo, driver="GPKG")
    rep = preflight._Report()
    assert preflight._check_classes({**cfg, "paths": {**cfg["paths"], "training": str(geo)}},
                                    {**layout, "grid": None}, rep) == {}
    assert codes({"checks": rep.checks}, "error") == {"classes.metric_crs"}
    # missing label field
    nolabel = tmp_path / "nolabel.gpkg"
    train.drop(columns=cfg["classes"]["label_field"]).to_file(nolabel, driver="GPKG")
    assert preflight._check_classes({**cfg, "paths": {**cfg["paths"], "training": str(nolabel)}}, layout,
                                    preflight._Report()) == {}
    # unreadable training layer
    junk = tmp_path / "junk.gpkg"
    junk.write_text("junk")
    assert preflight._check_classes({**cfg, "paths": {**cfg["paths"], "training": str(junk)}}, layout,
                                    preflight._Report()) == {}
    # every polygon in one spatial block
    big = _cfg(study, model={"spatial_block_m": 10_000_000})
    rep = preflight._Report()
    out = preflight._check_classes(big, layout, rep)
    assert out["spatial_groups"] == 1 and "classes.spatial_groups" in codes({"checks": rep.checks}, "error")


def test_classes_holdout_and_fold_failures_are_reported(study, monkeypatch):
    cfg = _cfg(study)
    layout = preflight.plan(cfg)
    monkeypatch.setattr(preflight, "_metric", lambda crs: True)
    from geocrop import model
    monkeypatch.setattr(model, "split_final_holdout", lambda *a, **k: (_ for _ in ()).throw(ValueError("tiny")))
    rep = preflight._Report()
    preflight._check_classes(cfg, layout, rep)
    assert "classes.holdout" in codes({"checks": rep.checks}, "error")
    monkeypatch.undo()
    monkeypatch.setattr(model, "selection_samples", lambda *a, **k: (_ for _ in ()).throw(ValueError("folds")))
    rep = preflight._Report()
    preflight._check_classes(cfg, layout, rep)
    assert "classes.folds" in codes({"checks": rep.checks}, "error")


def test_holdout_taking_a_whole_class_is_an_error(study, monkeypatch):
    cfg = _cfg(study, model={"holdout_fraction": 0.2})
    layout = preflight.plan(cfg)
    from geocrop import model
    real = model.split_final_holdout

    def greedy(s, *a, **k):
        dev, test = real(s, *a, **k)
        return dev.take(dev.y != dev.y[0]), test
    monkeypatch.setattr(model, "split_final_holdout", greedy)
    rep = preflight._Report()
    preflight._check_classes(cfg, layout, rep)
    assert "classes.development" in codes({"checks": rep.checks}, "error")
