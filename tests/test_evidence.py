"""Parcel evidence packages: schema, rule trail, observed vs interpolated sources, determinism (synthetic)."""
import json
import re
from html import unescape

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box
from test_temporal_quality import END, START, signatures

from geocrop import cli, contracts, evidence, provenance

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except ImportError:   # optional `plots` extra: packages are built without the PNG
    HAS_MPL = False


def _inputs(tmp_path):
    ids = ["A", "B", "C"]
    gdf = gpd.GeoDataFrame({
        "parcel_id": ids, "declared_class": ["maize", "maize", "soybean"],
        "predicted_class": ["maize", "maize", "maize"], "decision": ["accepted"] * 3, "decision_reasons": [""] * 3,
        "declaration_status": ["agreement", "agreement", "discrepancy"], "confidence": [0.9, 0.8, 0.7],
        "decision_confidence": [0.9, 0.8, 0.7], "margin": [0.8, 0.6, 0.4], "decision_margin": [0.8, 0.6, 0.4],
        "entropy": [0.1, 0.2, 0.3], "n_valid": [50, 50, 50], "p_maize": [0.9, 0.8, 0.7],
        "p_soybean": [0.1, 0.2, 0.3], "prediction_set": ["maize", "maize", "maize"], "set_size": [1, 1, 1],
        "cp_status": ["agreement", "agreement", "discrepancy"], "aoa_fraction": [1.0, 1.0, 0.2],
        "traffic_light": ["green", "green", "red"]},
        geometry=[box(i * 100, 0, i * 100 + 80, 80) for i in range(3)], crs="EPSG:32634")
    parcels = tmp_path / "parcels.gpkg"
    gdf.to_file(parcels, driver="GPKG")
    gdf.drop(columns="geometry").to_csv(tmp_path / "parcels.csv", index=False)
    (tmp_path / "parcels.summary.json").write_text(json.dumps({
        "strategy": "mean_probability", "thresholds": {"min_valid_pixels": 4, "min_confidence": 0.5},
        "conformal_alpha": 0.1, "min_aoa_fraction": 0.5, "calibration_model_sha256": "x" * 64}))
    sig = tmp_path / "signatures.csv"
    signatures().to_csv(sig, index=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.joblib").write_bytes(b"synthetic model bytes")
    manifest = provenance.run_manifest(tmp_path / "run", config={"seed": 1}, inputs=[sig])
    return {"parcels": parcels, "signatures": sig, "model": model, "run_manifest": manifest}


def _build(tmp_path, out, **kw):
    i = _inputs(tmp_path) if not (tmp_path / "parcels.gpkg").exists() else {
        "parcels": tmp_path / "parcels.gpkg", "signatures": tmp_path / "signatures.csv",
        "model": tmp_path / "model", "run_manifest": tmp_path / "run" / "run_manifest.json"}
    return evidence.build(i["parcels"], tmp_path / out, signatures=i["signatures"], start=START, end=END,
                          model=i["model"], run_manifest=i["run_manifest"],
                          temporal_rule_config={"enabled": True}, **kw)


def test_packages_schema_rules_and_sources(tmp_path):
    result = _build(tmp_path, "ev")
    assert result["packages"] == 3 and result["with_series"] and result["with_temporal_support"]
    body = evidence.load(tmp_path / "ev" / "B" / "evidence.json")
    assert body["schema_version"] == contracts.SCHEMA_VERSIONS["evidence"]
    assert evidence.validate(body) == []
    # weak temporal support routes the agreeing parcel B to review with explicit reason codes
    assert body["outcome"]["traffic_light"] == "yellow"
    assert "LONG_GAP_AT_PEAK" in body["outcome"]["traffic_light_reasons"]
    assert body["outcome"]["stored_traffic_light"] == "green" and body["outcome"]["consistent_with_table"] is False
    rules = {r["rule"]: r for r in body["rules"]}
    peak = rules["temporal_support.max_gap_at_peak_days"]
    assert peak["triggered"] and peak["threshold"] == 30 and peak["inputs"]["tq_peak_gap_days"] > 30
    assert rules["decision.min_confidence"]["threshold"] == 0.5 and not rules["decision.min_confidence"]["triggered"]
    assert rules["traffic_light.base"]["outcome"] == "green"
    # observed vs interpolated: cloudy acquisitions around the peak are rejected, the grid there is interpolated
    ds = body["data_sources"]
    assert "2023-07-20" in {o["date"] for o in ds["rejected_acquisitions"]}
    assert "2023-07-20" not in {o["date"] for o in ds["observed_acquisitions"]}
    grid = {g["date"]: g for g in ds["interpolated_grid"]}
    assert grid["2023-07-20"]["interpolated"] and grid["2023-07-20"]["nearest_observation_days"] > 10
    assert not grid["2023-04-16"]["interpolated"]
    # model and run references
    assert len(body["model"]["sha256"]) == 64 and body["model"]["matches_summary"] is False
    assert body["run_manifest"]["schema_version"] == 1 and body["run_manifest"]["sha256"]
    # parcel C: discrepancy outside AOA stays red, low observation count recorded
    c = evidence.load(tmp_path / "ev" / "C" / "evidence.json")
    assert c["outcome"]["traffic_light"] == "red" and "LOW_OBS" in c["outcome"]["traffic_light_reasons"]
    assert {r["rule"]: r for r in c["rules"]}["applicability.min_aoa_fraction"]["inputs"]["aoa_fraction"] == 0.2


def test_index_html_png_and_selection(tmp_path):
    _build(tmp_path, "ev", ids=["B"])
    out = tmp_path / "ev"
    assert sorted(p.name for p in out.iterdir()) == ["B", "evidence_manifest.json", "index.csv", "index.gpkg"]
    index = pd.read_csv(out / "index.csv")
    assert index.parcel_id.tolist() == ["B"] and index.traffic_light.tolist() == ["yellow"]
    assert len(gpd.read_file(out / "index.gpkg")) == 1
    html = unescape((out / "B" / "evidence.html").read_text())
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert "<script" not in html.lower() and "style=" not in html
    assert not re.findall(r"(?:src|href)=\"https?:", html)
    assert "<svg" in html and "observed (clear)" in html
    png = out / "B" / "timeseries.png"
    assert png.read_bytes()[:4] == b"\x89PNG" if HAS_MPL else not png.exists()
    manifest = json.loads((out / "evidence_manifest.json").read_text())
    assert contracts.check("evidence", manifest)["packages"] == 1
    with pytest.raises(ValueError, match="Unknown parcel"):
        _build(tmp_path, "ev2", ids=["nope"])


def test_packages_are_deterministic(tmp_path):
    _build(tmp_path, "one")
    _build(tmp_path, "two")
    for name in ("evidence.json", "evidence.html") + (("timeseries.png",) if HAS_MPL else ()):
        for pid in ("A", "B", "C"):
            assert (tmp_path / "one" / pid / name).read_bytes() == (tmp_path / "two" / pid / name).read_bytes()
    assert (tmp_path / "one" / "index.csv").read_bytes() == (tmp_path / "two" / "index.csv").read_bytes()


def test_csv_table_without_series_and_newer_schema_refused(tmp_path):
    _inputs(tmp_path)
    result = evidence.build(tmp_path / "parcels.csv", tmp_path / "ev", png=True)
    assert result["packages"] == 3 and not result["with_series"]
    body = evidence.load(tmp_path / "ev" / "A" / "evidence.json")
    assert body["outcome"]["traffic_light"] == "green" and body["data_sources"]["interpolated_grid"] == []
    assert not (tmp_path / "ev" / "A" / "timeseries.png").exists()
    body["schema_version"] = 99
    path = tmp_path / "future.json"
    path.write_text(json.dumps(body))
    with pytest.raises(contracts.SchemaVersionError):
        evidence.load(path)
    broken = {k: v for k, v in body.items() if k != "rules"} | {"schema_version": 1}
    assert "missing rules" in evidence.validate(broken)


def test_safe_names_are_unique():
    used = set()
    assert evidence.safe_name("a/b", used) == "a_b"
    assert evidence.safe_name("a?b", used) != "a_b"


def test_cli_temporal_support_and_evidence(tmp_path, capsys):
    i = _inputs(tmp_path)
    cli.main(["temporal-support", "--signatures", str(i["signatures"]), "--start", str(START), "--end", str(END),
              "--out", str(tmp_path / "support.csv"), "--parcels", str(i["parcels"]),
              "--routed", str(tmp_path / "routed.gpkg")])
    result = json.loads(capsys.readouterr().out)
    assert result["parcels"] == 3 and result["reason_counts"]["LONG_GAP_AT_PEAK"] >= 1
    routed = gpd.read_file(tmp_path / "routed.gpkg")
    assert routed.set_index("parcel_id").loc["B", "traffic_light"] == "yellow"
    cli.main(["evidence", "--parcels", str(tmp_path / "routed.gpkg"), "--out", str(tmp_path / "ev"),
              "--temporal-support", str(tmp_path / "support.csv"), "--temporal-rules", '{"enabled": true}',
              "--ids", "A", "B", "--no-png", "--summary", str(tmp_path / "parcels.summary.json")])
    result = json.loads(capsys.readouterr().out)
    assert result["packages"] == 2 and result["traffic_light_counts"] == {"green": 1, "yellow": 1}
    body = evidence.load(tmp_path / "ev" / "B" / "evidence.json")
    assert body["outcome"]["consistent_with_table"] is True
