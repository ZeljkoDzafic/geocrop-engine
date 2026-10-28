"""`geocrop intake` and the tabular parcel/observation adapters on SYNTHETIC data only (examples/intake_example)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from geocrop import adapters, cli, intake
from geocrop import config as cfgmod

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "intake_example"
CONFIG = EXAMPLE / "study.yaml"


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load(CONFIG)


def codes(report, status):
    return {c["code"] for c in report["checks"] if c["status"] == status}


def run_good(cfg, **kw):
    return intake.run(cfg, CONFIG, **kw)


def broken_args():
    b = EXAMPLE / "broken"
    return dict(training=b / "training.geojson", parcels=b / "parcels.geojson",
                observations=b / "observations.geojson", crop_mapping=b / "crop_mapping.csv")


def test_good_example_passes_and_report_is_aggregate_only(cfg):
    report = run_good(cfg)
    assert report["verdict"] == "pass", [c for c in report["checks"] if c["status"] != "pass"]
    assert set(report["layers"]) == {"training", "parcels", "observations"}
    assert report["classes"]["training"]["per_class"]["maize"]["polygons"] == 15
    text = json.dumps(report, default=str)
    assert "coordinates" not in text and "POLYGON" not in text and "POINT" not in text


def test_broken_example_finds_every_planted_fault(cfg):
    report = run_good(cfg, **broken_args())
    assert report["verdict"] == "error"
    errors, warns = codes(report, "error"), codes(report, "warn")
    assert {"audit.training.schema.unique", "audit.training.geometry.valid", "parcels.unmapped_codes",
            "observations.date_format", "training.class_name.target_spelling"} <= errors
    assert {"training.class_name.whitespace", "training.class_name.variants", "training.polygon_size",
            "training.aoi_overlap", "observations.season", "observations.training_overlap",
            "geometry.repairable"} <= warns
    variants = next(c for c in report["checks"] if c["code"] == "training.class_name.variants")
    assert variants["details"]["normalisation_map"] == {"Soybean": "soybean", "soybean ": "soybean"}
    unmapped = next(c for c in report["checks"] if c["code"] == "parcels.unmapped_codes")
    assert unmapped["details"]["unmapped_codes"] == {"TRAVA": 27, "NEPOZNATO": 6}
    repair = next(c for c in report["checks"] if c["code"] == "geometry.repairable")
    assert repair["details"] == {"invalid": 1, "repairable": 1}


def test_offender_listing_is_limited_and_attribute_free(cfg):
    few = run_good(cfg, list_offenders=2, **broken_args())
    assert len(few["offenders"]["parcels.unmapped_codes"]) == 2
    none = run_good(cfg, list_offenders=0, **broken_args())
    assert none["offenders"] == {}
    md = intake.to_markdown(few)
    assert md.startswith("# GeoCrop intake: FAIL") and "Soybean" in md


def test_cli_writes_reports_and_exit_code(tmp_path, capsys):
    out = tmp_path / "rep"
    cli.main(["intake", "--config", str(CONFIG), "--out", str(out)])
    assert json.loads((out / "intake_report.json").read_text(encoding="utf8"))["verdict"] == "pass"
    assert "# GeoCrop intake: PASS" in (out / "intake_report.md").read_text(encoding="utf8")
    b = EXAMPLE / "broken"
    with pytest.raises(SystemExit) as exc:
        cli.main(["intake", "--config", str(CONFIG), "--out", str(out), "--training", str(b / "training.geojson")])
    assert exc.value.code == 2
    capsys.readouterr()


def test_output_must_not_overwrite_input(tmp_path):
    inp = tmp_path / "intake_report.json"
    inp.write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        intake.write({"verdict": "pass"}, tmp_path, [inp])


def test_missing_layer_is_warning_or_error_with_require_all(cfg, tmp_path):
    c = json.loads(json.dumps(cfg))
    c["paths"]["parcels"] = str(tmp_path / "nope.gpkg")
    c["paths"]["observations"] = str(tmp_path / "nope2.gpkg")
    assert "parcels.supplied" in codes(intake.run(c, CONFIG), "warn")
    assert "parcels.supplied" in codes(intake.run(c, CONFIG, require_all=True), "error")


def test_unreadable_and_crsless_layers(cfg, tmp_path):
    bad = tmp_path / "t.geojson"
    bad.write_text("not json")
    assert "training.read" in codes(run_good(cfg, training=bad), "error")
    shp = tmp_path / "x.shp"
    shp.write_bytes(b"\0")
    assert "training.read" in codes(run_good(cfg, training=shp), "error")
    nocrs = tmp_path / "nocrs.geojson"
    nocrs.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"polygon_id": "a", "class_name": "maize"},
         "geometry": {"type": "Polygon", "coordinates": [[[17, 45], [17.01, 45], [17.01, 45.01], [17, 45]]]}}]}))
    rep = run_good(cfg, training=nocrs)
    assert rep["layers"]["training"]["crs"] == "EPSG:4326"          # GeoJSON default CRS
    gdf = gpd.read_file(EXAMPLE / "good" / "training.geojson").set_crs(None, allow_override=True)
    path = tmp_path / "nocrs.gpkg"
    gdf.to_file(path, driver="GPKG")
    assert "audit.training.vector.crs" in codes(run_good(cfg, training=path), "error")


def test_formats_gpkg_flatgeobuf_shapefile(cfg, tmp_path):
    gdf = gpd.read_file(EXAMPLE / "good" / "training.geojson")
    for name, driver in (("t.gpkg", "GPKG"), ("t.shp", "ESRI Shapefile"), ("t.fgb", "FlatGeobuf")):
        path = tmp_path / name
        try:
            gdf.to_file(path, driver=driver)
        except Exception:  # noqa: BLE001 - driver missing in this GDAL build
            continue
        rep = run_good(cfg, training=path)
        assert rep["layers"]["training"]["features"] == 60, name
        assert "training.read" not in codes(rep, "error")


def test_municipality_field_and_boundaries(cfg, tmp_path):
    rep = run_good(cfg)
    pm = next(c for c in rep["checks"] if c["code"] == "training.municipality")["details"]["per_municipality"]
    assert set(pm) == {"Gradiška", "Laktaši", "Srbac"}
    skipped = next(c for c in rep["checks"] if c["code"] == "parcels.municipality")
    assert skipped["details"] == {"skipped": True}
    bounds = gpd.GeoDataFrame({"shapeName": ["Gradiška", "Laktaši", "Srbac"]}, geometry=[
        box(675000, 4983000, 678700, 4993000), box(678700, 4983000, 681300, 4993000),
        box(681300, 4983000, 685000, 4993000)], crs="EPSG:32633")
    path = tmp_path / "municipalities.geojson"
    bounds.to_file(path, driver="GeoJSON")
    rep = run_good(cfg, municipalities=path)
    pm = next(c for c in rep["checks"] if c["code"] == "parcels.municipality")["details"]["per_municipality"]
    assert sum(v["features"] for v in pm.values()) == 108 and set(pm) <= {"Gradiška", "Laktaši", "Srbac"}


def test_training_inside_parcels_is_flagged(cfg, tmp_path):
    parcels = gpd.read_file(EXAMPLE / "good" / "parcels.geojson")
    training = gpd.read_file(EXAMPLE / "good" / "training.geojson")
    training.loc[0, "geometry"] = parcels.geometry.iloc[0].buffer(-20)
    path = tmp_path / "t.geojson"
    training.to_file(path, driver="GeoJSON")
    rep = run_good(cfg, training=path)
    check = next(c for c in rep["checks"] if c["code"] == "training.parcel_overlap")
    assert check["status"] == "warn" and check["details"]["inside"] >= 1


def test_text_issues_diacritics_and_mojibake():
    s = pd.Series(["Šuma", "suma", "SUMA ", "Å¡uma", "Đurđevak", "djurdjevak", "plain"])
    issues = intake.text_issues(s)
    assert issues["mojibake"] == ["Å¡uma"]
    assert issues["whitespace"] == ["SUMA "]
    assert set(issues["variants"]) <= {"suma", "SUMA ", "Šuma"} and len(set(issues["variants"].values())) == 1
    assert intake.fold("Đurđevak") == intake.fold("durdevak") and intake.fold("Ž-t") == "z_t"


def test_season_and_group_checks(cfg, tmp_path):
    obs = gpd.read_file(EXAMPLE / "good" / "observations.geojson")
    obs.loc[0, "group_id"] = ""
    obs.loc[1, "observed_at"] = "2023-11-30"
    obs.loc[2, "class_name"] = "Maize"
    path = tmp_path / "o.geojson"
    obs.to_file(path, driver="GeoJSON")
    rep = run_good(cfg, observations=path)
    assert "observations.groups" in codes(rep, "error") and "observations.season" in codes(rep, "warn")
    assert "observations.class_name.target_spelling" in codes(rep, "error")


# ------------------------------------------------------------------------------------------ tabular input
def test_encoding_and_delimiter_detection():
    tab = EXAMPLE / "tabular"
    assert adapters.detect_encoding(tab / "parcels_cp1250.csv") == "cp1250"
    assert adapters.detect_encoding(tab / "parcels_cp1250.csv", "utf-8") == "utf-8"
    frame, info = adapters.read_table(tab / "parcels_cp1250.csv")
    assert info == {"encoding": "cp1250", "delimiter": ";", "encoding_detected": True}
    assert "opština" in frame.columns and "ŠEĆERNA REPA" in set(frame["declared_crop"])
    assert adapters.detect_delimiter("a,b;c,d\n") == ","
    assert adapters.detect_delimiter('"a;b",c\n') == ","


def test_utf8_bom_and_comma_delimiter(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("parcel_id,declared_crop,wkt\nA1,ŽITARICE,\"POLYGON((0 0,1 0,1 1,0 0))\"\n", encoding="utf-8-sig")
    gdf, info = adapters.read_layer(path, wkt_field="wkt", crs="EPSG:32633")
    assert info["encoding"] == "utf-8-sig" and gdf.declared_crop[0] == "ŽITARICE" and gdf.crs.to_epsg() == 32633


def test_normalize_parcels_from_cp1250_wkt_csv(tmp_path):
    tab = EXAMPLE / "tabular"
    mapping = adapters.read_mapping(tab / "crop_mapping_cp1250.csv")
    assert mapping["ŠEĆERNA REPA"] == "sugar_beet"
    out = tmp_path / "parcels.gpkg"
    result = adapters.normalize_parcels(tab / "parcels_cp1250.csv", out, "parcel_id", "declared_crop", mapping,
                                        wkt_field="wkt", crs="EPSG:32633")
    gdf = gpd.read_file(out)
    assert result["parcels"] == 13 and result["unmapped_codes"] == 0 and result["input"]["encoding"] == "cp1250"
    assert list(gdf.columns) == ["parcel_id", "declared_crop", "geometry"]
    assert "sugar_beet" in set(gdf.declared_crop) and gdf.crs.to_epsg() == 32633


def test_normalize_observations_from_decimal_comma_xy(tmp_path):
    out = tmp_path / "obs.gpkg"
    result = adapters.normalize_observations(EXAMPLE / "tabular" / "observations.csv", out, "klasa", "datum",
                                             "grupa", x_field="x", y_field="y", crs="EPSG:32633")
    gdf = gpd.read_file(out)
    assert result["observations"] == 10 and list(gdf.columns) == ["class_name", "observed_at", "group_id", "geometry"]
    assert gdf.observed_at.str.match(r"^2023-\d\d-\d\d$").all() and gdf.geometry.x.between(676000, 684000).all()
    bad = tmp_path / "bad.csv"
    bad.write_text("klasa,datum,x,y\nmaize,juni 2023,1,2\n")
    with pytest.raises(ValueError, match="neither ISO"):
        adapters.normalize_observations(bad, tmp_path / "x.gpkg", "klasa", "datum", x_field="x", y_field="y",
                                        crs="EPSG:32633")


def test_table_errors(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="no geometry"):
        adapters.read_layer(path)
    with pytest.raises(ValueError, match="--crs"):
        adapters.read_layer(path, x_field="a", y_field="b")
    path.write_text("a;b\n1;x\n")
    with pytest.raises(ValueError, match="non-numeric"):
        adapters.read_layer(path, x_field="a", y_field="b", crs="EPSG:4326")
    path.write_text("lat;lon;c\n45,1;17,2;z\n")
    gdf, _ = adapters.read_layer(path)                 # lat/lon columns are recognised; EPSG:4326 assumed
    assert gdf.crs.to_epsg() == 4326 and (gdf.geometry.x[0], gdf.geometry.y[0]) == (17.2, 45.1)


def test_intake_reads_csv_tables(cfg, tmp_path):
    tab = EXAMPLE / "tabular"
    opts = {"parcels": {"wkt_field": "wkt", "crs": "EPSG:32633"},
            "observations": {"x_field": "x", "y_field": "y", "crs": "EPSG:32633"}}
    c = json.loads(json.dumps(cfg))
    c["parcels"]["declared_field"] = "declared_crop"
    c["validation"].update(label_field="klasa", date_field="datum", group_field="grupa")
    rep = intake.run(c, CONFIG, parcels=tab / "parcels_cp1250.csv", observations=tab / "observations.csv",
                     crop_mapping=tab / "crop_mapping_cp1250.csv", read_options=opts)
    assert rep["layers"]["parcels"]["encoding"] == "cp1250"
    assert "parcels.unmapped_codes" not in codes(rep, "error")
    assert rep["layers"]["observations"]["features"] == 10
    # dd.mm.yyyy. dates are rejected by intake and converted by normalize-observations
    assert any("day.month.year" in x["message"] for x in rep["checks"] if x["code"] == "observations.date_format")
    wrong = intake.run(c, CONFIG, parcels=tab / "parcels_cp1250.csv", crop_mapping=tab / "crop_mapping_cp1250.csv",
                       read_options={"parcels": {"wkt_field": "wkt", "crs": "EPSG:32633", "encoding": "utf-8"}})
    assert "parcels.read" in codes(wrong, "error")


def test_mapping_conflicts_and_uncovered_targets(cfg, tmp_path):
    mapping = tmp_path / "m.csv"
    mapping.write_text("source;class_name\nSOJA;soybean\nSOJA;maize\nKUKURUZ;maize\n", encoding="utf8")
    rep = run_good(cfg, crop_mapping=mapping)
    assert "parcels.mapping_conflict" in codes(rep, "error")
    assert "parcels.mapping_targets" in codes(rep, "warn")


def test_preflight_points_to_intake(cfg):
    from geocrop import preflight
    md = preflight.to_markdown({"verdict": "ready", "summary": {"error": 0, "warn": 0, "pass": 1},
                                "config": "c.yaml", "config_fingerprint": "x", "engine": "0", "generated_at": "t",
                                "intake": "geocrop intake --config c.yaml", "checks": [], "windows": [],
                                "coverage": [], "classes": {}, "assumptions": [], "scope": "s",
                                "resources": {"features": {}, "compression_ratio": 1}})
    assert "geocrop intake --config c.yaml" in md


def test_example_generator_is_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_example", EXAMPLE / "make_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE = tmp_path
    mod.main()
    for rel in ("good/training.geojson", "broken/observations.geojson", "tabular/parcels_cp1250.csv", "aoi.geojson"):
        assert (tmp_path / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel
    shutil.rmtree(tmp_path)
