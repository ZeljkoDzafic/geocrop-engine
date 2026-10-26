"""Edge and error paths of the parcel review backend, its HTTP context and the model registry (synthetic, fast)."""
import json
import os
import sys

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from geocrop import features, registry, review
from geocrop.samples import Samples

CRS = "EPSG:32634"
X0, Y0 = 500000.0, 5000000.0


# --------------------------------------------------------------------------- fixtures


def _layer(path, ids=("A", "B"), **extra):
    gdf = gpd.GeoDataFrame({"parcel_id": list(ids), "traffic_light": ["red", "green"][:len(ids)],
                            "predicted_class": ["maize", "soy"][:len(ids)], **extra},
                           geometry=[box(X0 + 20 + 100 * i, Y0 - 80, X0 + 80 + 100 * i, Y0 - 20)
                                     for i in range(len(ids))], crs=CRS)
    gdf.to_file(path, driver="GeoJSON")
    return path


def _raster(path, value, nodata=None, width=30, height=12):
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0, Y0, 10, 10), nodata=nodata) as ds:
        ds.write(np.full((1, height, width), value, "float32"))


def _imagery(folder, s2=True, s1=False, nodata=None):
    folder.mkdir(parents=True, exist_ok=True)
    if s2:
        for band, v in (("B02", 0.05), ("B03", 0.08), ("B04", 0.05), ("B08", 0.4)):
            _raster(folder / f"{band}.tif", v, nodata)
    if s1:
        _raster(folder / "VV.tif", 0.05)
        _raster(folder / "VH.tif", 0.01)
    return folder


# --------------------------------------------------------------------------- review: small helpers


def test_cache_evicts_oldest_and_rereads_changed_files(tmp_path):
    cache, calls = review._Cache(1), []
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("1")
    b.write_text("2")

    def load(p):
        calls.append(p.name)
        return p.read_text()
    assert cache.get(a, load) == "1" and cache.get(a, load) == "1"
    assert cache.get(b, load) == "2" and len(cache.data) == 1
    assert cache.get(a, load) == "1" and calls == ["a.txt", "b.txt", "a.txt"]


def test_columns_of_unreadable_and_parquet_files(tmp_path):
    bad = tmp_path / "bad.gpkg"
    bad.write_text("not a geopackage")
    assert review._columns(bad) == ()
    pytest.importorskip("pyarrow")
    pd.DataFrame({"parcel_id": ["1"], "mean": [0.5]}).to_parquet(tmp_path / "t.parquet")
    assert review._columns(tmp_path / "t.parquet") == ("parcel_id", "mean")


def test_scene_labels_from_sidecars_and_corrupt_metadata(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene.json").write_text("{broken")
    assert review._scene_label(scene) == ("scene", None)
    window = tmp_path / "P1"
    window.mkdir()
    (window / "composite.json").write_text(json.dumps({"start": "2023-04-01", "end": "2023-04-11"}))
    assert review._scene_label(window) == ("window P1", "2023-04-06")
    (window / "composite.json").write_text(json.dumps({"start": "2023-04-01"}))
    assert review._scene_label(window) == ("window P1", None)
    (window / "composite.json").write_text(json.dumps({"start": "bad", "end": "2023-04-11"}))
    assert review._scene_label(window) == ("window P1", None)


def test_model_sha256_skips_corrupt_and_malformed_manifests(tmp_path):
    corrupt, short, good = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (corrupt, short, good):
        d.mkdir()
    (corrupt / "prediction_manifest.json").write_text("{nope")
    (short / "parcels.summary.json").write_text(json.dumps({"calibration_model_sha256": "abc"}))
    (good / "train_report.json").write_text(json.dumps({"model_sha256": "f" * 64}))
    files = [corrupt / "prediction_manifest.json", short / "parcels.summary.json"]
    assert review.model_sha256(files) == ""
    assert review.model_sha256(files + [good / "train_report.json"]) == "f" * 64


def test_load_layer_requires_unique_ids(tmp_path):
    gpd.GeoDataFrame({"name": ["x"]}, geometry=[Point(0, 0)], crs=CRS).to_file(tmp_path / "a.geojson")
    with pytest.raises(ValueError, match="no parcel id column"):
        review._load_layer(tmp_path / "a.geojson")
    _layer(tmp_path / "b.geojson", ids=("A", "A"))
    with pytest.raises(ValueError, match="not unique"):
        review._load_layer(tmp_path / "b.geojson")


def test_plain_reasons_and_summary_rows():
    assert review._plain(pd.NA) is None and review._plain(np.int32(3)) == 3
    assert review._plain([1, 2]) == "[1, 2]"            # pd.isna on a list is ambiguous -> text
    row = pd.Series({"traffic_light_reasons": "a;b", "decision_reasons": "b;c", "declaration_status": "discrepancy",
                     "traffic_light": "red", "confidence": np.float64(0.5)})
    codes = [r["code"] for r in review.reasons(row)]
    assert codes == ["discrepancy", "a", "b", "c"]       # duplicate 'b' listed once, declaration first
    item = review.summary_row("A", row, {"A": {"decision": "agree", "label": "x", "timestamp_utc": "t", "o": 1}})
    assert item["review"] == {"decision": "agree", "label": "x", "timestamp_utc": "t"}
    assert review.summary_row("B", row, {})["review"] is None


def test_classes_facets_and_query_on_a_sparse_layer(tmp_path):
    gdf = review._load_layer(_layer(tmp_path / "l.geojson", ids=("A", "B"), entropy=[0.2, np.nan]))
    assert review.classes(gdf) == ["maize", "soy"]
    f = review._facets(gdf)
    assert f["declared_class"] == {} and f["decision"] == {} and f["uncertainty"]["ambiguous_set"] == 0
    res = review.query(gdf, sort="entropy")
    assert [i["parcel_id"] for i in res["items"]] == ["A", "B"]        # NaN last
    res = review.query(gdf, sort="area_m2", descending=True)          # sort key absent from the layer
    assert sorted(res["ids"]) == ["A", "B"]
    assert review.query(gdf, declared_class=["x"], max_confidence=0.5, min_set_size=2)["total"] == 0
    assert review.query(gdf, sort="parcel_id", descending=True)["ids"] == ["B", "A"]
    with pytest.raises(ValueError, match="sort must be"):
        review.query(gdf, sort="nope")


def test_detail_markers_and_geometry(tmp_path):
    gdf = review._load_layer(_layer(tmp_path / "l.geojson"))
    with pytest.raises(KeyError):
        review.detail(gdf, "Z")
    markers = tmp_path / "markers.csv"
    pd.DataFrame({"parcel_id": ["B"], "harvest_doy": [210.0]}).to_csv(markers, index=False)
    broken = tmp_path / "broken.csv"
    broken.write_text("x\n1\n")                  # no id column -> skipped
    body = review.detail(gdf, "A", [broken, markers])
    assert body["markers"] == {} and body["conformal_set"] == []
    assert review.detail(gdf, "B", [markers])["markers"] == {"harvest_doy": 210.0}
    geo = review.geometry_wgs84(gdf.to_crs("EPSG:4326"), "A")
    (s, w), (n, e) = geo["bounds"]
    assert s < n and w < e and -90 <= s <= 90


def test_series_from_signatures_names_orbits_and_skips_bad_tables(tmp_path):
    sig = tmp_path / "sig.csv"
    pd.DataFrame({"parcel_id": ["A", "A", "A", "B"], "date": ["2023-05-01", "2023-05-02", "2023-05-03", "2023-05-01"],
                  "index": ["NDVI", "VV_dB", "VH_dB", "NDVI"], "mean": [0.5, -12.0, -18.0, 0.4],
                  "n_valid": [10, 10, 0, 1], "orbit_direction": [None, "ASCENDING", "DESCENDING", None]}
                 ).to_csv(sig, index=False)
    out = review.series_from_signatures([tmp_path / "missing.csv", sig], "A")
    names = {p["name"]: [s["name"] for s in p["series"]] for p in out["panels"]}
    assert names == {"Sentinel-2": ["NDVI"], "Sentinel-1": ["VV_dB asc", "VH_dB des"]}
    s1 = next(p for p in out["panels"] if p["name"] == "Sentinel-1")
    assert [s["points"][0]["observed"] for s in s1["series"]] == [True, False]
    assert review.series_from_signatures([sig], "Z")["panels"] == []


def test_series_and_chips_from_imagery_edge_cases(tmp_path):
    gdf = review._load_layer(_layer(tmp_path / "l.geojson"))
    _imagery(tmp_path / "s2", nodata=-1.0)
    _imagery(tmp_path / "s1", s2=False, s1=True)
    folders = review.imagery(sorted(tmp_path.rglob("*.tif")))
    assert {(m["s2"], m["s1"]) for m in folders} == {(True, False), (False, True)}
    out = review.series_from_imagery(gdf, "A", folders)
    panels = {p["name"]: p for p in out["panels"]}
    ndvi = panels["Sentinel-2"]["series"][0]["points"][0]["mean"]
    assert ndvi == pytest.approx((0.4 - 0.05) / (0.4 + 0.05), abs=1e-6)
    vv = next(s for s in panels["Sentinel-1"]["series"] if s["name"] == "VV_dB")["points"][0]["mean"]
    assert vv == pytest.approx(10 * np.log10(0.05), abs=1e-4)
    # a degenerate (point) parcel has an empty window and is skipped; chips need S2 and a sane window
    pts = gdf.copy()
    pts["geometry"] = [Point(X0 + 50, Y0 - 50), Point(X0 + 150, Y0 - 50)]
    assert review.series_from_imagery(pts, "A", folders)["panels"] == []
    huge = gdf.copy()
    huge["geometry"] = [box(X0, Y0 - 60000, X0 + 60000, Y0), box(X0, Y0 - 10, X0 + 10, Y0)]
    assert review.chips(huge, "A", folders) == []
    with pytest.raises(ValueError, match="kind must be"):
        review.chips(gdf, "A", folders, kind="false-colour")
    chips = review.chips(gdf, "A", folders, kind="ndvi", size=4)
    assert len(chips) == 1 and chips[0]["rgba"].shape == (4, 4, 4) and 0 < chips[0]["valid_fraction"] <= 1
    assert review._shrink(np.zeros((3, 3)), 8).shape == (3, 3)
    blank = tmp_path / "blank"
    blank.mkdir()
    _raster(blank / "B04.tif", -1.0, nodata=-1.0)
    data = review._read(blank, "B04", rasterio.windows.Window(0, 0, 2, 2))
    assert np.isnan(data).all()


def test_evidence_discovery_and_series(tmp_path):
    def package(folder, body):
        folder.mkdir()
        (folder / "evidence.json").write_text(json.dumps(body) if isinstance(body, dict) else body)
        return folder / "evidence.json"
    files = [package(tmp_path / "a", {"schema": review.EVIDENCE_SCHEMA, "parcel_id": "A"}),
             package(tmp_path / "a2", {"schema": review.EVIDENCE_SCHEMA, "parcel_id": "A"}),
             package(tmp_path / "b", {"schema": review.EVIDENCE_SCHEMA, "parcel_id": "B"}),
             package(tmp_path / "c", {"schema": "other", "parcel_id": "C"}),
             package(tmp_path / "d", "{corrupt")]
    found = review.evidence_packages(files)
    assert found == {"A": files[0], "B": files[2]}
    assert len(review.evidence_packages(files, limit=1)) == 1
    body = {"data_sources": {"observed_acquisitions": [{"index": "VV_dB", "date": "2023-05-02", "mean": -11.0}],
                             "rejected_acquisitions": [{"index": "VV_dB", "date": "2023-05-01"}]}}
    path = package(tmp_path / "e", body)
    out = review.series_from_evidence(path)
    pts = out["panels"][0]["series"][0]["points"]
    assert out["panels"][0]["name"] == "Sentinel-1" and [p["observed"] for p in pts] == [False, True]


def test_validate_decision_errors(tmp_path):
    gdf = review._load_layer(_layer(tmp_path / "l.geojson"))
    gdf.loc["B", "predicted_class"] = None
    ok = {"parcel_id": "A", "decision": "disagree", "purpose": "training", "label": "soy"}
    assert review.validate_decision(ok, gdf, ["maize", "soy"])["new_class"] is False
    for change, message in (({"decision": "maybe"}, "decision must be"), ({"purpose": "fun"}, "purpose must be"),
                            ({"label": "maize"}, "equals the prediction"), ({"confidence": 2}, "between 0 and 1"),
                            ({"parcel_id": "B", "decision": "agree"}, "no predicted class")):
        with pytest.raises(ValueError, match=message):
            review.validate_decision({**ok, **change}, gdf, [])
    row = review.validate_decision({**ok, "decision": "unsure", "confidence": 0.25, "analyst": "=cmd"}, gdf, [])
    assert row["label"] == "" and row["confidence"] == "0.25" and row["analyst"] == "'=cmd"


# --------------------------------------------------------------------------- web_review context


class _Manager:
    def __init__(self, ws):
        self.workspace = ws.resolve()
        self.state_dir = (ws / ".state").resolve()
        self.state_dir.mkdir(exist_ok=True)
        self.jobs = {}

    def add(self, job_id, outputs, status="succeeded"):
        self.jobs[job_id] = {"id": job_id, "status": status, "outputs": [str(o) for o in outputs]}

    def get(self, job_id):
        return self.jobs[job_id]

    def output_files(self, job_id):
        files = []
        for raw in self.jobs[job_id]["outputs"]:
            p = os.path.realpath(raw)
            files += [p] if os.path.isfile(p) else [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
        from pathlib import Path
        return sorted(Path(f).resolve() for f in files)


@pytest.fixture
def web(tmp_path):
    pytest.importorskip("fastapi")
    from geocrop import web_review
    ws = tmp_path / "ws"
    out = ws / "out"
    out.mkdir(parents=True)
    _layer(out / "parcels.geojson")
    sig = ws / "other" / "sig.csv"
    sig.parent.mkdir()
    pd.DataFrame({"parcel_id": ["A"], "date": ["2023-05-01"], "index": ["NDVI"], "mean": [0.5]}).to_csv(sig, index=False)
    m = _Manager(ws)
    m.add("job", [out])
    m.add("other", [ws / "other"])
    m.add("running", [ws / "other"], status="running")
    return web_review, m, out


def test_context_refuses_unfinished_jobs_unknown_layers_and_evidence(web):
    wr, m, out = web
    from fastapi import HTTPException
    for args, status in ((("running",), 409), (("job", "nope.geojson"), 404), (("job", None, "missing"), 404),
                         (("job", None, "running"), 409)):
        with pytest.raises(HTTPException) as exc:
            wr.Context(m, *args)
        assert exc.value.status_code == status
    ctx = wr.Context(m, "job", "out/parcels.geojson", "job,other")
    assert ctx.evidence_jobs == ["other"] and [p.name for p in ctx.signatures] == ["sig.csv"]
    assert ctx.evidence() == {} and ctx.review_dir == out.resolve() / "review"
    assert ctx.latest() == {}


def test_review_dir_for_file_outputs_and_writable_guards(web, tmp_path):
    wr, m, out = web
    from fastapi import HTTPException
    m.add("file", [out / "parcels.geojson"])
    ctx = wr.Context(m, "file")
    assert ctx.review_dir == (out / "parcels_review").resolve()
    (ctx.review_dir / "decisions.csv").mkdir(parents=True)
    with pytest.raises(HTTPException) as exc:
        ctx.writable("decisions.csv")
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        ctx.writable("../escape.csv")
    assert exc.value.status_code == 409
    # an output that does not contain the layer gives no review folder; layer-less jobs have nothing to review
    m.add("split", [m.workspace / "other", out / "parcels.geojson"])
    assert wr.Context(m, "split").review_dir is not None
    m.add("nolayer", [m.workspace / "other"])
    ctx = wr.Context(m, "nolayer")
    assert ctx.layer is None and ctx.latest() == {}
    for call in (ctx.require_layer, lambda: ctx.writable("decisions.csv")):
        with pytest.raises(HTTPException):
            call()


def test_symlinked_review_folder_is_not_writable(web):
    wr, m, out = web
    target = m.workspace / "elsewhere"
    target.mkdir()
    os.symlink(target, out / "review")
    assert wr.Context(m, "job").review_dir is None


def test_cached_values_survive_corrupt_cache_and_unwritable_state(web):
    wr, m, out = web
    ctx = wr.Context(m, "job")
    ctx.signatures = [m.workspace / "gone.csv"]           # missing source file -> zero stamp, no crash
    calls = []
    assert wr._cached(ctx, ("k",), lambda: calls.append(1) or {"v": 1}) == {"v": 1}
    assert wr._cached(ctx, ("k",), lambda: calls.append(1) or {"v": 2}) == {"v": 1} and calls == [1]
    for f in (m.state_dir / "review-cache" / "job").glob("*.json"):
        f.write_text("{corrupt")
    assert wr._cached(ctx, ("k",), lambda: {"v": 3}) == {"v": 3}
    blocked = m.state_dir / "review-cache"
    for f in sorted(blocked.rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    blocked.rmdir()
    blocked.write_text("a file where the cache folder should be")
    assert wr._cached(ctx, ("k",), lambda: {"v": 4}) == {"v": 4}


def test_routes_report_unwritten_files_and_broken_evidence(web):
    wr, m, out = web
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    (out / "pkg").mkdir()
    (out / "pkg" / "evidence.json").write_text(json.dumps({"schema": review.EVIDENCE_SCHEMA, "parcel_id": "A",
                                                           "temporal_support": "not-a-dict"}))
    app = FastAPI()
    wr.register_review_routes(app, lambda request: m, lambda: None)
    client = TestClient(app)
    assert client.get("/api/jobs/job/review/file", params={"name": "decisions.csv"}).status_code == 404
    body = client.get("/api/jobs/job/review/parcel", params={"id": "A"}).json()
    assert body["evidence"] is None and body["parcel_id"] == "A"


# --------------------------------------------------------------------------- registry


def _payload(path, features_=("f1", "f2"), classes=("a", "b"), **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": list(features_), "classes": list(classes), "params": {"algorithm": "rf"}, "seed": 3,
                 **extra}, path)
    return path


def _stack(path, names=("f1", "f2"), crs=CRS, res=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=len(names), dtype="float32", crs=crs,
                       transform=from_origin(X0, Y0, res, res)) as ds:
        for i, n in enumerate(names, 1):
            ds.write(np.zeros((4, 4), "float32"), i)
            ds.set_band_description(i, n)
    return path


def test_root_recording_and_plain(monkeypatch, tmp_path):
    monkeypatch.setenv(registry.ENV, str(tmp_path / "r"))
    assert registry.default_root() == tmp_path / "r" and registry.recording_enabled()
    monkeypatch.setenv(registry.ENV, "off")
    assert registry.default_root().name == "registry" and not registry.recording_enabled()
    assert registry.default_root(tmp_path) == tmp_path
    assert registry._plain({"a": np.array([1.5, np.nan]), "b": {1, }, "c": object.__class__}) == {
        "a": [1.5, None], "b": [1], "c": "<class 'type'>"}
    assert registry._plain(np.float32("inf")) is None


def test_register_error_paths_and_companion_checks(tmp_path, monkeypatch):
    reg = tmp_path / "reg"
    with pytest.raises(registry.RegistryError, match="not found"):
        registry.register(tmp_path / "none.joblib", "m", root=reg)
    model = _payload(tmp_path / "src" / "model.joblib")
    (model.parent / "train_report.json").write_text(json.dumps({"model_sha256": "0" * 64}))
    (model.parent / "conformal.json").write_text(json.dumps({"level": 0.9}))
    with pytest.raises(registry.RegistryError, match="lacks 1 model features"):
        registry.register(model, "m", root=reg, features=_stack(tmp_path / "s.tif", names=("f1", "x")))
    s = Samples(np.zeros((2, 1), "float32"), np.array(["a", "b"]), np.array(["1", "2"]), np.zeros(2), np.zeros(2),
                ["other"], CRS, year=np.array(["2023", "2023"]))
    s.save(model.parent / "samples.npz")
    monkeypatch.setattr(registry.os, "link", lambda a, b: (_ for _ in ()).throw(OSError("cross-device")))
    e = registry.register(model, "m", root=reg, link=True)
    assert e["artifact"]["storage"] == "copy"
    assert any("model_sha256 mismatch" in n for n in e["notes"]) and any("hard link" in n for n in e["notes"])
    assert "train_report.json" not in e["files"] and "conformal.json" in e["files"]
    assert e["fingerprints"] == {"verified": False, "reason": "samples.npz lacks model features",
                                 "samples_file": "samples.npz"}
    assert e["validation"]["conformal"]["level"] == 0.9 and e["validation"]["cv"] is None
    assert e["taxonomy"]["hcat"] is None and e["training_data"] == {}
    monkeypatch.undo()
    other = _payload(tmp_path / "src2" / "model.joblib", seed=4)
    e2 = registry.register(other, "m", root=reg, link=True)
    assert e2["version"] == 2 and e2["artifact"]["storage"] == "hardlink"


def test_register_detects_a_model_changing_during_copy(tmp_path, monkeypatch):
    model = _payload(tmp_path / "src" / "model.joblib")
    monkeypatch.setattr(registry.shutil, "copyfile", lambda a, b: open(b, "wb").write(b"changed"))
    with pytest.raises(registry.RegistryError, match="changed while"):
        registry.register(model, "m", root=tmp_path / "reg")


def test_validation_licences_and_samples_summaries():
    v = registry._validation({"transfer_validation": {"strategy": "region", "out_of_domain": {"macro_f1": 0.4},
                                                      "transfer_gap": {"macro_f1": 0.1}, "worst_group": "R2"}}, None)
    assert v["transfer"]["out_of_domain_macro_f1"] == 0.4 and v["transfer"]["gap_macro_f1"] == 0.1
    assert v["transfer"]["out_of_domain_kappa"] is None and v["conformal"] is None
    assert registry._metrics({"macro_f1": 0.5, "bootstrap": {"macro_f1_ci": [0.4, 0.6]}}) == {
        "macro_f1": 0.5, "macro_f1_ci": [0.4, 0.6]}
    lic = registry._licences([{"group": "emb", "sensor": "AlphaEarth"}, {"group": "s2", "sensor": "S2"}], None)
    assert [x["licence"] for x in lic] == ["CC-BY-4.0"]
    assert registry._model_fingerprints(None, {}, None) == {"verified": False,
                                                           "reason": "no train_report.json for this model"}
    s = Samples(np.zeros((3, 1), "float32"), np.array(["a", "b", "a"]), np.array(["1", "2", "3"]), np.zeros(3),
                np.zeros(3), ["f"], CRS, region=np.array(["R1", "R1", "R2"]))
    assert registry._read_samples_summary(s) == {"regions": {"R1": 2, "R2": 1}}


def test_get_refs_and_pretrained_entries(tmp_path, monkeypatch):
    reg = tmp_path / "reg"
    with pytest.raises(registry.RegistryError, match="Invalid registry reference"):
        registry.parse_ref("registry:@1")
    with pytest.raises(registry.RegistryError, match="Unknown pretrained"):
        registry.get("ftw/none", reg)
    first = registry.pretrained_entries()[0]
    assert registry.get(first["name"], reg) == first
    with pytest.raises(registry.RegistryError, match="No model"):
        registry.get("m", reg)
    registry.register(_payload(tmp_path / "src" / "model.joblib"), "m", root=reg)
    for ref in ("m@7", "m@beta"):
        with pytest.raises(registry.RegistryError, match="unknown version or stage"):
            registry.get(ref, reg)
    with pytest.raises(registry.RegistryError, match="Stage must be"):
        registry.promote("m@1", "gold", reg)
    with pytest.raises(registry.RegistryError, match="no registry stages"):
        registry.promote(first["name"], "staging", reg)
    with pytest.raises(registry.RegistryError, match="not a classification model"):
        registry.resolve_for_predict(first["name"], tmp_path / "s.tif", reg)
    with pytest.raises(registry.RegistryError, match="no validation metrics"):
        registry.compare("m@1", first["name"], reg)
    monkeypatch.setitem(sys.modules, "geocrop.embedding_sources", None)    # Presto adapter unavailable
    assert all(not e["name"].startswith("presto") for e in registry.pretrained_entries())


def test_verify_detects_tampering_and_pretrained_cache_state(tmp_path):
    reg = tmp_path / "reg"
    model = _payload(tmp_path / "src" / "model.joblib")
    (model.parent / "aoa.json").write_text("{}")
    (model.parent / "feature_importance.csv").write_text("f,i\n")
    registry.register(model, "m", root=reg)
    folder = reg / "models" / "m" / "1"
    for f in folder.iterdir():
        f.chmod(0o644)
    (folder / "aoa.json").write_text('{"x": 1}')
    (folder / "feature_importance.csv").unlink()
    check = registry.verify_entry(registry.get("m@1", reg))
    assert not check["ok"] and set(check["problems"]) == {"aoa.json sha256 mismatch", "feature_importance.csv missing"}
    with pytest.raises(registry.RegistryError, match="failed verification"):
        registry.promote("m@1", "staging", reg)
    assert registry.promote("m@1", "archived", reg)["stage"] == "archived"   # archiving a broken version is allowed
    with pytest.raises(registry.RegistryError, match="failed verification"):
        registry.resolve_for_predict("m@latest", tmp_path / "s.tif", reg)
    (folder / "model.joblib").unlink()
    assert "model artefact missing" in registry.verify_entry(registry.get("m@1", reg))["problems"]
    cached = tmp_path / "ckpt.pt"
    entry = {"kind": "pretrained", "name": "x", "version": "1", "cache_path": str(cached),
             "artifact": {"sha256": "0" * 64}}
    assert registry.verify_entry(entry)["status"] == "not cached"
    cached.write_bytes(b"weights")
    res = registry.verify_entry(entry)
    assert res["status"] == "cached" and not res["ok"] and res["problems"]


def test_input_contract_mismatches(tmp_path):
    reg = tmp_path / "reg"
    stack = _stack(tmp_path / "train" / "stack.tif")
    features.write_manifest(stack, {"features": [{"name": "f1", "group": "s2", "sensor": "S2", "period": "P1"},
                                                 {"name": "f2", "group": "s2", "sensor": "S2", "period": "P2"}]})
    entry = registry.register(_payload(tmp_path / "src" / "model.joblib"), "m", root=reg, features=stack)
    assert entry["input_contract"]["windows"] == ["P1", "P2"] and entry["feature_manifest"]["groups"] == {"s2": 2}
    assert registry.check_input_contract(entry, stack)["checked"] == ["features", "crs", "resolution",
                                                                      "feature_manifest"]
    other = _stack(tmp_path / "pred" / "stack.tif", names=("f1", "f3"), crs="EPSG:32633", res=20)
    features.write_manifest(other, {"features": [{"name": "f1", "group": "s2", "sensor": "S2", "period": "P9"}]})
    with pytest.raises(registry.ContractError) as exc:
        registry.check_input_contract(entry, other)
    text = str(exc.value)
    assert "missing 1 model features" in text and "CRS" in text and "resolution" in text and "group/sensor" in text


def _record(reg, task="t", kind="benchmark", rows=None, split="s" * 64):
    return registry.record_run(kind, reg, rows or [{"candidate": "rf", "macro_f1": 0.5}], task=task,
                               split={"split": split}, dataset="d" * 64, root=reg)


def test_benchmark_records_lookup_and_leaderboard_filters(tmp_path):
    reg = tmp_path / "reg"
    a = registry._load(_record(reg, rows=[{"candidate": "rf", "macro_f1": 0.5, "macro_f1_ci": [0.4, 0.6]},
                                          {"candidate": "bad", "status": "failed", "macro_f1": 0.9},
                                          {"candidate": "none", "macro_f1": None}]))
    _record(reg, kind="ablation", task="u")
    assert [e["kind"] for e in registry.list_benchmarks(reg, kind="ablation")] == ["ablation"]
    assert registry.get_benchmark(a["id"], reg)["id"] == a["id"]
    with pytest.raises(registry.RegistryError, match="No benchmark run"):
        registry.get_benchmark("zzz", reg)
    with pytest.raises(registry.RegistryError, match="Ambiguous"):
        registry.get_benchmark("2", reg)        # every id starts with the year
    with pytest.raises(registry.RegistryError, match="with split fingerprint"):
        registry.leaderboard("t", reg, split="0000")
    with pytest.raises(registry.RegistryError, match="No benchmark runs"):
        registry.leaderboard("missing", reg)
    groups = registry.leaderboard("t", reg)
    assert [r["candidate"] for r in groups[0]["rows"]] == ["rf"]
    md = registry.leaderboard_markdown(groups)
    assert "[0.400, 0.600]" in md
    assert registry._rows([{"candidate": "x", "ci": [0.1, 0.2], "obj": {"a": 1}, "trio": [1, 2, 3]}], "candidate") == [
        {"candidate": "x", "ci": [0.1, 0.2]}]
    assert registry._fmt(None) == "" and registry._fmt("x") == "x"


def test_record_public_rows_and_runtime(tmp_path):
    reg = tmp_path / "reg"
    report = {"protocol": {"id": "p", "version": 1, "sha256": "h"}, "samples": {"fingerprint": "f" * 64},
              "started_utc": "bad", "finished_utc": "2023-01-01T00:00:00+00:00",
              "models": {"rf": {"status": "ok", "wall_s": 1.0, "parcel": {"macro_f1": 0.7}, "bootstrap": {},
                                "pixel": {"macro_f1": 0.6}, "conformal": {"coverage": 0.9},
                                "review": {"review_share": 0.1}, "time": {"fit_s": 0.5}},
                         "tl": {"status": "skipped", "wall_s": None}}}
    path = registry.record_public(report, tmp_path, roles="train:a", protocol_split={}, evaluation={}, root=reg)
    body = registry._load(path)
    assert body["runtime_seconds"] is None and body["task"] == "public/p"
    assert [r["candidate"] for r in body["rows"]] == ["rf", "tl"] and body["rows"][0]["macro_f1"] == 0.7


def test_compare_markdown_lists_refused_sections():
    md = registry.compare_markdown({"a": "m@1", "b": "m@2", "sections": {
        "cv": {"fingerprint": "f" * 64, "metrics": {"macro_f1": {"a": 0.5, "b": 0.6, "delta": 0.1}}}},
        "refused": {"holdout": "present in only one model"}})
    assert "| macro_f1 | 0.5000 | 0.6000 | +0.1000 |" in md and "* holdout: not compared" in md
