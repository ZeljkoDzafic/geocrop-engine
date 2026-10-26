"""Analyst parcel review (gap G7): queue, evidence, chips, append-only decisions and label export."""
import base64
import csv
import os
import time

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from geocrop import review  # noqa: E402
from geocrop.active_learning import LABEL_TEMPLATE_COLUMNS, ingest, read_labels  # noqa: E402
from geocrop.web import create_app  # noqa: E402

TOKEN = "review-token-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def wait(client, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(.1)
    pytest.fail(f"job {job_id} did not finish")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    ws = tmp_path_factory.mktemp("reviewws")
    with TestClient(create_app(ws, TOKEN), base_url="http://127.0.0.1") as client:
        demo = client.post("/api/jobs", json={"argv": ["demo", "--root", "outputs/demo", "--quick"]}, headers=AUTH).json()
        assert wait(client, demo["id"])["status"] == "succeeded"
        other = client.post("/api/jobs", json={"argv": ["indices"]}, headers=AUTH).json()
        assert wait(client, other["id"])["status"] == "succeeded"
        yield {"client": client, "ws": ws, "job": demo["id"], "other": other["id"], "root": ws / "outputs" / "demo"}


def get(env, path, **params):
    return env["client"].get(f"/api/jobs/{env['job']}/review{path}", params=params, headers=AUTH)


def post(env, path, body):
    return env["client"].post(f"/api/jobs/{env['job']}/review{path}", json=body, headers=AUTH)


def test_summary_facets_and_empty_state(env):
    s = get(env, "").json()
    assert s["available"] and s["layer"] == "outputs/demo/parcels/parcels.gpkg" and s["parcels"] == 144
    assert {"soybean", "maize"} <= set(s["classes"]) and s["facets"]["traffic_light"]["red"] >= 1
    assert s["sources"]["imagery"] == 4 and s["sources"]["signatures"] == []
    assert len(s["model_sha256"]) == 64 and s["review_dir"] == "outputs/demo/review"
    # a job without parcels (like a quicklook) explains what review needs instead of failing
    empty = env["client"].get(f"/api/jobs/{env['other']}/review", headers=AUTH).json()
    assert empty["available"] is False and "parcel" in empty["reason"]
    assert env["client"].get(f"/api/jobs/{env['other']}/review/parcels", headers=AUTH).status_code == 404


def test_queue_filters_sorting_and_paging(env):
    all_ = get(env, "/parcels", limit=500, review="all").json()
    assert all_["total"] == 144 and len(all_["items"]) == 144
    lights = [i["traffic_light"] for i in all_["items"]]
    assert lights == sorted(lights, key=lambda v: review.TRAFFIC_ORDER[v])          # priority: red first
    red = get(env, "/parcels", traffic_light="red", review="all").json()
    assert red["total"] == 11 and {i["traffic_light"] for i in red["items"]} == {"red"}
    both = get(env, "/parcels", traffic_light="red,yellow", review="all").json()
    assert both["total"] == 16
    soy = get(env, "/parcels", declared_class="soybean", review="all").json()
    assert soy["total"] and all(i["declared_class"] == "soybean" for i in soy["items"])
    disc = get(env, "/parcels", reason="discrepancy", review="all").json()
    assert disc["total"] == 11
    low = get(env, "/parcels", max_confidence=0.9, review="all").json()
    assert all(i["confidence"] < 0.9 for i in low["items"])
    assert get(env, "/parcels", min_set_size=2, review="all").json()["total"] == sum(
        1 for i in all_["items"] if (i["set_size"] or 0) >= 2)
    conf = get(env, "/parcels", sort="confidence", order="desc", limit=20, review="all").json()["items"]
    assert [i["confidence"] for i in conf] == sorted((i["confidence"] for i in conf), reverse=True)
    page1 = get(env, "/parcels", limit=10, offset=0, sort="parcel_id", review="all").json()["items"]
    page2 = get(env, "/parcels", limit=10, offset=10, sort="parcel_id", review="all").json()["items"]
    ids = [i["parcel_id"] for i in page1 + page2]
    assert ids == sorted(ids) and len(set(ids)) == 20
    assert get(env, "/parcels", q="P000", review="all").json()["total"] >= 1
    assert get(env, "/parcels", sort="geometry").status_code == 400
    assert get(env, "/parcels", review="maybe").status_code == 422


def test_parcel_detail_series_from_imagery_and_signatures(env):
    pid = get(env, "/parcels", traffic_light="red", review="all").json()["items"][0]["parcel_id"]
    d = get(env, "/parcel", id=pid).json()
    assert d["parcel_id"] == pid and d["predicted_class"] != d["declared_class"]
    assert abs(sum(d["probabilities"].values()) - 1) < 0.05 and d["conformal_set"]
    assert {"code": "discrepancy", "source": "declaration"} in d["reasons"]
    assert d["geometry"]["geometry"]["type"] == "Polygon" and 44 < d["geometry"]["bounds"][0][0] < 46
    s = d["series"]
    assert s["source"] == "imagery" and [p["name"] for p in s["panels"]] == ["Sentinel-2", "Sentinel-1"]
    ndvi = s["panels"][0]["series"][0]
    assert ndvi["name"] == "NDVI" and [p["label"] for p in ndvi["points"]] == ["P1", "P2", "P3", "P4"]
    assert all(p["observed"] and -1 <= p["mean"] <= 1 for p in ndvi["points"])
    assert get(env, "/parcel", id="no-such-parcel").status_code == 404

    # a signature table in the job's outputs replaces the imagery means; markers columns are picked up too
    evidence = env["root"] / "evidence"
    evidence.mkdir(exist_ok=True)
    pd.DataFrame({"parcel_id": [pid] * 3 + [pid] * 2, "date": ["2023-05-01", "2023-05-06", "2023-05-11"] * 1 + ["2023-05-01", "2023-05-13"],
                  "index": ["NDVI"] * 3 + ["VV_dB"] * 2, "mean": [0.3, None, 0.5, -11.0, -10.5],
                  "std": [0.05, None, 0.04, 1.0, 1.1], "n_valid": [10, 0, 12, 10, 10]}).to_csv(evidence / "sig.csv", index=False)
    pd.DataFrame({"parcel_id": [pid], "peak_doy": [182.0], "harvest_detected": [True]}).to_csv(evidence / "events.csv", index=False)
    d = get(env, "/parcel", id=pid).json()
    s = d["series"]
    assert s["source"] == "signatures" and s["files"] == ["outputs/demo/evidence/sig.csv"]
    ndvi = s["panels"][0]["series"][0]["points"]
    assert [p["observed"] for p in ndvi] == [True, False, True]
    assert s["panels"][1]["series"][0]["name"] == "VV_dB"
    assert d["markers"] == {"harvest_detected": True, "peak_doy": 182.0}
    for f in ("sig.csv", "events.csv"):
        (evidence / f).unlink()


def test_chips_png_outline_and_cache_outside_outputs(env):
    pid = get(env, "/parcels", review="all", limit=1).json()["items"][0]["parcel_id"]
    for kind in ("true", "ndvi"):
        r = get(env, "/chips", id=pid, kind=kind, size=64).json()
        assert len(r["chips"]) == 4
        chip = r["chips"][0]
        png = base64.b64decode(chip["url"].split(",", 1)[1])
        assert chip["url"].startswith("data:image/png;base64,") and png.startswith(b"\x89PNG")
        assert int.from_bytes(png[16:20], "big") <= 64                                # downsampled
        xs = [x for ring in chip["outline"] for x, _ in ring]
        assert 0 < min(xs) < max(xs) < 1 and chip["valid_fraction"] == 1.0
    cached = list((env["ws"] / ".geocrop-web" / "review-cache" / env["job"]).glob("*.json"))
    assert cached and not (env["root"] / "review" / "cache").exists()
    assert get(env, "/chips", id=pid, kind="false").status_code == 422
    assert get(env, "/chips", id=pid, size=4096).status_code == 422
    assert get(env, "/chips", id="../../etc/passwd").status_code == 404


def test_decisions_append_only_and_validation(env):
    items = get(env, "/parcels", traffic_light="red", review="all").json()["items"]
    a, b, c = (i["parcel_id"] for i in items[:3])
    assert post(env, "/decisions", {"parcel_id": a, "decision": "agree"}).status_code == 422       # purpose explicit
    assert post(env, "/decisions", {"parcel_id": a, "decision": "maybe", "purpose": "training"}).status_code == 422
    assert post(env, "/decisions", {"parcel_id": a, "decision": "disagree", "purpose": "training"}).status_code == 400
    assert post(env, "/decisions", {"parcel_id": a, "decision": "disagree", "purpose": "training",
                                    "label": "=HYPERLINK(1)"}).status_code == 400
    assert post(env, "/decisions", {"parcel_id": "nope", "decision": "unsure", "purpose": "training"}).status_code == 404
    assert post(env, "/decisions", {"parcel_id": a, "decision": "agree", "purpose": "training", "extra": 1}).status_code == 422
    r = post(env, "/decisions", {"parcel_id": a, "decision": "agree", "purpose": "training", "analyst": "=cmd|' /C calc'!A0",
                                 "comment": "line\nbreak"})
    assert r.status_code == 201 and r.json()["decision"]["label"] == items[0]["predicted_class"]
    rec = r.json()["decision"]
    assert rec["analyst"].startswith("'=") and "\n" not in rec["comment"] and len(rec["model_sha256"]) == 64
    declared = items[1]["declared_class"]
    assert post(env, "/decisions", {"parcel_id": b, "decision": "disagree", "label": declared, "purpose": "training",
                                    "analyst": "ana", "confidence": 0.8}).status_code == 201
    assert post(env, "/decisions", {"parcel_id": c, "decision": "unsure", "purpose": "validation"}).status_code == 201
    # a second decision on the same parcel is appended, never edits the first
    assert post(env, "/decisions", {"parcel_id": b, "decision": "unsure", "purpose": "training", "analyst": "ana"}).status_code == 201
    log = env["root"] / "review" / "decisions.csv"
    rows = list(csv.DictReader(log.open()))
    assert [r["parcel_id"] for r in rows][-4:] == [a, b, c, b] and tuple(rows[0]) == review.DECISION_COLUMNS
    assert all(r["timestamp_utc"].endswith("+00:00") for r in rows)
    assert get(env, "/parcels", review="pending", limit=500).json()["total"] == 141
    s = get(env, "").json()["progress"]
    assert s["reviewed"] == 3 and s["unsure"] == 2 and s["agree"] == 1
    hist = get(env, "/parcel", id=b).json()["history"]
    assert [h["decision"] for h in hist] == ["disagree", "unsure"]
    got = get(env, "/decisions").json()
    assert got["rows"] == 4 and got["path"] == "outputs/demo/review/decisions.csv"


def test_export_is_an_ingestable_label_table(env, tmp_path):
    items = get(env, "/parcels", traffic_light="red", review="all").json()["items"]
    pid, declared = items[3]["parcel_id"], items[3]["declared_class"]
    post(env, "/decisions", {"parcel_id": pid, "decision": "disagree", "label": declared, "purpose": "training", "analyst": "zd"})
    r = post(env, "/export", {"purpose": "training", "round": 1})
    assert r.status_code == 200
    body = r.json()
    path = env["ws"] / body["path"]
    assert body["path"] == "outputs/demo/review/labels_training_round1.csv" and body["ingest_argv"][:2] == ["active-learning", "ingest"]
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert tuple(table.columns) == LABEL_TEMPLATE_COLUMNS
    assert set(table["purpose"]) == {"training"} and set(table["source"]) == {"analyst"} and set(table["al_round"]) == {"1"}
    labels = read_labels(path)
    assert pid in set(labels["parcel_id"])
    out = tmp_path / "training_r1.gpkg"
    report = ingest(path, env["root"] / "parcels" / "parcels.gpkg", env["root"] / "reference" / "training.gpkg", out, 1)
    assert out.exists() and report["actions"].get("added", 0) >= 1      # analyst labels became training polygons
    assert "unknown_parcel" not in report["actions"]
    post(env, "/decisions", {"parcel_id": items[4]["parcel_id"], "decision": "unsure", "purpose": "validation"})
    val = post(env, "/export", {"purpose": "validation"}).json()
    vt = pd.read_csv(env["ws"] / val["path"], dtype=str, keep_default_na=False)
    assert val["ingest_argv"] is None and set(vt["purpose"]) == {"validation"}
    dl = env["client"].get(f"/api/jobs/{env['job']}/review/file", params={"name": "labels_training_round1.csv"}, headers=AUTH)
    assert dl.status_code == 200 and dl.text.startswith("parcel_id,label,source")
    assert post(env, "/export", {"purpose": "both"}).status_code == 422


def test_review_confinement(env, tmp_path):
    c, job = env["client"], env["job"]
    for layer in ("../../etc/passwd", "/etc/passwd", "outputs/demo/reference/parcels.gpkg", str(env["root"] / "parcels" / "parcels.gpkg")):
        assert get(env, "/parcels", layer=layer).status_code == 404
    for name in ("../decisions.csv", "decisions.csv/..", "x.csv", "/etc/passwd", "labels_training.csv.exe"):
        assert c.get(f"/api/jobs/{job}/review/file", params={"name": name}, headers=AUTH).status_code == 422
    assert get(env, "/parcel", id="P0000", evidence="nosuchjob").status_code == 404
    # files reached through a symlink that leaves the workspace are never review sources
    outside = tmp_path / "outside_sig.csv"
    pd.DataFrame({"parcel_id": ["P0000"], "date": ["2023-05-01"], "index": ["NDVI"], "mean": [0.9]}).to_csv(outside, index=False)
    link = env["root"] / "evidence_link.csv"
    os.symlink(outside, link)
    try:
        assert get(env, "").json()["sources"]["signatures"] == []
        assert get(env, "/parcel", id="P0000").json()["series"]["source"] == "imagery"
    finally:
        link.unlink()
    # a review folder that is a symlink to elsewhere is refused for writing
    review_dir = env["root"] / "review"
    moved = env["root"] / "review_moved"
    review_dir.rename(moved)
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.symlink(target, review_dir)
    try:
        r = post(env, "/decisions", {"parcel_id": "P0000", "decision": "unsure", "purpose": "training"})
        assert r.status_code == 409 and not list(target.iterdir())
    finally:
        review_dir.unlink()
        moved.rename(review_dir)
    # decisions.csv replaced by a symlink is refused as well
    log = review_dir / "decisions.csv"
    kept = review_dir / "decisions.keep"
    log.rename(kept)
    os.symlink(tmp_path / "evil.csv", log)
    try:
        assert post(env, "/decisions", {"parcel_id": "P0000", "decision": "unsure", "purpose": "training"}).status_code == 409
        assert not (tmp_path / "evil.csv").exists()
    finally:
        log.unlink()
        kept.rename(log)


def test_review_requires_token_and_versioned_alias(env):
    c, job = env["client"], env["job"]
    paths = ["", "/parcels", "/parcel?id=P0000", "/chips?id=P0000", "/decisions", "/file?name=decisions.csv"]
    for path in paths:
        assert c.get(f"/api/jobs/{job}/review{path}").status_code == 401
        assert c.get(f"/api/v1/jobs/{job}/review{path}").status_code == 401
    assert c.post(f"/api/jobs/{job}/review/decisions", json={}).status_code == 401
    assert c.post(f"/api/v1/jobs/{job}/review/export", json={"purpose": "training"}).status_code == 401
    assert c.get(f"/api/v1/jobs/{job}/review", headers=AUTH).json()["available"] is True
    assert c.get(f"/api/jobs/{job}/review", headers={**AUTH, "Origin": "https://evil.example"}).status_code == 403
    schema = c.get("/api/openapi.json", headers=AUTH).json()["paths"]
    assert "/api/v1/jobs/{job_id}/review/parcels" in schema and "/api/jobs/{job_id}/review/parcels" not in schema


def test_review_static_ui(env):
    c = env["client"]
    page = c.get("/").text
    assert 'id="view-review"' in page and 'data-view="review"' in page and '<script defer src="/static/review.js"></script>' in page
    assert "<script>" not in page and " onclick=" not in page and " style=" not in page
    js = c.get("/static/review.js")
    assert js.status_code == 200 and "innerHTML" not in js.text and "eval(" not in js.text
    for key in ('k === "a"', 'k === "d"', 'k === "u"', 'k === "j"', 'k === "k"'):
        assert key in js.text
    assert "focusParcel" in c.get("/static/map.js").text
    css = c.get("/static/style.css").text
    assert ".review-grid" in css and "--series-1" in css


def test_export_labels_unit():
    rows = [
        {"parcel_id": "A", "decision": "agree", "label": "maize", "analyst": "x", "purpose": "training", "timestamp_utc": "t1", "comment": ""},
        {"parcel_id": "A", "decision": "disagree", "label": "soybean", "analyst": "x", "purpose": "training", "timestamp_utc": "t2", "comment": "c"},
        {"parcel_id": "A", "decision": "agree", "label": "maize", "analyst": "y", "purpose": "training", "timestamp_utc": "t3", "comment": ""},
        {"parcel_id": "B", "decision": "unsure", "label": "", "analyst": "x", "purpose": "validation", "timestamp_utc": "t4", "comment": ""},
    ]
    t = review.export_labels(rows, "training", 2)
    assert tuple(t.columns) == LABEL_TEMPLATE_COLUMNS and len(t) == 2          # latest per (parcel, analyst)
    assert set(zip(t["observer"], t["label"])) == {("x", "soybean"), ("y", "maize")}
    v = review.export_labels(rows, "validation")
    assert list(v["label"]) == [""] and list(v["al_round"]) == [""]
    with pytest.raises(ValueError):
        review.export_labels(rows, "test")
    assert review._text("=1+1", 10) == "'=1+1" and review._text("a\x00b", 10) == "a b"
