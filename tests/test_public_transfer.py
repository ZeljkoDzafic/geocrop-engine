"""Public transfer protocol (region/year) and the scale-benchmark tool, on synthetic data (no network)."""
from __future__ import annotations

import copy
import importlib.util
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geocrop import applicability, public_data, public_transfer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "benchmarks" / "si_transfer_pomurje.v1.yaml"


def _tool():
    spec = importlib.util.spec_from_file_location("scale_benchmark", ROOT / "tools" / "scale_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- protocol

def test_frozen_protocol_is_valid_and_pins_every_label_source():
    p, digest = public_transfer.load_protocol(PROTOCOL)
    assert len(digest) == 64
    assert p["source_domain"] == "pomurje_2021"
    assert {d["year"] for d in p["domains"].values()} == {2021, 2025}
    for s in p["sources"].values():
        assert s["checksum"].split(":")[0] in ("md5", "sha256") and s["licence"]
    # regions: source and targets do not share tiles within the same year
    tiles = {}
    size = p["tile_size_m"]
    for d in p["domains"].values():
        for origin in d["tiles"].values():
            tiles.setdefault(d["year"], []).append(box(*origin, origin[0] + size, origin[1] + size))
    for boxes in tiles.values():
        for i, a in enumerate(boxes):
            assert all(a.intersection(b).area == 0 for b in boxes[i + 1:])


def test_domain_windows_follow_the_domain_year_and_reject_reversed_windows():
    p, _ = public_transfer.load_protocol(PROTOCOL)
    w = public_transfer.domain_protocol(p, "pomurje_2025")["windows"]
    assert w["P1"] == {"start": "2025-03-15", "end": "2025-04-30"}
    bad = copy.deepcopy(p)
    bad["windows"]["P1"] = {"start": "05-01", "end": "04-01"}
    with pytest.raises(ValueError, match="before it starts"):
        public_transfer.validate_protocol(bad)


@pytest.mark.parametrize("edit, match", [
    (lambda p: p.update(source_domain="nowhere"), "source_domain"),
    (lambda p: p["sources"]["kmrs_si_2025"].pop("checksum"), "checksum"),
    (lambda p: p.update(common_classes=["rice"]), "common_classes"),
    (lambda p: p["domains"]["podravje_2021"].update(source="other"), "unknown label source"),
])
def test_invalid_protocols_are_rejected(edit, match):
    p, _ = public_transfer.load_protocol(PROTOCOL)
    edit(p)
    with pytest.raises(ValueError, match=match):
        public_transfer.validate_protocol(p)


def test_code_map_takes_the_majority_and_reports_consistency():
    table = pd.DataFrame({"SIFRA": ["005", "005", "005", "204", "777"],
                          "HC": ["3301010600", "3301010600", "3301090400", "3302000000", None]})
    mapping, info = public_transfer.code_map(table, "SIFRA", "HC")
    assert mapping.to_dict() == {"005": "3301010600", "204": "3302000000"}
    assert info["inconsistent_codes"] == 1 and info["min_consistency"] == pytest.approx(2 / 3)
    assert public_transfer.code_map(table, "SIFRA", "HC")[1]["sha256"] == info["sha256"]


def test_url_layer_extracts_a_checksum_pinned_archive(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    gpd.GeoDataFrame({"ID": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3794").to_file(src / "L.shp")
    zpath = tmp_path / "L.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f.name)

    def fake_fetch(url, dest, checksum=None, **kw):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(zpath.read_bytes())
        return Path(dest)
    monkeypatch.setattr(public_data, "fetch", fake_fetch)
    layer = public_data.url_layer("https://example.org/data/L.zip", "sha256:" + "0" * 64, tmp_path / "cache", "kmrs")
    assert layer.name == "L.shp" and layer.parent.parent.name == "kmrs"


# --------------------------------------------------------------------------- scoring

def _domain(rng, n_parcels, shift, classes, prefix):
    rows, X, poly = [], [], []
    for i in range(n_parcels):
        c = classes[i % len(classes)]
        k = classes.index(c)
        uid = f"{prefix}:{i}"
        rows.append({"uid": uid, "class_name": c, "block": f"{prefix}/{i // 4}"})
        X.append(rng.normal(k * 3 + shift, 1.0, size=(10, 4)))
        poly += [uid] * 10
    return pd.DataFrame(rows), np.vstack(X), np.array(poly)


def test_score_reports_shift_as_lower_accuracy_coverage_and_aoa():
    rng = np.random.default_rng(0)
    classes = ["a", "b", "c"]
    p = {"evaluation": {"bootstrap": 50, "bootstrap_seed": 1, "confidence": 0.9, "conformal_alpha": 0.1,
                        "aoa_min_fraction": 0.5}, "common_classes": ["a", "b"]}
    train, Xt, pt = _domain(rng, 60, 0.0, classes, "src")
    near, Xn, pn = _domain(rng, 30, 0.0, classes, "near")
    far, Xf, pf = _domain(rng, 30, 2.5, classes, "far")
    allp = pd.concat([train, near, far]).set_index("uid")
    from sklearn.ensemble import RandomForestClassifier
    yt = allp["class_name"].reindex(pt).to_numpy()
    est = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xt, yt)
    proba = public_transfer._proba_fn(est, classes)
    thresholds = public_transfer._calibrated(est, classes, Xt, pt, allp["class_name"], 0.1)
    aoa = applicability.fit(Xt, np.arange(len(Xt)) % 3, [f"f{i}" for i in range(4)])
    a = public_transfer.score(proba, classes, Xn, pn, allp["class_name"], allp["block"], thresholds, aoa, p)
    b = public_transfer.score(proba, classes, Xf, pf, allp["class_name"], allp["block"], thresholds, aoa, p)
    assert a["parcels"] == b["parcels"] == 30
    assert a["macro_f1"] > b["macro_f1"]
    assert a["conformal"]["coverage"] >= b["conformal"]["coverage"]
    assert a["aoa"]["pixel_share_inside"] > b["aoa"]["pixel_share_inside"]
    assert a["common_classes"]["classes"] == ["a", "b"] and a["common_classes"]["parcels"] == 20
    lo, hi = a["macro_f1_ci"]
    assert lo <= a["macro_f1"] <= hi
    report = {"protocol": {"id": "x", "version": 1, "sha256": "0" * 64}, "engine": {"version": "t", "git_commit": "c",
              "git_dirty": False}, "started_utc": "s", "finished_utc": "f", "note": "synthetic", "command": "cmd",
              "source_domain": "src", "classes": classes,
              "domains": {k: {"region": k, "year": 2021, "source": "s", "tiles": ["T"],
                              "labels": {"sampled_per_class": {"a": 1}, "classes_dropped_below_minimum": {}}}
                          for k in ("src", "far")},
              "results": {"src": {"kind": "in-domain", **a},
                          "far": {"kind": "out-of-domain", "all_parcels": b, "local_baseline": {"status": "n/a"}}},
              "transfer_cv": None, "sources": {}, "code_maps": {},
              "samples": {"pixels": 1, "features": 4, "fingerprint": "f" * 64}, "stages": {}, "peak_rss_mb": 1}
    md = public_transfer.markdown(json.loads(json.dumps(report)))
    assert "| far | out-of-domain | 30 |" in md and "Drop vs source" in md


# --------------------------------------------------------------------------- scale benchmark tool

def test_scale_benchmark_square_dir_bytes_and_report(tmp_path):
    tool = _tool()
    w, s, e, n = tool.square(100)
    assert e - w == pytest.approx(10000) and n - s == pytest.approx(10000)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.bin").write_bytes(b"x" * 10000)
    assert tool._dir_bytes(tmp_path) >= 10000
    stage = {"wall_s": 10.0, "cpu_s": 20.0, "avg_busy_cores": 2.0, "peak_rss_mb": 500.0, "disk_written_mb": 5.0,
             "dir_peak_mb": 50.0, "net_in_mb_lower": 1.0, "net_in_mb_host_upper": 2.0, "host_load1_mean": 1.0}
    for area in (25, 100):
        d = tmp_path / "out" / f"area_{area}km2"
        d.mkdir(parents=True)
        stages = [{"stage": n, "status": "ok", **{k: v * area / 25 for k, v in stage.items()}}
                  for n in ("s2_import", "features", "training", "prediction")]
        (d / "benchmark.json").write_text(json.dumps({"area": area, "error": None, "total_wall_s": 40.0 * area / 25,
            "stages": stages, "info": {"area_km2": area, "pixels": area * 10000, "scenes": {"s2": 10, "s1": 0},
                                       "features": 97, "samples": 1000}}))
    summary = tool.report(tmp_path / "out", tmp_path / "img")
    assert (tmp_path / "img" / "scale_benchmark.png").is_file()
    a, b = summary["linear_fits_a_plus_b_area"]["features"]["wall"]
    assert b == pytest.approx(0.4) and a == pytest.approx(0, abs=1e-9)
