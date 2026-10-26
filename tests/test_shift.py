"""Domain-shift detection and shift-aware conformal sets on synthetic data (synthetic results only)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from geocrop import applicability, shift, uncertainty

CLASSES = ["a", "b", "c"]
MEANS = np.array([[0, 0, 0, 0, 0, 0], [2, 1, 0, 1, 0, 2], [0, 2, 2, 0, 1, 1]], dtype=float)


def draw(rng, counts, offset=0.0, prefix="p", px=10):
    """Pixels grouped in polygons (polygon-level random effect), `counts` polygons per class."""
    X, y, poly = [], [], []
    for k, n in enumerate(counts):
        for j in range(n):
            centre = MEANS[k] + offset + rng.normal(0, 0.3, 6)
            X.append(centre + rng.normal(0, 0.5, (px, 6)))
            y += [CLASSES[k]] * px
            poly += [f"{prefix}{k}_{j}"] * px
    return np.vstack(X), np.array(y), np.array(poly)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    X, y, poly = draw(rng, [60, 60, 60], prefix="fit")
    Xc, yc, pc = draw(rng, [40, 40, 40], prefix="cal")
    model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1).fit(X, y)
    proba = lambda Z: model.predict_proba(Z)  # noqa: E731 - classes_ are sorted like CLASSES
    pcal = pd.DataFrame(proba(Xc)).groupby(pc).mean()
    lab = pd.Series(yc).groupby(pc).first().reindex(pcal.index)
    conf = uncertainty.conformal_thresholds(pcal.to_numpy(), lab.to_numpy(), CLASSES, (0.1,),
                                            sparse_class_policy="include")
    folds = pd.Series(poly).factorize()[0] % 5
    aoa = applicability.fit(X, folds, [f"f{i}" for i in range(6)], seed=0)
    ref = shift.build_reference(np.vstack([X, Xc]), np.r_[poly, pc], aoa.features, CLASSES, pcal.index,
                                pcal.to_numpy(), lab.to_numpy(), train_di=aoa.train_di,
                                aoa_threshold=aoa.threshold, importance=model.feature_importances_,
                                year=np.full(len(X) + len(Xc), "2021"), region=np.full(len(X) + len(Xc), "r1"))
    return {"model": model, "proba": proba, "conf": conf, "aoa": aoa, "ref": ref, "rng": rng}


def run(setup, X, g, **kw):
    return shift.detect(setup["ref"], X, g, setup["proba"], aoa=setup["aoa"], conformal=setup["conf"], seed=1, **kw)


def test_in_domain_control_is_not_flagged(setup):
    X, _, g = draw(np.random.default_rng(5), [40, 40, 40], prefix="t")
    r = run(setup, X, g, metadata={"year": "2021", "region": "r1"})
    assert r["verdict"] == "in-domain", r["reasons"]
    assert r["signals"]["domain_classifier"]["value"] < 0.7
    assert r["conformal"]["status"].startswith("assumed exchangeable")
    assert shift.applied_thresholds(r, 0.1) is None


def test_covariate_shift_is_severe(setup):
    X, _, g = draw(np.random.default_rng(6), [40, 40, 40], offset=1.5, prefix="t")
    r = run(setup, X, g)
    assert r["verdict"] == "severe"
    assert r["signals"]["domain_classifier"]["value"] >= 0.85
    assert r["signals"]["aoa"]["level"] >= 1
    assert r["conformal"]["status"] == shift.NOT_ESTABLISHED
    json.dumps(r, allow_nan=False)


def test_prior_shift_alone_is_mild(setup):
    X, _, g = draw(np.random.default_rng(7), [100, 10, 10], prefix="t")
    r = run(setup, X, g, metadata={"year": "2021", "region": "r1"})
    assert r["signals"]["class_prior"]["level"] == 1
    assert r["signals"]["domain_classifier"]["level"] == 0, r["signals"]["domain_classifier"]
    assert r["signals"]["domain_classifier"]["unadjusted_auc"] > r["signals"]["domain_classifier"]["value"]
    assert r["verdict"] == "mild"


def test_metadata_year_mismatch_is_mild(setup):
    X, _, g = draw(np.random.default_rng(8), [40, 40, 40], prefix="t")
    r = run(setup, X, g, config={"year": 2025})
    assert r["signals"]["metadata"]["level"] == 1 and r["verdict"] == "mild"
    assert any("year 2025" in x["message"] for x in r["reasons"])


def test_adjusted_thresholds_are_wider_and_used_only_under_shift(setup):
    X, _, g = draw(np.random.default_rng(6), [40, 40, 40], offset=1.5, prefix="t")
    r = run(setup, X, g, config={"shift_conformal": "conservative"})
    t = r["conformal"]["thresholds"]["0.1"]
    assert all(t["conservative"][c] >= t["none"][c] for c in CLASSES)
    assert shift.applied_thresholds(r, 0.1) == t["conservative"]
    assert set(t["weighted"]) == set(CLASSES)


def test_weighted_threshold_equal_weights_matches_split_conformal():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=57)
    for a in (0.05, 0.1, 0.2):
        k = int(np.ceil((len(s) + 1) * (1 - a)))
        assert shift.weighted_threshold(s, np.ones(len(s)), 1.0, a) == pytest.approx(np.sort(s)[k - 1])
    assert shift.weighted_threshold(s[:5], np.ones(5), 1.0, 0.1) == 1.0     # too few objects: include all
    # a heavy test weight pushes the quantile to +inf (always include)
    assert shift.weighted_threshold(s, np.ones(len(s)), 1e6, 0.1) == 1.0


def test_policy_routing():
    frame = pd.DataFrame({"aoa_fraction": [0.9, 0.1, 0.9, None, 0.9],
                          "decision": ["accepted"] * 5})
    light = pd.Series(["green", "green", "red", "yellow", "grey"])
    codes = [[], [], ["conformal_discrepancy"], ["review_decision"], ["insufficient_data"]]
    mild = {"level": 1, "policy": "review"}
    out, c, w = shift.apply_policy(frame, light, codes, mild)
    assert list(out) == ["yellow", "yellow", "yellow", "yellow", "grey"]
    assert all(shift.REASON_CODE in c[i] for i in range(4)) and shift.REASON_CODE not in c[4]
    out, c, w = shift.apply_policy(frame, light, codes, {"level": 2, "policy": "abstain"})
    assert list(out) == ["yellow", "grey", "yellow", "grey", "grey"] and list(w) == [0, 1, 0, 1, 0]
    out, c, w = shift.apply_policy(frame, light, codes, {"level": 2, "policy": "warn"})
    assert list(out) == list(light) and c == codes
    calendar_red = shift.apply_policy(frame.iloc[:1], pd.Series(["red"]), [["calendar_inconsistent"]], mild)
    assert list(calendar_red[0]) == ["red"]


def test_reference_roundtrip_and_settings(tmp_path, setup):
    p = shift.save_reference(setup["ref"], tmp_path / shift.REFERENCE_FILE)
    back = shift.load_reference(p)
    assert back["meta"] == setup["ref"]["meta"] and back["X"].shape == setup["ref"]["X"].shape
    with pytest.raises(ValueError):
        shift.settings({"shift_policy": "ignore"})
    with pytest.raises(ValueError):
        shift.settings({"shift_thresholds": {"bogus": [1, 2]}})
    assert shift.coverage_statement(None, 0.1).startswith("target 90 %")
    assert shift.coverage_statement({"level": 2, "verdict": "severe", "conformal": {"applied": "weighted"}},
                                    0.1).startswith(shift.NOT_ESTABLISHED)


# --------------------------------------------------------------------------- engine integration (synthetic demo)

@pytest.fixture(scope="module")
def trained(samples, stack, demo, tmp_path_factory):
    from geocrop import model
    root = tmp_path_factory.mktemp("shift")
    model.train(samples, root / "m", params={"n_estimators": 20, "n_jobs": 1}, n_splits=3, spatial_block_m=500,
                bootstrap=0)
    return root


def test_train_writes_reference_and_predict_reports_shift(trained, stack, demo):
    from geocrop import model
    assert (trained / "m" / shift.REFERENCE_FILE).exists()
    out = model.predict(stack, trained / "m" / "model.joblib", trained / "pred", block_size=64,
                        shift={"config": {}, "parcels": demo["parcels"], "alpha": 0.1})
    manifest = json.loads((trained / "pred" / "prediction_manifest.json").read_text())
    report = json.loads((trained / "pred" / "shift_report.json").read_text())
    assert manifest["domain_shift"]["verdict"] == report["verdict"]
    assert report["sample"]["target"] == "pixels inside parcels"
    assert report["verdict"] in shift.VERDICTS
    # same stack as training: the in-domain control must not be flagged as severe
    assert report["verdict"] != "severe", report["reasons"]
    plain = model.predict(stack, trained / "m" / "model.joblib", trained / "plain", block_size=64)
    assert "domain_shift" not in json.loads(open(plain["manifest"]).read())
    assert out["manifest"]


def test_parcels_mark_coverage_and_route_by_policy(trained, demo):
    from geocrop import evidence, parcels
    pred = trained / "pred"
    conf = trained / "m" / "conformal.json"
    report = json.loads((pred / "shift_report.json").read_text())
    base = parcels.aggregate(pred / "probabilities.tif", demo["parcels"], trained / "base.gpkg", conformal=conf,
                             aoa=pred / "aoa.tif")
    severe = {**report, "level": 2, "verdict": "severe", "policy": "review",
              "conformal": {**report["conformal"], "status": shift.NOT_ESTABLISHED, "applied": "conservative"}}
    summ = parcels.aggregate(pred / "probabilities.tif", demo["parcels"], trained / "rev.gpkg", conformal=conf,
                             aoa=pred / "aoa.tif", domain_shift=severe)
    table = pd.read_csv(trained / "rev.csv")
    assert set(table["conformal_coverage"]) == {shift.coverage_statement(severe, 0.1)}
    assert "not established" in summ["domain_shift"]["conformal_coverage"]
    assert summ["domain_shift"]["conformal_thresholds"] == "shift-adjusted (conservative)"
    assert "green" not in set(table["traffic_light"])
    assert (table["traffic_light"] != "grey").sum() == sum(v for k, v in base["traffic_light_counts"].items()
                                                           if k != "grey")
    assert table.loc[table.traffic_light != "grey", "traffic_light_reasons"].str.contains("DOMAIN_SHIFT").all()
    warn = parcels.aggregate(pred / "probabilities.tif", demo["parcels"], trained / "warn.gpkg", conformal=conf,
                             aoa=pred / "aoa.tif", domain_shift={**severe, "policy": "warn",
                                                                  "conformal": {**severe["conformal"],
                                                                                "applied": "none"}})
    assert warn["traffic_light_counts"] == base["traffic_light_counts"]
    assert warn["decision_counts"] == base["decision_counts"]
    # evidence packages reproduce the stored, shift-routed traffic light
    ev = evidence.build(trained / "rev.gpkg", trained / "evidence", png=False)
    idx = pd.read_csv(ev["index_csv"])
    assert (idx["traffic_light"] == idx["stored_traffic_light"]).all()
    one = json.loads((trained / "evidence" / idx["evidence_json"].iloc[0]).read_text())
    assert one["classification"]["conformal_coverage"].startswith(shift.NOT_ESTABLISHED)
    assert any(r["rule"] == "applicability.domain_shift" and r["triggered"] for r in one["rules"])


def test_abstain_withholds_out_of_aoa_parcels(trained, demo):
    from geocrop import parcels
    pred = trained / "pred"
    report = json.loads((pred / "shift_report.json").read_text())
    severe = {**report, "level": 2, "verdict": "severe", "policy": "abstain",
              "conformal": {**report["conformal"], "applied": "none"}}
    parcels.aggregate(pred / "probabilities.tif", demo["parcels"], trained / "abs.gpkg",
                      conformal=trained / "m" / "conformal.json", aoa=pred / "aoa.tif", min_aoa_fraction=0.99,
                      domain_shift=severe)
    t = pd.read_csv(trained / "abs.csv")
    out = t["aoa_fraction"].isna() | (t["aoa_fraction"] < 0.99)
    assert (t.loc[out & (t.n_valid > 0), "traffic_light"] == "grey").all()
    assert t.loc[out & t.model_decision_withheld, "traffic_light_reasons"].str.contains("DOMAIN_SHIFT").all()


def test_config_section_validates():
    from geocrop import config
    cfg = config.load(overrides={"uncertainty": {"shift_policy": "review", "shift_conformal": "weighted"}})
    assert cfg["uncertainty"]["shift_policy"] == "review"
    with pytest.raises(config.ConfigError):
        config.load(overrides={"uncertainty": {"shift_policy": "maybe"}})
    assert config.fingerprint(config.load()) == config.fingerprint(config.load(overrides={"uncertainty": {}}))
