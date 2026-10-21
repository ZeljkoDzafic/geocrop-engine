import numpy as np
import pytest

from geocrop import metrics


def test_summary_known_values():
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    s = metrics.summary(y, p)
    assert s["overall_accuracy"] == 0.75
    assert s["per_class"]["a"]["producers_accuracy"] == 0.5
    assert s["per_class"]["b"]["users_accuracy"] == pytest.approx(2 / 3)
    assert s["kappa"] == pytest.approx(0.5)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    y = rng.choice(["a", "b", "c"], 300)
    p = np.where(rng.random(300) < 0.8, y, "a")
    b = metrics.bootstrap(y, p, n_boot=200)
    oa = metrics.summary(y, p)["overall_accuracy"]
    assert b["overall_accuracy_ci"][0] <= oa <= b["overall_accuracy_ci"][1]


def test_area_adjusted_perfect_map():
    cm = [[10, 0], [0, 10]]
    r = metrics.area_adjusted(cm, ["a", "b"], {"a": 300.0, "b": 700.0})
    assert r["overall_accuracy"] == 1 and r["overall_accuracy_se"] == 0
    assert r["per_class"]["a"]["adjusted_area"] == pytest.approx(300)


def test_area_adjusted_hand_computed():
    # reference rows, map columns; map stratum a: 8 a + 2 b; stratum b: 1 a + 9 b
    cm = [[8, 1], [2, 9]]
    r = metrics.area_adjusted(cm, ["a", "b"], {"a": 100.0, "b": 900.0})
    # p(ref=a) = 0.1*0.8 + 0.9*0.1 = 0.17
    assert r["per_class"]["a"]["adjusted_area"] == pytest.approx(170)
    assert r["overall_accuracy"] == pytest.approx(0.1 * 0.8 + 0.9 * 0.9)
    assert r["per_class"]["a"]["producers_accuracy"] == pytest.approx(0.08 / 0.17)


def test_calibration_perfect():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.calibration(probs, ["a", "b"], ["a", "b"])["ece"] == 0


def test_mcnemar_and_reject_curve():
    y = ["a"] * 40
    assert metrics.mcnemar(y, ["a"] * 40, ["b"] * 40)["p_value"] < 1e-6
    assert metrics.mcnemar(y, y, y)["p_value"] == 1
    curve = metrics.reject_curve([0.9, 0.4, 0.8], [True, False, True], thresholds=[0, 0.5])
    assert curve[1]["accuracy"] == 1 and curve[1]["review_count"] == 1
