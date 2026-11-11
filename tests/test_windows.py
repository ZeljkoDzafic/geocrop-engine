"""Temporal windows: generation, JM per window, single/cumulative/optimised CV, earliest reliable window and the
in-season leakage guard, on synthetic series of three classes with different phenology (no real data)."""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from geocrop import timeseries as ts
from geocrop import windows as W

START, END = date(2001, 4, 1), date(2001, 9, 28)      # arbitrary synthetic season


def _curve(doy, on, peak, off, amp=0.7):
    """Smooth bump: rises from `on` to `peak`, falls to `off` (day offsets from START)."""
    up = 1 / (1 + np.exp(-(doy - (on + peak) / 2) / ((peak - on) / 6)))
    down = 1 / (1 + np.exp((doy - (peak + off) / 2) / ((off - peak) / 6)))
    return 0.15 + amp * up * down


def synthetic(seed=0, per_class=18, noise=0.03):
    """A emerges early (day 20), B late (day 70); C looks like B until it senesces early (day 120 vs 170).
    S1 VH separates B from C a little earlier than optical does."""
    rng = np.random.default_rng(seed)
    spec = {"A": (20, 80, 175), "B": (70, 120, 170), "C": (70, 120, 125)}
    rows = []
    s2_days = np.arange(0, 180, 5)
    s1_days = np.arange(2, 180, 6)
    for ci, (cls, (on, pk, off)) in enumerate(spec.items()):
        for i in range(per_class):
            uid = f"{cls}{i:02d}"
            x, y = rng.uniform(0, 12000, 2)
            shift = rng.normal(0, 3)
            for d in s2_days:
                if rng.random() < 0.25:                      # cloud gap
                    continue
                v = _curve(d + shift, on, pk, off) + rng.normal(0, noise)
                rows.append((uid, cls, x, y, START + timedelta(days=int(d)), "NDVI", v))
            for d in s1_days:
                v = -18 + 4 * _curve(d + shift, on + 10, pk, off - 20) + (1.5 if cls == "C" else 0) + rng.normal(0, 0.5)
                rows.append((uid, cls, x, y, START + timedelta(days=int(d)), "VH", v))
    return pd.DataFrame(rows, columns=["unit_id", "class_name", "x", "y", "date", "variable", "value"])


def config(**over):
    c = {"seed": 7, "season": {"start": START.isoformat(), "end": END.isoformat()},
         "generate": {"length_days": 30, "prefix": "T"}, "groups": {"S1": ["VH"], "S2": ["NDVI"]},
         "cv": {"block_m": 3000, "n_splits": 4}, "model": {"n_estimators": 30, "n_jobs": 1},
         "bootstrap": {"n": 100}, "permutation_repeats": 1, "kinds": ["single", "cumulative", "optimised"],
         "criteria": {"target_classes": ["A", "B", "C"], "min_f1": 0.7, "jm": {"pairs": [["A", "B"]], "min": 1.5},
                      "lower_bound": {"metric": "f1", "min": 0.4, "method": "bootstrap"},
                      "stability": {"mode": "all_later"}}}
    c.update(over)
    return c


# --- window definition ----------------------------------------------------------------------------------------

def test_explicit_windows_arbitrary_names_and_overlap():
    ws = W.parse_windows({"early": {"start": "2001-04-01", "end": "2001-05-15"},
                          "mid": {"start": "2001-05-01", "end": "2001-06-30"}})
    assert [w.name for w in ws] == ["early", "mid"] and ws[0].days == 45
    with pytest.raises(ValueError, match="unique"):
        W.parse_windows([{"name": "a", "start": "2001-01-01", "end": "2001-01-02"}] * 2)
    with pytest.raises(ValueError, match="precedes"):
        W.parse_windows({"a": {"start": "2001-02-01", "end": "2001-01-01"}})


def test_generated_windows_length_step_overlap_partial():
    ws = W.generate_windows("2001-04-01", "2001-06-30", 30)
    assert [w.days for w in ws] == [30, 30, 30] and ws[1].start == ws[0].end + timedelta(days=1)
    ov = W.generate_windows("2001-04-01", "2001-06-30", 30, overlap_days=10)
    assert ov[1].start == date(2001, 4, 21)                       # step = 20
    assert W.generate_windows("2001-04-01", "2001-06-30", 40, step_days=40, partial="clip")[-1].end == date(2001, 6, 30)
    assert len(W.generate_windows("2001-04-01", "2001-06-30", 40)) == 2   # partial dropped
    with pytest.raises(ValueError, match="disagree"):
        W.generate_windows("2001-04-01", "2001-06-30", 30, step_days=7, overlap_days=10)
    # any year or season, nothing hard-coded
    assert W.generate_windows("2033-11-01", "2034-01-30", 30)[0].start == date(2033, 11, 1)


def test_cumulative_windows_start_at_season_start():
    ws = W.generate_windows(START, END, 30)
    cum = W.cumulative_windows(ws, START)
    assert all(c.start == START for c in cum) and [c.end for c in cum] == [w.end for w in ws]


# --- leakage guard ----------------------------------------------------------------------------------------------

def test_guard_raises_on_late_data_and_names_offenders():
    W.check_observation_dates([date(2001, 5, 1), date(2001, 5, 31)], "2001-05-31")
    with pytest.raises(W.LeakageError, match="after the cutoff"):
        W.check_observation_dates(pd.to_datetime(["2001-05-01", "2001-06-01"]), "2001-05-31")
    with pytest.raises(W.LeakageError, match="f_bad"):
        W.check_provenance({"f_ok": date(2001, 5, 31), "f_bad": date(2001, 6, 5), "f_none": None}, "2001-05-31")
    ends = {"P1": date(2001, 5, 31), "P2": date(2001, 6, 30)}
    W.check_feature_names(["P1_NDVI", "ts_NDVI_d120"], "2001-05-31", ends)
    for bad in (["P2_NDVI"], ["season_NDVI_mean"], ["harm_NDVI_a0"], ["ts_NDVI_d200"], ["unknown_col"]):
        with pytest.raises(W.LeakageError):
            W.check_feature_names(bad, "2001-05-31", ends)


def _matrix(df, var="NDVI"):
    sub = df[df.variable == var]
    dates = sorted(sub.date.unique())
    units = sorted(sub.unit_id.unique())
    mat = np.full((len(units), len(dates)), np.nan)
    for (u, d), v in sub.groupby(["unit_id", "date"]).value.mean().items():
        mat[units.index(u), dates.index(d)] = v
    return [pd.Timestamp(d).date() for d in dates], mat


def test_causal_smoothing_ignores_the_future_but_full_season_smoothing_does_not():
    obs = W.load_observations(synthetic(1))
    dates, mat = _matrix(obs)
    cutoff = START + timedelta(days=90)
    late = np.array([d > cutoff for d in dates])
    perturbed = mat.copy()
    perturbed[:, late] += np.random.default_rng(3).normal(0, 0.5, perturbed[:, late].shape)   # change only the future

    a = W.causal_smooth(dates, mat, cutoff, START)
    b = W.causal_smooth(dates, perturbed, cutoff, START)
    assert a["last_observation"] <= cutoff
    np.testing.assert_array_equal(a["z"], b["z"])                          # bit-identical
    assert np.isfinite(a["z"]).any()

    # the non-causal smoother (whole season, then read at the cutoff) is changed by future observations
    grid = ts.make_grid(START, END, 5)
    def full(m):
        y, w = ts.bin_to_grid(dates, m, np.isfinite(m), grid)
        return ts.whittaker(y, w, 20.0)
    i = (cutoff - START).days // 5
    assert not np.array_equal(full(mat)[:, i], full(perturbed)[:, i])
    # refitting per cutoff yields one result per cutoff, each limited to its own cutoff
    per = W.causal_smooth_per_cutoff(dates, mat, [START + timedelta(days=60), cutoff], START)
    assert all(r["last_observation"] <= c for c, r in per.items())


def test_window_features_and_pipeline_do_not_use_observations_after_the_cutoff():
    obs = W.load_observations(synthetic(2))
    units = W.build_units(obs)
    win = W.Window("w", START + timedelta(days=30), START + timedelta(days=89))
    for stats in (("mean", "std", "min", "max", "median"), ("mean", "smoothed_mean")):
        base, used = W.window_features(obs, units, ["NDVI", "VH"], win, stats, START)
        W.check_provenance(used, win.end)
        bad = obs.copy()
        future = bad.date > pd.Timestamp(win.end)
        bad.loc[future, "value"] = bad.loc[future, "value"] * -3 + 100
        again, _ = W.window_features(bad, units, ["NDVI", "VH"], win, stats, START)
        pd.testing.assert_frame_equal(base, again, check_exact=True)       # bit-identical

    # end to end: results of every window ending on or before D are unchanged when later data are destroyed
    D = START + timedelta(days=89)
    r1 = W.evaluate(obs, config())
    bad = obs.copy()
    future = bad.date > pd.Timestamp(D)
    bad.loc[future, "value"] = np.random.default_rng(0).normal(0, 50, int(future.sum()))
    r2 = W.evaluate(bad, config())
    for key in ("performance", "performance_by_class", "jm_best"):
        a = r1[key][pd.to_datetime(r1[key]["end"]) <= pd.Timestamp(D)].reset_index(drop=True)
        b = r2[key][pd.to_datetime(r2[key]["end"]) <= pd.Timestamp(D)].reset_index(drop=True)
        assert len(a) > 0
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    # ... whereas windows after D do change
    late1 = r1["performance"][pd.to_datetime(r1["performance"]["end"]) > pd.Timestamp(D)]["macro_f1"].to_numpy()
    late2 = r2["performance"][pd.to_datetime(r2["performance"]["end"]) > pd.Timestamp(D)]["macro_f1"].to_numpy()
    assert not np.array_equal(late1, late2)
    assert r1["leakage_guard"]["ok"].all()


def test_guard_trips_if_a_feature_would_use_future_data(monkeypatch):
    obs = W.load_observations(synthetic(3))
    units = W.build_units(obs)
    original = W._Data._select
    # simulate a regression: the window filter ignores its end date
    monkeypatch.setattr(W._Data, "_select", lambda self, var, start, end: original(self, var, start, END))
    with pytest.raises(W.LeakageError):
        W.window_features(obs, units, ["NDVI"], W.Window("w", START, START + timedelta(days=30)), ("mean",), START)


# --- separability and comparison -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return W.evaluate(synthetic(0), config())


def test_early_window_separates_a_from_b_but_not_b_from_c(result):
    jm = result["jm_best"]
    pick = lambda pair, window: jm[(jm.configuration == "S2") & (jm.kind == "single") & (jm.pair == pair)
                                   & (jm.window == window)].jm.iloc[0]
    assert pick("A-B", "T02") > 1.5          # days 30-59: A has emerged, B has not
    assert pick("B-C", "T02") < 1.0          # B and C look alike before C senesces
    assert pick("B-C", "T05") > 1.5          # days 120-149: C has senesced


def test_same_folds_and_all_kinds_and_groups_present(result):
    perf = result["performance"]
    assert perf["fold_signature"].nunique() == 1
    assert set(perf.kind) == {"single", "cumulative", "optimised"}
    assert set(perf.configuration) == {"S1", "S2", "combined"}
    assert not result["optimised_selection"].empty
    # cumulative windows start at the season start, single windows do not
    assert (pd.to_datetime(perf[perf.kind == "cumulative"].start) == pd.Timestamp(START)).all()
    # nested optimisation: every fold records its own chosen start
    assert result["optimised_selection"].groupby(["configuration", "window"]).fold.nunique().min() == 4


def test_contribution_analysis_has_drop_group_and_permutation(result):
    c = result["contribution"]
    assert set(c.group) == {"S1", "S2"}
    assert {"drop_group_gain", "permutation_drop", "macro_f1_group_only"} <= set(c.columns)
    # in the late cumulative windows the optical series carries the signal: removing S2 hurts at least as much as S1
    late = c[(c.kind == "cumulative") & (pd.to_datetime(c.end) == pd.to_datetime(c.end).max())].set_index("group")
    assert late.loc["S2", "drop_group_gain"] >= late.loc["S1", "drop_group_gain"] - 0.05


def test_earliest_reliable_window_is_early_for_ab_and_later_with_bc(result):
    first = result["earliest"]
    assert first["window"] is not None
    pairs_ab = config(criteria={**config()["criteria"], "target_classes": ["A"], "min_f1": 0.7,
                                "lower_bound": {"metric": "f1", "min": 0.3, "method": "bootstrap"},
                                "jm": {"pairs": [["A", "B"]], "min": 1.5}})
    early = W.evaluate(synthetic(0), pairs_ab, None)["earliest"]
    assert early["window"] is not None and early["window"] <= first["window"]
    assert pd.Timestamp(early["end"]) < pd.Timestamp(START + timedelta(days=100))
    # requiring B-C separation as well cannot make the answer earlier
    both = config(criteria={**pairs_ab["criteria"], "target_classes": ["A", "B", "C"],
                            "jm": {"pairs": [["A", "B"], ["B", "C"]], "min": 1.5}})
    assert W.evaluate(synthetic(0), both)["earliest"]["window"] >= early["window"]


def test_earliest_reliable_returns_none_when_criteria_unreachable():
    c = config(criteria={**config()["criteria"], "min_f1": 1.01}, kinds=["cumulative"])
    r = W.evaluate(synthetic(0), c)
    assert r["earliest"]["window"] is None and "No window" in r["earliest"]["note"]


def test_stability_requires_later_windows_to_hold():
    table = pd.DataFrame({"configuration": "combined", "kind": "cumulative",
                          "window": ["a", "b", "c", "d"], "class_name": "X", "f1": [0.5, 0.9, 0.5, 0.9],
                          "recall": 1.0, "f1_lower": 0.9, "recall_lower": 0.9, "f1_fold_min": 0.9, "recall_fold_min": 0.9})
    base = {**W.CRITERIA_DEFAULTS, "min_f1": 0.8, "target_classes": ["X"]}
    ends = dict(zip("abcd", pd.date_range("2001-05-01", periods=4, freq="30D").date))
    none = W.earliest_reliable(table, pd.DataFrame(), {**base, "stability": {"mode": "none", "k": 0}}, list("abcd"), ends)
    allat = W.earliest_reliable(table, pd.DataFrame(), {**base, "stability": {"mode": "all_later", "k": 0}}, list("abcd"), ends)
    nextk = W.earliest_reliable(table, pd.DataFrame(), {**base, "stability": {"mode": "next_k", "k": 1}}, list("abcd"), ends)
    assert none["window"] == "b"                 # first hit, although window c falls back below the threshold
    assert allat["window"] == "d"                # only d and everything after it holds
    assert nextk["window"] is None               # b is followed by a failing window; d has no later window


# --- reproducibility and exports -----------------------------------------------------------------------------

def test_reproducible_and_exports_with_criteria_registered_first(tmp_path):
    obs = tmp_path / "obs.csv"
    synthetic(0).to_csv(obs, index=False)
    a = W.evaluate(obs, config(), tmp_path / "a")
    b = W.evaluate(obs, config(), tmp_path / "b")
    pd.testing.assert_frame_equal(a["performance"], b["performance"], check_exact=True)
    out = tmp_path / "a"
    for name in ("jm_best.csv", "jm_best.md", "performance.csv", "performance_by_class.md", "contribution.csv",
                 "earliest_reliable.csv", "leakage_guard.csv", "resolved_config.json", "criteria.json",
                 "jm_by_window.png", "performance_by_window.png", "group_contribution.png"):
        assert (out / name).stat().st_size > 0, name
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "completed" and m["seed"] == 7 and m["inputs"][0]["sha256"]
    assert m["criteria_sha256"] == W.criteria_hash(json.loads((out / "criteria.json").read_text()))
    assert m["registered_utc"] <= m["completed_utc"]
    assert "recommendation for human review" in m["caveat"].lower()


def test_manifest_with_criteria_exists_before_any_result(tmp_path, monkeypatch):
    seen = {}
    original = W._oof

    def spy(*a, **k):
        seen.setdefault("manifest", json.loads((tmp_path / "manifest.json").read_text()))
        return original(*a, **k)
    monkeypatch.setattr(W, "_oof", spy)
    W.evaluate(synthetic(0), config(kinds=["cumulative"]), tmp_path)
    assert seen["manifest"]["status"] == "criteria_registered" and seen["manifest"]["criteria"]["min_f1"] == 0.7


def test_config_validation():
    with pytest.raises(ValueError, match="groups"):
        W.resolve_config({"windows": {"a": {"start": "2001-01-01", "end": "2001-01-31"}}})
    with pytest.raises(ValueError, match="inside the season"):
        W.resolve_config(config(season={"start": "2001-04-01", "end": "2001-05-01"},
                                generate={"start": "2001-04-01", "end": "2001-09-01", "length_days": 30}))
    rc = W.resolve_config(config())
    assert rc["criteria"]["jm"]["pairs"] == ["A-B"] and len(rc["windows"]) == 6


def test_cli_list_guard_and_run(tmp_path, capsys):
    import yaml

    from geocrop import cli
    cfg = config(kinds=["cumulative"])
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    synthetic(0).to_csv(tmp_path / "o.csv", index=False)
    cli.main(["windows", "list", "--config", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "w.yaml")])
    assert "T01:" in (tmp_path / "w.yaml").read_text()
    cli.main(["windows", "guard", "--config", str(tmp_path / "c.yaml"), "--cutoff", "2001-05-30", "--names", "T01_NDVI"])
    with pytest.raises(SystemExit, match="LEAKAGE"):
        cli.main(["windows", "guard", "--config", str(tmp_path / "c.yaml"), "--cutoff", "2001-05-30",
                  "--names", "T01_NDVI", "T04_NDVI"])
    cli.main(["windows", "run", "--config", str(tmp_path / "c.yaml"), "--observations", str(tmp_path / "o.csv"),
              "--out", str(tmp_path / "out"), "--seed", "3"])
    assert json.loads((tmp_path / "out" / "resolved_config.json").read_text())["seed"] == 3
    assert (tmp_path / "out" / "resolved_config.yaml").exists()
