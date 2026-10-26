"""Active-learning loop (gap G9) on SYNTHETIC data: selection, ingestion with provenance, budget simulation."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from shapely.geometry import Point, box

from geocrop import active_learning as al

CRS = "EPSG:32634"


def _parcels(n_side=12, size=100.0, seed=0):
    """Grid of square parcels with geocrop-parcels-like columns (entropy, predicted_class, p_*)."""
    rng = np.random.default_rng(seed)
    classes = ["maize", "soybean", "wheat", "forest"]
    rows = []
    for iy in range(n_side):
        for ix in range(n_side):
            p = rng.dirichlet(np.ones(len(classes)) * 0.7)
            ent = float(-(p * np.log(p)).sum() / np.log(len(classes)))
            row = {"parcel_id": f"P{iy:02d}{ix:02d}", "n_valid": 50, "entropy": ent,
                   "predicted_class": classes[int(p.argmax())], "mean_class": classes[int(p.argmax())],
                   "true_class": classes[(ix + iy) % len(classes)],
                   "geometry": box(500000 + ix * size, 4900000 + iy * size,
                                   500000 + (ix + 1) * size, 4900000 + (iy + 1) * size)}
            row.update({f"p_{c}": float(v) for c, v in zip(classes, p)})
            rows.append(row)
    return gpd.GeoDataFrame(rows, crs=CRS)


def _centroids(gdf):
    return np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])


# ----------------------------------------------------------------------------------------- selection

def test_select_respects_budget_spacing_and_exclusions(tmp_path):
    gdf = _parcels()
    labelled = ["P0000", "P0101"]
    excluded = ["P0505", "P0606"]
    picked, summary = al.select(gdf, 10, tmp_path / "sel.gpkg", min_spacing_m=250, labelled_ids=labelled,
                                exclude_ids=excluded, trips=3)
    assert len(picked) == summary["selected"] == 10
    assert pdist(_centroids(picked)).min() >= 250
    assert not set(picked.parcel_id) & set(labelled + excluded)
    assert list(picked.al_rank) == list(range(1, 11))
    assert set(picked.trip_id) == {1, 2, 3}
    for _, g in picked.groupby("trip_id"):
        assert sorted(g.trip_order) == list(range(1, len(g) + 1))
    form = pd.read_csv(summary["label_form"])
    assert list(form.parcel_id) == list(picked.parcel_id)
    assert {"label", "source", "date", "confidence", "purpose"} <= set(form.columns)
    assert (form.purpose == "training").all()
    assert gpd.read_file(tmp_path / "sel.gpkg").shape[0] == 10


def test_select_reports_shortfall_instead_of_relaxing_spacing():
    gdf = _parcels(n_side=4)                      # 400 m x 400 m area
    picked, summary = al.select(gdf, 10, min_spacing_m=350, candidate_factor=10)
    assert summary["shortfall"] == 10 - len(picked) > 0
    assert len(picked) < 2 or pdist(_centroids(picked)).min() >= 350
    assert "hint" in summary


def test_select_is_more_diverse_than_top_uncertainty():
    gdf = _parcels(n_side=14, seed=3)
    picked, _ = al.select(gdf, 12, min_spacing_m=0, stratify="none", candidate_factor=4, uncertainty_weight=0.0)
    top = gdf.sort_values("entropy", ascending=False).head(12)
    cols = [c for c in gdf if c.startswith("p_")]
    Z = al.standardise(gdf[cols].to_numpy())
    z = pd.DataFrame(Z, index=gdf.parcel_id)
    spread = pdist(z.loc[picked.parcel_id].to_numpy()).min()
    assert spread > pdist(z.loc[top.parcel_id].to_numpy()).min()
    assert picked.entropy.min() >= gdf.entropy.quantile(1 - 4 * 12 / len(gdf)) - 1e-9   # from the top candidates


def test_select_stratifies_by_predicted_class():
    gdf = _parcels(n_side=14, seed=5)
    picked, summary = al.select(gdf, 8, min_spacing_m=0, stratify="equal", candidate_factor=8)
    counts = picked.predicted_class.value_counts()
    assert set(counts.index) == set(gdf.predicted_class)    # every predicted class gets visits
    assert counts.max() - counts.min() <= 1
    assert summary["quota_relaxed"] is False


def test_class_quota_modes():
    strata = ["a"] * 10 + ["b"] * 2 + ["c"] * 1
    assert al.class_quota(strata, 9, "equal") == {"a": 6, "b": 2, "c": 1}
    q = al.class_quota(strata, 6, "proportional")
    assert sum(q.values()) == 6 and q["a"] >= q["b"] >= q["c"]
    assert al.class_quota(strata, 5, "none") == {}
    with pytest.raises(ValueError):
        al.class_quota(strata, 5, "bogus")


def test_kcenter_uses_labelled_centres():
    Z = np.array([[0.0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    r = al.kcenter_batch(Z, 1, centres=np.array([[0.0, 0.0]]), uncertainty_weight=0.0)
    assert r.order[0] in (2, 3)                 # far from what is already labelled


# ----------------------------------------------------------------------------------------- ingestion

@pytest.fixture
def campaign(tmp_path):
    gdf = _parcels(n_side=6)
    parcels = tmp_path / "parcels.gpkg"
    gdf.to_file(parcels, driver="GPKG")
    tr = gdf[gdf.parcel_id.isin(["P0000", "P0001", "P0002"])][["geometry"]].copy()
    tr["geometry"] = tr.geometry.buffer(-10)        # interpreted polygons without a parcel_id column
    tr["polygon_id"] = ["T1", "T2", "T3"]
    tr["class_name"] = ["maize", "soybean", "wheat"]
    training = tmp_path / "training.gpkg"
    gpd.GeoDataFrame(tr, crs=CRS).to_file(training, driver="GPKG")
    c = gdf.set_index("parcel_id").geometry.centroid
    val = gpd.GeoDataFrame({"obs_id": ["O1"], "class_name": ["forest"]}, geometry=[Point(c["P0505"].x, c["P0505"].y)],
                           crs=CRS)
    validation = tmp_path / "observations.gpkg"
    val.to_file(validation, driver="GPKG")
    return {"parcels": parcels, "training": training, "validation": validation, "dir": tmp_path}


def _labels(rows):
    return pd.DataFrame(rows, columns=["parcel_id", "label", "source", "date", "confidence", "purpose"])


def test_ingest_provenance_conflicts_and_validation_separation(campaign):
    d = campaign["dir"]
    labels = _labels([
        ("P0100", "maize", "field", "2024-06-01", 0.9, "training"),
        ("P0101", "wheat", "field", "2024-06-01", 0.9, "training"),
        ("P0101", "soybean", "analyst", "2024-06-03", 0.6, "training"),     # disagreement within the round
        ("P0001", "maize", "analyst", "2024-06-02", 0.8, "training"),       # disagrees with existing training
        ("P0002", "wheat", "field", "2024-06-02", 0.9, "training"),         # already labelled, same class
        ("P0505", "forest", "field", "2024-06-02", 0.9, "training"),        # validation parcel -> never training
        ("P0303", "forest", "field", "2024-06-02", 0.9, "validation"),      # reserved for validation
        ("P0404", "cant_tell", "analyst", "2024-06-02", 0.2, "training"),
        ("P0402", "soybean", "analyst", "2024-06-02", 0.3, "training"),     # below min confidence
        ("ZZZ", "maize", "field", "2024-06-02", 0.9, "training"),
    ])
    out = d / "training_r1.gpkg"
    rep = al.ingest(labels, campaign["parcels"], campaign["training"], out, 1, validation=campaign["validation"],
                    min_confidence=0.5)
    acts = rep["actions"]
    assert acts["added"] == 1 and acts["conflict_excluded"] == 2 and acts["conflict_kept_existing"] == 1
    assert acts["already_labelled"] == 1 and acts["validation_excluded"] == 1 and acts["validation_reserved"] == 1
    assert acts["no_label"] == 1 and acts["low_confidence"] == 1 and acts["unknown_parcel"] == 1
    assert {c["type"] for c in rep["conflicts"]} == {"within_round", "with_training"}
    merged = gpd.read_file(out)
    assert len(merged) == 4
    new = merged[merged.al_round == 1]
    assert list(new.parcel_id) == ["P0100"] and new.label_source.iloc[0] == "field"
    assert new.polygon_id.iloc[0] == "AL1_P0100"
    assert not {"P0505", "P0303"} & set(merged.parcel_id.dropna())
    assert (merged[merged.al_round == 0].label_source == "initial").all()
    assert merged.loc[merged.polygon_id == "T2", "class_name"].iloc[0] == "soybean"   # kept existing
    prov = pd.read_csv(rep["provenance"], dtype={"parcel_id": str})
    assert len(prov) == len(labels) and (prov["round"] == 1).all()
    assert pd.read_csv(rep["validation_labels"]).parcel_id.tolist() == ["P0303"]
    assert json.loads((d / "ingest_report_round1.json").read_text())["added"] == 1
    # round ids are single-use; a second round appends to the provenance log
    with pytest.raises(ValueError, match="already recorded"):
        al.ingest(labels, campaign["parcels"], out, out, 1)
    rep2 = al.ingest(_labels([("P0200", "wheat", "field", "2024-07-01", 1.0, "training")]), campaign["parcels"],
                     out, d / "training_r2.gpkg", 2, validation=campaign["validation"])
    prov = pd.read_csv(rep2["provenance"])
    assert sorted(prov["round"].unique()) == [1, 2]
    assert set(gpd.read_file(d / "training_r2.gpkg").al_round) == {0, 1, 2}


def test_ingest_replace_and_conflict_policies(campaign):
    d = campaign["dir"]
    labels = _labels([
        ("P0001", "maize", "analyst", "2024-06-02", 0.8, "training"),
        ("P0101", "wheat", "field", "2024-06-01", 0.9, "training"),
        ("P0101", "soybean", "analyst", "2024-06-03", 0.6, "training"),
    ])
    rep = al.ingest(labels, campaign["parcels"], campaign["training"], d / "t.gpkg", 1,
                    conflict_policy="highest_confidence", existing_policy="replace")
    merged = gpd.read_file(d / "t.gpkg")
    assert rep["actions"]["replaced"] == 1 and "T2" not in set(merged.polygon_id)
    assert merged.set_index("parcel_id").loc["P0101", "class_name"] == "wheat"
    prov = pd.read_csv(rep["provenance"])
    assert prov.loc[prov.action == "replaced", "previous_label"].iloc[0] == "soybean"
    rep = al.ingest(labels, campaign["parcels"], campaign["training"], d / "t2.gpkg", 1,
                    provenance=d / "p2.csv", conflict_policy="latest")
    assert gpd.read_file(d / "t2.gpkg").set_index("parcel_id").loc["P0101", "class_name"] == "soybean"


def test_ingest_refuses_training_that_overlaps_validation(campaign):
    c = gpd.read_file(campaign["training"]).geometry.centroid.iloc[0]
    bad = campaign["dir"] / "bad_val.gpkg"
    gpd.GeoDataFrame({"obs_id": ["X"]}, geometry=[c], crs=CRS).to_file(bad, driver="GPKG")
    with pytest.raises(ValueError, match="overlaps validation"):
        al.ingest(_labels([("P0100", "maize", "field", "", 1, "training")]), campaign["parcels"],
                  campaign["training"], campaign["dir"] / "x.gpkg", 1, validation=bad)


def test_ingested_training_feeds_sample_extraction(demo, stack, tmp_path):
    from geocrop.samples import extract
    par = gpd.read_file(demo["parcels"])
    tr = gpd.read_file(demo["training"])
    known = set(al.training_parcel_ids(tr, par, "parcel_id").dropna())
    new = par[~par.parcel_id.isin(known)].head(3)
    labels = _labels([(p, c, "field", "2023-07-20", 1.0, "training") for p, c in zip(new.parcel_id, new.true_class)])
    rep = al.ingest(labels, par, tr, tmp_path / "train.gpkg", 1, validation=demo["observations"], inner_buffer_m=20)
    assert rep["added"] + rep["actions"].get("validation_excluded", 0) == 3
    s = extract(stack, tmp_path / "train.gpkg", max_per_polygon=10)
    assert len(np.unique(s.polygon)) == len(tr) + rep["added"]


# ----------------------------------------------------------------------------------------- simulation

def test_simulation_reproducible_and_uncertainty_not_worse_than_random(tmp_path):
    pool = al.synthetic_pool(n_side=20, noise=0.10, seed=3)
    kw = dict(initial=8, batch=8, rounds=5, repeats=3, seed=11, block_m=1000, params={"n_estimators": 60, "n_jobs": 1},
              figure=False)
    a = al.simulate(pool, tmp_path / "a", **kw)
    b = al.simulate(pool, tmp_path / "b", **kw)
    ca, cb = pd.read_csv(tmp_path / "a" / "curves.csv"), pd.read_csv(tmp_path / "b" / "curves.csv")
    pd.testing.assert_frame_equal(ca, cb)
    assert a["comparison"] == b["comparison"]
    assert set(ca.strategy) == set(al.STRATEGIES)
    assert sorted(ca.n_labelled.unique()) == [8 + 8 * i for i in range(6)]
    # all strategies of one repeat share the initial set and the same test set
    first = ca[ca["round"] == 0].groupby("repeat").macro_f1.nunique()
    assert (first == 1).all()
    summ = pd.read_csv(tmp_path / "a" / "summary.csv")
    f1 = summ[summ.metric == "macro_f1"]
    assert (f1.ci95_low <= f1["mean"]).all() and (f1["mean"] <= f1.ci95_high).all()
    comp = a["comparison"]
    assert comp["uncertainty"]["alc_mean"] >= comp["random"]["alc_mean"] - 0.02
    assert comp["uncertainty_diversity"]["alc_mean"] >= comp["random"]["alc_mean"] - 0.02
    assert a["mode"].startswith("SYNTHETIC")


def test_spatial_test_split_is_fixed_and_buffered():
    pool = al.synthetic_pool(n_side=16, seed=1)
    t1, p1 = al.spatial_test_split(pool, 1000, 0.25, buffer_m=300, seed=4)
    t2, p2 = al.spatial_test_split(pool, 1000, 0.25, buffer_m=300, seed=4)
    assert np.array_equal(t1, t2) and np.array_equal(p1, p2)
    assert not set(t1) & set(p1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.column_stack([pool.x[t1], pool.yy[t1]])).query(np.column_stack([pool.x[p1], pool.yy[p1]]))
    assert d.min() >= 300


def test_simulate_from_samples_and_cli(samples, tmp_path):
    from geocrop.cli import main
    pool = al.pool_from_samples(samples)
    assert len(pool.y) == len(np.unique(samples.polygon)) and pool.X.shape[1] == len(samples.features)
    path = samples.save(tmp_path / "samples.npz")
    main(["active-learning", "simulate", "--samples", str(path), "--out", str(tmp_path / "sim"), "--initial", "6",
          "--batch", "3", "--rounds", "2", "--repeats", "2", "--block-m", "500", "--test-fraction", "0.34",
          "--params", '{"n_estimators": 30, "n_jobs": 1}', "--no-figure"])
    meta = json.loads((tmp_path / "sim" / "simulation.json").read_text())
    assert meta["mode"] == "labelled pool" and (tmp_path / "sim" / "alc.csv").exists()


def test_figure_written_when_matplotlib_available(tmp_path):
    pytest.importorskip("matplotlib")
    pool = al.synthetic_pool(n_side=12, seed=2)
    meta = al.simulate(pool, tmp_path, initial=6, batch=4, rounds=2, repeats=2, block_m=800,
                       params={"n_estimators": 20, "n_jobs": 1})
    assert (tmp_path / "budget_curves.png").stat().st_size > 1000
    assert meta["outputs"]["figure"].endswith("budget_curves.png")


def test_cli_select_and_ingest(campaign, tmp_path):
    from geocrop.cli import main
    sel = tmp_path / "round1" / "select.gpkg"
    main(["active-learning", "select", "--parcels", str(campaign["parcels"]), "--out", str(sel), "--budget", "5",
          "--min-spacing-m", "150", "--training", str(campaign["training"]), "--exclude", str(campaign["validation"]),
          "--trips", "2"])
    picked = gpd.read_file(sel)
    assert len(picked) == 5
    assert not {"P0000", "P0001", "P0002", "P0505"} & set(picked.parcel_id)
    form = pd.read_csv(tmp_path / "round1" / "select_label_form.csv", dtype=str)
    truth = gpd.read_file(campaign["parcels"]).set_index("parcel_id").true_class
    form["label"] = [truth[p] for p in form.parcel_id]
    form["source"], form["date"], form["confidence"] = "field", "2024-06-10", "1"
    form.to_csv(tmp_path / "round1" / "labels.csv", index=False)
    main(["active-learning", "ingest", "--labels", str(tmp_path / "round1" / "labels.csv"), "--parcels",
          str(campaign["parcels"]), "--training", str(campaign["training"]), "--out", str(tmp_path / "t1.gpkg"),
          "--round", "1", "--validation", str(campaign["validation"])])
    assert len(gpd.read_file(tmp_path / "t1.gpkg")) == 3 + 5
