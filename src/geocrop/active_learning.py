"""Closed active-learning loop: batch selection, label ingestion with provenance, and budget simulation.

`geocrop.fieldwork.rank` orders parcels by predictive uncertainty (entropy, ambiguous conformal sets,
extrapolation outside the area of applicability). A field campaign, however, visits a *batch* of parcels,
and the most uncertain parcels are often near-duplicates (same crop, same confusion, same neighbourhood).
This module completes the loop:

* `select`    batch selection: the top-uncertain candidates are thinned by k-center greedy in a standardised
              feature space (diversity; Sener & Savarese, 2018), stratified by predicted class, with a minimum
              distance between selected parcels and optional grouping into field trips by area.
* `ingest`    new field observations / analyst decisions (parcel id, label, source, date, confidence) are
              merged into the training polygons with per-round provenance; label disagreements are reported and
              validation observations can never enter the training set.
* `simulate`  retrospective simulation on a fully labelled pool: uncertainty+diversity vs pure uncertainty vs
              random sampling over rounds, all scored on the same spatially separated test set, repeated over
              seeds -> label-budget curves (macro-F1 vs number of labelled parcels) with confidence intervals.

A simulation counts labelled parcels; it does not model travel time, access, or observer error, so it is not a
field-cost estimate. Uncertainty sampling background: Settles (2009), Tuia et al. (2011).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .fieldwork import priority_scores

STRATEGIES = ("uncertainty_diversity", "uncertainty", "random")
STRATIFY = ("equal", "proportional", "none")
NO_LABEL = frozenset({"", "nan", "none", "null", "unknown", "cant_tell", "can't tell", "cannot_tell", "?"})
LABEL_TEMPLATE_COLUMNS = ("parcel_id", "label", "source", "date", "confidence", "purpose", "observer", "notes",
                          "al_round")


# --------------------------------------------------------------------------------------------- selection core

def standardise(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Z-scores using the mean/std of `reference` (default X); constant and all-missing columns are dropped."""
    X = np.asarray(X, dtype=float)
    ref = X if reference is None else np.asarray(reference, dtype=float)
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
    keep = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    return np.nan_to_num(Z, nan=0.0)


def class_quota(strata, budget: int, mode: str = "equal") -> dict:
    """Budget split over strata (predicted classes), capped by availability; spare budget is redistributed.

    `equal` gives every predicted class the same share (rare classes are not starved); `proportional` follows the
    candidate frequencies (largest remainder). `none` returns no quota.
    """
    if mode not in STRATIFY:
        raise ValueError(f"stratify must be one of {STRATIFY}")
    strata = pd.Series(np.asarray(strata).astype(str))
    avail = strata.value_counts().to_dict()
    if mode == "none" or not avail:
        return {}
    budget = min(int(budget), int(sum(avail.values())))
    quota = dict.fromkeys(avail, 0)
    if mode == "proportional":
        total = sum(avail.values())
        raw = {c: budget * n / total for c, n in avail.items()}
        quota = {c: min(avail[c], int(math.floor(v))) for c, v in raw.items()}
        for c in sorted(raw, key=lambda c: raw[c] - math.floor(raw[c]), reverse=True):
            if sum(quota.values()) >= budget:
                break
            if quota[c] < avail[c]:
                quota[c] += 1
    left = budget - sum(quota.values())
    while left > 0:   # water-filling: one slot at a time to the least-served class that still has candidates
        open_ = [c for c in sorted(avail) if quota[c] < avail[c]]
        if not open_:
            break
        c = min(open_, key=lambda c: (quota[c], c))
        quota[c] += 1
        left -= 1
    return quota


@dataclass
class BatchResult:
    order: list[int]                    # indices into the candidate arrays, in selection order
    feature_distance: list[float]       # standardised feature distance to the nearest labelled/selected parcel
    quota: dict = field(default_factory=dict)
    quota_relaxed: bool = False
    shortfall: int = 0


def kcenter_batch(Z: np.ndarray, budget: int, *, priority=None, centres: np.ndarray | None = None,
                  uncertainty_weight: float = 0.3, coords: np.ndarray | None = None, min_spacing_m: float = 0.0,
                  fixed_coords: np.ndarray | None = None, strata=None, stratify: str = "none") -> BatchResult:
    """Greedy k-center selection among candidates with spatial spacing and per-stratum quotas.

    Each step takes the eligible candidate maximising ``(1 - w) * d / max(d) + w * priority / max(priority)``
    where ``d`` is the distance (standardised features) to the nearest labelled or already selected parcel.
    The first pick (when nothing is labelled yet) is the most uncertain candidate. Eligible = not chosen, at
    least `min_spacing_m` from every selected parcel (and from `fixed_coords`, e.g. already scheduled visits),
    and within its stratum's quota. Quotas are soft: if they cannot be filled under the spacing rule the
    remaining budget is filled ignoring them (`quota_relaxed`); spacing is never relaxed.
    """
    n = len(Z)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not 0 <= uncertainty_weight <= 1:
        raise ValueError("uncertainty_weight must be within [0, 1]")
    pr = np.zeros(n) if priority is None else np.asarray(priority, dtype=float)
    pr = np.nan_to_num(pr - np.nanmin(pr) if n else pr, nan=0.0)
    prn = pr / pr.max() if n and pr.max() > 0 else np.zeros(n)
    d = np.full(n, np.inf)
    if centres is not None and len(centres):
        for c in np.asarray(centres, dtype=float):
            d = np.minimum(d, np.linalg.norm(Z - c, axis=1))
    sd = np.full(n, np.inf)
    if min_spacing_m and coords is None:
        raise ValueError("min_spacing_m needs coordinates")
    if min_spacing_m and fixed_coords is not None:
        for c in np.asarray(fixed_coords, dtype=float):
            sd = np.minimum(sd, np.hypot(*(coords - c).T))
    strata = np.asarray(strata).astype(str) if strata is not None else np.full(n, "all")
    quota = class_quota(strata, budget, stratify)
    taken: dict[str, int] = {}
    chosen = np.zeros(n, dtype=bool)
    order, dist = [], []
    relaxed, n_relaxed = False, 0
    while len(order) < budget:
        ok = ~chosen & (sd >= min_spacing_m) if min_spacing_m else ~chosen
        if quota and not relaxed:
            within = np.array([taken.get(s, 0) < quota.get(s, 0) for s in strata], dtype=bool)
            if not (ok & within).any():
                relaxed, n_relaxed = True, len(order)
            else:
                ok = ok & within
        if not ok.any():
            break
        idx = np.flatnonzero(ok)
        finite = np.isfinite(d[idx])
        if not finite.any():
            pick = idx[np.argmax(prn[idx])]
        else:
            dd = np.where(finite, d[idx], 0.0)
            scale = dd.max() if dd.max() > 0 else 1.0
            score = (1 - uncertainty_weight) * dd / scale + uncertainty_weight * prn[idx]
            pick = idx[np.argmax(score)]
        order.append(int(pick))
        dist.append(float(d[pick]))
        chosen[pick] = True
        taken[strata[pick]] = taken.get(strata[pick], 0) + 1
        d = np.minimum(d, np.linalg.norm(Z - Z[pick], axis=1))
        if min_spacing_m:
            sd = np.minimum(sd, np.hypot(*(coords - coords[pick]).T))
    return BatchResult(order, dist, quota, relaxed and len(order) > n_relaxed, budget - len(order))


def plan_trips(coords: np.ndarray, trips: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict]:
    """Group selected parcels into `trips` areas (k-means on coordinates) and order each by nearest neighbour.

    Returns 1-based trip ids, 1-based visit order within the trip, and straight-line route length per trip (km;
    centroid to centroid, no road network, start/end depot not included).
    """
    from sklearn.cluster import KMeans

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), {}
    k = max(1, min(int(trips), n))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords) if k > 1 else np.zeros(n, int)
    # renumber trips west -> east by centroid so ids are stable and readable on a map
    cx = {t: coords[labels == t, 0].mean() for t in np.unique(labels)}
    remap = {t: i + 1 for i, t in enumerate(sorted(cx, key=cx.get))}
    trip = np.array([remap[t] for t in labels])
    order = np.zeros(n, int)
    length = {}
    for t in np.unique(trip):
        members = list(np.flatnonzero(trip == t))
        cur = min(members, key=lambda i: (coords[i, 0], coords[i, 1]))
        route, km = [cur], 0.0
        members.remove(cur)
        while members:
            nxt = min(members, key=lambda i: np.hypot(*(coords[i] - coords[cur])))
            km += float(np.hypot(*(coords[nxt] - coords[cur]))) / 1000
            route.append(nxt)
            members.remove(nxt)
            cur = nxt
        for pos, i in enumerate(route, 1):
            order[i] = pos
        length[int(t)] = round(km, 3)
    return trip, order, length


# --------------------------------------------------------------------------------------------- select (CLI)

def _read_any(path_or_frame, as_text: bool = False) -> pd.DataFrame:
    """CSV/Parquet as DataFrame, vector formats as GeoDataFrame, frames copied (`as_text`: CSV cells as str)."""
    if isinstance(path_or_frame, pd.DataFrame):
        return path_or_frame.copy()
    p = Path(path_or_frame)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""]) if as_text else pd.read_csv(p)
    return gpd.read_file(p)


def _has_geometry(t) -> bool:
    return isinstance(t, gpd.GeoDataFrame) and "geometry" in t and t.geometry.notna().any()


def _read_table(path_or_frame, id_field: str, as_text: bool = False) -> pd.DataFrame:
    t = _read_any(path_or_frame, as_text)
    if "geometry" in t:
        t = pd.DataFrame(t.drop(columns="geometry"))
    if id_field not in t:
        raise ValueError(f"Table needs a {id_field!r} column")
    t[id_field] = t[id_field].astype(str)
    return t


def _feature_matrix(gdf: gpd.GeoDataFrame, features, feature_columns, id_field: str) -> tuple[np.ndarray, list[str]]:
    if features is not None:
        t = _read_table(features, id_field)
        if t[id_field].duplicated().any():
            raise ValueError(f"Feature table has duplicate {id_field} values")
        cols = list(feature_columns) if feature_columns else [c for c in t.columns
                                                               if c != id_field and pd.api.types.is_numeric_dtype(t[c])]
        t = t.set_index(id_field).reindex(gdf[id_field].astype(str))
    else:
        t = gdf
        cols = list(feature_columns) if feature_columns else [c for c in gdf.columns if c.startswith("p_")]
    missing = [c for c in cols if c not in t]
    if missing:
        raise ValueError(f"Feature columns not found: {missing[:5]}")
    if not cols:
        raise ValueError("No feature columns: pass --features (table keyed by parcel id) or parcels with p_<class> "
                         "probability columns (output of geocrop parcels)")
    return t[cols].to_numpy(dtype=float), cols


def select(parcels: str | Path | gpd.GeoDataFrame, budget: int, output: str | Path | None = None, *,
           features=None, feature_columns=None, id_field: str = "parcel_id", candidate_factor: float = 3.0,
           uncertainty_weight: float = 0.3, min_spacing_m: float = 1000.0, stratify: str = "equal",
           trips: int | None = None, labelled_ids=(), exclude_ids=(), scheduled=None, round_id: int = 1,
           seed: int = 42) -> tuple[gpd.GeoDataFrame, dict]:
    """Select one field-campaign batch of `budget` parcels (uncertain, diverse, stratified, spatially spread).

    parcels       output of `geocrop parcels` (needs `entropy`; uses `set_size`/`aoa_fraction` when present)
    features      optional table keyed by `id_field` (CSV/Parquet/GPKG, e.g. parcel signatures or markers); its
                  numeric columns span the diversity space. Default: the parcel's mean class probabilities `p_*`.
    labelled_ids  parcels already in the training set: excluded, and used as k-center centres so the batch
                  covers feature space the model has not seen.
    exclude_ids   further parcels never to propose (e.g. validation parcels, inaccessible fields).
    scheduled     optional vector file of visits already planned: selected parcels keep `min_spacing_m` from them.
    trips         group the batch into this many field trips by area and order each trip.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if candidate_factor < 1:
        raise ValueError("candidate_factor must be >= 1")
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf:
        raise ValueError(f"Parcels need a {id_field!r} column")
    if "entropy" not in gdf:
        raise ValueError("Parcels need an 'entropy' column (run geocrop parcels on a probability raster)")
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError("Parcels need a projected CRS")
    gdf = gdf.reset_index(drop=True)
    gdf[id_field] = gdf[id_field].astype(str)
    X, cols = _feature_matrix(gdf, features, feature_columns, id_field)
    has_features = np.isfinite(X).any(axis=1)
    Z_all = standardise(X)
    labelled = gdf[id_field].isin({str(i) for i in labelled_ids}).to_numpy()
    excluded = gdf[id_field].isin({str(i) for i in exclude_ids}).to_numpy()
    valid = gdf["n_valid"].fillna(0).to_numpy() > 0 if "n_valid" in gdf else np.ones(len(gdf), bool)
    pool = valid & ~labelled & ~excluded & has_features
    priority = priority_scores(gdf)
    pool_idx = np.flatnonzero(pool)
    pool_idx = pool_idx[np.argsort(-priority[pool_idx], kind="stable")]
    n_cand = min(len(pool_idx), max(budget, int(math.ceil(candidate_factor * budget))))
    cand = pool_idx[:n_cand]
    cls_col = "predicted_class" if "predicted_class" in gdf else "mean_class" if "mean_class" in gdf else None
    strata = (gdf[cls_col].astype(object).where(gdf[cls_col].notna(), "unclassified").astype(str).to_numpy()
              if cls_col else np.full(len(gdf), "all"))
    pts = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    fixed = None
    if scheduled is not None:
        sch = gpd.read_file(scheduled) if not isinstance(scheduled, gpd.GeoDataFrame) else scheduled
        sch = sch.to_crs(gdf.crs)
        fixed = np.column_stack([sch.geometry.centroid.x, sch.geometry.centroid.y])
    res = kcenter_batch(Z_all[cand], budget, priority=priority[cand], centres=Z_all[labelled & has_features],
                        uncertainty_weight=uncertainty_weight, coords=pts[cand], min_spacing_m=min_spacing_m,
                        fixed_coords=fixed, strata=strata[cand], stratify=stratify)
    rows = cand[res.order]
    picked = gdf.iloc[rows].copy()
    picked["priority"] = priority[rows]
    picked["feature_distance"] = [None if not np.isfinite(v) else round(v, 4) for v in res.feature_distance]
    picked["stratum"] = strata[rows]
    picked["al_round"] = int(round_id)
    picked["al_rank"] = np.arange(1, len(rows) + 1)
    trip_length = {}
    if trips and len(rows):
        t, o, trip_length = plan_trips(pts[rows], trips, seed)
        picked["trip_id"], picked["trip_order"] = t, o
    min_gap = None
    if len(rows) > 1:
        from scipy.spatial.distance import pdist
        min_gap = float(pdist(pts[rows]).min())
    summary = {
        "round": int(round_id), "budget": int(budget), "selected": int(len(rows)), "shortfall": int(res.shortfall),
        "pool": int(pool.sum()), "candidates": int(len(cand)), "labelled_centres": int((labelled & has_features).sum()),
        "excluded": int(excluded.sum()), "without_features": int((~has_features).sum()),
        "feature_space": cols if len(cols) <= 20 else cols[:20] + [f"... ({len(cols)} columns)"],
        "stratify": stratify, "quota": res.quota, "quota_relaxed": bool(res.quota_relaxed),
        "selected_per_stratum": picked["stratum"].value_counts().sort_index().to_dict(),
        "min_spacing_m": min_spacing_m, "min_distance_between_selected_m": None if min_gap is None else round(min_gap, 1),
        "mean_priority": float(picked["priority"].mean()) if len(rows) else None,
        "trips": trip_length, "trip_length_note": "straight-line km between parcel centroids, not road distance",
    }
    if res.shortfall:
        summary["hint"] = ("Fewer parcels than the budget satisfy the spacing rule among the candidates; lower "
                           "--min-spacing-m or raise --candidate-factor.")
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        picked.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
        form = pd.DataFrame({"parcel_id": picked[id_field].to_numpy()})
        for c in LABEL_TEMPLATE_COLUMNS[1:]:
            form[c] = ""
        form["purpose"], form["al_round"] = "training", int(round_id)
        if "trip_id" in picked:
            form["trip_id"], form["trip_order"] = picked["trip_id"].to_numpy(), picked["trip_order"].to_numpy()
        form_path = out.with_name(out.stem + "_label_form.csv")
        form.to_csv(form_path, index=False)
        summary["out"], summary["label_form"] = str(out), str(form_path)
        out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    return picked, summary


# --------------------------------------------------------------------------------------------- ingestion

def read_labels(path_or_frame, id_field: str = "parcel_id") -> pd.DataFrame:
    """Field observations / analyst decisions: parcel id, label, source, date, confidence (+ optional purpose).

    `purpose` is `training` (default) or `validation`. Missing `source` becomes "unknown"; missing confidence
    stays empty (accepted only when no minimum confidence is required).
    """
    t = _read_table(path_or_frame, id_field, as_text=True)
    if "label" not in t:
        raise ValueError("Label table needs a 'label' column")
    for col, default in (("source", "unknown"), ("date", ""), ("purpose", "training")):
        if col not in t:
            t[col] = default
        t[col] = t[col].fillna(default).astype(str).str.strip()
    t["purpose"] = t["purpose"].str.lower().replace("", "training")
    bad = sorted(set(t["purpose"]) - {"training", "validation"})
    if bad:
        raise ValueError(f"purpose must be 'training' or 'validation', got {bad}")
    t["confidence"] = pd.to_numeric(t["confidence"], errors="coerce") if "confidence" in t else np.nan
    t["label"] = t["label"].astype(object).where(t["label"].notna(), "").astype(str).str.strip()
    return t.reset_index(drop=True)


def validation_parcel_ids(validation, parcels: gpd.GeoDataFrame, id_field: str, buffer_m: float) -> set[str]:
    if validation is None:
        return set()
    v = _read_any(validation, as_text=True)
    if not _has_geometry(v):
        if id_field not in v:
            raise ValueError(f"Validation table needs geometries or a {id_field!r} column")
        return set(v[id_field].astype(str))
    ids = set(v[id_field].astype(str)) if id_field in v else set()
    geoms = v.to_crs(parcels.crs).geometry
    if buffer_m:
        geoms = geoms.buffer(buffer_m)
    hit = gpd.sjoin(parcels[[id_field, "geometry"]], gpd.GeoDataFrame(geometry=geoms, crs=parcels.crs),
                    how="inner", predicate="intersects")
    return ids | set(hit[id_field].astype(str))


def training_parcel_ids(training: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame, id_field: str) -> pd.Series:
    """Parcel id of every training polygon (own column, else the parcel containing its representative point)."""
    if id_field in training:
        return training[id_field].astype(object).where(training[id_field].notna(), None)
    pts = gpd.GeoDataFrame(geometry=training.geometry.representative_point(), crs=training.crs).to_crs(parcels.crs)
    hit = gpd.sjoin(pts, parcels[[id_field, "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    return hit[id_field].reindex(training.index).astype(object).where(lambda s: s.notna(), None)


def _resolve(group: pd.DataFrame, policy: str) -> tuple[str | None, str]:
    labels = set(group["label"])
    if len(labels) == 1:
        return group["label"].iloc[0], ""
    if policy == "highest_confidence":
        conf = group["confidence"].fillna(-1)
        top = group[conf == conf.max()]
        if len(set(top["label"])) == 1:
            return top["label"].iloc[0], "resolved_by_confidence"
    elif policy == "latest":
        dates = pd.to_datetime(group["date"], errors="coerce", utc=True)
        if dates.notna().any():
            top = group[dates == dates.max()]
            if len(set(top["label"])) == 1:
                return top["label"].iloc[0], "resolved_by_date"
    elif policy == "majority":
        counts = group["label"].value_counts()
        if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
            return counts.index[0], "resolved_by_majority"
    return None, "unresolved"


def ingest(labels, parcels, training, output, round_id: int, *, validation=None, provenance=None,
           id_field: str = "parcel_id", label_field: str = "class_name", polygon_id_field: str = "polygon_id",
           min_confidence: float = 0.0, conflict_policy: str = "exclude", existing_policy: str = "keep",
           validation_buffer_m: float = 0.0, inner_buffer_m: float = 0.0) -> dict:
    """Merge one round of new labels into the training polygons, with provenance and validation separation.

    labels           read by `read_labels` (CSV/GPKG/Parquet/DataFrame)
    parcels          parcel geometries keyed by `id_field` (new training polygons are taken from here)
    training         current training polygons (`polygon_id`, `class_name`, geometry) as used by `samples.extract`
    output           updated training polygons (may equal `training`)
    validation       independent validation observations (points/polygons, or a table of parcel ids). Every parcel
                     they touch (plus `validation_buffer_m`) and every label row with purpose=validation is kept
                     out of training. If the *existing* training set already contains such a parcel the run stops.
    conflict_policy  disagreement between labels of one parcel within this round: `exclude` (default; report and
                     add nothing), `highest_confidence`, `latest`, `majority` (unresolved ties are excluded)
    existing_policy  new label disagrees with the existing training label: `keep` (default; report) or `replace`
    provenance       CSV log appended per round (default: <output dir>/al_provenance.csv); a round id can be
                     ingested once.
    """
    if conflict_policy not in ("exclude", "highest_confidence", "latest", "majority"):
        raise ValueError("conflict_policy must be exclude, highest_confidence, latest or majority")
    if existing_policy not in ("keep", "replace"):
        raise ValueError("existing_policy must be keep or replace")
    round_id = int(round_id)
    if round_id < 1:
        raise ValueError("round_id must be >= 1 (round 0 denotes the initial training set)")
    out = Path(output)
    prov_path = Path(provenance) if provenance else out.parent / "al_provenance.csv"
    old_prov = pd.read_csv(prov_path, dtype={"parcel_id": str}) if prov_path.exists() else None
    if old_prov is not None and (old_prov["round"] == round_id).any():
        raise ValueError(f"Round {round_id} is already recorded in {prov_path}; use a new round id")
    lab = read_labels(labels, id_field)
    par = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in par:
        raise ValueError(f"Parcels need a {id_field!r} column")
    par[id_field] = par[id_field].astype(str)
    if par[id_field].duplicated().any():
        raise ValueError(f"Parcel {id_field} values must be unique")
    tr = gpd.read_file(training) if not isinstance(training, gpd.GeoDataFrame) else training.copy()
    for col in (label_field, polygon_id_field):
        if col not in tr:
            raise ValueError(f"Training polygons need a {col!r} column")
    tr = tr.reset_index(drop=True)
    tr_pid = training_parcel_ids(tr, par, id_field)
    val_ids = validation_parcel_ids(validation, par, id_field, validation_buffer_m)
    val_ids |= set(lab.loc[lab["purpose"] == "validation", id_field])
    leaked = sorted({p for p in tr_pid.dropna() if p in val_ids})
    if validation is not None:   # unmapped training polygons: check geometry directly
        v = _read_any(validation)
        if _has_geometry(v):
            vg = v.to_crs(tr.crs).geometry
            vg = vg.buffer(validation_buffer_m) if validation_buffer_m else vg
            hit = gpd.sjoin(tr[["geometry"]], gpd.GeoDataFrame(geometry=vg, crs=tr.crs), predicate="intersects")
            leaked = sorted(set(leaked) | {f"{polygon_id_field}={tr.loc[i, polygon_id_field]}" for i in hit.index.unique()})
    if leaked:
        raise ValueError(f"Existing training data overlaps validation observations ({len(leaked)}: {leaked[:5]}); "
                         "remove them from training before ingesting new labels")
    existing = {str(p): str(c) for p, c in zip(tr_pid, tr[label_field]) if p is not None}
    known = set(par[id_field])
    action = pd.Series("", index=lab.index, dtype=object)
    detail = pd.Series("", index=lab.index, dtype=object)
    previous = pd.Series("", index=lab.index, dtype=object)
    action[~lab[id_field].isin(known)] = "unknown_parcel"
    action[(action == "") & lab["label"].str.lower().isin(NO_LABEL)] = "no_label"
    action[(action == "") & (lab["purpose"] == "validation")] = "validation_reserved"
    action[(action == "") & lab[id_field].isin(val_ids)] = "validation_excluded"
    if min_confidence > 0:
        low = (action == "") & ~(lab["confidence"].fillna(-np.inf) >= min_confidence)
        action[low] = "low_confidence"
    conflicts = []
    accepted: dict[str, tuple[str, int]] = {}
    for pid, g in lab[action == ""].groupby(id_field, sort=False):
        chosen, how = _resolve(g, conflict_policy)
        if len(set(g["label"])) > 1:
            conflicts.append({"parcel_id": pid, "type": "within_round", "labels": sorted(set(g["label"])),
                              "sources": sorted(set(g["source"])), "resolution": chosen or "excluded"})
        if chosen is None:
            action[g.index] = "conflict_excluded"
            detail[g.index] = "labels disagree: " + "|".join(sorted(set(g["label"])))
            continue
        keep = g.index[g["label"] == chosen]
        if how == "resolved_by_confidence":
            keep = [g.loc[keep, "confidence"].fillna(-1).idxmax()]
        first = keep[0]
        action[g.index.difference([first])] = np.where(g.loc[g.index.difference([first]), "label"] == chosen,
                                                       "duplicate_in_round", "overruled")
        detail[g.index] = how
        old = existing.get(pid)
        if old is not None and old == chosen:
            action[first] = "already_labelled"
        elif old is not None:
            previous[first] = old
            conflicts.append({"parcel_id": pid, "type": "with_training", "labels": [old, chosen],
                              "sources": sorted(set(g["source"])),
                              "resolution": chosen if existing_policy == "replace" else old})
            if existing_policy == "replace":
                action[first] = "replaced"
                accepted[pid] = (chosen, first)
            else:
                action[first] = "conflict_kept_existing"
        else:
            action[first] = "added"
            accepted[pid] = (chosen, first)
    # build the new training table
    for col, default in (("al_round", 0), ("label_source", "initial"), ("label_date", ""), ("label_confidence", np.nan)):
        if col not in tr:
            tr[col] = default
    if id_field not in tr:
        tr[id_field] = tr_pid.to_numpy()
    replaced = {pid for pid, (_, i) in accepted.items() if action[i] == "replaced"}
    if replaced:
        tr = tr[~tr_pid.isin(replaced).to_numpy()]
    geo = par.set_index(id_field).to_crs(tr.crs).geometry
    new_rows = []
    for pid, (label, i) in accepted.items():
        geom = geo[pid]
        if inner_buffer_m:
            geom = geom.buffer(-abs(inner_buffer_m))
        if geom is None or geom.is_empty:
            action[i], detail[i] = "empty_after_buffer", f"inner_buffer_m={inner_buffer_m}"
            continue
        r = lab.loc[i]
        new_rows.append({polygon_id_field: f"AL{round_id}_{pid}", label_field: label, id_field: pid,
                         "al_round": round_id, "label_source": r["source"], "label_date": r["date"],
                         "label_confidence": r["confidence"], "geometry": geom})
    merged = pd.concat([tr, gpd.GeoDataFrame(new_rows, geometry="geometry", crs=tr.crs)], ignore_index=True) \
        if new_rows else tr.reset_index(drop=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=tr.crs)
    if merged[polygon_id_field].astype(str).duplicated().any():
        raise ValueError(f"Duplicate {polygon_id_field} after merge; check earlier rounds")
    final_pids = set(merged[id_field].dropna().astype(str))
    assert not final_pids & val_ids, "validation parcels reached the training set"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    merged.to_file(out, driver="GPKG" if out.suffix.lower() == ".gpkg" else "GeoJSON")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prov = pd.DataFrame({"round": round_id, "parcel_id": lab[id_field], "label": lab["label"], "source": lab["source"],
                         "date": lab["date"], "confidence": lab["confidence"], "purpose": lab["purpose"],
                         "action": action, "detail": detail, "previous_label": previous, "ingested_at": now})
    prov = pd.concat([old_prov, prov], ignore_index=True) if old_prov is not None else prov
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov.to_csv(prov_path, index=False)
    reserved = lab[action == "validation_reserved"]
    report = {
        "round": round_id, "labels_read": int(len(lab)),
        "actions": {k: int(v) for k, v in action.value_counts().sort_index().items()},
        "training_before": int(len(tr) + len(replaced)), "training_after": int(len(merged)),
        "added": int((action == "added").sum()), "replaced": int((action == "replaced").sum()),
        "conflicts": conflicts, "validation_parcels_protected": int(len(val_ids)),
        "classes_after": merged[label_field].astype(str).value_counts().sort_index().to_dict(),
        "out": str(out), "provenance": str(prov_path),
    }
    if len(reserved):
        vpath = prov_path.with_name(f"validation_labels_round{round_id}.csv")
        reserved.to_csv(vpath, index=False)
        report["validation_labels"] = str(vpath)
    (prov_path.parent / f"ingest_report_round{round_id}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf8")
    return report


# --------------------------------------------------------------------------------------------- simulation

@dataclass
class Pool:
    """One row per labelled reference unit (parcel/polygon): features, label, centroid."""
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    x: np.ndarray
    yy: np.ndarray
    features: list[str]
    crs: str
    synthetic: bool = False


def pool_from_samples(samples) -> Pool:
    """Aggregate pixel samples (`geocrop.samples.Samples`) to polygon means (one unit = one field visit)."""
    frame = pd.DataFrame(samples.X, columns=samples.features)
    frame["_p"], frame["_x"], frame["_y"] = samples.polygon, samples.x, samples.yy
    g = frame.groupby("_p", sort=True)
    agg = g[samples.features].mean()
    lab = pd.Series(samples.y, index=samples.polygon).groupby(level=0).agg(lambda s: s.mode().iloc[0])
    xy = g[["_x", "_y"]].mean()
    return Pool(agg.to_numpy(float), lab.reindex(agg.index).to_numpy().astype(str), agg.index.to_numpy().astype(str),
                xy["_x"].to_numpy(), xy["_y"].to_numpy(), list(samples.features), samples.crs)


# SYNTHETIC class signatures: NDVI in four seasonal windows, SWIR offset, VH offset (dB) — mirrors geocrop.demo
_SYN = {
    "winter_crops": ((0.75, 0.55, 0.15, 0.20), 0.00, -1.0, 0.24),
    "maize": ((0.20, 0.60, 0.82, 0.50), 0.02, 1.5, 0.28),
    "soybean": ((0.15, 0.50, 0.85, 0.35), -0.03, 0.0, 0.18),
    "sunflower": ((0.20, 0.72, 0.55, 0.22), 0.04, 2.0, 0.12),
    "grassland": ((0.55, 0.62, 0.55, 0.50), 0.01, -1.5, 0.12),
    "forest": ((0.80, 0.86, 0.85, 0.75), -0.05, 3.0, 0.06),
}


def synthetic_pool(n_side: int = 24, spacing_m: float = 250.0, noise: float = 0.10, seed: int = 7) -> Pool:
    """SYNTHETIC parcel pool on a regular grid (imbalanced classes, regional offsets, label-free noise).

    Exists to exercise and test the simulation; numbers obtained on it say nothing about real performance.
    """
    rng = np.random.default_rng(seed)
    names = list(_SYN)
    share = np.array([_SYN[c][3] for c in names])
    n = n_side * n_side
    k = rng.choice(len(names), size=n, p=share / share.sum())
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    x = 600000.0 + gx.ravel() * spacing_m + spacing_m / 2
    yy = 4980000.0 - gy.ravel() * spacing_m - spacing_m / 2
    # smooth regional offset (sowing date / soil effects): spatial autocorrelation the test split must respect
    region = 0.06 * np.sin(gx.ravel() / n_side * 2 * np.pi) * np.cos(gy.ravel() / n_side * 2 * np.pi)
    ndvi = np.array([_SYN[names[i]][0] for i in k]) + region[:, None] + rng.normal(0, noise, (n, 4))
    swir = np.array([_SYN[names[i]][1] for i in k]) + rng.normal(0, noise / 3, n)
    vh = -23 + 9 * ndvi + np.array([_SYN[names[i]][2] for i in k])[:, None] + rng.normal(0, 2.0, (n, 4))
    X = np.column_stack([np.clip(ndvi, 0.02, 0.95), swir, vh])
    feats = [f"NDVI_P{i}" for i in range(1, 5)] + ["SWIR_offset"] + [f"VH_P{i}" for i in range(1, 5)]
    ids = np.array([f"S{i:05d}" for i in range(n)])
    return Pool(X, np.array(names)[k], ids, x, yy, feats, "EPSG:32634", synthetic=True)


def spatial_test_split(pool: Pool, block_m: float, test_fraction: float = 0.25, buffer_m: float = 0.0,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed test set of whole spatial blocks (class-stratified where possible) and the purged labelling pool.

    Pool units closer than `buffer_m` to any test unit are removed, so neither strategy can label a test
    unit's neighbour. Returns (test_idx, pool_idx).
    """
    from scipy.spatial import cKDTree
    from sklearn.model_selection import StratifiedGroupKFold

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be within (0, 1)")
    blocks = (np.floor(pool.x / block_m).astype(int).astype(str).astype(object) + ":" +
              np.floor(pool.yy / block_m).astype(int).astype(str).astype(object))
    k = max(2, int(round(1 / test_fraction)))
    if len(np.unique(blocks)) < k:
        raise ValueError(f"Need at least {k} spatial blocks of {block_m} m; reduce block_m")
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, test = next(splitter.split(np.zeros(len(pool.y)), pool.y, blocks))
    is_test = np.zeros(len(pool.y), bool)
    is_test[test] = True
    rest = np.flatnonzero(~is_test)
    if buffer_m > 0:
        tree = cKDTree(np.column_stack([pool.x[test], pool.yy[test]]))
        dist, _ = tree.query(np.column_stack([pool.x[rest], pool.yy[rest]]))
        rest = rest[dist >= buffer_m]
    return np.sort(test), rest


def _entropy(proba: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1) / math.log(max(n_classes, 2))


def _initial(y: np.ndarray, idx: np.ndarray, n: int, rng) -> np.ndarray:
    """Random initial labels with at least one unit per class (when n allows): the existing reference set."""
    chosen = []
    classes = rng.permutation(np.unique(y[idx]))
    for c in classes[:n]:
        chosen.append(rng.choice(idx[y[idx] == c]))
    rest = np.setdiff1d(idx, chosen)
    chosen += list(rng.choice(rest, size=max(0, min(n - len(chosen), len(rest))), replace=False))
    return np.array(sorted(chosen))


def _ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    from scipy import stats

    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), m, m
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * sd / math.sqrt(len(v)))
    return m, sd, m - h, m + h


def simulate(pool: Pool, output_dir: str | Path, *, strategies=STRATEGIES, initial: int = 12, batch: int = 12,
             rounds: int = 8, repeats: int = 5, seed: int = 42, params: dict | None = None, block_m: float = 1000.0,
             test_fraction: float = 0.25, test_buffer_m: float = 0.0, candidate_factor: float = 3.0,
             uncertainty_weight: float = 0.3, min_spacing_m: float = 0.0, stratify: str = "none",
             figure: bool = True) -> dict:
    """Retrospective active-learning simulation -> label-budget curves with repeat-level confidence intervals.

    The test set (whole spatial blocks, optional buffer) is fixed once and shared by all strategies and repeats.
    Each repeat draws one initial labelled set shared by all strategies (paired comparison); each round the
    model is refit on the labelled units and scored (macro-F1, OA) on the test set, then `batch` units are
    revealed according to the strategy. Writes curves.csv (per repeat), summary.csv (mean, sd, 95 % t-CI per
    budget), alc.csv (area under the budget curve per repeat and paired difference to random),
    simulation.json and budget_curves.png (needs matplotlib).
    """
    from .estimators import make_estimator
    from .metrics import summary as score

    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies {unknown}; choose from {STRATEGIES}")
    if min(initial, batch, repeats) < 1 or rounds < 0:
        raise ValueError("initial, batch and repeats must be >= 1, rounds >= 0")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {"algorithm": "rf", "n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced",
              "n_jobs": -1, **(params or {})}
    test, avail = spatial_test_split(pool, block_m, test_fraction, test_buffer_m, seed)
    if len(avail) < initial + batch * rounds:
        raise ValueError(f"Pool has {len(avail)} units after removing the test set; need initial + batch * rounds = "
                         f"{initial + batch * rounds}")
    classes = sorted(np.unique(pool.y))
    Z = standardise(pool.X, pool.X[avail])
    coords = np.column_stack([pool.x, pool.yy])
    records = []
    for r in range(repeats):
        seed_r = seed + 1000 * r
        start = _initial(pool.y, avail, initial, np.random.default_rng(seed_r))
        for strat in strategies:
            rng = np.random.default_rng(seed_r + 1)
            labelled = start.copy()
            for rd in range(rounds + 1):
                model = make_estimator(params, seed_r)
                model.fit(pool.X[labelled], pool.y[labelled])
                pred = model.predict(pool.X[test])
                s = score(pool.y[test], pred, labels=classes)
                records.append({"strategy": strat, "repeat": r, "seed": seed_r, "round": rd,
                                "n_labelled": int(len(labelled)), "macro_f1": s["macro_f1"],
                                "overall_accuracy": s["overall_accuracy"],
                                "classes_labelled": int(len(np.unique(pool.y[labelled])))})
                if rd == rounds:
                    break
                unl = np.setdiff1d(avail, labelled)
                if strat == "random":
                    new = rng.choice(unl, size=batch, replace=False)
                else:
                    proba = model.predict_proba(pool.X[unl])
                    ent = _entropy(proba, len(classes))
                    o = np.argsort(-ent, kind="stable")
                    rank = unl[o]
                    if strat == "uncertainty":
                        new = rank[:batch]
                    else:
                        n_c = min(len(rank), max(batch, int(math.ceil(candidate_factor * batch))))
                        cand, ent_c = rank[:n_c], ent[o][:n_c]
                        strata = np.asarray(model.classes_)[proba[o][:n_c].argmax(1)]
                        res = kcenter_batch(Z[cand], batch, priority=ent_c, centres=Z[labelled],
                                            uncertainty_weight=uncertainty_weight, coords=coords[cand],
                                            min_spacing_m=min_spacing_m, fixed_coords=coords[labelled]
                                            if min_spacing_m else None, strata=strata, stratify=stratify)
                        new = cand[res.order]
                        if len(new) < batch:   # spacing left gaps: fill with the next most uncertain units
                            taken = set(new.tolist())
                            new = np.concatenate([new, [u for u in rank if u not in taken][:batch - len(new)]])
                labelled = np.sort(np.concatenate([labelled, new]))
    curves = pd.DataFrame(records)
    curves.to_csv(out / "curves.csv", index=False)
    rows = []
    for (strat, n_lab), g in curves.groupby(["strategy", "n_labelled"], sort=False):
        for metric in ("macro_f1", "overall_accuracy"):
            m, sd, lo, hi = _ci(g[metric].to_numpy())
            rows.append({"strategy": strat, "n_labelled": n_lab, "metric": metric, "mean": m, "sd": sd,
                         "ci95_low": lo, "ci95_high": hi, "repeats": len(g)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    alc_rows = []
    for (strat, r), g in curves.groupby(["strategy", "repeat"], sort=False):
        g = g.sort_values("n_labelled")
        span = g.n_labelled.max() - g.n_labelled.min()
        f, n_ = g.macro_f1.to_numpy(float), g.n_labelled.to_numpy(float)
        val = float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(n_)) / span) if span else float(f[0])
        alc_rows.append({"strategy": strat, "repeat": r, "alc_macro_f1": val,
                         "final_macro_f1": float(g.macro_f1.iloc[-1])})
    alc = pd.DataFrame(alc_rows)
    comparison = {}
    for strat in strategies:
        a = alc[alc.strategy == strat].sort_values("repeat")
        m, sd, lo, hi = _ci(a.alc_macro_f1.to_numpy())
        entry = {"alc_mean": round(m, 4), "alc_ci95": [round(lo, 4), round(hi, 4)],
                 "final_macro_f1_mean": round(float(a.final_macro_f1.mean()), 4)}
        if "random" in strategies and strat != "random":
            b = alc[alc.strategy == "random"].sort_values("repeat")
            diff = a.alc_macro_f1.to_numpy() - b.alc_macro_f1.to_numpy()
            dm, _, dlo, dhi = _ci(diff)
            entry["alc_minus_random"] = round(dm, 4)
            entry["alc_minus_random_ci95"] = [round(dlo, 4), round(dhi, 4)]
            for i, d in zip(a.index, diff):
                alc.loc[i, "alc_minus_random"] = d
        comparison[strat] = entry
    alc.to_csv(out / "alc.csv", index=False)
    meta = {
        "mode": "SYNTHETIC pool - software/method check, not real accuracy" if pool.synthetic else "labelled pool",
        "units_total": int(len(pool.y)), "test_units": int(len(test)), "pool_units": int(len(avail)),
        "purged_by_buffer": int(len(pool.y) - len(test) - len(avail)), "classes": classes,
        "test_class_counts": pd.Series(pool.y[test]).value_counts().sort_index().to_dict(),
        "protocol": {"strategies": list(strategies), "initial": initial, "batch": batch, "rounds": rounds,
                     "repeats": repeats, "seed": seed, "block_m": block_m, "test_fraction": test_fraction,
                     "test_buffer_m": test_buffer_m, "candidate_factor": candidate_factor,
                     "uncertainty_weight": uncertainty_weight, "min_spacing_m": min_spacing_m, "stratify": stratify,
                     "params": params, "features": pool.features},
        "comparison": comparison,
        "note": "x axis = labelled reference units (one unit ~ one field visit); travel, access and observer "
                "error are not simulated, so this is not a field-cost estimate.",
        "outputs": {"curves": str(out / "curves.csv"), "summary": str(out / "summary.csv"),
                    "alc": str(out / "alc.csv")},
    }
    if figure:
        meta["outputs"]["figure"] = plot_curves(summary, out / "budget_curves.png", synthetic=pool.synthetic)
    (out / "simulation.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf8")
    return meta


def plot_curves(summary: pd.DataFrame, path: str | Path, metric: str = "macro_f1", synthetic: bool = False) -> str | None:
    """Mean budget curve with 95 % CI band per strategy (PNG). Returns None when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn("Figure skipped: install matplotlib (pip install matplotlib) for budget_curves.png",
                      stacklevel=2)
        return None
    s = summary[summary.metric == metric]
    styles = {"uncertainty_diversity": ("#1b6ca8", "o"), "uncertainty": ("#d9822b", "s"), "random": ("#5b5b5b", "^")}
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    for strat, g in s.groupby("strategy", sort=False):
        g = g.sort_values("n_labelled")
        colour, marker = styles.get(strat, (None, "o"))
        ax.plot(g.n_labelled, g["mean"], marker=marker, ms=4, lw=1.6, color=colour, label=strat.replace("_", " + "))
        if g["ci95_low"].notna().any():
            ax.fill_between(g.n_labelled, g.ci95_low, g.ci95_high, color=colour, alpha=0.15, lw=0)
    ax.set_xlabel("labelled parcels (field visits)")
    ax.set_ylabel("macro-F1 on fixed spatial test set" if metric == "macro_f1" else metric.replace("_", " "))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    if synthetic:
        ax.set_title("SYNTHETIC data - method check only", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
