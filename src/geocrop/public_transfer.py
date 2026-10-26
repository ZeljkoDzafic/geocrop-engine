"""Frozen-protocol public TRANSFER benchmark: train in one region/year, test in other regions and years.

A versioned protocol YAML (``configs/benchmarks/*transfer*.yaml``, schema ``geocrop-public-transfer-protocol/1``)
fixes the label sources (checksum-pinned), the domains (region + year + 10 km tiles), the HCAT classes, the
sampling, the source-domain split, imagery, windows (month-day, applied per year), features, the model and the
metrics. ``run`` reuses the building blocks of :mod:`geocrop.public_benchmark`:

    labels per domain (EuroCrops; national open layers with codes mapped to HCAT via the EuroCrops table)
    -> per tile and year: STAC import -> window composites -> feature stack -> pixel samples
    -> source domain: spatial-block split (fit | calibration | test, distance buffer)
    -> one model fitted on source "fit" pixels, conformal thresholds on source calibration parcels,
       area of applicability (AOA) on source fit pixels
    -> scored on the source test parcels (IN-DOMAIN) and on every target domain's parcels (OUT-OF-DOMAIN):
       parcel OA / macro-F1 / per-class F1 with spatial-block bootstrap CIs, conformal coverage under shift,
       share of pixels and parcels inside the AOA
    -> per target, a LOCAL baseline (trained and tested inside the target with the same split rules) on the
       same target test parcels, so that the transfer gap is measured on identical parcels
    -> the engine's grouped transfer CV (:func:`geocrop.experiments.transfer`, leave-one-region/year-out) on the
       pooled pixel samples of all domains.

Scores describe this public European task only (declared, not field-verified labels).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from . import __version__, hcat, metrics, public_data, uncertainty
from . import public_benchmark as pb
from .benchmark import _json_safe

SCHEMA = "geocrop-public-transfer/1"
PROTOCOL_SCHEMA = "geocrop-public-transfer-protocol/1"
REQUIRED = ("protocol", "version", "sources", "grid", "tile_size_m", "domains", "source_domain", "classes",
            "sampling", "split", "imagery", "windows", "features", "model", "evaluation")


# --------------------------------------------------------------------------- protocol

def load_protocol(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    p = yaml.safe_load(raw) or {}
    validate_protocol(p)
    return p, hashlib.sha256(raw).hexdigest()


def validate_protocol(p: dict) -> None:
    if p.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Protocol schema must be {PROTOCOL_SCHEMA!r}")
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise ValueError(f"Protocol lacks {missing}")
    hcat.rules_from_spec(p["classes"])
    if p["source_domain"] not in p["domains"]:
        raise ValueError("source_domain must be one of the domains")
    if len(p["domains"]) < 2:
        raise ValueError("A transfer protocol needs a source and at least one target domain")
    for name, d in p["domains"].items():
        if d.get("source") not in p["sources"]:
            raise ValueError(f"Domain {name}: unknown label source {d.get('source')!r}")
        if not d.get("tiles"):
            raise ValueError(f"Domain {name} has no tiles")
        domain_windows(p, int(d["year"]))
    for name, s in p["sources"].items():
        if s.get("kind") not in ("eurocrops", "url"):
            raise ValueError(f"Source {name}: kind must be 'eurocrops' or 'url'")
        if s["kind"] == "url" and not (s.get("checksum") and s.get("code_map")):
            raise ValueError(f"Source {name}: a url source needs a pinned checksum and a code_map")
    unknown = set(p.get("common_classes", [])) - set(p["classes"])
    if unknown:
        raise ValueError(f"common_classes not in classes: {sorted(unknown)}")


def domain_windows(p: dict, year: int) -> dict:
    """Protocol windows (month-day) as ISO dates of `year`; windows must not cross the year boundary."""
    out = {}
    for name, w in p["windows"].items():
        start, end = date.fromisoformat(f"{year}-{w['start']}"), date.fromisoformat(f"{year}-{w['end']}")
        if end < start:
            raise ValueError(f"Window {name} ends before it starts")
        out[name] = {"start": start.isoformat(), "end": end.isoformat()}
    return out


def domain_protocol(p: dict, domain: str) -> dict:
    """A public-benchmark-shaped protocol for one domain (tiles, year, windows), for `public_benchmark` helpers."""
    d = p["domains"][domain]
    src = p["sources"][d["source"]]
    return {"grid": p["grid"], "classes": p["classes"], "sampling": p["sampling"], "imagery": p["imagery"],
            "compositing": p.get("compositing", {}), "features": p["features"], "year": int(d["year"]),
            "windows": domain_windows(p, int(d["year"])),
            "tiles": {"size_m": p["tile_size_m"], "origins": d["tiles"]},
            "dataset": {"id_field": src["id_field"], "code_field": "_hcat_code"}}


# --------------------------------------------------------------------------- labels

def _layer(p: dict, source: str, cache, layers: dict | None) -> Path:
    if layers and source in layers:
        return Path(layers[source])
    s = p["sources"][source]
    if s["kind"] == "eurocrops":
        return public_data.eurocrops_layer(s["archive"], s.get("checksum"), str(s["zenodo_record"]), cache)
    return public_data.url_layer(s["url"], s["checksum"], cache, subdir=source)


def code_map(table: pd.DataFrame, national: str, hcat_field: str) -> tuple[pd.Series, dict]:
    """National code -> HCAT code (majority per national code) and its consistency."""
    t = table[[national, hcat_field]].dropna().astype(str)
    counts = t.groupby([national, hcat_field]).size().rename("n").reset_index()
    counts = counts.sort_values(["n", hcat_field], ascending=[False, True])
    best = counts.drop_duplicates(national).set_index(national)
    total = t.groupby(national).size()
    share = (best["n"] / total.reindex(best.index)).astype(float)
    mapping = best[hcat_field]
    return mapping, {"national_codes": int(len(mapping)), "inconsistent_codes": int((share < 1).sum()),
                     "min_consistency": float(share.min()) if len(share) else None,
                     "sha256": hashlib.sha256(mapping.sort_index().to_json().encode()).hexdigest()}


def _hcat_codes(p: dict, source: str, raw: gpd.GeoDataFrame, maps: dict) -> pd.Series:
    s = p["sources"][source]
    if s["kind"] == "eurocrops":
        return raw[s["code_field"]].astype(str)
    return raw[s["national_code_field"]].astype(str).map(maps[source][0])


def domain_labels(p: dict, domain: str, cache, layers, maps) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """(population, sample, info) of one domain; target classes below min_parcels_per_class are dropped."""
    d = p["domains"][domain]
    src = p["sources"][d["source"]]
    dp = domain_protocol(p, domain)
    layer = _layer(p, d["source"], cache, layers)
    national = src.get("national_code_field")
    cols = [src["id_field"], src["code_field"] if src["kind"] == "eurocrops" else national]
    frames, unmapped = [], 0
    for tile, footprint in pb.tile_boxes(dp).items():
        raw = public_data.read_parcels(layer, footprint.bounds, p["grid"]["crs"], columns=cols)
        raw["_hcat_code"] = _hcat_codes(p, d["source"], raw, maps)
        unmapped += int(raw["_hcat_code"].isna().sum())
        raw["_hcat_code"] = raw["_hcat_code"].fillna("unmapped")
        frames.append(pb.label_parcels(raw, dp, tile, footprint))
    pop = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=p["grid"]["crs"]).drop_duplicates("parcel_id")
    counts = pop["class_name"].value_counts()
    keep_min = int(p["sampling"].get("min_parcels_per_class", 1))
    dropped = {str(c): int(counts.get(c, 0)) for c in p["classes"] if counts.get(c, 0) < keep_min}
    if domain == p["source_domain"] and dropped:
        raise ValueError(f"Source domain lacks classes {dropped}; change the protocol")
    pop = pop[~pop["class_name"].isin(dropped)]
    sample = pb.sample_parcels(pop, int(p["sampling"]["parcels_per_class"]), int(p["sampling"]["seed"]))
    sample["domain"] = domain
    info = {"population_per_class": {str(k): int(v) for k, v in counts.sort_index().items()},
            "sampled_per_class": {str(k): int(v) for k, v in sample["class_name"].value_counts().sort_index().items()},
            "classes_dropped_below_minimum": dropped, "parcels_with_unmapped_code": unmapped}
    return pop, sample, info


# --------------------------------------------------------------------------- evaluation

def _aoa_folds(blocks: np.ndarray, k: int, seed: int) -> np.ndarray:
    units = np.unique(blocks)
    order = np.random.default_rng(seed).permutation(len(units))
    fold_of = {u: int(i % k) for i, u in zip(range(len(units)), units[order])}
    return np.array([fold_of[b] for b in blocks])


def score(est_proba, classes, X, poly, truth, block, thresholds, aoa, p, labels=None) -> dict:
    """Parcel-level metrics with block-bootstrap CIs, conformal coverage and AOA shares for one parcel set."""
    ev = p["evaluation"]
    proba = est_proba(X)
    pt = pb.parcel_probabilities(proba, poly, classes)
    y = truth.reindex(pt.index).to_numpy()
    pred = np.array(classes)[pt.to_numpy().argmax(axis=1)]
    s = metrics.summary(y, pred, labels=classes)
    blocks = block.reindex(pt.index).to_numpy()
    boot = pb.block_bootstrap(y, pred, blocks, classes, int(ev["bootstrap"]), int(ev["bootstrap_seed"]),
                              float(ev["confidence"]))
    out = {"parcels": int(len(y)), "pixels": int(len(X)),
           "classes_present": sorted(set(map(str, y))),
           "overall_accuracy": s["overall_accuracy"], "macro_f1": s["macro_f1"], "kappa": s["kappa"],
           "overall_accuracy_ci": boot["overall_accuracy_ci"], "macro_f1_ci": boot["macro_f1_ci"],
           "per_class_f1": {c: v["f1"] for c, v in s["per_class"].items() if v["reference_count"]},
           "per_class_f1_ci": {c: v for c, v in boot["per_class_f1_ci"].items() if v is not None},
           "per_class_parcels": {c: v["reference_count"] for c, v in s["per_class"].items() if v["reference_count"]},
           "bootstrap_units": boot["n_units"], "confusion_matrix": s["confusion_matrix"], "labels": s["labels"]}
    common = [c for c in p.get("common_classes", []) if c in classes]
    if common:
        m = np.isin(y, common)
        if m.any():
            sc = metrics.summary(y[m], pred[m], labels=classes)
            bc = pb.block_bootstrap(y[m], pred[m], blocks[m], classes, int(ev["bootstrap"]),
                                    int(ev["bootstrap_seed"]), float(ev["confidence"]))
            out["common_classes"] = {"classes": common, "parcels": int(m.sum()),
                                     "overall_accuracy": sc["overall_accuracy"], "macro_f1": sc["macro_f1"],
                                     "overall_accuracy_ci": bc["overall_accuracy_ci"],
                                     "macro_f1_ci": bc["macro_f1_ci"]}
    sets = uncertainty.prediction_sets(pt.to_numpy(), classes, thresholds)
    cov = uncertainty.coverage(sets, y)
    cov["per_class_coverage"] = {c: float(np.mean([c in st for st, t in zip(sets, y) if t == c]))
                                 for c in classes if (y == c).any()}
    cov["target"] = 1 - float(ev["conformal_alpha"])
    out["conformal"] = cov
    di = aoa.dissimilarity(X)
    inside = di <= aoa.threshold
    frac = pd.Series(inside, index=poly).groupby(level=0).mean().reindex(pt.index)
    out["aoa"] = {"pixel_share_inside": float(inside.mean()), "median_di": float(np.median(di)),
                  "parcel_share_inside": float((frac >= float(ev["aoa_min_fraction"])).mean()),
                  "threshold": aoa.threshold}
    if (~(frac >= float(ev["aoa_min_fraction"]))).any():
        ok = (frac >= float(ev["aoa_min_fraction"])).to_numpy()
        out["aoa"]["oa_inside_parcels"] = float((pred[ok] == y[ok]).mean()) if ok.any() else None
        out["aoa"]["oa_outside_parcels"] = float((pred[~ok] == y[~ok]).mean()) if (~ok).any() else None
    return out


def _fit(params, X, y, seed):
    from .estimators import make_estimator
    est = make_estimator(dict(params), seed)
    est.fit(X, y)
    return est


def _proba_fn(est, classes):
    order = [list(map(str, est.classes_)).index(c) for c in classes]

    def f(X):
        return np.asarray(est.predict_proba(X))[:, order]
    return f


def _calibrated(est, classes, X_cal, poly_cal, truth, alpha):
    pc = pb.parcel_probabilities(_proba_fn(est, classes)(X_cal), poly_cal, classes)
    conf = uncertainty.conformal_thresholds(pc.to_numpy(), truth.reindex(pc.index).to_numpy(), classes,
                                            alphas=(alpha,), mondrian=True, sparse_class_policy="include")
    return conf["thresholds"][str(alpha)]


# --------------------------------------------------------------------------- run

def run(protocol_path: str | Path, work_dir: str | Path, cache: str | Path | None = None, workers: int = 4,
        layers: dict | None = None, skip_transfer_cv: bool = False, log=None) -> dict:
    """Execute a transfer protocol; write ``results.json``/``results.md`` into `work_dir` and return the report."""
    from . import applicability
    from .experiments import transfer
    from .provenance import _run, _versions
    from .samples import Samples, extract

    log = log or pb._log
    started = datetime.now(timezone.utc)
    here = Path(__file__).parent
    engine = {"version": __version__, "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=here),
              "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here))}
    p, digest = load_protocol(protocol_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stages, t0 = {}, time.perf_counter()

    def stage(name):
        nonlocal t0
        stages[name] = {"seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": pb.peak_rss_mb()}
        log(f"stage {name}: {stages[name]['seconds']} s")
        t0 = time.perf_counter()

    # ---- labels
    maps = {}
    for name, s in p["sources"].items():
        if s["kind"] == "url":
            cm = s["code_map"]
            ref_layer = _layer(p, cm["source"], cache, layers)
            import pyogrio
            table = pyogrio.read_dataframe(ref_layer, columns=[cm["national_code_field"], cm["hcat_field"]],
                                           read_geometry=False)
            maps[name] = code_map(table, cm["national_code_field"], cm["hcat_field"])
    labels, samples_by_domain = {}, {}
    for domain in p["domains"]:
        _, sample, info = domain_labels(p, domain, cache, layers, maps)
        labels[domain] = info
        samples_by_domain[domain] = sample
    stage("labels")

    # ---- imagery and pixel samples
    samp = p["sampling"]
    parts, imagery, parcels = [], {}, []
    for domain, d in p["domains"].items():
        dp = domain_protocol(p, domain)
        mine_all = samples_by_domain[domain]
        for tile, footprint in pb.tile_boxes(dp).items():
            key = f"{d['year']}_{tile}"
            stack, meta = pb.tile_stack(dp, key, footprint, work, workers, log)
            imagery[key] = {k: meta[k] for k in ("bbox_wgs84", "items_found")} | {
                "windows": {w: v["used"] for w, v in meta["windows"].items()}}
            mine = mine_all[mine_all["tile"] == tile]
            if not len(mine):
                continue
            mine = mine.assign(uid=domain + ":" + mine["parcel_id"].astype(str))
            s = extract(stack, mine[["uid", "class_name", "geometry"]], label_field="class_name", id_field="uid",
                        max_per_polygon=int(samp["max_pixels_per_parcel"]),
                        inner_buffer_m=float(samp["inner_buffer_m"]), seed=int(samp["seed"]))
            parts.append(s.take(np.arange(len(s.y))))
            parts[-1].region = np.full(len(s.y), d["region"])
            parts[-1].year = np.full(len(s.y), str(d["year"]))
            parcels.append(mine)
    stage("imagery_and_samples")
    if any(q.features != parts[0].features for q in parts):
        raise RuntimeError("Tiles produced different feature sets")
    S = Samples(np.concatenate([q.X for q in parts]), np.concatenate([q.y for q in parts]),
                np.concatenate([q.polygon for q in parts]), np.concatenate([q.x for q in parts]),
                np.concatenate([q.yy for q in parts]), parts[0].features, parts[0].crs,
                region=np.concatenate([q.region for q in parts]), year=np.concatenate([q.year for q in parts]))
    allp = gpd.GeoDataFrame(pd.concat(parcels, ignore_index=True), crs=p["grid"]["crs"])
    allp = allp[allp["uid"].isin(set(S.polygon))].reset_index(drop=True)
    allp["block"] = pb.spatial_block(allp.geometry.centroid, float(p["split"]["spatial_block_m"]))
    truth = allp.set_index("uid")["class_name"]
    block = allp.set_index("uid")["domain"] + "/" + allp.set_index("uid")["block"]
    domain_of_pixel = pd.Series(allp["domain"].to_numpy(), index=allp["uid"]).reindex(S.polygon).to_numpy()

    # ---- source model, calibration, AOA
    ev = p["evaluation"]
    alpha = float(ev["conformal_alpha"])
    seed = int(p["split"]["seed"])
    src = p["source_domain"]
    src_parcels = allp[allp["domain"] == src].reset_index(drop=True)
    src_parcels["role"] = pb.split_parcels(src_parcels, p["split"]).to_numpy()
    role = pd.Series(src_parcels["role"].to_numpy(), index=src_parcels["uid"]).reindex(S.polygon).to_numpy()
    classes = sorted(src_parcels["class_name"].unique())
    fit = role == "fit"
    est = _fit(p["model"], S.X[fit], S.y[fit], seed)
    cal = role == "calibration"
    thresholds = _calibrated(est, classes, S.X[cal], S.polygon[cal], truth, alpha)
    imp = getattr(est, "feature_importances_", None)
    aoa = applicability.fit(S.X[fit], _aoa_folds(block.reindex(S.polygon[fit]).to_numpy(), int(ev["aoa_folds"]), seed),
                            S.features, imp, seed=seed)
    stage("source_model")

    proba = _proba_fn(est, classes)
    results = {}
    test = role == "test"
    results[src] = {"kind": "in-domain (source test parcels)",
                    **score(proba, classes, S.X[test], S.polygon[test], truth, block, thresholds, aoa, p)}
    for domain in p["domains"]:
        if domain == src:
            continue
        m = domain_of_pixel == domain
        d = p["domains"][domain]
        kind = "other year, same region" if d["region"] == p["domains"][src]["region"] else "other region"
        r = {"kind": f"out-of-domain ({kind})",
             "all_parcels": score(proba, classes, S.X[m], S.polygon[m], truth, block, thresholds, aoa, p)}
        # local baseline on the same target test parcels
        tp = allp[allp["domain"] == domain].reset_index(drop=True)
        try:
            tp["role"] = pb.split_parcels(tp, p["split"]).to_numpy()
            trole = pd.Series(tp["role"].to_numpy(), index=tp["uid"]).reindex(S.polygon).to_numpy()
            tfit, tcal, ttest = trole == "fit", trole == "calibration", trole == "test"
            local = _fit(p["model"], S.X[tfit], S.y[tfit], seed)
            lclasses = sorted(tp["class_name"].unique())
            lthr = _calibrated(local, lclasses, S.X[tcal], S.polygon[tcal], truth, alpha)
            laoa = applicability.fit(S.X[tfit], _aoa_folds(block.reindex(S.polygon[tfit]).to_numpy(),
                                                           int(ev["aoa_folds"]), seed),
                                     S.features, getattr(local, "feature_importances_", None), seed=seed)
            r["test_parcels_transferred"] = score(proba, classes, S.X[ttest], S.polygon[ttest], truth, block,
                                                  thresholds, aoa, p)
            r["test_parcels_local"] = score(_proba_fn(local, lclasses), lclasses, S.X[ttest], S.polygon[ttest],
                                            truth, block, lthr, laoa, p)
            r["local_baseline"] = {"fit_parcels": int((tp.role == "fit").sum()), "status": "ok"}
        except ValueError as exc:
            r["local_baseline"] = {"status": f"not available: {exc}"}
        results[domain] = r
        log(f"{domain}: macro-F1 {r['all_parcels']['macro_f1']:.3f}")
    stage("transfer_scoring")

    cv = None
    if not skip_transfer_cv and p.get("transfer_cv"):
        tc = p["transfer_cv"]
        table = transfer(S, work / "transfer_cv", params=dict(tc["params"]), strategies=tuple(tc["strategies"]),
                         n_splits=int(tc["n_splits"]), spatial_block_m=float(tc["spatial_block_m"]), seed=seed)
        cv = json.loads((work / "transfer_cv" / "transfer.json").read_text(encoding="utf8"))
        cv = {"in_domain_spatial_cv": cv["in_domain_spatial_cv"],
              "strategies": {k: {kk: v[kk] for kk in ("in_domain", "out_of_domain", "transfer_gap",
                                                      "group_macro_f1_mean", "worst_group")}
                             | {"per_group": [{kk: g[kk] for kk in ("held_out_group", "n_test", "in_domain_macro_f1",
                                                                     "out_of_domain_macro_f1", "gap_macro_f1",
                                                                     "missing_train_classes")} for g in v["per_group"]]}
                             for k, v in cv["strategies"].items()},
              "rows": int(len(table))}
        for v in cv["strategies"].values():
            v["out_of_domain"] = {k: v["out_of_domain"][k] for k in ("overall_accuracy", "kappa", "macro_f1")}
        stage("transfer_cv")

    report = {
        "schema": SCHEMA,
        "protocol": {"id": p["protocol"], "version": p["version"], "file": Path(protocol_path).name, "sha256": digest},
        "command": f"geocrop benchmark-transfer --protocol {protocol_path} --work {work_dir}",
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "cpu_count": os.cpu_count(), **_versions()},
        "sources": {k: {kk: v.get(kk) for kk in ("kind", "zenodo_record", "archive", "url", "checksum", "licence",
                                                 "citation")} for k, v in p["sources"].items()},
        "code_maps": {k: v[1] for k, v in maps.items()},
        "domains": {k: {"region": v["region"], "year": v["year"], "source": v["source"], "tiles": list(v["tiles"]),
                        "labels": labels[k]} for k, v in p["domains"].items()},
        "source_domain": src, "classes": classes,
        "source_split": {r: int((src_parcels.role == r).sum()) for r in sorted(src_parcels.role.unique())},
        "samples": {"pixels": int(len(S.y)), "features": len(S.features),
                    "fingerprint": pb._fingerprint(S.X, S.y.astype(str), S.polygon.astype(str))},
        "results": results, "transfer_cv": cv, "imagery": imagery, "stages": stages,
        "peak_rss_mb": pb.peak_rss_mb(),
        "note": "European public transfer task (Slovenia; declared, not field-verified labels). Not Republic of "
                "Srpska data and not evidence of accuracy there.",
    }
    report = _json_safe(json.loads(json.dumps(report, default=pb._json_default)))
    (work / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    (work / "results.md").write_text(markdown(report), encoding="utf8")
    return report


# --------------------------------------------------------------------------- markdown

def _ci(v):
    return "" if not v else f" [{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}"


def markdown(r: dict) -> str:
    src = r["source_domain"]
    lines = [f"# Public transfer benchmark `{r['protocol']['id']}` v{r['protocol']['version']}", "",
             f"Protocol SHA-256 `{r['protocol']['sha256']}` · engine {r['engine']['version']} "
             f"(`{r['engine']['git_commit']}`{', dirty' if r['engine']['git_dirty'] else ''}) · "
             f"{r['started_utc']} – {r['finished_utc']}", "",
             f"> {r['note']}", "", f"Reproduce: `{r['command']}`", "",
             "## Domains", "", "| Domain | Region | Year | Labels | Tiles | Sampled parcels per class | Dropped classes |",
             "|---|---|---|---|---|---|---|"]
    for k, d in r["domains"].items():
        lab = d["labels"]
        lines.append(f"| {k}{' (source)' if k == src else ''} | {d['region']} | {d['year']} | {d['source']} | "
                     f"{', '.join(d['tiles'])} | {', '.join(f'{c} {n}' for c, n in lab['sampled_per_class'].items())} | "
                     f"{', '.join(f'{c} ({n})' for c, n in lab['classes_dropped_below_minimum'].items()) or '-'} |")
    lines += ["", "## In-domain vs out-of-domain (one model trained on the source)", "",
              "Parcel level, 95 % spatial-block bootstrap intervals. Macro-F1 over the classes present in the "
              "domain; *common* = grassland, maize, winter cereals only (comparable across domains).", "",
              "| Domain | Kind | Parcels | OA % | Macro-F1 % | Common-class macro-F1 % | Conformal coverage % "
              "(target) | Mean set size | Parcels inside AOA % | Pixels inside AOA % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["results"].items():
        s = v if k == src else v["all_parcels"]
        c = s.get("common_classes", {})
        lines.append(f"| {k} | {v['kind']} | {s['parcels']} | {_pct(s['overall_accuracy'])}{_ci(s['overall_accuracy_ci'])} | "
                     f"{_pct(s['macro_f1'])}{_ci(s['macro_f1_ci'])} | {_pct(c.get('macro_f1'))}{_ci(c.get('macro_f1_ci'))} | "
                     f"{_pct(s['conformal']['coverage'])} ({_pct(s['conformal']['target'])}) | "
                     f"{s['conformal']['mean_set_size']:.2f} | {_pct(s['aoa']['parcel_share_inside'])} | "
                     f"{_pct(s['aoa']['pixel_share_inside'])} |")
    lines += ["", "## Per-class F1 (%)", ""]
    classes = r["classes"]
    lines += ["| Domain | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for k, v in r["results"].items():
        s = v if k == src else v["all_parcels"]
        lines.append(f"| {k} | " + " | ".join(_pct(s["per_class_f1"].get(c)) if c in s["per_class_f1"] else "-"
                                             for c in classes) + " |")
    lines += ["", "Drop vs source (percentage points, source F1 minus target F1):", "",
              "| Domain | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    base = r["results"][src]["per_class_f1"]
    for k, v in r["results"].items():
        if k == src:
            continue
        f = v["all_parcels"]["per_class_f1"]
        lines.append(f"| {k} | " + " | ".join(f"{100 * (base[c] - f[c]):+.1f}" if c in f and f[c] is not None
                                             and base.get(c) is not None else "-" for c in classes) + " |")
    lines += ["", "## Transferred vs local model on the same target test parcels", "",
              "| Target | Test parcels | Transferred macro-F1 % | Local macro-F1 % | Gap (pp) | Transferred OA % | "
              "Local OA % | Coverage transferred / local % |", "|---|---|---|---|---|---|---|---|"]
    for k, v in r["results"].items():
        if k == src:
            continue
        if "test_parcels_local" not in v:
            lines.append(f"| {k} | - | - | - | - | - | - | {v['local_baseline']['status']} |")
            continue
        a, b = v["test_parcels_transferred"], v["test_parcels_local"]
        lines.append(f"| {k} | {a['parcels']} | {_pct(a['macro_f1'])}{_ci(a['macro_f1_ci'])} | "
                     f"{_pct(b['macro_f1'])}{_ci(b['macro_f1_ci'])} | {100 * (b['macro_f1'] - a['macro_f1']):+.1f} | "
                     f"{_pct(a['overall_accuracy'])} | {_pct(b['overall_accuracy'])} | "
                     f"{_pct(a['conformal']['coverage'])} / {_pct(b['conformal']['coverage'])} |")
    cv = r.get("transfer_cv")
    if cv:
        lines += ["", "## Engine transfer CV (pixel level, pooled samples of all domains)", "",
                  f"In-domain spatial-block CV macro-F1: {_pct(cv['in_domain_spatial_cv']['macro_f1'])} %.", "",
                  "| Strategy | Held-out group | Test pixels | In-domain macro-F1 % | Out-of-domain macro-F1 % | Gap (pp) | "
                  "Classes missing in training |", "|---|---|---|---|---|---|---|"]
        for strat, v in cv["strategies"].items():
            for g in v["per_group"]:
                lines.append(f"| {strat} | {g['held_out_group']} | {g['n_test']} | {_pct(g['in_domain_macro_f1'])} | "
                             f"{_pct(g['out_of_domain_macro_f1'])} | {100 * g['gap_macro_f1']:+.1f} | "
                             f"{', '.join(g['missing_train_classes']) or '-'} |")
    lines += ["", "## Label sources and licences", ""]
    for k, s in r["sources"].items():
        lines.append(f"- **{k}** ({s['kind']}): {s['licence']}. {s['citation'].strip()} Checksum `{s['checksum']}`.")
    for k, m in r.get("code_maps", {}).items():
        lines.append(f"- {k}: national codes mapped to HCAT with the EuroCrops table: {m['national_codes']} codes, "
                     f"{m['inconsistent_codes']} inconsistent (min. consistency {m['min_consistency']}), "
                     f"table SHA-256 `{m['sha256'][:16]}…`.")
    lines += ["", f"Samples: {r['samples']['pixels']} pixels x {r['samples']['features']} features, fingerprint "
              f"`{r['samples']['fingerprint'][:16]}…`; stages: " +
              ", ".join(f"{k} {v['seconds']} s" for k, v in r["stages"].items()) +
              f"; peak RSS {r['peak_rss_mb']} MB.", ""]
    return "\n".join(lines)
