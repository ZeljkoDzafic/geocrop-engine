"""Config-driven, resumable pipeline.

    composite -> [embeddings] -> features (+ time series) -> samples -> train -> predict -> cbm
              -> [parcel_classify] -> parcels -> validate -> experiments

Stages in brackets, and the optional parts of the others (transfer validation, probability smoothing, event
markers and marker rules, `model.params`), run only when their configuration enables them. While they are
off they contribute nothing to any fingerprint, so existing configurations keep their stage fingerprints.

Each stage records in `<output>/stages.json` its status and a fingerprint made of its configuration,
the SHA-256 content digests of its inputs (cached per path/size/mtime, see `geocrop.cache`), the engine
source code and the fingerprints of the stages it depends on, plus the identity of every output. A stage
is skipped only if it succeeded, its fingerprint is unchanged and all outputs are present and intact, so
long regional runs resume after interruption while any input, code or upstream change re-runs dependent
stages. `force=True` reruns everything. A lock prevents concurrent runs on one output directory and
`run.log` records timing and peak memory per stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from . import __version__, parallel, scenes
from . import aoi as aoimod
from . import config as cfgmod
from .cache import HashCache, RunLock

STAGES = ("composite", "embeddings", "features", "samples", "train", "predict", "cbm", "parcel_classify", "parcels",
          "validate", "experiments")
CACHE_SCHEMA = 3
_HASHED_SUFFIXES = {".json", ".yaml", ".yml", ".csv", ".txt", ".py"}
DEPENDS = {"composite": (), "embeddings": (), "features": ("composite", "embeddings"), "samples": ("features",),
           "train": ("samples",), "predict": ("train", "features"), "cbm": (), "parcel_classify": ("cbm",),
           "parcels": ("predict", "train", "cbm", "parcel_classify"), "validate": ("predict",),
           "experiments": ("samples", "features")}


def _embeddings_attached(cfg: dict) -> bool:
    opts = cfgmod.embedding_options(cfg)
    return bool(opts and opts["attach"])


#: Dependencies that exist only while a configuration option is on. When off they are left out of the
#: upstream part of the fingerprint (as before they existed) and never block a stage.
OPTIONAL_DEPENDS = {
    ("features", "embeddings"): _embeddings_attached,
    ("parcels", "parcel_classify"): lambda cfg: (cfg["parcel_classify"]["enabled"]
                                                and cfg["parcel_classify"]["decision"] == "replace"),
}
_HASHER: HashCache | None = None   # content digests (SHA-256, cached by path/size/mtime) when a run is active


def _peak_rss_mb() -> float | None:
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, AttributeError):
        return None


def _file_identity(path: Path) -> dict:
    """Content identity: SHA-256 (cached) during runs, so touching or copying a file does not invalidate a stage;
    without an active hash cache, size + mtime plus a hash of small metadata files."""
    st = path.stat()
    if _HASHER is not None:
        return {"bytes": st.st_size, "sha256": _HASHER.file(path)}
    identity = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if path.suffix.lower() in _HASHED_SUFFIXES and st.st_size <= 1024 * 1024:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _part_identity(part):
    if isinstance(part, (str, Path)):
        path = Path(part)
        if path.is_file():
            return {"path": str(path.resolve()), **_file_identity(path)}
        if path.is_dir():
            return {"directory": str(path.resolve()), "files": {
                str(p.relative_to(path)): _file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}}
    if isinstance(part, dict):
        return {str(k): _part_identity(v) for k, v in part.items()}
    if isinstance(part, (list, tuple)):
        return [_part_identity(v) for v in part]
    return part


def _fp(*parts) -> str:
    body = json.dumps([_part_identity(p) for p in parts], sort_keys=True, default=str).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def _fp_with(parts: tuple, extra: dict):
    """Fingerprint of `parts`, plus `extra` only when it is non-empty: options that are off add nothing."""
    return _fp(*parts, extra) if extra else _fp(*parts)


def _grid_identity(path) -> dict:
    """CRS, transform and shape of a grid raster: the grid, independent of the raster's pixel values."""
    import rasterio
    with rasterio.open(path) as ds:
        return {"crs": ds.crs.to_wkt() if ds.crs else None, "transform": list(ds.transform)[:6],
                "shape": [ds.height, ds.width]}


def _provider_identity(provider) -> dict:
    """What a feature provider contributes to the embeddings fingerprint: name, kind, dims, licence, where it
    is registered, its pinned weights (SHA-256) and, for plugins, their source file (engine code is hashed
    separately)."""
    import inspect

    from . import plugins
    identity = {"name": provider.name, "kind": provider.contract.kind, "dims": int(provider.dims),
                "licence": provider.licence, "registered": plugins.feature_providers.source(provider.name),
                "class": f"{type(provider).__module__}.{type(provider).__qualname__}",
                "weights": [w.sha256 for w in getattr(provider, "weights", ()) or ()]}
    if not type(provider).__module__.startswith("geocrop."):
        try:
            identity["code"] = inspect.getsourcefile(type(provider))
        except TypeError:          # built-in or dynamically created class
            pass
    return identity


def _output_identity(outputs) -> dict | None:
    result = {}
    for raw in outputs:
        path = Path(raw)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        result[str(path.resolve())] = _file_identity(path)
    return result or None


class Runner:
    def __init__(self, cfg: dict, force: bool = False, log=print):
        self.cfg, self.force, self.log = cfg, force, log
        self.paths = {k: Path(v) for k, v in cfg["paths"].items()}
        self.out = self.paths["output"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "stages.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if not isinstance(self.state, dict):
            raise ValueError(f"Invalid stage state in {self.state_path}: expected a JSON object")
        global _HASHER
        _HASHER = HashCache(self.out / ".hashcache.json", cfg["cache"]["hash"])
        self.hashes = _HASHER
        self.code_fp = _fp(sorted(Path(__file__).parent.glob("*.py")))
        #: AOI mask of the study (docs/aoi-aware.md) or None: then every stage runs on the whole grid as before.
        self.aoi = aoimod.from_config(cfg, self.out)

    def _aoi_extra(self) -> dict:
        """Fingerprint part of the AOI mask (content and halo); empty without a mask, so unmasked studies keep
        their fingerprints."""
        return {"aoi": {"mask": self.aoi.path, "halo_px": self.aoi.halo}} if self.aoi is not None else {}

    def _note(self, message: str, echo: bool = True):
        """Console message via `self.log` (replaceable) and, always, a timestamped line in run.log."""
        from datetime import datetime, timezone
        if echo:
            self.log(message)
        with (self.out / "run.log").open("a", encoding="utf8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")

    def _done(self, stage, fp, outputs) -> bool:
        entry = self.state.get(stage, {})
        identity = _output_identity(outputs)
        return (not self.force and entry.get("cache_schema") == CACHE_SCHEMA and entry.get("status") == "success"
                and entry.get("fingerprint") == fp and identity is not None and entry.get("outputs") == identity)

    def _save_state(self):
        # The previous JSON remains readable if the process stops before replacement.
        with tempfile.NamedTemporaryFile("w", dir=self.out, prefix=".stages-", suffix=".tmp", delete=False,
                                         encoding="utf8") as fh:
            tmp = Path(fh.name)
            try:
                json.dump(self.state, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp, self.state_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _record(self, stage, fp, started, status, info=None, outputs=None, error=None):
        self.state[stage] = {"cache_schema": CACHE_SCHEMA, "status": status, "fingerprint": fp,
                             "seconds": round(time.time() - started, 1), "peak_rss_mb": _peak_rss_mb(),
                             "engine": __version__, "info": info or {}}
        if outputs is not None:
            self.state[stage]["outputs"] = outputs
        if error is not None:
            self.state[stage]["error"] = {"type": type(error).__name__, "message": str(error)}
        self._save_state()

    def depends(self, name) -> tuple[str, ...]:
        """Upstream stages of `name` under this configuration (optional dependencies only while enabled)."""
        return tuple(d for d in DEPENDS.get(name, ())
                     if OPTIONAL_DEPENDS.get((name, d), lambda cfg: True)(self.cfg))

    def stage(self, name, fp, outputs, fn):
        # Never consume the possibly partial outputs of an upstream stage whose last run failed or was killed.
        incomplete = {d: self.state[d].get("status") for d in self.depends(name)
                      if self.state.get(d, {}).get("status") not in (None, "success")
                      and not (d == "cbm" and not self.cfg["cbm"]["enabled"])}
        if incomplete:
            raise RuntimeError(f"Stage {name} depends on {incomplete} whose last run did not complete, so their "
                               f"outputs may be partial; rerun them first (geocrop run --stages "
                               f"{' '.join(sorted(incomplete))} {name})")
        upstream = {d: self.state.get(d, {}).get("fingerprint") for d in self.depends(name)}
        fp = _fp(CACHE_SCHEMA, self.code_fp, upstream, fp)
        if self._done(name, fp, outputs):
            self._note(f"[skip] {name} (unchanged)")
            return self.state[name].get("info")
        self._disk_guard(name, outputs)
        self._note(f"[run ] {name}")
        started, started_ns = time.time(), time.time_ns() - 2_000_000_000   # 2 s: coarse filesystem clocks
        self._record(name, fp, started, "running")
        try:
            info = fn()
            identity = _output_identity(outputs)
            if identity is None:
                missing = [str(p) for p in outputs if not Path(p).is_file() or Path(p).stat().st_size == 0]
                raise RuntimeError(f"Stage {name} did not produce its complete nonempty output set: {missing}")
            self._record(name, fp, started, "success", info, identity)
        except BaseException as exc:
            from .cleanup import quarantine_partial
            moved = quarantine_partial(outputs, started_ns, self.out, name)   # partial outputs never look final
            self._record(name, fp, started, "failed", {"quarantined": moved} if moved else None, error=exc)
            self._note(f"[fail] {name}: {type(exc).__name__}: {exc}", echo=False)
            if moved:
                self._note(f"[clean] {name}: {len(moved)} partial output(s) moved to {self.out / '.quarantine'}")
            raise
        finally:
            self.hashes.save()
        self._note(f"[done] {name} in {self.state[name]['seconds']} s, peak RSS {self.state[name]['peak_rss_mb']} MB",
                   echo=False)
        return info

    def _disk_guard(self, name, outputs):
        """Fail before a heavy stage starts when its output volume lacks the estimated space (see preflight)."""
        from .preflight import HEAVY_STAGES, DiskSpaceError, check_disk
        if name not in HEAVY_STAGES or not self.cfg.get("resources", {}).get("disk_guard", True):
            return
        existing = sum(Path(p).stat().st_size for p in outputs if Path(p).is_file())
        try:
            info = check_disk(self.cfg, name, existing)
        except DiskSpaceError as exc:
            self._note(f"[fail] {name}: {exc}")
            raise
        if info:
            self._note(f"[disk] {name}: {info['free_gb']} GB free, {info['need_gb']} GB required", echo=False)

    # ------------------------------------------------------------------ stages
    def composite(self):
        from . import s1, s2
        windows = cfgmod.parse_windows(self.cfg["windows"])
        if not windows:
            raise cfgmod.ConfigError("Define windows: in the configuration")
        c = self.cfg["compositing"]
        comp = self.paths["composites"]
        s2_scenes = scenes.assign(scenes.discover(self.paths["s2_scenes"]), windows) if self.paths["s2_scenes"].exists() else {}
        s1_scenes = scenes.assign(scenes.discover(self.paths["s1_scenes"]), windows) if self.paths["s1_scenes"].exists() else {}
        grid = self.cfg["grid"].get("reference")
        missing = [w for w in windows if not s2_scenes.get(w) and not s1_scenes.get(w)]
        if missing:
            raise RuntimeError(f"No scenes for windows {missing}; inspect the inventory and adjust windows")
        outputs = []
        for w in windows:
            if s2_scenes.get(w):
                outputs.extend(comp / w / name for name in
                               [*[f"{band}.tif" for band in s2.S2_BANDS], "valid_count.tif", "composite.json"])
            if s1_scenes.get(w):
                prefixes = {s1.ORBITS.get(str(m.get("orbit_direction", "")).upper(), "")
                            for m in s1_scenes[w]} if self.cfg["s1"]["separate_orbits"] else {""}
                outputs.extend(comp / w / f"{prefix + '_' if prefix else ''}{pol}.tif"
                               for prefix in sorted(prefixes) for pol in ("VV", "VH"))
                outputs.append(comp / w / "composite_s1.json")
        known_products = {*[f"{band}.tif" for band in s2.S2_BANDS], "valid_count.tif", "composite.json",
                          "composite_s1.json", *[f"{prefix}{pol}.tif" for prefix in ("", "ASC_", "DESC_")
                                                  for pol in ("VV", "VH")]}
        expected = set(outputs)
        conflicting = [comp / w / name for w in windows for name in sorted(known_products)
                       if (comp / w / name).is_file() and comp / w / name not in expected]
        if conflicting:
            raise RuntimeError("Composite products conflict with the current sensor/orbit selection: "
                               f"{[str(p) for p in conflicting]}. Choose a new paths.composites directory "
                               "or deliberately clean the conflicting generated products before retrying.")
        fp = _fp_with((c, self.cfg["s1"], self.cfg["windows"], grid,
                       [s["path"] for v in list(s2_scenes.values()) + list(s1_scenes.values()) for s in v]),
                      self._aoi_extra())

        def run():
            info = {"s2": scenes.inventory(s2_scenes) if s2_scenes else [],
                    "s1": scenes.inventory(s1_scenes) if s1_scenes else []}
            for w in windows:
                if s2_scenes.get(w):
                    s2.composite([s["path"] for s in s2_scenes[w]], comp / w, reference=grid,
                                 statistic=c["statistic"], scl_invalid=c["scl_invalid"],
                                 cloud_probability_max=c["cloud_probability_max"],
                                 mask_dilation_px=c["mask_dilation_px"],
                                 min_valid_observations=c["min_valid_observations"],
                                 temporal_screening=c["temporal_screening"], aoi=self.aoi)
                if s1_scenes.get(w):
                    ref = grid or (comp / w / "B02.tif")
                    s1.composite([s["path"] for s in s1_scenes[w]], comp / w, reference=ref,
                                 separate_orbits=self.cfg["s1"]["separate_orbits"], statistic=self.cfg["s1"]["statistic"],
                                 aoi=self.aoi)
            return info
        return self.stage("composite", fp, outputs, run)

    def _grid_reference(self, stage: str) -> Path:
        ref = self.cfg["grid"].get("reference")
        if ref:
            return Path(ref)
        found = sorted(self.paths["composites"].glob("*/B02.tif"))
        if not found:
            raise cfgmod.ConfigError(f"{stage} needs the study grid: set grid.reference or run the composite stage")
        return found[0]

    def _monthly_composites(self, reference, monthly: Path, start: str, months: int) -> dict:
        """Calendar-month S2/S1 composites from the scene folders: the one input of every on-VM provider."""
        from . import embeddings as emb
        c = self.cfg["compositing"]
        return emb.monthly_composites(reference, monthly, start, months, self.paths["s2_scenes"],
                                      self.paths["s1_scenes"], statistic=c["statistic"], scl_invalid=c["scl_invalid"],
                                      cloud_probability_max=c["cloud_probability_max"],
                                      mask_dilation_px=c["mask_dilation_px"])

    def _embedding_rasters(self, eo: dict) -> list[Path]:
        from .embeddings import output_path
        return [output_path(eo["dir"], src, eo["year"]) for src in cfgmod.embedding_sources(eo)]

    def embeddings(self):
        """Optional foundation-model embeddings on the study grid (`features.embeddings`, docs/embeddings.md).

        One or several feature providers (`source` may be a list); every provider that encodes monthly
        composites (Presto, Galileo, AnySat, plugin MonthlyProviders) reads the same composites: `monthly`, or
        `<dir>/monthly` built here once from the scene folders. Licences that need an acknowledgement are
        checked before anything runs (`allow_license`, as on the CLI). Network access (AlphaEarth/TESSERA reads,
        pinned weight downloads) happens only here and only when the section is configured."""
        from . import feature_providers as fpmod
        from . import features
        eo = cfgmod.embedding_options(self.cfg)
        if not eo:
            return None
        sources = cfgmod.embedding_sources(eo)
        providers = [fpmod.get(src) for src in sources]
        for provider in providers:          # refuse before any download or computation
            fpmod.check_licence(provider, eo["allow_license"])
        if eo["checkpoint"] and len(sources) > 1:
            raise cfgmod.ConfigError("features.embeddings.checkpoint applies to one source; list one source")
        namespace = cfgmod.embedding_namespace(eo)
        monthly_kind = {p.name for p in providers if p.contract.kind != "precomputed"}
        parcels = self.paths["parcels"] if eo["parcels"] else None
        outputs = []
        for raster in self._embedding_rasters(eo):
            outputs += [raster, features.manifest_path(raster)]
            if parcels is not None:
                outputs.append(raster.with_name(raster.stem + "_parcels.csv"))
        reference = self._grid_reference("embeddings")
        monthly = Path(eo["monthly"]) if eo["monthly"] else Path(eo["dir"]) / "monthly"
        inputs = {}
        if monthly_kind:
            c = self.cfg["compositing"]
            # the composites' content: the given folder itself, or everything it is built from
            inputs = ({"monthly": monthly} if eo["monthly"] else
                      {"s2_scenes": self.paths["s2_scenes"], "s1_scenes": self.paths["s1_scenes"],
                       "compositing": {k: c[k] for k in ("statistic", "scl_invalid", "cloud_probability_max",
                                                         "mask_dilation_px")}})
        if parcels is not None:
            inputs.update(parcels=parcels, id_field=self.cfg["parcels"]["id_field"])
        if eo["checkpoint"]:
            inputs["checkpoint"] = eo["checkpoint"]
        # dir is where outputs go (part of their identity), not an input; monthly and checkpoint are hashed in
        # `inputs`; acknowledging a licence does not change any output
        options = {k: v for k, v in eo.items()
                   if k not in ("attach", "monthly", "dir", "allow_license", "checkpoint", "source", "namespace")}
        identity = {"providers": [_provider_identity(p) for p in providers], "namespace": namespace}
        fp = _fp(options, identity, _grid_identity(reference), inputs)

        def run():
            from .embedding_sources import extract  # adapter: imported only when embeddings are configured
            if monthly_kind and not eo["monthly"]:
                self._monthly_composites(reference, monthly, eo["start"] or f"{eo['year']}-01", eo["months"])
            results = []
            for src in sources:
                r = extract(src, reference, eo["dir"], eo["year"], monthly=monthly if src in monthly_kind else None,
                            start=eo["start"], months=eo["months"], parcels=parcels,
                            id_field=self.cfg["parcels"]["id_field"], checkpoint=eo["checkpoint"],
                            mirror=eo["mirror"], index=eo["index"], resampling=eo["resampling"],
                            namespace=namespace, allow_license=eo["allow_license"])
                results.append({k: r[k] for k in ("source", "year", "raster", "parcels", "feature_names") if k in r})
            return results[0] if len(results) == 1 else {"sources": results}
        return self.stage("embeddings", fp, outputs, run)

    def features(self):
        from . import features
        f = self.cfg["features"]
        periods = list(cfgmod.parse_windows(self.cfg["windows"])) or None
        comp = self.paths["composites"]
        inputs = sorted(p for p in comp.rglob("*.tif")) if comp.exists() else []
        t = self.cfg["timeseries"]
        target = self.paths["features"]
        outputs = [target, features.manifest_path(target)]
        ts_inputs = []
        if t["enabled"]:
            ts_target = target.with_name(target.stem + "_ts.tif")
            outputs.extend([ts_target, features.manifest_path(ts_target)])
            from .tsfeatures import quality_path
            outputs.extend([quality_path(ts_target), quality_path(ts_target).with_suffix(".json")])
            if periods and comp.exists():
                win = target.with_name(target.stem + "_windows.tif")
                outputs.extend([win, features.manifest_path(win)])
            ts_inputs = [self.paths["s2_scenes"], self.paths["s1_scenes"], self.cfg["grid"].get("reference"),
                         self.cfg["compositing"]["scl_invalid"], self.cfg["compositing"]["cloud_probability_max"]]
        attach = []
        if _embeddings_attached(self.cfg):
            attach = self._embedding_rasters(cfgmod.embedding_options(self.cfg))
            for raster in attach:
                if not raster.is_file():
                    raise FileNotFoundError(f"{raster} missing; run the embeddings stage first "
                                            "(geocrop run --stages embeddings features)")
        # the embeddings options enter through the attached raster's identity (None = as before they existed)
        support = t["enabled"] and t["temporal_support"]
        tolerance = self.cfg["cbm"]["temporal_support"]["tolerance_days"]
        if support:
            from .tsfeatures import support_path
            sp = support_path(target.with_name(target.stem + "_ts.tif"))
            outputs.extend([sp, sp.with_suffix(".json")])
        extra = {"embeddings": [attach[0], features.manifest_path(attach[0])] if len(attach) == 1 else
                 [[a, features.manifest_path(a)] for a in attach]} if attach else {}
        if support:
            extra["support_tolerance_days"] = tolerance
        extra.update(self._aoi_extra())
        fp = _fp_with(({**f, "embeddings": None}, self._section("timeseries"), periods, inputs, ts_inputs), extra)

        def run():
            info = build()
            for raster in attach:
                features.attach_embeddings(target, raster, aoi=self.aoi)
            if attach:
                info["embeddings"] = str(attach[0]) if len(attach) == 1 else [str(a) for a in attach]
            return info

        def build():
            parts = []
            if periods and comp.exists():
                win = target.with_name(target.stem + "_windows.tif") if t["enabled"] else target
                parts.append(features.build_stack(comp, win, periods=periods, s2_bands=f["s2_bands"],
                                                  s2_indices=f["s2_indices"], s1=f["s1"], deltas=f["deltas"],
                                                  seasonal=f["seasonal"], aoi=self.aoi))
            if t["enabled"]:
                from datetime import date

                from .tsfeatures import build_timeseries_stack, merge_stacks
                if not (t["start"] and t["end"]):
                    raise cfgmod.ConfigError("timeseries.start and timeseries.end are required")
                ref = self.cfg["grid"].get("reference") or (comp / periods[0] / "B02.tif")
                tsp = build_timeseries_stack(
                    [self.paths["s2_scenes"], self.paths["s1_scenes"]], ref, target.with_name(target.stem + "_ts.tif"),
                    date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"])), t["s2_indices"],
                    t["s1_indices"], t["step_days"], t["output_step_days"], t["lambda"], t["envelope_iterations"],
                    t["harmonic_order"], self.cfg["compositing"]["scl_invalid"],
                    self.cfg["compositing"]["cloud_probability_max"],
                    min_observed_fraction=t["min_observed_fraction"], max_gap_days=t["max_gap_days"],
                    **({"temporal_support": True, "support_tolerance_days": tolerance} if support else {}),
                    aoi=self.aoi)
                parts.append(tsp)
                if len(parts) > 1:
                    merge_stacks(parts, target, aoi=self.aoi)
                else:
                    import shutil
                    shutil.copy(tsp, target)
                    shutil.copy(features.manifest_path(tsp), features.manifest_path(target))
            if not parts:
                raise cfgmod.ConfigError("Features require window composites or enabled time-series inputs")
            return {"stack": str(target), "parts": [str(p) for p in parts]}
        return self.stage("features", fp, outputs, run)

    def samples(self):
        from .samples import extract
        target = self.out / "samples.npz"
        m, cl = self.cfg["model"], self.cfg["classes"]
        from .features import manifest_path
        cv = self.cfg["cv"]
        groups = {k: cv[k] for k in ("region_field", "year_field", "year") if cv[k] is not None}
        extra = {"cv_groups": groups} if groups else {}
        if m.get("feature_groups"):
            extra["feature_groups"] = list(m["feature_groups"])
        fp = _fp_with((m["max_samples_per_polygon"], cl, self.cfg["seed"], self.paths["features"],
                       manifest_path(self.paths["features"]), self.paths["training"]), extra)

        def run():
            selected = None
            if m.get("feature_groups"):
                from .features import load_manifest, select
                selected = select(load_manifest(self.paths["features"]), groups=m["feature_groups"])
            s = extract(self.paths["features"], self.paths["training"], label_field=cl["label_field"],
                        id_field=cl["id_field"], max_per_polygon=m["max_samples_per_polygon"], seed=self.cfg["seed"],
                        features=selected, **groups)
            s.save(target)
            return {"samples": int(len(s.y)), "polygons": int(len(set(s.polygon)))}
        return self.stage("samples", fp, [target], run)

    def train(self):
        from . import model
        from .samples import Samples
        m = self.cfg["model"]
        target = self.out / "model" / "model.joblib"
        outputs = [target.with_name(name) for name in (
            "model.joblib", "train_report.json", "feature_importance.csv", "cv_confusion.csv",
            "oof_polygons.csv", "conformal.json", "suspect_labels.csv", "aoa.json")]
        if m.get("holdout_fraction"):
            outputs.append(target.with_name("holdout_predictions.csv"))
        if m.get("calibration_mode") == "holdout":
            outputs.extend(target.with_name(name) for name in ("calibration_polygons.csv", "calibration_split.json"))
        strategy = self._cv_strategy()
        if strategy != "spatial_blocks":
            outputs.extend(target.with_name(name) for name in ("transfer_per_group.csv", "transfer_report.json"))
        fp = _fp_with((self._model_section(), self.cfg["seed"], self.out / "samples.npz"),
                      {"cv_strategy": strategy} if strategy != "spatial_blocks" else {})

        def run():
            params = model.config_params(m, keep_none=True)
            extra = {"cv_strategy": strategy} if strategy != "spatial_blocks" else {}
            r = model.train(Samples.load(self.out / "samples.npz"), self.out / "model", params=params,
                            n_splits=m["cv_folds"], spatial_block_m=m["spatial_block_m"], seed=self.cfg["seed"],
                            tune_grid=m.get("tune_grid"), provenance={"config_fingerprint": cfgmod.fingerprint(self.cfg)},
                            calibrate=m["calibrate"], spatial_buffer_m=m["spatial_buffer_m"],
                            calibration_mode=m.get("calibration_mode", "oof"),
                            calibration_fraction=m.get("calibration_fraction", 0.2),
                            holdout_fraction=m["holdout_fraction"], inner_splits=m["inner_splits"],
                            allow_missing_classes=m.get("allow_missing_classes", False), **extra)
            info = {"cv_macro_f1": r["cross_validation"]["macro_f1"],
                    "cv_oa": r["cross_validation"]["overall_accuracy"]}
            if "transfer_validation" in r:
                tv = r["transfer_validation"]
                report = {"schema": "geocrop-transfer/1", "cv_strategy": strategy,
                          **{k: tv[k] for k in tv if k != "nested"},
                          "note": "Leave-one-group-out transfer validation of the training samples; "
                                  "cross_validation, calibration and conformal thresholds keep spatial blocks."}
                target.with_name("transfer_report.json").write_text(
                    json.dumps(report, indent=2, default=str), encoding="utf8")
                info["transfer_gap_macro_f1"] = tv["transfer_gap"].get("macro_f1")
            return info
        return self.stage("train", fp, outputs, run)

    def _section(self, name: str) -> dict:
        """A config section for fingerprints, without keys added later that still hold their default
        (`config.FINGERPRINT_NEUTRAL`)."""
        return cfgmod.without_neutral_defaults(self.cfg)[name]

    def _model_section(self) -> dict:
        return self._section("model")

    def _cv_strategy(self) -> str:
        cv = self.cfg["cv"]
        strategy = cv["strategy"]
        if strategy in ("region", "region_year") and not cv["region_field"]:
            raise cfgmod.ConfigError(f"cv.strategy {strategy} needs cv.region_field (a polygon attribute)")
        if strategy in ("year", "region_year") and not (cv["year_field"] or cv["year"] is not None):
            raise cfgmod.ConfigError(f"cv.strategy {strategy} needs cv.year_field or cv.year; leave-one-year-out "
                                     "needs samples from at least two years")
        return strategy

    def predict(self):
        from . import model
        pred = self.out / "prediction"
        outputs = [pred / name for name in ("classes.tif", "probabilities.tif", "uncertainty.tif", "class_legend.json",
                                           "prediction_manifest.json")]
        aoa = self.out / "model" / "aoa.json"
        if aoa.exists():
            outputs.append(pred / "aoa.tif")
        smoothing = model.smoothing_settings(self.cfg["predict"]["smoothing"])
        if smoothing is not None:
            outputs.append(pred / "probabilities_unsmoothed.tif")
        shift = self._shift_options()
        extra = {"smoothing": smoothing} if smoothing is not None else {}
        if cfgmod.without_neutral_defaults(self.cfg).get("uncertainty") is not None:
            extra["shift"] = {**shift, "parcels": shift["parcels"] and str(shift["parcels"])}
        extra.update(self._aoi_extra())
        fp = _fp_with((self.paths["features"], self.out / "model" / "model.joblib", aoa), extra)

        def run():
            if smoothing is None:   # a previous smoothed run's raw probabilities no longer describe this map
                (pred / "probabilities_unsmoothed.tif").unlink(missing_ok=True)
            return model.predict(self.paths["features"], self.out / "model" / "model.joblib", pred,
                                 smoothing=smoothing, shift=shift, aoi=self.aoi,
                                 workers=self._block_workers())
        return self.stage("predict", fp, outputs, run)

    def _block_workers(self) -> int:
        """Processes for blockwise prediction: resources.workers, else a positive model.n_jobs, else min(CPUs, 4)."""
        return parallel.resolve_workers(self.cfg["resources"].get("workers"), self.cfg["model"].get("n_jobs"))

    def _shift_options(self) -> dict:
        """`model.predict` shift-check options (docs/domain-shift.md): sample inside the parcels when available."""
        u, p = self.cfg["uncertainty"], self.cfg["parcels"]
        return {"config": u, "parcels": self.paths["parcels"] if self.paths["parcels"].exists() else None,
                "id_field": p["id_field"], "alpha": p["conformal_alpha"], "seed": self.cfg["seed"]}

    def cbm(self):
        from datetime import date

        from . import cbm
        c, t = self.cfg["cbm"], self.cfg["timeseries"]
        if not c["enabled"]:
            return None
        if not self.paths["parcels"].exists():
            self.log("[skip] cbm (no parcel layer)")
            return None
        if not (t["start"] and t["end"]):
            raise cfgmod.ConfigError("cbm needs timeseries.start/end for the season grid")
        start, end = date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"]))
        target = self.out / "cbm" / "markers.csv"
        ref = self.cfg["grid"].get("reference") or next(self.paths["composites"].glob("*/B02.tif"))
        events = self._event_markers_on()
        outputs = [target, self.out / "cbm" / "signatures.csv"]
        s1_period = []
        if events:
            outputs.append(self.out / "cbm" / "event_markers.csv")
            if self.paths["s1_scenes"].exists():
                s1_period = [m for m in scenes.discover(self.paths["s1_scenes"])
                             if m.get("sensor") == "S1" and start <= m["date"] <= end]
            if s1_period:
                outputs.append(self.out / "cbm" / "s1_signatures.csv")
        tq = self._temporal_rules()
        if tq:
            outputs.append(self.out / "cbm" / "temporal_support.csv")
        fp = _fp_with((self._cbm_section(), self._section("timeseries"), self.paths["parcels"],
                       self.paths["s2_scenes"], ref,
                       self.cfg["parcels"]["id_field"], self.cfg["parcels"]["interior_buffer_px"]),
                      {"event_markers": True, "s1_scenes": self.paths["s1_scenes"] if s1_period else None}
                      if events else {})

        def run():
            idx = [c["index"], "BSI"] if c["index"] != "BSI" else ["BSI"]
            if events and c["events"]["index"] not in idx:
                idx.append(c["events"]["index"])
            if tq and tq["index"] not in idx:
                idx.append(tq["index"])
            sig = cbm.parcel_signatures(self.paths["s2_scenes"], self.paths["parcels"], ref,
                                        self.out / "cbm" / "signatures.csv", start, end, idx,
                                        id_field=self.cfg["parcels"]["id_field"],
                                        interior_buffer_px=self.cfg["parcels"]["interior_buffer_px"],
                                        **({"moments": True} if events else {}))
            m = cbm.markers(sig, start, end, index=c["index"], min_coverage=c["min_coverage"])
            m.to_csv(target, index=False)
            info = {"parcels": int(len(m))}
            if tq:
                from .temporal_quality import parcel_support
                sup = parcel_support(sig, start, end, tq, markers=m)
                sup.to_csv(self.out / "cbm" / "temporal_support.csv", index=False)
                info["temporal_support_parcels"] = int(len(sup))
            if events:
                from .cbm_markers import event_markers
                s1sig = None
                if s1_period:
                    s1sig = cbm.s1_signatures(self.paths["s1_scenes"], self.paths["parcels"], ref,
                                              self.out / "cbm" / "s1_signatures.csv", start, end,
                                              id_field=self.cfg["parcels"]["id_field"],
                                              interior_buffer_px=self.cfg["parcels"]["interior_buffer_px"])
                ev = event_markers(sig, start, end, s1sig, c["events"])
                ev.to_csv(self.out / "cbm" / "event_markers.csv", index=False)
                info.update({"event_markers": int(len(ev)), "s1_acquisitions": len(s1_period),
                             "harvest_detected": int(ev.harvest_detected.sum())})
            return info
        return self.stage("cbm", fp, outputs, run)

    def _event_markers_on(self) -> bool:
        c = self.cfg["cbm"]
        return bool(c["enabled"] and (c["event_markers"] or c["rules"]["enabled"]))

    def _cbm_section(self) -> dict:
        return self._section("cbm")

    def _temporal_rules(self) -> dict | None:
        """Enabled `cbm.temporal_support` (docs/temporal-quality.md), else None."""
        rules = self.cfg["cbm"]["temporal_support"]
        return rules if self.cfg["cbm"]["enabled"] and rules["enabled"] else None

    def _temporal_support_table(self):
        import pandas as pd
        path = self.out / "cbm" / "temporal_support.csv"
        return pd.read_csv(path, dtype={"parcel_id": str}) if self._temporal_rules() and path.exists() else None

    def _marker_table(self):
        """cbm phenology markers, joined with the event markers when they are computed (None without cbm)."""
        import pandas as pd
        markers_csv = self.out / "cbm" / "markers.csv"
        if not (self.cfg["cbm"]["enabled"] and markers_csv.exists()):
            return None
        table = pd.read_csv(markers_csv)
        events_csv = self.out / "cbm" / "event_markers.csv"
        if self._event_markers_on() and events_csv.exists():
            ev = pd.read_csv(events_csv, dtype={"parcel_id": str})
            table = table.assign(parcel_id=table.parcel_id.astype(str)).merge(ev, on="parcel_id", how="outer")
        return table

    def _marker_rules(self) -> dict | None:
        rules = self.cfg["cbm"]["rules"]
        return rules if self.cfg["cbm"]["enabled"] and rules["enabled"] else None

    def _calendar(self):
        if not self.cfg["cbm"]["calendar"]:
            return None
        import yaml
        return yaml.safe_load(Path(self.cfg["cbm"]["calendar"]).read_text(encoding="utf8"))

    def _crop_mapping(self):
        if not self.cfg["parcels"].get("crop_mapping"):
            return None
        from .adapters import read_mapping
        return read_mapping(self.cfg["parcels"]["crop_mapping"])

    def parcel_classify(self):
        """Optional parcel-level classifier on the cbm signatures (`parcel_classify`, docs/parcel-classification.md)."""
        from datetime import date

        from . import parcel_classify as pc
        q, p, m, t = self.cfg["parcel_classify"], self.cfg["parcels"], self.cfg["model"], self.cfg["timeseries"]
        if not q["enabled"]:
            return None
        if not self.paths["parcels"].exists():
            self.log("[skip] parcel_classify (no parcel layer)")
            return None
        root = self.out / "parcel_classify"
        model_dir = root / "model"
        target = root / "parcels.gpkg"
        outputs = [model_dir / name for name in ("model.joblib", "train_report.json", "conformal.json", "aoa.json",
                                                 "oof_polygons.csv", "parcel_features.csv")]
        outputs += [target, target.with_suffix(".csv"), target.with_suffix(".summary.json")]
        params = self._parcel_params()
        signatures, rules = self.out / "cbm" / "signatures.csv", self._marker_rules()
        events_csv = self.out / "cbm" / "event_markers.csv"
        mapping, calendar = self._crop_mapping(), self._calendar()
        cv = {k: m[k] for k in ("cv_folds", "spatial_block_m", "spatial_buffer_m", "holdout_fraction",
                                "calibration_mode", "calibration_fraction")}
        fp = _fp(q, params, cv, self.cfg["seed"], t["start"], t["end"], self.cfg["cbm"]["index"], p,
                 self.paths["parcels"], signatures, self.out / "cbm" / "markers.csv", mapping, calendar,
                 {"rules": rules, "events": events_csv} if rules else None,
                 *([{"temporal_rules": self._temporal_rules(), "support": self.out / "cbm" / "temporal_support.csv"}]
                   if self._temporal_rules() else []))

        def run():
            import geopandas as gpd
            import pandas as pd
            gdf = gpd.read_file(self.paths["parcels"])
            start, end = date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"]))
            frame = pc.parcel_features(pd.read_csv(signatures), gdf, start, end, p["id_field"],
                                       q["indices"] or [self.cfg["cbm"]["index"]], q["stats"], q["groups"],
                                       q["step_days"], q["output_step_days"], q["window_days"], q["lambda"])
            rep = pc.train(frame, gdf, q["label_field"], model_dir, params, q["groups"], p["id_field"],
                           m["cv_folds"], m["spatial_block_m"], self.cfg["seed"], m["holdout_fraction"],
                           m["calibration_mode"], m["calibration_fraction"], m["spatial_buffer_m"], q["bootstrap"])
            summary = pc.predict(model_dir, frame, gdf, target, p["id_field"], p["declared_field"],
                                 p["conformal_alpha"], p["min_valid_pixels"], p["min_confidence"], p["min_margin"],
                                 p["min_aoa_fraction"], mapping, self._marker_table(), calendar)
            if rules or self._temporal_rules():
                summary = _apply_marker_rules(target, self._marker_table(), rules, p["id_field"],
                                              self._temporal_support_table(), self._temporal_rules())
            return {"cv_macro_f1": rep["cross_validation"]["macro_f1"], "parcels_trained": rep["samples"],
                    "decision_counts": summary["decision_counts"]}
        return self.stage("parcel_classify", fp, outputs, run)

    def _parcel_params(self) -> dict:
        from . import model
        from .parcel_classify import DEFAULT_PARAMS
        extra = dict(self.cfg["parcel_classify"]["params"] or {})
        algorithm = extra.get("algorithm", "rf")
        params = {**DEFAULT_PARAMS, "algorithm": algorithm} if algorithm in ("rf", "extra_trees") else \
            {"algorithm": algorithm}
        params.update(extra)
        tunable = {k: v for k, v in extra.items() if k != "algorithm"}
        if tunable:
            model.check_params(params, tunable, where="parcel_classify.params")
        return params

    def parcels(self):
        from . import parcels
        p = self.cfg["parcels"]
        if not self.paths["parcels"].exists():
            self.log("[skip] parcels (no parcel layer)")
            return None
        target = self.out / "parcels" / "parcels.gpkg"
        fp = _fp(p, self.paths["parcels"], self.out / "prediction" / "probabilities.tif")
        mapping = self._crop_mapping()
        conformal = self.out / "model" / "conformal.json"
        aoa = self.out / "prediction" / "aoa.tif"
        markers_csv = self.out / "cbm" / "markers.csv"
        calendar = self._calendar()
        rules = self._marker_rules()
        replace = self.cfg["parcel_classify"]["enabled"] and self.cfg["parcel_classify"]["decision"] == "replace"
        extra = {}
        if rules:
            extra.update(rules=rules, events=self.out / "cbm" / "event_markers.csv")
        temporal = self._temporal_rules()
        if temporal:
            extra.update(temporal_rules=temporal, temporal_support=self.out / "cbm" / "temporal_support.csv")
        if replace:
            extra["decision"] = "parcel_classifier"
        shift_report = self.out / "prediction" / "shift_report.json"
        if self.cfg["uncertainty"]["shift_detection"] and shift_report.exists():
            extra["domain_shift"] = shift_report

        def run():
            if replace:   # the parcel classifier's decisions are the parcel product (same schema)
                import shutil
                source = self.out / "parcel_classify" / "parcels.gpkg"
                target.parent.mkdir(parents=True, exist_ok=True)
                for suffix in (".gpkg", ".csv", ".summary.json"):
                    shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))
                return json.loads(target.with_suffix(".summary.json").read_text(encoding="utf8"))
            return parcels.aggregate(
                self.out / "prediction" / "probabilities.tif", self.paths["parcels"], target,
                id_field=p["id_field"], declared_field=p["declared_field"], strategy=p["strategy"],
                interior_buffer_px=p["interior_buffer_px"], min_valid_pixels=p["min_valid_pixels"],
                min_confidence=p["min_confidence"], min_majority_share=p["min_majority_share"],
                min_margin=p["min_margin"], crop_mapping=mapping,
                conformal=conformal if conformal.exists() else None, alpha=p["conformal_alpha"],
                aoa=aoa if aoa.exists() else None, min_aoa_fraction=p["min_aoa_fraction"],
                markers=self._marker_table(), calendar=calendar, **({"marker_rules": rules} if rules else {}),
                **({"temporal_support": self._temporal_support_table(), "temporal_rules": temporal}
                   if temporal else {}),
                **({"domain_shift": json.loads(shift_report.read_text(encoding="utf8"))}
                   if "domain_shift" in extra else {}))
        fp = _fp_with((fp, mapping, conformal, aoa, markers_csv, calendar, self.cfg["cbm"]["enabled"],
                       self.out / "prediction" / "prediction_manifest.json"), extra)
        return self.stage("parcels", fp, [target, target.with_suffix(".csv"), target.with_suffix(".summary.json")], run)

    def validate(self):
        from . import validation
        v = self.cfg["validation"]
        if not self.paths["observations"].exists():
            self.log("[skip] validate (no independent observations)")
            return None
        target = self.out / "validation" / "validation_report.json"
        fp = _fp(self._section("validation"), self.cfg["seed"], self.paths["observations"], self.paths["training"],
                 [self.out / "prediction" / name for name in
                  ("classes.tif", "probabilities.tif", "class_legend.json", "prediction_manifest.json")])

        def run():
            r = validation.evaluate_points(
                self.out / "prediction", self.paths["observations"], self.out / "validation",
                label_field=v["label_field"], date_field=v["date_field"], year=v["year"],
                training=self.paths["training"] if self.paths["training"].exists() else None,
                training_exclusion_m=v["training_exclusion_m"], window_px=v["window_px"],
                bootstrap=v["bootstrap"], seed=self.cfg["seed"], group_field=v["group_field"])
            return {"oa": r["accuracy"]["overall_accuracy"], "macro_f1": r["accuracy"]["macro_f1"],
                    "selection": r["selection"]}
        return self.stage("validate", fp, [target, target.with_name("observations_evaluated.csv"),
                                           target.with_name("confusion_matrix.csv")], run)

    def experiments(self):
        from . import experiments, features
        from .samples import Samples
        m = self.cfg["model"]
        exps = self.cfg["experiments"] or experiments.DEFAULT_EXPERIMENTS
        target = self.out / "experiments" / "ablation.csv"
        strategy = self.cfg["cv"]["strategy"]
        fp = _fp_with((exps, self._model_section(), self.cfg["seed"], self.cfg["classes"]["targets"],
                       self.out / "samples.npz", features.manifest_path(self.paths["features"])),
                      {"cv_strategy": strategy} if strategy != "spatial_blocks" else {})

        def run():
            from .model import config_params
            params = config_params(m)
            extra = {"cv_strategy": self._cv_strategy()} if strategy != "spatial_blocks" else {}
            t = experiments.run(Samples.load(self.out / "samples.npz"), features.load_manifest(self.paths["features"]),
                                self.out / "experiments", exps, params, m["cv_folds"], m["spatial_block_m"],
                                self.cfg["seed"], self.cfg["classes"]["targets"],
                                spatial_buffer_m=m["spatial_buffer_m"], calibration_mode=m["calibration_mode"],
                                calibration_fraction=m["calibration_fraction"],
                                holdout_fraction=m["holdout_fraction"], **extra)
            return {"rows": int(len(t))}
        return self.stage("experiments", fp, [target, target.with_suffix(".json")], run)


def _apply_marker_rules(target: Path, markers, rules: dict | None, id_field: str, support=None,
                        temporal_rules: dict | None = None) -> dict:
    """Re-evaluate the traffic light of a parcel product with the event markers and `cbm.rules`, and/or the
    temporal support and `cbm.temporal_support`."""
    from collections import Counter

    import geopandas as gpd

    from .cbm import traffic_light_details
    from .cbm_markers import EVENT_COLUMNS
    gdf = gpd.read_file(target)
    if markers is not None:
        mk = markers.assign(parcel_id=markers.parcel_id.astype(str)).set_index("parcel_id")
        mk = mk.reindex(gdf[id_field].astype(str))
        for col in EVENT_COLUMNS:
            if col in mk:
                gdf[col] = mk[col].to_numpy()
    if support is not None:
        from .temporal_quality import attach
        gdf = attach(gdf, support, id_field)
    details = traffic_light_details(gdf, rules, temporal_rules)
    gdf["traffic_light"] = details["traffic_light"].to_numpy()
    gdf["traffic_light_reasons"] = details["traffic_light_reasons"].to_numpy()
    target.unlink(missing_ok=True)
    gdf.to_file(target, driver="GPKG")
    gdf.drop(columns=gdf.geometry.name).to_csv(target.with_suffix(".csv"), index=False)
    summary_path = target.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    summary["traffic_light_counts"] = gdf["traffic_light"].value_counts().to_dict()
    summary["traffic_light_reason_counts"] = dict(sorted(Counter(
        code for codes in gdf["traffic_light_reasons"] for code in str(codes).split(";") if code).items()))
    summary["marker_rules"] = rules
    if temporal_rules:
        summary["temporal_rules"] = temporal_rules
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    return summary


def run(config_path: str | Path, stages=None, force: bool = False, log=print) -> dict:
    cfg = cfgmod.load(config_path)
    runner = Runner(cfg, force=force, log=log)
    chosen = list(stages) if stages else list(STAGES)
    unknown = set(chosen) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}; choose from {STAGES}")
    if cfg["model"]["params"]:
        from .model import config_params
        config_params(cfg["model"])          # fail before hours of compositing, not at the train stage
    if cfg["parcel_classify"]["enabled"]:
        runner._parcel_params()
    results = {}
    global _HASHER
    try:
        with RunLock(runner.out):
            runner._note(f"geocrop {__version__}: stages {chosen}, config {config_path}", echo=False)
            for name in STAGES:
                if name in chosen:
                    results[name] = getattr(runner, name)()
            from .provenance import run_manifest
            run_manifest(runner.out, config=cfg,
                         inputs=[runner.paths[k] for k in ("training", "parcels", "observations")])
    finally:
        runner.hashes.save()
        _HASHER = None
    return results
