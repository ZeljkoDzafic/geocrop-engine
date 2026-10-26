"""`geocrop run` embeddings stage with feature providers: every on-VM provider gets the same monthly composites,
lists of providers (namespaced names), `namespace`, licence acknowledgement as on the CLI, the stage
fingerprint, licences in the model registry and the pins re-checked by tools/check_weights.py.

The providers are tiny random-weight stand-ins registered through the plugin registry (no downloads); the
torch-based one is skipped cleanly without torch. All data are synthetic (geocrop.demo).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml

from geocrop import config, embeddings, feature_providers, features, pipeline, plugins, registry

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["embeddings", "features", "samples", "train"]
NC = "CC-BY-NC-4.0"


def _s2_mean(s2) -> np.ndarray:
    """(N, 10) mean over the months of the S2 series; NaN rows for pixels without any valid observation."""
    v = feature_providers._stack_s2(s2)                                   # (N, T, 10)
    ok = np.isfinite(v).all(axis=-1, keepdims=True)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, v, 0).sum(axis=1) / n, np.nan)


class TinyMonthly(feature_providers.MonthlyProvider):
    """numpy stand-in encoder: a fixed random projection of the mean monthly S2 spectrum (no weights)."""
    name = "tiny_monthly"
    dims = 4
    sensor = "TinyMonthly"
    product = "random-projection stand-in (tests)"
    licence = "MIT"
    code_licence = "MIT"
    attribution = "test stand-in"
    contract = feature_providers.InputContract("monthly", ("Sentinel-2 (monthly median)",), ("B2",), "months",
                                               "none")

    def load_model(self, weights=None, **_):
        return np.random.default_rng(0).normal(size=(10, self.dims))

    def model_dims(self, model) -> int:
        return int(model.shape[1])

    def encode(self, model, s2, s1, latlons, month0, batch_size, state=None):
        if not s2:
            raise ValueError("needs S2 composites")
        return (_s2_mean(s2) @ model).astype("float32")


class TinyTorch(TinyMonthly):
    """torch stand-in encoder: a randomly initialised linear layer (skipped without torch)."""
    name = "tiny_torch"
    dims = 3
    sensor = "TinyTorch"

    def load_model(self, weights=None, **_):
        import torch
        torch.manual_seed(0)
        return torch.nn.Linear(10, self.dims)

    def model_dims(self, model) -> int:
        return int(model.out_features)

    def encode(self, model, s2, s1, latlons, month0, batch_size, state=None):
        import torch
        x = _s2_mean(s2)
        ok = np.isfinite(x).all(axis=1)
        out = np.full((len(x), self.dims), np.nan, dtype="float32")
        with torch.no_grad():
            out[ok] = model(torch.from_numpy(x[ok].astype("float32"))).numpy()
        return out


class TinyRestricted:
    """Pre-computed stand-in whose data licence is not permissive (needs --allow-license / allow_license)."""
    name = "tiny_restricted"
    dims = 2
    sensor = "TinyRestricted"
    product = "restricted stand-in (tests)"
    licence = NC
    code_licence = "MIT"
    attribution = "test stand-in"
    dequantisation = "none"
    weights = ()
    requires_licence_ack = True
    legacy_names = False
    contract = feature_providers.InputContract("precomputed", ("test",), ("a", "b"), "annual", "none")

    def load_model(self, weights=None, **_):
        return None

    def model_dims(self, model) -> int:
        return self.dims

    def embed_pixels(self, reference, output, *, year, names=None, **_):
        with rasterio.open(reference) as ref:
            shape = ref.shape
        data = np.random.default_rng(1).normal(size=(self.dims, *shape)).astype("float32")
        with embeddings.EmbeddingWriter(output, reference, self.name, year, self.dims, names=names) as sink:
            sink.write(data)
        return Path(output)

    def embed_parcels(self, reference, parcels, output, *, year, raster=None, id_field="parcel_id", **_):
        return embeddings.raster_parcel_means(raster, parcels, output, id_field)


STAND_INS = (TinyMonthly, TinyTorch, TinyRestricted)


@pytest.fixture(scope="module", autouse=True)
def stand_ins():
    for cls in STAND_INS:
        plugins.feature_providers.register(cls.name, cls(), replace=True)
    yield
    for cls in STAND_INS:
        plugins.feature_providers.unregister(cls.name)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from geocrop.demo import generate_scenes
    root = tmp_path_factory.mktemp("embstage")
    d = generate_scenes(root / "data", parcels_per_side=3, parcel_px=8)
    cfg = {"windows": {"P1": {"start": "2023-04-01", "end": "2023-05-31"},
                       "P2": {"start": "2023-06-01", "end": "2023-07-15"},
                       "P3": {"start": "2023-07-16", "end": "2023-08-31"}},
           "grid": {"reference": d["reference"]},
           "paths": {"composites": d["composites"], "s2_scenes": str(Path(d["scene_root"]) / "s2"),
                     "s1_scenes": str(Path(d["scene_root"]) / "s1"), "training": d["training"],
                     "parcels": d["parcels"], "observations": d["observations"]},
           "model": {"n_estimators": 10, "n_jobs": 1, "cv_folds": 2, "spatial_block_m": 200,
                     "max_samples_per_polygon": 20},
           "experiments": {"S2": {"groups": ["s2_index"]}}}
    return {"root": root, "cfg": cfg, "data": d}


def _config(study, name: str, emb: dict | None) -> Path:
    cfg = copy.deepcopy(study["cfg"])
    out = study["root"] / name
    cfg["paths"].update(output=str(out / "run"), features=str(out / "stack.tif"))
    if emb is not None:
        cfg["features"] = {"embeddings": {"year": 2023, "start": "2023-05", "months": 3, **emb}}
    path = study["root"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(path: Path, stages=STAGES):
    logs = []
    result = pipeline.run(path, stages, log=logs.append)
    return result, {line.split()[-1] for line in logs if line.startswith("[run ]")}


def _stack_names(path: Path) -> list[str]:
    return [f.name for f in features.load_manifest(path)]


# --------------------------------------------------------------------------- end to end


def test_run_two_providers_one_stack_via_cli(study, capsys):
    """`geocrop run` with a list (on-VM encoder + pre-computed source): one raster each, namespaced names, both
    attached; the monthly composites are built once from the scene folders; an unchanged rerun skips."""
    from geocrop.cli import main
    path = _config(study, "two", {"source": ["tiny_monthly", "tiny_restricted"], "allow_license": [NC]})
    main(["run", "--config", str(path), "--stages", *STAGES])
    text = capsys.readouterr().out                                  # stage log lines, then the JSON result
    printed = json.loads(text[text.index("\n{") + 1:] if not text.startswith("{") else text)
    sources = printed["embeddings"]["sources"]
    assert [s["source"] for s in sources] == ["tiny_monthly", "tiny_restricted"]
    out = study["root"] / "two"
    assert sorted(p.name for p in (out / "embeddings" / "monthly").iterdir()) == ["2023-05", "2023-06", "2023-07"]
    names = _stack_names(out / "stack.tif")
    assert [n for n in names if n.startswith("emb_")] == [*(f"emb_tiny_monthly_{i:02d}" for i in range(4)),
                                                          "emb_tiny_restricted_00", "emb_tiny_restricted_01"]
    with rasterio.open(out / "embeddings" / "tiny_monthly_2023.tif") as ds:
        data = ds.read(masked=True)
        assert ds.tags()["start"] == "2023-05" and ds.tags()["months"] == "3"
    assert np.isfinite(data.compressed()).all() and data.count() > 0
    assert printed["features"]["embeddings"] == [str(out / "embeddings" / "tiny_monthly_2023.tif"),
                                                 str(out / "embeddings" / "tiny_restricted_2023.tif")]
    _, again = _run(path)
    assert again == set()


def test_licence_acknowledgement_refused_then_accepted(study):
    path = _config(study, "licence", {"source": "tiny_restricted"})
    with pytest.raises(feature_providers.LicenceNotAcknowledged, match=f"--allow-license {NC}"):
        _run(path)
    out = study["root"] / "licence"
    assert not (out / "embeddings").exists()            # refused before anything was computed or downloaded
    path = _config(study, "licence", {"source": "tiny_restricted", "allow_license": [NC]})
    _, ran = _run(path)
    assert {"embeddings", "features"} <= ran
    with rasterio.open(out / "embeddings" / "tiny_restricted_2023.tif") as ds:
        assert ds.tags()["licence"] == NC
    # acknowledging a further licence changes no output: nothing re-runs; removing the acknowledgement refuses
    _, again = _run(_config(study, "licence", {"source": "tiny_restricted", "allow_license": [NC, "other"]}))
    assert again == set()
    with pytest.raises(feature_providers.LicenceNotAcknowledged):
        _run(_config(study, "licence", {"source": "tiny_restricted", "allow_license": ["other"]}))


@pytest.mark.parametrize("namespace, expected", [(None, "emb_tiny_monthly_00"), (False, "emb_00"),
                                                 (True, "emb_tiny_monthly_00")])
def test_namespace_is_honoured(study, namespace, expected):
    emb = {"source": "tiny_monthly", "attach": False}
    if namespace is not None:
        emb["namespace"] = namespace
    path = _config(study, f"ns_{namespace}", emb)
    _run(path, ["embeddings"])
    with rasterio.open(study["root"] / f"ns_{namespace}" / "embeddings" / "tiny_monthly_2023.tif") as ds:
        assert ds.descriptions[0] == expected


def test_torch_stand_in_end_to_end(study):
    pytest.importorskip("torch")
    path = _config(study, "torch", {"source": "tiny_torch"})
    _, ran = _run(path)
    assert {"embeddings", "features", "samples", "train"} <= ran
    assert {"emb_tiny_torch_00", "emb_tiny_torch_02"} <= set(_stack_names(study["root"] / "torch" / "stack.tif"))


def test_galileo_and_anysat_get_the_shared_monthly_composites(study, monkeypatch):
    """Every on-VM provider (here the built-in Galileo and AnySat, without downloads) receives the same monthly
    composites, built once; namespace and licence acknowledgements are passed through."""
    from geocrop import embedding_sources
    calls, built = [], []

    def extract(source, reference, out_dir, year, monthly=None, **kwargs):
        calls.append({"source": source, "monthly": monthly, **kwargs})
        target = embeddings.output_path(out_dir, source, year)
        names = feature_providers.feature_names(feature_providers.get(source), TinyRestricted.dims,
                                               kwargs["namespace"])
        TinyRestricted().embed_pixels(reference, target, year=year, names=names)
        return {"source": source, "year": year, "raster": str(target)}

    real = embeddings.monthly_composites
    monkeypatch.setattr(embedding_sources, "extract", extract)
    monkeypatch.setattr(embeddings, "monthly_composites", lambda *a, **k: built.append(a) or real(*a, **k))
    path = _config(study, "galileo_anysat", {"source": ["galileo", "anysat"], "attach": False})
    result, ran = _run(path, ["embeddings"])
    assert ran == {"embeddings"} and len(built) == 1
    monthly = study["root"] / "galileo_anysat" / "embeddings" / "monthly"
    assert [c["source"] for c in calls] == ["galileo", "anysat"]
    assert all(Path(c["monthly"]) == monthly and c["namespace"] is True and c["start"] == "2023-05"
               and c["months"] == 3 for c in calls)
    assert [s["raster"].endswith(f"{n}_2023.tif") for s, n in zip(result["embeddings"]["sources"],
                                                                  ("galileo", "anysat"))] == [True, True]


# --------------------------------------------------------------------------- fingerprints


def _stage_fp(study, name: str, emb: dict | None) -> str | None:
    cfg = config.load(_config(study, name, emb))
    runner = pipeline.Runner(cfg, log=lambda *_: None)
    runner.stage = lambda stage, fp, outputs, fn: fp
    try:
        return runner.embeddings()
    finally:
        pipeline._HASHER = None


def test_fingerprint_covers_providers_weights_namespace_and_monthly(study, monkeypatch, tmp_path):
    assert _stage_fp(study, "fp", None) is None                     # embeddings off: no stage, no fingerprint
    monthly = tmp_path / "monthly"
    embeddings.monthly_composites(study["data"]["reference"], monthly, "2023-05", 3,
                                  study["cfg"]["paths"]["s2_scenes"])
    base_opts = {"source": "tiny_monthly", "monthly": str(monthly)}
    base = _stage_fp(study, "fp", base_opts)
    assert base == _stage_fp(study, "fp", base_opts)
    assert _stage_fp(study, "fp", {**base_opts, "allow_license": [NC]}) == base     # no output changes
    assert _stage_fp(study, "fp", {**base_opts, "attach": False}) == base           # attach is the features' part
    changed = {
        "provider set": _stage_fp(study, "fp", {**base_opts, "source": ["tiny_monthly", "tiny_restricted"],
                                                "allow_license": [NC]}),
        "namespace": _stage_fp(study, "fp", {**base_opts, "namespace": False}),
        "months": _stage_fp(study, "fp", {**base_opts, "months": 2}),
    }
    monkeypatch.setattr(TinyMonthly, "weights", (feature_providers.WeightFile("w.pt", "https://example.org/w.pt",
                                                                              "0" * 64),))
    changed["weights"] = _stage_fp(study, "fp", base_opts)
    monkeypatch.setattr(TinyMonthly, "weights", (feature_providers.WeightFile("w.pt", "https://example.org/w.pt",
                                                                              "1" * 64),))
    changed["weights sha256"] = _stage_fp(study, "fp", base_opts)
    monkeypatch.undo()
    assert _stage_fp(study, "fp", base_opts) == base
    tif = next(monthly.rglob("B04.tif"))
    with rasterio.open(tif, "r+") as ds:                            # same files, different content
        ds.write(ds.read(1) * 1.01, 1)
    changed["monthly content"] = _stage_fp(study, "fp", base_opts)
    assert len({base, *changed.values()}) == len(changed) + 1, changed


def test_default_run_fingerprints_unchanged():
    """With embeddings off, the features fingerprint does not see the embeddings options (see also
    tests/test_pipeline_options.py::test_default_fingerprints_unchanged_from_main)."""
    cfg = config.load(overrides={"paths": {"output": "run"}})
    assert config.embedding_options(cfg) is None
    assert not pipeline._embeddings_attached(cfg)


def test_checkpoint_with_several_sources_is_rejected():
    with pytest.raises(config.ConfigError, match="checkpoint"):
        config.load(overrides={"features": {"embeddings": {"source": ["presto", "galileo"], "year": 2023,
                                                           "checkpoint": "w.pt"}}})


# --------------------------------------------------------------------------- licences in the model registry


def test_registry_records_every_provider_licence():
    manifest = [{"name": f"emb_{p}_00", "group": "emb", "sensor": s}
                for p, s in (("galileo", "Galileo"), ("anysat", "AnySat"), ("tiny_restricted", "TinyRestricted"))]
    manifest += [{"name": "emb_00", "group": "emb", "sensor": "Presto"}, {"name": "NDVI_P1", "group": "s2_index"}]
    lic = {r["component"]: r for r in registry._licences(manifest, {"extra": "MIT"})}
    assert lic["galileo embeddings"]["licence"] == "MIT" and lic["anysat embeddings"]["licence"] == "MIT"
    assert lic["galileo embeddings"]["weights"][1]["sha256"] == feature_providers.GALILEO.weights[1].sha256
    assert lic["tiny_restricted embeddings"]["licence"] == NC
    assert lic["presto embeddings"]["licence"].startswith("MIT") and lic["extra"]["licence"] == "MIT"


def test_pretrained_entries_include_every_provider_with_weights(monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setattr(TinyMonthly, "weights", (feature_providers.WeightFile("w.pt", "https://example.org/w.pt",
                                                                              "a" * 64, 10),))
    entries = {e["name"]: e for e in registry.pretrained_entries()}
    assert entries["anysat/default"]["licence"] == "MIT"
    assert entries["anysat/default"]["artifact"]["sha256"] == feature_providers.ANYSAT.weights[0].sha256
    galileo = [e for name, e in entries.items() if name.startswith("galileo/")]
    assert len(galileo) == 2 and {e["licence"] for e in galileo} == {"MIT"}
    assert entries["tiny_monthly/default"]["artifact"]["url"] == "https://example.org/w.pt"
    assert not any(name.startswith("presto/") and name != "presto/default" for name in entries)
    assert registry.verify_entry(entries["tiny_monthly/default"])["status"] == "not cached"


# --------------------------------------------------------------------------- tools/check_weights.py


def test_check_weights_reads_transfer_protocol_sources(monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    sys.modules.pop("check_weights", None)
    import check_weights
    pins = check_weights.pins()
    by_url = {p["url"]: p for p in pins}
    kmrs = by_url["https://rkg.gov.si/razno/portal_analysis/KMRS_2025.zip"]
    assert kmrs["checksum"].startswith("sha256:2977a4163ce6") and kmrs["name"].startswith("kmrs_si_2025:")
    assert len(by_url) == len(pins)                                     # each pinned file checked once
    assert any(p["name"].startswith("anysat:") for p in pins)
    body = {"sources": {"a": {"kind": "eurocrops", "zenodo_record": "1", "archive": "X.zip", "checksum": "md5:0"},
                        "b": {"kind": "url", "url": "https://example.org/b.zip", "checksum": "sha256:1"}}}
    assert [p["name"] for p in check_weights.protocol_sources(body, "p.yaml")] == ["a:X.zip (p.yaml)",
                                                                                   "b:b.zip (p.yaml)"]
