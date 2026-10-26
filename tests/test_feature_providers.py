"""Feature-provider adapter interface: registry, contracts, namespaced feature names (several embedding sources in
one stack, legacy stacks still readable), licence acknowledgement, and Galileo / AnySat end to end on synthetic
monthly composites with random-weight stand-ins. No downloads in this tier; real access is marked `network`."""
import json
import shutil

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from geocrop import config, embedding_sources, embeddings, feature_providers, plugins
from geocrop.features import attach_embeddings, load_manifest, manifest_path, select

BUILTINS = ("alphaearth", "tessera", "presto", "galileo", "anysat")


def test_registry_lists_builtins_with_contracts():
    names = feature_providers.names()
    assert set(BUILTINS) <= set(names)
    for n in BUILTINS:
        p = feature_providers.get(n)
        assert isinstance(p, feature_providers.FeatureProvider) and p.name == n and p.dims > 0
        assert p.contract.kind in ("precomputed", "monthly") and p.contract.bands and p.licence
        for w in p.weights:
            assert w.url.startswith("https://") and len(w.sha256) == 64
        assert p.requires_licence_ack is False                  # every built-in is permissive
    assert feature_providers.get("galileo").weights[1].url.endswith("/data/models/nano/encoder.pt")
    assert "@" not in feature_providers.get("anysat").weights[0].url and "/resolve/63f2521" in \
        feature_providers.get("anysat").weights[0].url
    assert plugins.describe()["feature_providers"]["presto"] == plugins.BUILTIN
    rows = {r["name"]: r for r in feature_providers.table()}
    assert rows["anysat"]["dims"] == 768 and rows["tessera"]["kind"] == "precomputed"


def test_registry_rejects_incomplete_providers():
    with pytest.raises(TypeError, match="missing"):
        plugins.feature_providers.register("broken", object())
    with pytest.raises(ValueError, match="names must match"):
        plugins.feature_providers.register("alias", feature_providers.GalileoProvider())


def test_feature_names_and_parsing():
    assert embeddings.band_names(3) == ["emb_00", "emb_01", "emb_02"]
    assert embeddings.band_names(2, "galileo") == ["emb_galileo_00", "emb_galileo_01"]
    assert embeddings.band_names(768, "anysat")[-1] == "emb_anysat_767"
    assert embeddings.parse_band_name("emb_07") == (None, 7)
    assert embeddings.parse_band_name("emb_presto_127") == ("presto", 127)
    assert embeddings.parse_band_name("emb_my_model_03") == ("my_model", 3)
    assert embeddings.parse_band_name("P1_B04") is None and embeddings.parse_band_name("emb_x") is None
    presto, galileo = feature_providers.get("presto"), feature_providers.get("galileo")
    assert feature_providers.feature_names(presto)[0] == "emb_00"                 # legacy default kept
    assert feature_providers.feature_names(presto, namespace=True)[0] == "emb_presto_00"
    assert feature_providers.feature_names(galileo)[0] == "emb_galileo_00"
    assert feature_providers.feature_names(galileo, namespace=False)[0] == "emb_00"


def _write_embedding(path, reference, source, dims, names=None, seed=0):
    with rasterio.open(reference) as ref:
        shape = ref.shape
    data = np.random.default_rng(seed).normal(size=(dims, *shape)).astype("float32")
    with embeddings.EmbeddingWriter(path, reference, source, 2023, dims, names=names) as sink:
        sink.write(data)
    return path


def test_two_sources_in_one_stack_and_legacy_reading(tmp_path, stack):
    """A legacy stack (emb_NN from AlphaEarth) accepts a second legacy source under namespaced names, and
    providers can be selected by name in both naming schemes."""
    copy = tmp_path / "stack.tif"
    shutil.copy(stack, copy)
    shutil.copy(manifest_path(stack), manifest_path(copy))
    n0 = len(load_manifest(copy))
    ae = _write_embedding(tmp_path / "e" / "alphaearth_2023.tif", stack, "alphaearth", 64)
    pr = _write_embedding(tmp_path / "e" / "presto_2023.tif", stack, "presto", 128, seed=1)
    ga = _write_embedding(tmp_path / "e" / "galileo_2023.tif", stack, "galileo", 128,
                          names=embeddings.band_names(128, "galileo"), seed=2)
    attach_embeddings(copy, ae)
    attach_embeddings(copy, pr)                          # emb_00 collides -> attached as emb_presto_NN
    attach_embeddings(copy, ga)
    feats = load_manifest(copy)
    assert len(feats) == n0 + 64 + 128 + 128 and len({f.name for f in feats}) == len(feats)
    assert len(select(feats, groups=["emb"])) == 64 + 128 + 128
    assert embeddings.embedding_features(feats, "alphaearth") == embeddings.band_names(64)       # legacy names
    assert embeddings.embedding_features(feats, "presto") == embeddings.band_names(128, "presto")
    assert embeddings.embedding_features(feats, "galileo") == embeddings.band_names(128, "galileo")
    with rasterio.open(copy) as ds, rasterio.open(pr) as src:
        names = list(ds.descriptions)
        np.testing.assert_array_equal(ds.read(names.index("emb_presto_05") + 1), src.read(6))
    with rasterio.open(pr) as ds:                                  # the source raster itself is untouched
        assert ds.descriptions[0] == "emb_00"


def test_raster_parcel_means(tmp_path, monthly):
    raster = _write_embedding(tmp_path / "tessera_2023.tif", monthly["reference"], "tessera", 4)
    table = embeddings.raster_parcel_means(raster, monthly["parcels"], tmp_path / "p.csv")
    assert len(table) == len(gpd.read_file(monthly["parcels"])) and list(table.columns[2:]) == embeddings.band_names(4)
    assert np.isfinite(table[embeddings.band_names(4)].to_numpy()).all()


# --------------------------------------------------------------------------- licence acknowledgement


class _RestrictedProvider:
    """A plugin-style pre-computed provider whose data licence is not permissive."""
    name = "restricted"
    dims = 3
    sensor = "Restricted"
    product = "test product"
    licence = "CC-BY-NC-4.0"
    code_licence = "MIT"
    attribution = "test"
    dequantisation = "none"
    weights = ()
    requires_licence_ack = True
    legacy_names = False
    contract = feature_providers.InputContract("precomputed", ("test",), ("a", "b", "c"), "annual", "none")

    def load_model(self, weights=None, **_):
        return None

    def embed_pixels(self, reference, output, *, year, names=None, **_):
        return _write_embedding(output, reference, self.name, self.dims, names)

    def embed_parcels(self, reference, parcels, output, **_):
        raise NotImplementedError


def test_licence_must_be_acknowledged(tmp_path, stack):
    plugins.feature_providers.register("restricted", _RestrictedProvider())
    try:
        with pytest.raises(feature_providers.LicenceNotAcknowledged, match="--allow-license CC-BY-NC-4.0"):
            embedding_sources.extract("restricted", stack, tmp_path, 2023)
        res = embedding_sources.extract("restricted", stack, tmp_path, 2023, allow_license=["CC-BY-NC-4.0"])
        with rasterio.open(res["raster"]) as ds:
            assert ds.descriptions == ("emb_restricted_00", "emb_restricted_01", "emb_restricted_02")
            assert ds.tags()["licence"] == "CC-BY-NC-4.0"
        cfg = config.load(overrides={"features": {"embeddings": {"source": "restricted", "year": 2023,
                                                                 "allow_license": ["CC-BY-NC-4.0"]}}})
        assert config.embedding_options(cfg)["allow_license"] == ["CC-BY-NC-4.0"]
    finally:
        plugins.feature_providers.unregister("restricted")


def test_config_accepts_provider_lists():
    cfg = config.load(overrides={"features": {"embeddings": {"source": ["presto", "galileo"], "year": 2023}}})
    opts = config.embedding_options(cfg)
    assert config.embedding_sources(opts) == ["presto", "galileo"] and config.embedding_namespace(opts) is True
    single = config.embedding_options(config.load(overrides={"features": {"embeddings": {"source": "anysat",
                                                                                          "year": 2023}}}))
    assert config.embedding_sources(single) == ["anysat"] and config.embedding_namespace(single) is None
    for bad in ({"source": ["presto", "presto"], "year": 2023}, {"source": ["presto", "satmae"], "year": 2023},
                {"source": ["presto", "galileo"], "year": 2023, "namespace": False},
                {"source": "galileo", "year": 2023, "allow_license": "MIT"}):
        with pytest.raises(config.ConfigError):
            config.load(overrides={"features": {"embeddings": bad}})


# --------------------------------------------------------------------------- Galileo / AnySat end to end


@pytest.fixture(scope="module")
def monthly(tmp_path_factory):
    from geocrop.demo import generate_scenes

    root = tmp_path_factory.mktemp("fp")
    sc = generate_scenes(root / "demo", parcels_per_side=3, parcel_px=8)
    embeddings.monthly_composites(sc["reference"], root / "monthly", "2023-03", 6, sc["scene_root"], sc["scene_root"])
    return {**sc, "monthly": root / "monthly"}


def test_galileo_inputs_contract():
    T, N = 2, 3
    s2 = {b: np.full((T, N), 0.1, "float32") for b in embeddings.GEOCROP_S2.values()}
    s2["B08"][:] = 0.3
    s2["B12"][1, 2] = np.nan
    s1 = {"VV": np.full((T, N), 0.01, "float32"), "VH": np.full((T, N), 0.001, "float32")}
    s1["VH"][0, 1] = 0
    x, mask = feature_providers.galileo_inputs(s2, s1)
    assert x.shape == (N, T, 13) and mask.shape == (N, T, 7)
    np.testing.assert_allclose(x[0, 0, :2], [-20, -30], rtol=1e-5)          # dB, unshifted
    np.testing.assert_allclose(x[0, 0, 8], 3000)                             # B8 in DN
    np.testing.assert_allclose(x[0, 0, 12], 0.5, rtol=1e-6)                  # NDVI
    assert mask[2, 1, 1:].all() and not mask[2, 1, 0] and (x[2, 1, 2:] == 0).all()
    assert mask[1, 0, 0] and not mask[1, 0, 1:].any()
    _, m2 = feature_providers.galileo_inputs(s2, None)
    assert m2[:, :, 0].all()


def test_month_day_of_year():
    np.testing.assert_array_equal(feature_providers.month_day_of_year(0, 3), [15, 46, 74])
    np.testing.assert_array_equal(feature_providers.month_day_of_year(11, 2), [349, 15])      # wraps into January


def test_galileo_provider_pixels_and_parcels(tmp_path, monthly):
    pytest.importorskip("torch")
    import torch

    from geocrop import galileo_encoder

    torch.manual_seed(0)
    p = feature_providers.get("galileo")
    model = galileo_encoder.load(None)
    out = p.embed_pixels(monthly["reference"], tmp_path / "galileo_2023.tif", monthly=monthly["monthly"],
                         start="2023-03", months=6, model=model, block_size=10)
    with rasterio.open(out) as ds, rasterio.open(monthly["reference"]) as ref:
        assert ds.count == 128 and ds.shape == ref.shape and ds.descriptions[0] == "emb_galileo_00"
        v = ds.read()
        assert (v != ds.nodata).all(axis=0).mean() > 0.9 and ds.tags()["embedding_source"] == "galileo"
        assert ds.tags()["licence"] == "MIT"
    feats = load_manifest(out)
    assert feats[0].group == "emb" and feats[0].sensor == "Galileo"
    table = p.embed_parcels(monthly["reference"], monthly["parcels"], tmp_path / "parcels.csv",
                            monthly=monthly["monthly"], start="2023-03", months=6, model=model)
    cols = embeddings.band_names(128, "galileo")
    assert np.isfinite(table[cols].to_numpy()).all() and table["n_pixels"].min() > 0


def test_anysat_provider_pixels_and_parcels(tmp_path, monthly):
    pytest.importorskip("torch")
    import torch

    from geocrop import anysat_encoder

    torch.manual_seed(0)
    p = feature_providers.get("anysat")
    model = anysat_encoder.load(None, size="tiny")
    out = p.embed_pixels(monthly["reference"], tmp_path / "anysat_2023.tif", monthly=monthly["monthly"],
                         start="2023-03", months=6, model=model, block_size=16)
    with rasterio.open(out) as ds:
        assert ds.count == model.embedding_size and ds.descriptions[-1] == f"emb_anysat_{model.embedding_size - 1:02d}"
        assert (ds.read() != ds.nodata).all(axis=0).mean() > 0.9
        state = json.loads(ds.tags()["provider_state"])
    assert len(state["s2_mean"]) == 10 and all(s > 0 for s in state["s2_std"]) and state["s2_observations"] > 0
    table = p.embed_parcels(monthly["reference"], monthly["parcels"], tmp_path / "parcels.parquet",
                            monthly=monthly["monthly"], start="2023-03", months=6, model=model, state=state)
    assert np.isfinite(table[embeddings.band_names(model.embedding_size, "anysat")].to_numpy()).all()
    with pytest.raises(ValueError, match="statistics"):
        p.encode(model, {b: np.zeros((2, 1), "float32") for b in embeddings.GEOCROP_S2.values()}, None,
                 np.zeros((1, 2), "float32"), 0, 8, {})


def test_cli_galileo_with_local_weights(tmp_path, monthly):
    pytest.importorskip("torch")
    import torch

    from geocrop import galileo_encoder
    from geocrop.cli import main

    folder = tmp_path / "nano"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": {"encoder": galileo_encoder.NANO_CONFIG}}))
    torch.save(galileo_encoder.load(None).state_dict(), folder / "encoder.pt")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths: {{features: {tmp_path / 'features' / 'stack.tif'}}}\n"
                   f"features:\n  embeddings: {{source: galileo, year: 2023, start: '2023-03', months: 6, "
                   f"monthly: {monthly['monthly']}}}\n")
    main(["embeddings", "--config", str(cfg), "--reference", str(monthly["reference"]), "--checkpoint", str(folder),
          "--parcels", str(monthly["parcels"])])
    out = tmp_path / "features" / "embeddings" / "galileo_2023.tif"
    with rasterio.open(out) as ds:
        assert ds.descriptions[0] == "emb_galileo_00" and ds.tags()["checkpoint"] == str(folder)
    assert out.with_name("galileo_2023_parcels.csv").exists()
    main(["embeddings", "--list"])


def test_fetch_weights_verifies_every_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(embedding_sources, "download", lambda url, target, sha256=None, session=None, size=None:
                        calls.append((url, target.name, sha256)) or target)
    paths = embedding_sources.fetch_weights(feature_providers.get("galileo"), tmp_path)
    assert [c[1] for c in calls] == [w.name for w in feature_providers.get("galileo").weights]
    assert all(len(c[2]) == 64 for c in calls) and len(paths) == 2
    with pytest.raises(ValueError, match="no weights"):
        embedding_sources.fetch_weights(feature_providers.get("alphaearth"))


# --------------------------------------------------------------------------- real downloads (network)


@pytest.mark.network
def test_galileo_weights_download_and_load(tmp_path):
    pytest.importorskip("torch")
    try:
        paths = embedding_sources.fetch_weights(feature_providers.get("galileo"), tmp_path)
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"GitHub not reachable: {exc}")
    model = feature_providers.get("galileo").load_model(paths)
    assert model.embedding_size == 128


@pytest.mark.network
@pytest.mark.slow
def test_anysat_weights_download_and_load(tmp_path):
    pytest.importorskip("torch")
    try:
        path = embedding_sources.fetch_weights(feature_providers.get("anysat"))      # 504 MB, cached
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"Hugging Face not reachable: {exc}")
    assert feature_providers.get("anysat").load_model(path).embedding_size == 768
