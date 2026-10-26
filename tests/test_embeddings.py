"""Foundation-model embeddings (gaps G2/G3): dequantisation, regridding, output contract, Presto input contract
and inference with a random-weight stand-in, feature-stack/ablation integration. No downloads in this tier;
real access tests are marked `network`."""
import io
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from geocrop import config, embedding_sources, embeddings
from geocrop.features import load_manifest, select

UTM = CRS.from_epsg(32634)


def _grid(path: Path, width=12, height=10, origin=(350000.0, 4955000.0)) -> Path:
    profile = {"driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "float32",
               "crs": UTM, "transform": Affine(10, 0, origin[0], 0, -10, origin[1]), "nodata": -9999.0}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.zeros((1, height, width), "float32"))
    return path


# --------------------------------------------------------------------------- dequantisation / regridding


def test_dequantise_alphaearth_matches_spec_and_masks_nodata():
    raw = np.array([-128, -127, -64, 0, 64, 127], dtype="int8")
    got = embeddings.dequantise_alphaearth(raw)
    assert np.isnan(got[0])
    expected = (raw[1:] / 127.5) ** 2 * np.sign(raw[1:])
    np.testing.assert_allclose(got[1:], expected, rtol=1e-6)
    # unit vectors survive the published quantisation (sign * sqrt(|v|) * 127.5) to within ~1 %
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 64))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = np.clip(np.round(np.sign(v) * np.sqrt(np.abs(v)) * 127.5), -127, 127).astype("int8")
    norms = np.linalg.norm(embeddings.dequantise_alphaearth(q), axis=1)
    assert np.all(np.abs(norms - 1) < 0.02)


def test_dequantise_tessera_broadcasts_scales_and_masks_invalid():
    q = np.arange(-3, 3, dtype="int8").reshape(1, 2, 3)
    scales = np.array([[0.5, np.inf]], dtype="float32")
    got = embeddings.dequantise_tessera(q, scales)
    np.testing.assert_allclose(got[0, 0], [-1.5, -1.0, -0.5])
    assert np.isnan(got[0, 1]).all()
    full = embeddings.dequantise_tessera(q, np.full((1, 2, 3), 2.0, "float32"))
    np.testing.assert_allclose(full, q * 2.0)


def test_regrid_handles_bottom_up_sources():
    """AlphaEarth COGs are stored bottom-up (positive y pixel size); values must land on the right rows."""
    data = np.arange(1 * 4 * 3, dtype="float32").reshape(1, 4, 3)          # row 0 = southernmost
    bottom_up = Affine(10, 0, 0, 0, 10, 1000)                               # origin = south-west corner
    north_up = Affine(10, 0, 0, 0, -10, 1040)
    out = embeddings.regrid(data, bottom_up, UTM, north_up, UTM, (4, 3))
    np.testing.assert_array_equal(out[0], data[0][::-1])
    window = embedding_sources._source_window(bottom_up, (4, 3), Affine(10, 0, 0, 0, -10, 1020), UTM, (2, 3), UTM,
                                              margin=0)
    assert (window.row_off, window.height) == (0, 2)                        # the two southern rows


def test_tessera_tile_centres():
    assert embedding_sources.tessera_tile_centres((19.10, 44.72, 19.16, 44.76)) == [(19.15, 44.75)]
    assert len(embedding_sources.tessera_tile_centres((19.08, 44.68, 19.21, 44.76))) == 3 * 2


class _RangeSession:
    """Serves byte ranges of a local file like an HTTP server answering Range requests with 206."""

    def __init__(self, blob: bytes):
        self.blob = blob

    def get(self, url, headers=None, timeout=None):
        a, b = (int(x) for x in headers["Range"].split("=")[1].split("-"))

        class R:
            status_code = 206
            content = self.blob[a:b + 1]

            def raise_for_status(self):
                return None
        return R()


def test_remote_npy_reads_row_ranges():
    arr = np.random.default_rng(1).integers(-127, 127, size=(7, 5, 128)).astype("int8")
    buf = io.BytesIO()
    np.save(buf, arr)
    npy = embedding_sources.RemoteNpy("mem://x.npy", _RangeSession(buf.getvalue()))
    assert npy.shape == (7, 5, 128)
    np.testing.assert_array_equal(npy.rows(2, 5), arr[2:5])
    scales = np.random.default_rng(2).random((7, 5)).astype("float32")
    buf = io.BytesIO()
    np.save(buf, scales)
    np.testing.assert_array_equal(embedding_sources.RemoteNpy("mem://s.npy", _RangeSession(buf.getvalue())).rows(6, 7),
                                  scales[6:7])


def test_download_rejects_wrong_checksum(tmp_path):
    class Session:
        def get(self, url, stream=True, timeout=None):
            class R:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def raise_for_status(self):
                    return None

                def iter_content(self, n):
                    yield b"not a checkpoint"
            return R()

    with pytest.raises(ValueError, match="SHA-256"):
        embedding_sources.download("https://example.org/x.pt", tmp_path / "x.pt", "0" * 64, Session())
    assert not (tmp_path / "x.pt").exists()


# --------------------------------------------------------------------------- output contract / features


def test_writer_contract_and_stack_integration(tmp_path, stack):
    with rasterio.open(stack) as s:
        ref_shape = s.shape
    out = embeddings.output_path(tmp_path / "embeddings", "alphaearth", 2023)
    rng = np.random.default_rng(3)
    data = rng.normal(size=(64, *ref_shape)).astype("float32")
    data[:, 0, 0] = np.nan
    with embeddings.EmbeddingWriter(out, stack, "alphaearth", 2023, extra={"tiles": "synthetic"}) as sink:
        sink.write(data)
    with rasterio.open(out) as ds:
        assert ds.count == 64 and ds.descriptions[:2] == ("emb_00", "emb_01") and ds.descriptions[-1] == "emb_63"
        assert "Google DeepMind" in ds.tags()["attribution"] and ds.tags()["licence"] == "CC-BY-4.0"
        assert ds.read(1)[0, 0] == ds.nodata
    body = json.loads(embeddings.write_manifest(out, "alphaearth", 2023, 64).read_text())
    assert body["embedding"]["licence"] == "CC-BY-4.0" and body["features"][0]["group"] == "emb"
    # attach to a copy of the handcrafted stack: manifest gains the emb group, selection works
    from geocrop.features import attach_embeddings, manifest_path
    copy = tmp_path / "stack.tif"
    shutil.copy(stack, copy)
    shutil.copy(manifest_path(stack), manifest_path(copy))
    n_before = len(load_manifest(copy))
    attach_embeddings(copy, out)
    feats = load_manifest(copy)
    assert len(feats) == n_before + 64
    assert select(feats, groups=["emb"]) == embeddings.band_names(64)
    periods = sorted({f.period for f in feats if f.period})
    assert not any(n.startswith("emb_") for n in select(feats, periods=periods[:1]))
    with rasterio.open(copy) as ds:
        assert ds.count == n_before + 64 and ds.descriptions[-1] == "emb_63"
    s = embeddings.summary(out)
    assert s["bands"] == 64 and 0.99 < s["valid_fraction"] < 1


def test_config_embeddings_validation():
    cfg = config.load(overrides={"features": {"embeddings": {"source": "presto", "year": 2023, "start": "2022-10",
                                                             "months": 12}}})
    opts = config.embedding_options(cfg)
    assert opts["source"] == "presto" and opts["dir"].endswith("embeddings") and opts["resampling"] == "nearest"
    assert config.embedding_options(config.load()) is None
    for bad in ({"source": "satmae", "year": 2023}, {"source": "alphaearth"}, {"source": "tessera", "year": 2023,
                                                                                "typo": 1},
                {"source": "presto", "year": 2023, "months": 30}, {"source": "presto", "year": 2023, "start": "2023"}):
        with pytest.raises(config.ConfigError):
            config.load(overrides={"features": {"embeddings": bad}})


def test_ablation_adds_embedding_experiments(tmp_path, samples, stack):
    from geocrop import experiments
    from geocrop.features import Feature
    from geocrop.samples import Samples

    manifest = load_manifest(stack)
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(len(samples.y), 8)).astype("float32")
    names = embeddings.band_names(8)
    s = Samples(np.hstack([samples.X, emb]), samples.y, samples.polygon, samples.x, samples.yy,
                samples.features + names, samples.crs)
    manifest = manifest + [Feature(n, "emb", "AlphaEarth") for n in names]
    t = experiments.run(s, manifest, tmp_path, params={"n_estimators": 15, "n_jobs": 1}, n_splits=3,
                        spatial_block_m=500)
    rows = dict(zip(t.experiment, t.status))
    assert rows["E6_embeddings_only"] == "ok" and rows["E7_handcrafted_plus_embeddings"] == "ok"
    n = dict(zip(t.experiment, t.n_features))
    assert n["E7_handcrafted_plus_embeddings"] == n["E5_handcrafted_all"] + 8
    plain = experiments.run(samples, load_manifest(stack), tmp_path / "plain", params={"n_estimators": 5, "n_jobs": 1},
                            n_splits=3, spatial_block_m=500)
    assert "E6_embeddings_only" not in set(plain.experiment)


# --------------------------------------------------------------------------- Presto


def test_presto_input_contract():
    T, N = 3, 2
    s2 = {b: np.full((T, N), 0.1, "float32") for b in embeddings.GEOCROP_S2.values()}
    s2["B08"][:] = 0.3
    s2["B04"][:] = 0.1
    s2["B11"][1, 0] = np.nan                                     # one S2 band missing -> whole S2 step masked
    s1 = {"VV": np.full((T, N), 10 ** -1.5, "float32"), "VH": np.full((T, N), 10 ** -2.0, "float32")}
    s1["VH"][2, 1] = 0                                           # invalid linear value -> S1 step masked
    x, mask = embeddings.presto_inputs(s2, s1)
    assert x.shape == (N, T, 17) and embeddings.PRESTO_BANDS[16] == "NDVI"
    np.testing.assert_allclose(x[0, 0, :2], [(-15 + 25) / 25, (-20 + 25) / 25], rtol=1e-5)
    np.testing.assert_allclose(x[0, 0, 8], 0.3)                  # B8 = reflectance (= DN / 10000)
    np.testing.assert_allclose(x[0, 0, 16], 0.5, rtol=1e-6)      # NDVI from B8/B4
    assert mask[0, 1, 2:12].all() and mask[0, 1, 16] and not mask[0, 1, :2].any()
    assert mask[1, 2, :2].all() and not mask[1, 2, 2:12].any()
    assert mask[:, :, 12:16].all()                               # ERA5 and SRTM not available on-VM
    assert not mask[0, 0].any() or mask[0, 0, 12:16].all()
    only_s2, m2 = embeddings.presto_inputs(s2, None)
    assert m2[:, :, :2].all()


@pytest.fixture(scope="module")
def stand_in():
    pytest.importorskip("torch")
    import torch
    torch.manual_seed(0)
    return embeddings.load_presto(None)          # random weights, same architecture as the checkpoint


def test_presto_embed_groups_mask_patterns(stand_in):
    rng = np.random.default_rng(5)
    T, N = 6, 40
    s2 = {b: rng.uniform(0.02, 0.4, (T, N)).astype("float32") for b in embeddings.GEOCROP_S2.values()}
    s1 = {q: rng.uniform(0.005, 0.2, (T, N)).astype("float32") for q in ("VV", "VH")}
    for b in s2:
        s2[b][rng.integers(0, T, 15), rng.integers(0, N, 15)] = np.nan
    s1["VV"][:, 3] = np.nan
    for b in s2:
        s2[b][:, 7] = np.nan
    s1["VV"][:, 7] = np.nan                                      # pixel 7 has no observation at all
    x, mask = embeddings.presto_inputs(s2, s1)
    latlons = np.column_stack([np.full(N, 44.7), np.full(N, 19.1)]).astype("float32")
    out = embeddings.presto_embed(stand_in, x, mask, latlons, month=2, batch_size=7)
    assert out.shape == (N, 128) and np.isnan(out[7]).all()
    assert np.isfinite(np.delete(out, 7, axis=0)).all()
    one = embeddings.presto_embed(stand_in, x[[5]], mask[[5]], latlons[[5]], month=2)
    np.testing.assert_allclose(one[0], out[5], atol=1e-5)
    other_month = embeddings.presto_embed(stand_in, x[[5]], mask[[5]], latlons[[5]], month=8)
    assert not np.allclose(other_month, one)


def test_presto_pixels_and_parcels_from_scene_folders(tmp_path, stand_in):
    from geocrop.demo import generate_scenes

    sc = generate_scenes(tmp_path / "demo", parcels_per_side=3, parcel_px=8)
    monthly = tmp_path / "monthly"
    report = embeddings.monthly_composites(sc["reference"], monthly, "2023-03", 8, sc["scene_root"], sc["scene_root"])
    assert any(r["s2"] for r in report.values()) and any(r["s1"] for r in report.values())
    out = embeddings.presto_pixels(monthly, sc["reference"], tmp_path / "emb" / "presto_2023.tif", "2023-03", 8,
                                   model=stand_in, block_size=10)
    with rasterio.open(out) as ds, rasterio.open(sc["reference"]) as ref:
        assert ds.count == 128 and ds.shape == ref.shape and ds.descriptions[0] == "emb_00"
        values = ds.read()
        assert (values != ds.nodata).all(axis=0).mean() > 0.9
        assert ds.tags()["start"] == "2023-03" and ds.tags()["embedding_source"] == "presto"
    assert load_manifest(out)[0].group == "emb"
    table = embeddings.presto_parcels(monthly, sc["reference"], sc["parcels"], tmp_path / "emb" / "parcels.csv",
                                      "2023-03", 8, model=stand_in)
    assert len(table) == len(gpd.read_file(sc["parcels"])) and table["n_pixels"].min() > 0
    assert np.isfinite(table[embeddings.band_names(128)].to_numpy()).all()


def test_cli_presto_with_local_checkpoint(tmp_path, stand_in):
    import torch

    from geocrop.cli import main
    from geocrop.demo import generate_scenes

    sc = generate_scenes(tmp_path / "demo", parcels_per_side=3, parcel_px=8)
    ckpt = tmp_path / "presto.pt"
    torch.save({f"encoder.{k}": v for k, v in stand_in.state_dict().items()}, ckpt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths: {{s2_scenes: {sc['scene_root']}, s1_scenes: {sc['scene_root']}, "
                   f"features: {tmp_path / 'features' / 'stack.tif'}}}\n"
                   "features:\n  embeddings: {source: presto, year: 2023, start: '2023-04', months: 6}\n")
    main(["embeddings", "--config", str(cfg), "--reference", str(sc["reference"]), "--checkpoint", str(ckpt),
          "--make-monthly", "--parcels", str(sc["parcels"])])
    out = tmp_path / "features" / "embeddings" / "presto_2023.tif"
    assert out.exists() and out.with_name("presto_2023_parcels.csv").exists()


# --------------------------------------------------------------------------- real access (network)

SEMBERIJA_SMALL = box(19.12, 44.73, 19.125, 44.734)


def _small_grid(tmp_path: Path) -> Path:
    from geocrop.raster import make_grid
    aoi = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(geometry=[SEMBERIJA_SMALL], crs="EPSG:4326").to_file(aoi)
    return make_grid(aoi, "EPSG:32634", 10, tmp_path / "grid.tif")


@pytest.mark.network
def test_alphaearth_real_window(tmp_path):
    pytest.importorskip("pyarrow")
    try:
        out = embedding_sources.fetch_alphaearth(_small_grid(tmp_path), tmp_path / "alphaearth_2023.tif", 2023)
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"AlphaEarth not reachable: {exc}")
    s = embeddings.summary(out)
    assert s["bands"] == 64 and s["valid_fraction"] > 0.99
    assert -1 <= s["min"] and s["max"] <= 1 and abs(s["vector_norm_mean"] - 1) < 0.02


@pytest.mark.network
def test_tessera_real_window(tmp_path):
    try:
        out = embedding_sources.fetch_tessera(_small_grid(tmp_path), tmp_path / "tessera_2023.tif", 2023)
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"TESSERA not reachable: {exc}")
    s = embeddings.summary(out)
    assert s["bands"] == 128 and s["valid_fraction"] > 0.99


@pytest.mark.network
def test_presto_checkpoint_download_and_load(tmp_path):
    pytest.importorskip("torch")
    try:
        path = embedding_sources.presto_checkpoint(tmp_path)
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"GitHub not reachable: {exc}")
    model = embeddings.load_presto(path)
    assert model.embedding_size == 128
