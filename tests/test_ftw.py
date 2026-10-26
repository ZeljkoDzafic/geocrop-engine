"""Fields of The World adapter: input stack, blockwise inference without seams, licence gate, registry use.

All default-tier tests use synthetic composites and tiny random-weight stand-in networks (no download).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import rasterio

from geocrop import benchmark, ftw, segmenters

torch = pytest.importorskip("torch")


def _tiny_net(classes: int = 3, seed: int = 0):
    """Random-weight stand-in with a 5 x 5 receptive field (radius 2)."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
                               torch.nn.Conv2d(8, classes, 3, padding=1)).eval()


@pytest.fixture(scope="module")
def windows(demo):
    comps = Path(demo["composites"])
    return comps / "P1", comps / "P3"


@pytest.fixture(scope="module")
def ftw_stack(windows, tmp_path_factory):
    return ftw.build_stack(*windows, tmp_path_factory.mktemp("ftw") / "stack.tif", block_size=64)["path"]


def test_stack_band_order_values_and_tags(windows, ftw_stack):
    with rasterio.open(ftw_stack) as ds:
        assert ds.count == 8
        assert ds.descriptions == ("A_B04", "A_B03", "A_B02", "A_B08", "B_B04", "B_B03", "B_B02", "B_B08")
        assert ds.tags()[ftw.UNITS_TAG] == "reflectance"
        stack = ds.read()
    for i, (window, band) in enumerate((w, b) for w in windows for b in ftw.WINDOW_BANDS):
        with rasterio.open(Path(window) / f"{band}.tif") as src:
            ref = src.read(1)
        ok = stack[i] != ftw.NODATA
        assert ok.mean() > 0.9
        np.testing.assert_allclose(stack[i][ok], ref[ok])


def test_stack_warps_other_grids_onto_the_reference(windows, tmp_path):
    from rasterio.enums import Resampling
    coarse = tmp_path / "coarse"
    coarse.mkdir()
    for band in ftw.WINDOW_BANDS:
        with rasterio.open(Path(windows[1]) / f"{band}.tif") as src:
            data = src.read(out_shape=(1, src.height // 2, src.width // 2), resampling=Resampling.average)
            prof = src.profile | {"height": data.shape[1], "width": data.shape[2],
                                  "transform": src.transform * src.transform.scale(2, 2)}
        with rasterio.open(coarse / f"{band}.tif", "w", **prof) as dst:
            dst.write(data)
    out = ftw.build_stack(windows[0], coarse, tmp_path / "s.tif", block_size=50)
    with rasterio.open(out["path"]) as ds, rasterio.open(Path(windows[0]) / "B04.tif") as ref:
        assert ds.shape == ref.shape and ds.transform == ref.transform
        assert (ds.read(5) != ftw.NODATA).mean() > 0.8


def test_missing_band_is_a_clear_error(windows, tmp_path):
    with pytest.raises(FileNotFoundError, match="B04"):
        ftw.build_stack(tmp_path, windows[1], tmp_path / "s.tif")


def test_tiled_inference_has_no_seams(ftw_stack, tmp_path):
    net = _tiny_net()
    whole = ftw.predict(ftw_stack, tmp_path / "whole", net, patch=512, overlap=4)
    tiled = ftw.predict(ftw_stack, tmp_path / "tiled", net, patch=48, overlap=4)
    naive = ftw.predict(ftw_stack, tmp_path / "naive", net, patch=48, overlap=0)
    assert whole["tiles"] == 1 and tiled["tiles"] == 16
    with rasterio.open(whole["probabilities"]) as a, rasterio.open(tiled["probabilities"]) as b, \
            rasterio.open(naive["probabilities"]) as c:
        pa, pb, pc = a.read(), b.read(), c.read()
        assert a.descriptions == ("p_background", "p_field", "p_boundary")
    np.testing.assert_allclose(pa, pb, atol=1e-5)
    assert not np.allclose(pa, pc, atol=1e-5)        # the halo is what removes the seams
    ok = pa[0] != ftw.NODATA
    np.testing.assert_allclose(pa[:, ok].sum(axis=0), 1, atol=1e-5)
    with rasterio.open(tiled["not_field"]) as ds:
        np.testing.assert_allclose(ds.read(1)[ok], 1 - pb[1][ok], atol=1e-6)


def test_input_scaling_follows_the_ftw_contract(ftw_stack, tmp_path):
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen["max"] = float(x.max())
            return torch.zeros((x.shape[0], 2, *x.shape[2:]))

    ftw.predict(ftw_stack, tmp_path, Probe(), patch=512, overlap=0)
    with rasterio.open(ftw_stack) as ds:
        raw = ds.read()
    assert seen["max"] == pytest.approx(raw[raw != ftw.NODATA].max() * 10000 / 3000, rel=1e-5)


def test_untagged_stack_needs_explicit_units(ftw_stack, tmp_path):
    with rasterio.open(ftw_stack) as src:
        prof, data = src.profile, src.read()
    plain = tmp_path / "plain.tif"
    with rasterio.open(plain, "w", **prof) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="units"):
        ftw.predict(plain, tmp_path / "o", _tiny_net(), patch=512)
    assert ftw.predict(plain, tmp_path / "o", _tiny_net(2), patch=512, units="dn")["classes"] == ["background", "field"]


def test_licence_gate_and_pinned_registry():
    assert ftw.MODELS[ftw.DEFAULT_MODEL].licence == ftw.CC_BY
    for m in ftw.MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", m.sha256) and m.size > 1_000_000
        assert m.url.startswith("https://github.com/fieldsoftheworld/ftw-baselines/releases/download/")
        if m.commercial_use:
            assert ftw.check_licence(m.name) is m
        else:
            with pytest.raises(ftw.LicenceError):
                ftw.check_licence(m.name)
            assert ftw.check_licence(m.name, allow_noncommercial=True) is m
    with pytest.raises(ValueError, match="Unknown FTW model"):
        ftw.check_licence("nope")


def test_cache_is_verified_and_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-only"):
        ftw.cached_checkpoint(ftw.DEFAULT_MODEL, tmp_path)
    path = ftw.checkpoint_path(ftw.DEFAULT_MODEL, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"0" * ftw.MODELS[ftw.DEFAULT_MODEL].size)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ftw.cached_checkpoint(ftw.DEFAULT_MODEL, tmp_path)
    with pytest.raises(ftw.LicenceError):
        ftw.cached_checkpoint("FTW_PRUE_EFNET_B5", tmp_path)


def test_cli_refuses_noncommercial_without_flag(capsys):
    from geocrop.cli import main
    with pytest.raises(SystemExit, match="allow-noncommercial"):
        main(["ftw-boundaries", "--model", "FTW_PRUE_EFNET_B5", "--download-only"])
    main(["ftw-boundaries", "--list-models"])
    assert "FTW_PRUE_EFNET_B3_CCBY" in capsys.readouterr().out


def test_ftw_segmenter_is_registered_and_benchmarked(demo, windows, tmp_path):
    seg = segmenters.get("ftw")
    assert isinstance(seg, segmenters.Segmenter) and "ftw" in segmenters.available()
    raster = Path(windows[0]) / "B04.tif"
    spec = {"segmenter": "ftw", "checkpoint": _tiny_net(), "window_a": windows[0], "window_b": windows[1],
            "patch": 64, "overlap": 4, "threshold": 1.01, "min_area_pixels": 4, "work_dir": tmp_path / "work"}
    table = benchmark.segmentation(raster, demo["parcels"], tmp_path, tile_size=96, segments_per_km2=40,
                                   methods={"ftw_tiny": spec})
    row = table.set_index("method").loc["ftw_tiny"]
    assert row.status == "ok" and row.segments > 0
    assert (tmp_path / "work" / "ftw_not_field.tif").exists()


def _fake_ftw_checkpoint(path: Path, classes: int = 3) -> Path:
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.manual_seed(0)
    net = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=8, classes=classes)
    torch.save({"hyper_parameters": {"model": "unet", "backbone": "efficientnet-b3", "in_channels": 8,
                                     "num_classes": classes},
                "state_dict": {f"model.{k}": v for k, v in net.state_dict().items()}}, path)
    return path


def test_checkpoint_loading_and_cli_end_to_end(demo, windows, tmp_path, capsys):
    from geocrop.cli import main
    ckpt = _fake_ftw_checkpoint(tmp_path / "fake.ckpt")
    net, hp = ftw.load_checkpoint(ckpt)
    assert hp["num_classes"] == 3 and not net.training
    out = tmp_path / "run"
    main(["ftw-boundaries", "--window-a", str(windows[0]), "--window-b", str(windows[1]), "--out", str(out),
          "--checkpoint", str(ckpt), "--patch", "64", "--overlap", "16", "--threshold", "1.01",
          "--reference", demo["parcels"]])
    assert "Licence of checkpoint fake.ckpt is unknown" in capsys.readouterr().err
    for name in ("ftw_stack.tif", "ftw_probabilities.tif", "ftw_not_field.tif", "ftw_parcels.gpkg",
                 "ftw_summary.json", "segment_metrics.json"):
        assert (out / name).exists(), name


@pytest.mark.network
def test_real_cc_by_checkpoint_download_and_inference(windows, ftw_stack, tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from geocrop.ftw_weights import download
    path = download(ftw.DEFAULT_MODEL, tmp_path / "cache")
    assert ftw.identify(path).name == ftw.DEFAULT_MODEL
    assert ftw.cached_checkpoint(ftw.DEFAULT_MODEL, tmp_path / "cache") == path
    res = ftw.predict(ftw_stack, tmp_path / "pred", path, patch=128, overlap=32)
    with rasterio.open(res["probabilities"]) as ds:
        p = ds.read()
    ok = p[0] != ftw.NODATA
    np.testing.assert_allclose(p[:, ok].sum(axis=0), 1, atol=1e-4)
