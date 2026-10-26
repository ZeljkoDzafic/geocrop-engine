"""Galileo encoder port (geocrop.galileo_encoder): random nano-sized weights, no download."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geocrop import galileo_encoder as ge  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ge.load(None)


def _inputs(n=12, t=6, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, t, 13))
    raw[:, :, 0] = rng.uniform(-20, -5, (n, t))
    raw[:, :, 1] = rng.uniform(-28, -12, (n, t))
    raw[:, :, 2:12] = rng.uniform(100, 5000, (n, t, 10))
    raw[:, :, 12] = (raw[:, :, 8] - raw[:, :, 4]) / (raw[:, :, 8] + raw[:, :, 4])
    mask = np.zeros((n, t, 7), dtype=bool)
    mask[1, 2, 1:] = True                     # one S2 month missing
    mask[2, :, 0] = True                      # no S1
    mask[3] = True                            # nothing observed
    latlons = np.column_stack([np.full(n, 44.7), np.full(n, 19.1)])
    return ge.normalise_space_time(raw), mask, latlons, raw


def test_normalisation_matches_official_formula():
    _, _, _, raw = _inputs()
    x = ge.normalise_space_time(raw)
    i = 4                                      # B4
    lo = ge.SPACE_TIME_MEAN[i] - 2 * ge.SPACE_TIME_STD[i]
    np.testing.assert_allclose(x[..., i], (raw[..., i] - lo) / (4 * ge.SPACE_TIME_STD[i]), rtol=1e-5)
    np.testing.assert_allclose(x[..., 12], raw[..., 12], rtol=1e-6)      # NDVI is not rescaled
    assert ge.SPACE_TIME_BANDS[12] == "NDVI" and len(ge.SPACE_TIME_GROUPS) == 7


def test_encode_pixels_shapes_nan_and_batching(model):
    x, mask, latlons, _ = _inputs()
    x[1, 2, 2:] = np.nan                       # values under the mask must not leak
    months = np.arange(6) + 3
    out = ge.encode_pixels(model, x, mask, months, latlons, batch_size=5)
    assert out.shape == (12, 128) and np.isnan(out[3]).all()
    assert np.isfinite(np.delete(out, 3, axis=0)).all()
    one = ge.encode_pixels(model, x[[1]], mask[[1]], months, latlons[[1]])
    np.testing.assert_allclose(one[0], out[1], atol=1e-5)
    other = ge.encode_pixels(model, x[[1]], mask[[1]], (months + 4) % 12, latlons[[1]])
    assert not np.allclose(other, one)
    no_loc = ge.encode_pixels(model, x[[0]], mask[[0]], months, None)
    assert np.isfinite(no_loc).all() and not np.allclose(no_loc, out[[0]])


def test_load_roundtrip(tmp_path, model):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model": {"encoder": ge.NANO_CONFIG}}))
    torch.save(model.state_dict(), tmp_path / "encoder.pt")
    again = ge.load(tmp_path)
    x, mask, latlons, _ = _inputs(n=4)
    np.testing.assert_allclose(ge.encode_pixels(again, x, mask, [0] * 6, latlons),
                               ge.encode_pixels(model, x, mask, [0] * 6, latlons), atol=1e-6, equal_nan=True)


def test_spatial_forward_shapes(model):
    b, h, t = 2, 4, 3
    out = model(torch.randn(b, h, h, t, 13), torch.zeros(b, h, h, 16), torch.zeros(b, t, 6), torch.zeros(b, 18),
                torch.zeros(b, h, h, t, 7), torch.ones(b, h, h, 3), torch.ones(b, t, 3), torch.ones(b, 4),
                torch.zeros(b, t, dtype=torch.long), patch_size=2)
    assert out[0].shape == (b, 2, 2, t, 7, 128)
    assert model.average_tokens(*out[:8]).shape == (b, 128)
