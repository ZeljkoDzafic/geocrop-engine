"""AnySat encoder port (geocrop.anysat_encoder) with a tiny random-weight stand-in: shapes, missing-step
handling, pattern grouping, input standardisation. No download; equivalence with the official model on the
released checkpoint is documented in docs/embeddings.md."""
from datetime import date

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geocrop import anysat_encoder as ae  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return ae.load(None, size="tiny", modalities=("s2", "s1"))


def test_forward_shapes(tiny):
    x = {"s2": torch.randn(2, 5, 10, 6, 6), "s2_dates": torch.randint(1, 366, (2, 5))}
    with torch.inference_mode():
        assert tiny(x, patch_size=10, output="tile").shape == (2, 64)
        assert tiny(x, patch_size=10, output="patch").shape == (2, 6, 6, 64)
        assert tiny(x, patch_size=20, output="patch").shape == (2, 3, 3, 64)
        both = {**x, "s1": torch.randn(2, 7, 3, 6, 6), "s1_dates": torch.randint(1, 366, (2, 7))}
        assert tiny(both, patch_size=20, output="tile").shape == (2, 64)
    with pytest.raises(ValueError):
        tiny(x, output="dense")
    with pytest.raises(ValueError):
        ae.AnySatEncoder("tiny", modalities=("aerial",))


def test_rpe_buckets_use_skip_bucket():
    ids = ae.rpe_bucket_ids(3, 3, skip=1)
    assert ids.shape == (10, 10)
    assert (ids[0] == ae._NUM_BUCKETS - 1).all() and (ids[:, 0] == ae._NUM_BUCKETS - 1).all()
    assert ids[1:, 1:].diagonal().unique().tolist() == [int(ae._BETA)]      # distance 0 -> centre bucket
    assert int(ids[1:, 1:].max()) < ae._NUM_BUCKETS - 1


def test_encode_pixels_groups_patterns_and_masks_empty_rows(tiny):
    rng = np.random.default_rng(1)
    N, T = 25, 6
    s2 = rng.normal(size=(N, T, 10)).astype("float32")
    valid = rng.random((N, T)) > 0.3
    valid[3] = False
    s2[5, 2, 4] = np.nan                                 # a non-finite band invalidates that step
    doy = np.array([ae.day_of_year(date(2023, m, 15)) for m in (1, 2, 3, 5, 6, 7)])
    assert doy[0] == 15
    out = ae.encode_pixels(tiny, s2, valid, doy, batch_size=4)
    assert out.shape == (N, 64) and np.isnan(out[3]).all()
    rows = [i for i in range(N) if i != 3]
    assert np.isfinite(out[rows]).all()
    for i in (0, 5, 11):
        steps = np.flatnonzero(valid[i] & np.isfinite(s2[i]).all(axis=1))
        with torch.inference_mode():
            ref = tiny({"s2": torch.from_numpy(s2[i, steps])[None, :, :, None, None],
                        "s2_dates": torch.from_numpy(doy[steps].astype("float32"))[None]}).numpy()[0]
        np.testing.assert_allclose(out[i], ref, atol=1e-5)
    with pytest.raises(ValueError):
        ae.encode_pixels(tiny, s2, valid, doy[:3])


def test_channel_stats_and_standardise():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=[100.0, 5.0], scale=[10.0, 2.0], size=(500, 4, 2))
    valid = np.ones((500, 4), bool)
    valid[:50] = False
    x[:50] = 1e6                                         # ignored through `valid`
    x[60, 1, 0] = np.nan                                 # ignored as non-finite
    mean, std = ae.channel_stats(x, valid)
    np.testing.assert_allclose(mean, [100, 5], atol=1.0)
    np.testing.assert_allclose(std, [10, 2], rtol=0.1)
    z = ae.standardise(x[50:], mean, std)
    ok = np.isfinite(z).all(axis=-1)
    np.testing.assert_allclose(z[ok].mean(axis=0), 0, atol=0.05)
    const = np.ones((10, 3, 2))
    assert (ae.channel_stats(const)[1] == 1).all()       # zero variance does not divide by zero
    with pytest.raises(ValueError):
        ae.channel_stats(x, np.zeros((500, 4), bool))


def test_load_random_and_strict_subset(tmp_path, tiny):
    path = tmp_path / "anysat_tiny.pth"
    state = {k: v for k, v in tiny.state_dict().items()}
    state["projector_aerial.proj.weight"] = torch.zeros(1)      # other modalities are skipped
    torch.save({"state_dict": state}, path)
    again = ae.load(str(path), size="tiny", modalities=("s2", "s1"))
    for k, v in tiny.state_dict().items():
        assert torch.equal(v, again.state_dict()[k])
    del state["cls_token"]
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError):
        ae.load(str(path), size="tiny", modalities=("s2", "s1"))
