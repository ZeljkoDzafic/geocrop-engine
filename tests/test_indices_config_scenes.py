from datetime import date

import numpy as np
import pytest

from geocrop import config, indices, scenes


def test_index_values():
    b = {"B02": np.array([0.04]), "B03": np.array([0.06]), "B04": np.array([0.05]), "B05": np.array([0.1]),
         "B06": np.array([0.2]), "B07": np.array([0.25]), "B08": np.array([0.35]), "B8A": np.array([0.36]),
         "B11": np.array([0.2]), "B12": np.array([0.12])}
    assert np.isclose(indices.get("NDVI").compute(b), 0.3 / 0.4)
    assert np.isclose(indices.get("GCVI").compute(b), 0.35 / 0.06 - 1)
    assert np.isclose(indices.get("RECI").compute(b), 0.35 / 0.1 - 1)
    msavi = indices.get("MSAVI2").compute(b)[0]
    assert 0 < msavi < indices.get("NDVI").compute(b)[0]


def test_radar_indices_db():
    pol = {"VV": np.array([0.1, 0.0]), "VH": np.array([0.01, 0.01])}
    assert np.isclose(indices.get("VV_dB").compute(pol)[0], -10)
    assert np.isnan(indices.get("VV_dB").compute(pol)[1])
    assert np.isclose(indices.get("VHVV_dB").compute(pol)[0], -10)


def test_safe_divide_zero():
    out = indices.safe_divide(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
    assert out[0] == 0.5 and np.isnan(out[1])


def test_index_table_complete():
    rows = indices.table()
    assert {"NDVI", "EVI", "MSAVI2", "NDRE", "RECI", "GCVI", "NDMI", "BSI"} <= {r["name"] for r in rows}
    assert all(r["reference"] and r["formula"] for r in rows)


def test_config_defaults_and_errors(tmp_path):
    cfg = config.load()
    assert cfg["model"]["spatial_block_m"] == 5000
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  n_estimatorz: 3\n")
    with pytest.raises(config.ConfigError, match="n_estimatorz"):
        config.load(bad)
    with pytest.raises(config.ConfigError, match="overlap"):
        config.parse_windows({"P1": {"start": "2023-04-01", "end": "2023-06-10"},
                              "P2": {"start": "2023-06-01", "end": "2023-07-01"}})


def test_config_fingerprint_stable():
    assert config.fingerprint(config.load()) == config.fingerprint(config.load())


def test_product_names():
    s2 = scenes.parse_product_name("S2B_MSIL2A_20230612T093549_N0509_R036_T34TCQ_20230612T122108.SAFE")
    assert s2["tile"] == "34TCQ" and s2["relative_orbit"] == 36 and s2["datetime"].startswith("2023-06-12")
    s1 = scenes.parse_product_name("S1A_IW_GRDH_1SDV_20230615T164512_20230615T164537_049001_05E4A1_7C3F.SAFE")
    assert s1["absolute_orbit"] == 49001 and s1["relative_orbit"] == ((49001 - 73) % 175) + 1
    with pytest.raises(ValueError):
        scenes.parse_product_name("random.tif")


def test_assign_windows(tmp_path):
    for d in ("2023-04-10", "2023-06-20", "2023-11-01"):
        f = tmp_path / d
        f.mkdir()
        scenes.write(f, sensor="S2", datetime=d + "T10:00:00")
    windows = {"P1": (date(2023, 4, 1), date(2023, 5, 31)), "P2": (date(2023, 6, 1), date(2023, 7, 15))}
    groups = scenes.assign(scenes.discover(tmp_path), windows)
    assert [len(groups[w]) for w in windows] == [1, 1]
    assert scenes.inventory(groups)[0]["first"] == "2023-04-10"
