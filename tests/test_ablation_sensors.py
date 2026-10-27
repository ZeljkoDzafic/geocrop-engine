"""Optical ablations must not silently include radar-derived temporal features."""
import json
from pathlib import Path

import numpy as np
import pytest

from geocrop import config, experiments, features
from geocrop.samples import Samples


def test_sensor_filter_reaches_ablation_model_and_report(samples, stack, tmp_path):
    radar = features.Feature('delta_VH_dB_P1_P2', 'delta', 'S1', None, ('P1', 'P2'))
    manifest = features.load_manifest(stack) + [radar]
    data = Samples(np.column_stack([samples.X, samples.X[:, 0]]), samples.y, samples.polygon,
                   samples.x, samples.yy, [*samples.features, radar.name], samples.crs)
    experiments.run(data, manifest, tmp_path, params={'n_estimators': 5, 'n_jobs': 1},
                    n_splits=2, spatial_block_m=500)
    report = json.loads((tmp_path / 'ablation.json').read_text())['experiments']
    optical = report['E2_S2_full_temporal']['features']
    assert radar.name not in optical
    assert 'delta_NDVI_P1_P2' in optical
    assert radar.name in report['E4_S1_S2_fusion']['features']
    assert all(f.sensor == 'S2' for f in manifest if f.name in optical)
    assert len({r['cv']['fold_signature'] for r in report.values()}) == 1


@pytest.mark.parametrize('path', ['configs/rs_north_2023.yaml',
                                 'configs/studies/lijevce_polje/study_2023.yaml'])
def test_study_optical_experiment_is_sensor_restricted(path):
    cfg = config.load(Path(__file__).parents[1] / path)
    assert cfg['experiments']['E2_S2_temporal']['sensors'] == ['S2']


@pytest.mark.parametrize('sensors', ['S2', [], ['S3'], [None]])
def test_invalid_experiment_sensors_are_rejected(sensors):
    with pytest.raises(config.ConfigError, match='sensors'):
        config.load(overrides={'experiments': {'E2': {'groups': ['delta'], 'sensors': sensors}}})
