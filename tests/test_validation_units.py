"""Inference uses reference units and confidence belonging to the evaluated decision."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geocrop import cli, config, experiments, features, metrics, pipeline, validation


def test_repeating_pixels_cannot_make_grouped_mcnemar_more_significant():
    y = np.array(['a'] * 3)
    b = np.array(['b'] * 3)
    groups = np.array(['p1', 'p2', 'p3'])
    single = metrics.grouped_mcnemar(y, y, b, groups)
    repeated = metrics.grouped_mcnemar(np.repeat(y, 100), np.repeat(y, 100), np.repeat(b, 100),
                                       np.repeat(groups, 100))
    assert single == repeated
    order = np.random.default_rng(17).permutation(300)
    assert repeated == metrics.grouped_mcnemar(np.repeat(y, 100)[order], np.repeat(y, 100)[order],
                                                np.repeat(b, 100)[order], np.repeat(groups, 100)[order])
    assert single['p_value'] == 0.25 and single['n_units'] == 3
    assert metrics.mcnemar(np.repeat(y, 100), np.repeat(y, 100), np.repeat(b, 100))['p_value'] < 1e-10


def test_grouped_test_has_explicit_ties_and_refuses_conflicting_truth():
    r = metrics.grouped_mcnemar(['a', 'a'], ['b', 'a'], ['b', 'b'], ['p', 'p'])
    assert r['a_only_correct'] == 1 and r['tie_rule'] == 'lexicographic'
    with pytest.raises(ValueError, match='conflicting labels'):
        metrics.grouped_mcnemar(['a', 'b'], ['a', 'a'], ['b', 'b'], ['p', 'p'])
    with pytest.raises(ValueError, match='equally sized'):
        metrics.grouped_mcnemar([], [], [], [])


def test_holm_known_values_order_and_invalid_input():
    assert metrics.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert metrics.holm_adjust([]) == []
    assert metrics.holm_adjust([0.8, 0.9]) == [1, 1]
    for p in ([float('nan')], [-0.1], [1.1]):
        with pytest.raises(ValueError, match='P-values'):
            metrics.holm_adjust(p)


def test_ablation_reports_polygon_units_and_baseline_excluded_holm(samples, stack, tmp_path):
    specs = {'base': {'groups': ['s2_band']}, 'optical': {'groups': ['s2_index']},
             'radar': {'groups': ['s1']}, 'empty': {'features': ['absent']}}
    frame = experiments.run(samples, features.load_manifest(stack), tmp_path, experiments=specs,
                            params={'n_estimators': 5, 'n_jobs': 1}, n_splits=2, spatial_block_m=500)
    table = frame.set_index('experiment')
    raw = [table.loc[k, 'mcnemar_p'] for k in ('optical', 'radar')]
    assert table.loc[['optical', 'radar'], 'mcnemar_p_holm'].tolist() == metrics.holm_adjust(raw)
    assert table.loc['base', 'mcnemar_n_units'] == len(np.unique(samples.polygon))
    assert table.loc['base', 'mcnemar_p_holm'] == 1
    assert pd.isna(table.loc['empty', 'mcnemar_p_holm'])
    report = json.loads((tmp_path / 'ablation.json').read_text())
    assert report['paired_comparison']['unit'] == 'reference_polygon'
    assert report['experiments']['radar']['paired_test']['n_units'] < len(samples.y)


@pytest.fixture
def tiny_prediction(tmp_path):
    root = tmp_path / 'prediction'
    root.mkdir()
    labels = np.zeros((3, 3), dtype='uint8')
    labels[1, 1] = 1  # centre says b with probability .95, its neighbours unanimously say a
    profile = {'driver': 'GTiff', 'height': 3, 'width': 3, 'crs': 'EPSG:32634',
               'transform': from_origin(500000, 5000000, 10, 10)}
    with rasterio.open(root / 'classes.tif', 'w', count=1, dtype='uint8', nodata=255, **profile) as ds:
        ds.write(labels, 1)
    probabilities = np.stack([np.where(labels == 0, .95, .05), np.where(labels == 0, .05, .95)]).astype('float32')
    with rasterio.open(root / 'probabilities.tif', 'w', count=2, dtype='float32', **profile) as ds:
        ds.write(probabilities)
        ds.descriptions = ('a', 'b')
    (root / 'class_legend.json').write_text(json.dumps({'0': 'a', '1': 'b'}))
    obs = gpd.GeoDataFrame({'class_name': ['a', 'a', 'a'], 'parcel id': ['p1', 'p1', 'p2']},
                           geometry=[Point(500015, 4999985), Point(500015, 4999985), Point(500005, 4999995)],
                           crs='EPSG:32634')
    return root, obs


def test_cluster_bootstrap_and_confidence_follow_window_decision(tiny_prediction, tmp_path):
    pred, obs = tiny_prediction
    result = validation.evaluate_points(pred, obs, tmp_path / 'v', window_px=1, group_field='parcel id', bootstrap=30)
    assert result['accuracy']['overall_accuracy'] == 1
    assert result['bootstrap']['units'] == 2 and result['bootstrap']['resampling'] == 'group'
    table = pd.read_csv(tmp_path / 'v/observations_evaluated.csv')
    assert table.loc[0, 'decision_confidence'] == pytest.approx(8 / 9)
    assert table.loc[0, 'max_probability'] == pytest.approx(.95)
    assert table.bootstrap_group.tolist() == ['p1', 'p1', 'p2']
    assert next(r for r in result['reject_curve'] if r['threshold'] == pytest.approx(.9))['review_count'] == 3
    assert result['validation_protocol']['confidence'] == 'window_vote_share'
    # The central-pixel calibration remains separate, and correctly records two wrong central predictions.
    assert sum(b['accuracy'] * b['count'] for b in result['calibration']['bins']) == 1


@pytest.mark.parametrize('bad_groups', [['p1', None, 'p2'], ['p1', '', 'p2']])
def test_missing_cluster_identifiers_fail(tiny_prediction, tmp_path, bad_groups):
    pred, obs = tiny_prediction
    obs['parcel id'] = bad_groups
    with pytest.raises(ValueError, match='identifiers'):
        validation.evaluate_points(pred, obs, tmp_path / 'v', group_field='parcel id')


def test_missing_field_and_single_cluster_fail(tiny_prediction, tmp_path):
    pred, obs = tiny_prediction
    with pytest.raises(ValueError, match='grouping field'):
        validation.evaluate_points(pred, obs, tmp_path / 'v', group_field='absent')
    obs['parcel id'] = 'only-one'
    with pytest.raises(ValueError, match='at least two'):
        validation.evaluate_points(pred, obs, tmp_path / 'v', group_field='parcel id')
    assert validation.evaluate_points(pred, obs, tmp_path / 'description', group_field='parcel id', bootstrap=0)


def test_probability_grid_mismatch_is_rejected(tiny_prediction, tmp_path):
    pred, obs = tiny_prediction
    with rasterio.open(pred / 'probabilities.tif', 'r+') as ds:
        ds.transform = from_origin(500001, 5000000, 10, 10)
    with pytest.raises(ValueError, match='Grid mismatch'):
        validation.evaluate_points(pred, obs, tmp_path / 'v')


def test_group_field_config_and_cli_are_wired(tmp_path, monkeypatch):
    assert config.fingerprint(config.load()) == '09ff103f5880447d'
    cfg = config.load(overrides={'validation': {'group_field': 'parcel_id'}})
    assert cli.build_parser().parse_args(['validate', '--prediction', 'p', '--observations', 'o', '--out', 'v',
                                         '--group-field', 'parcel_id']).group_field == 'parcel_id'
    obs = tmp_path / 'obs.gpkg'
    obs.touch()
    cfg['paths']['observations'] = str(obs)
    cfg['paths']['output'] = str(tmp_path / 'run')
    runner = pipeline.Runner(cfg, log=lambda *_: None)
    monkeypatch.setattr(runner, 'stage', lambda name, fp, outputs, run: run())
    seen = {}
    def fake(*args, **kwargs):
        seen.update(kwargs)
        return {'accuracy': {'overall_accuracy': 1, 'macro_f1': 1}, 'selection': {}}
    monkeypatch.setattr(validation, 'evaluate_points', fake)
    runner.validate()
    assert seen['group_field'] == 'parcel_id'


def test_invalid_probabilities_do_not_contaminate_window_accuracy(tiny_prediction, tmp_path):
    pred, obs = tiny_prediction
    with rasterio.open(pred / 'probabilities.tif', 'r+') as ds:
        ds.write(np.full((2, 1, 1), np.nan, dtype='float32'), window=rasterio.windows.Window(1, 1, 1, 1))
    result = validation.evaluate_points(pred, obs, tmp_path / 'v', window_px=1, bootstrap=0)
    assert result['accuracy']['n'] == 3 and result['accuracy']['overall_accuracy'] == 1
    assert result['calibration']['n'] == 1 and result['selection']['invalid_centre_probabilities'] == 2
    assert result['selection']['with_decision_confidence'] == 3
    centre = validation.evaluate_points(pred, obs, tmp_path / 'centre', bootstrap=0)
    assert centre['selection']['with_decision_confidence'] == 1


def test_holdout_comparisons_use_held_out_polygons(samples, stack, tmp_path):
    specs = {'base': {'groups': ['s2_band']}, 'optical': {'groups': ['s2_index']}}
    frame = experiments.run(samples, features.load_manifest(stack), tmp_path, experiments=specs,
                            params={'n_estimators': 5, 'n_jobs': 1}, n_splits=2, spatial_block_m=500,
                            holdout_fraction=.25)
    report = json.loads((tmp_path / 'ablation.json').read_text())
    comparison = report['experiments']['optical']['holdout_paired_test']
    assert 0 < comparison['n_units'] < len(np.unique(samples.polygon))
    assert comparison['p_value_holm'] == comparison['p_value']  # one non-baseline comparison
    assert frame.iloc[1]['holdout_mcnemar_n_units'] == comparison['n_units']


@pytest.mark.parametrize('group_field', ['', [], 123])
def test_group_field_config_rejects_invalid_names(group_field):
    with pytest.raises(config.ConfigError, match='group_field'):
        config.load(overrides={'validation': {'group_field': group_field}})


def test_model_comparisons_use_polygons_and_separate_holdout_family(samples, tmp_path):
    from geocrop import benchmark
    candidates = {'base': {'n_estimators': 5, 'n_jobs': 1},
                  'small': {'n_estimators': 5, 'n_jobs': 1, 'max_depth': 1},
                  'medium': {'n_estimators': 5, 'n_jobs': 1, 'max_depth': 2}}
    frame = benchmark.models(samples, tmp_path, candidates=candidates, n_splits=2, spatial_block_m=500,
                             repeats=2, holdout_fraction=.25)
    report = json.loads((tmp_path / 'benchmark_models.json').read_text())
    assert report['paired_comparison']['repeat'] == 'first repeat only'
    assert report['holdout_reference_model'] == report['reference_model'] == 'base'
    assert set(frame.mcnemar_unit) == {'reference_polygon'}
    for col in ['mcnemar_p_vs_reference', 'holdout_mcnemar_p_vs_reference']:
        expected = metrics.holm_adjust(frame.loc[frame.model != 'base', col].tolist())
        assert frame.loc[frame.model != 'base', col + '_holm'].tolist() == expected
        assert report['details']['small']['runs'][0][col + '_holm'] == expected[0]
        assert col not in report['details']['small']['runs'][1]
    assert frame.loc[0, 'mcnemar_n_units'] < len(samples.y)


@pytest.mark.parametrize('key,value', [('window_px', -1), ('window_px', True), ('bootstrap', -1), ('bootstrap', 1.5)])
def test_invalid_validation_counts_fail_before_a_pipeline_run(key, value):
    with pytest.raises(config.ConfigError, match=key):
        config.load(overrides={'validation': {key: value}})


def test_descriptive_only_cli_option_is_available():
    args = cli.build_parser().parse_args(['validate', '--prediction', 'p', '--observations', 'o', '--out', 'v',
                                         '--group-field', 'parcel_id', '--bootstrap', '0'])
    assert args.bootstrap == 0 and args.group_field == 'parcel_id'
