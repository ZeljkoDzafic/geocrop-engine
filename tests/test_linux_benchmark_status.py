"""Partial or failed scale runs must never produce a successful command status."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location('scale_status_test', ROOT / 'tools/scale_benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(benchmark, area=100, **kw):
    return {'area': area, 'error': None, 'total_wall_s': 60,
            'stages': [{'stage': s, 'status': 'ok'} for s in benchmark.STAGE_ORDER], **kw}


@pytest.mark.parametrize('case', ['exit', 'timeout', 'missing', 'error', 'partial', 'success'])
def test_orchestrator_status_matches_completed_evidence(benchmark, tmp_path, monkeypatch, case):
    def run(*args, **kwargs):
        if case == 'timeout':
            raise subprocess.TimeoutExpired(args[0], 1)
        target = tmp_path / 'area_100km2'
        target.mkdir()
        if case != 'missing':
            payload = result(benchmark)
            if case == 'error':
                payload['error'] = 'training failed'
            if case == 'partial':
                payload['stages'].pop()
            (target / 'benchmark.json').write_text(json.dumps(payload))
        return SimpleNamespace(returncode=7 if case == 'exit' else 0)
    monkeypatch.setattr(benchmark.subprocess, 'run', run)
    assert benchmark.run_all(tmp_path, [100], [100], 1, 'python') == (case == 'success')
    assert (json.loads((tmp_path / 'runs.json').read_text())['100']['status'] == 'ok') == (case == 'success')


def test_reused_success_and_budget_skip_are_recorded(benchmark, tmp_path):
    target = tmp_path / 'area_100km2'
    target.mkdir()
    (target / 'benchmark.json').write_text(json.dumps(result(benchmark, total_wall_s=3600)))
    assert not benchmark.run_all(tmp_path, [100, 500], [100, 500], 2, 'python')
    runs = json.loads((tmp_path / 'runs.json').read_text())
    assert runs['100']['reused'] and runs['500']['status'] == 'skipped'


@pytest.mark.parametrize(('run_status', 'report_status', 'expected'), [(0, 0, 0), (7, 0, 7), (0, 8, 8), (7, 8, 7)])
def test_container_exports_partial_results_without_hiding_failures(tmp_path, run_status, report_status, expected):
    script = (ROOT / 'tools/linux_benchmark.sh').read_text().split("<<'BENCHMARK' || status=$?\n")[1].split('\nBENCHMARK')[0]
    bench, export, bin_dir = (tmp_path / n for n in ('bench', 'export', 'bin'))
    for p in (bench / 'scale/area_100km2', export, bin_dir):
        p.mkdir(parents=True)
    for name in ('runs.json', 'scale_tables.md', 'scale_summary.json', 'area_100km2/benchmark.json'):
        (bench / 'scale' / name).write_text('{}')
    fake = bin_dir / 'python'
    fake.write_text('#!/bin/sh\nif [ "$2" = run ]; then exit "$RUN_STATUS"; else exit "$REPORT_STATUS"; fi\n')
    fake.chmod(0o755)
    script = script.replace('/bench/', str(bench) + '/').replace('/export', str(export))
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'RUN_STATUS': str(run_status), 'REPORT_STATUS': str(report_status)}
    proc = subprocess.run(['bash', '-s', '--', '2', '100'], input=script, text=True, env=env, capture_output=True)
    assert proc.returncode == expected, proc.stderr
    assert (export / 'area_100km2/benchmark.json').exists()
    assert int((export / 'exit_status.txt').read_text()) == expected
