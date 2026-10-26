"""Versioned JSON file contracts: stamped on write, validated (and migrated) on read."""
from __future__ import annotations

import json

import pytest
from joblib import dump

from geocrop import contracts, model, provenance, scenes


def test_scene_json_is_stamped_and_validated(tmp_path):
    scenes.write(tmp_path, sensor="S2", datetime="2023-07-01T10:00:00")
    body = json.loads((tmp_path / "scene.json").read_text())
    assert body["schema_version"] == contracts.SCHEMA_VERSIONS["scene"]
    assert scenes.read(tmp_path)["date"].isoformat() == "2023-07-01"


def test_unversioned_scene_json_still_reads(tmp_path):
    (tmp_path / "scene.json").write_text(json.dumps({"sensor": "S1", "datetime": "2023-07-02T05:00:00Z"}))
    meta = scenes.read(tmp_path)
    assert meta["sensor"] == "S1" and meta["schema_version"] == 1


def test_newer_scene_json_is_refused_with_upgrade_hint(tmp_path):
    (tmp_path / "scene.json").write_text(json.dumps({"schema_version": 99, "sensor": "S2",
                                                     "datetime": "2023-07-01"}))
    with pytest.raises(contracts.SchemaVersionError, match="Upgrade geocrop-engine"):
        scenes.read(tmp_path)


@pytest.mark.parametrize("bad", ["1", -1, True, 1.5])
def test_invalid_schema_version(bad):
    with pytest.raises(contracts.SchemaVersionError, match="Invalid schema_version"):
        contracts.check("scene", {"schema_version": bad})


def test_missing_migration_is_reported(monkeypatch):
    monkeypatch.setitem(contracts.SCHEMA_VERSIONS, "scene", 2)
    with pytest.raises(contracts.SchemaVersionError, match="no migration to 2"):
        contracts.check("scene", {"schema_version": 1}, "x/scene.json")
    monkeypatch.setitem(contracts.MIGRATIONS, ("scene", 1), lambda b: {"renamed": b.pop("old"), **b})
    assert contracts.check("scene", {"schema_version": 1, "old": 3}) == {"schema_version": 2, "renamed": 3}


def test_run_manifest_round_trip(tmp_path):
    path = provenance.run_manifest(tmp_path, config=None)
    body = provenance.read_run_manifest(path)
    assert body["schema_version"] == contracts.SCHEMA_VERSIONS["run_manifest"]
    assert body["schema"] == "geocrop-run/1"
    legacy = {k: v for k, v in json.loads(path.read_text()).items() if k != "schema_version"}
    path.write_text(json.dumps(legacy))
    assert provenance.read_run_manifest(path)["schema_version"] == 1


def test_model_payload_version(samples, stack, tmp_path):
    model.train(samples, tmp_path / "m", params={"n_estimators": 5, "n_jobs": 1}, n_splits=2,
                spatial_block_m=500, bootstrap=10)
    from joblib import load
    payload = load(tmp_path / "m" / "model.joblib")
    assert payload["schema_version"] == contracts.SCHEMA_VERSIONS["model"]
    payload.pop("schema_version")                         # models trained before versioning still predict
    dump(payload, tmp_path / "legacy.joblib")
    assert model.predict(stack, tmp_path / "legacy.joblib", tmp_path / "p", block_size=64)["classes"]
    dump({**payload, "schema_version": 2}, tmp_path / "future.joblib")
    with pytest.raises(contracts.SchemaVersionError, match="model in .*future.joblib has schema_version 2"):
        model.predict(stack, tmp_path / "future.joblib", tmp_path / "p2", block_size=64)
