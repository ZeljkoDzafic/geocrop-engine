"""Frozen public plugin contracts (versioned snapshot in tests/golden/plugin_contracts.json).

External adapters depend on these names and signatures: the entry-point groups, the registry API, the
built-in names a plugin cannot shadow, the `Segmenter` protocol, the `Index` record, the exporter/validator
call signatures and the versioned JSON file contracts. Any change - including an addition - fails this test
until the snapshot is regenerated (which bumps its ``version``) AND CHANGELOG.md mentions the new
``plugin-contracts/<version>``:

    GEOCROP_UPDATE_CONTRACTS=1 python -m pytest tests/test_plugin_contracts.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path

from geocrop import contracts, indices, plugins, segmenters

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).with_name("golden") / "plugin_contracts.json"
CHANGELOG = ROOT / "CHANGELOG.md"
SCHEMA = "geocrop-plugin-contracts/1"


def _signature(obj) -> str:
    return str(inspect.signature(obj))


def _public_methods(cls) -> dict[str, str]:
    return {name: _signature(fn) for name, fn in sorted(vars(cls).items())
            if not name.startswith("_") and inspect.isfunction(fn)}


def _builtin_names(registry: plugins.Registry) -> list[str]:
    registry._ensure_provider()
    return sorted(n for n in (*registry._items, *registry._lazy) if registry._sources.get(n) == plugins.BUILTIN)


def current_contract() -> dict:
    """Everything an external plugin may rely on, as plain JSON."""
    registries = {key: {"kind": reg.kind, "group": reg.group, "builtins": _builtin_names(reg)}
                  for key, reg in sorted(plugins.REGISTRIES.items())}
    builtin_callables = {}
    for key in ("exporters", "validators"):
        reg = plugins.REGISTRIES[key]
        builtin_callables[key] = {name: _signature(reg.get(name)) for name in _builtin_names(reg)}
    return {
        "registries": registries,
        "registry_api": _public_methods(plugins.Registry),
        "module_api": {"describe": _signature(plugins.describe), "BUILTIN": plugins.BUILTIN,
                       "PluginWarning_bases": [b.__name__ for b in plugins.PluginWarning.__mro__[1:]]},
        "segmenter_protocol": {
            "runtime_checkable": bool(getattr(segmenters.Segmenter, "_is_runtime_protocol", False)),
            "attributes": sorted(getattr(segmenters.Segmenter, "__annotations__", {})),
            "segment": _signature(segmenters.Segmenter.segment),
        },
        "index_record": {"fields": [f.name for f in dataclasses.fields(indices.Index)],
                         "compute": _signature(indices.Index.compute)},
        "estimator_factory": "factory(params: dict, seed: int) -> estimator with fit, predict_proba, classes_",
        "builtin_callables": builtin_callables,
        "file_contracts": {"key": contracts.KEY, "versions": dict(sorted(contracts.SCHEMA_VERSIONS.items()))},
    }


def _digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _load() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf8"))


def _maybe_update() -> None:
    if os.environ.get("GEOCROP_UPDATE_CONTRACTS") != "1":
        return
    contract = current_contract()
    old = _load() if SNAPSHOT.exists() else {"version": 0, "sha256": ""}
    version = old["version"] if old.get("sha256") == _digest(contract) else old["version"] + 1
    SNAPSHOT.write_text(json.dumps({"schema": SCHEMA, "version": version, "sha256": _digest(contract),
                                    "contract": contract}, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _diff(old, new, path="") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in new:
                out.append(f"removed {path}/{key}")
            elif key not in old:
                out.append(f"added {path}/{key}")
            else:
                out += _diff(old[key], new[key], f"{path}/{key}")
        return out
    return [] if old == new else [f"changed {path}: {old!r} -> {new!r}"]


def test_plugin_contracts_match_the_frozen_snapshot():
    _maybe_update()
    snap = _load()
    assert snap["schema"] == SCHEMA
    contract = current_contract()
    changes = _diff(snap["contract"], contract)
    assert not changes, (
        "Public plugin contract changed:\n  " + "\n  ".join(changes) + "\nIf intended, regenerate with "
        "GEOCROP_UPDATE_CONTRACTS=1 python -m pytest tests/test_plugin_contracts.py (bumps the version) and add "
        "'plugin-contracts/<version>' with a migration note to CHANGELOG.md.")
    assert snap["sha256"] == _digest(contract), "snapshot digest does not match its contract body"


def test_changelog_records_the_contract_version():
    version = _load()["version"]
    assert f"plugin-contracts/{version}" in CHANGELOG.read_text(encoding="utf8"), (
        f"CHANGELOG.md must mention plugin-contracts/{version} (the current plugin contract snapshot)")


def test_example_plugin_satisfies_the_contract():
    """The shipped example adapter is checked against the frozen contract, not just against the code."""
    import sys
    sys.path.insert(0, str(ROOT / "examples" / "plugin_example" / "src"))
    try:
        import geocrop_plugin_example as ex
    finally:
        sys.path.pop(0)
    contract = _load()["contract"]
    assert isinstance(ex.GridSegmenter(), segmenters.Segmenter)
    assert [f.name for f in dataclasses.fields(ex.NDTI)] == contract["index_record"]["fields"]
    est = ex.make_nearest_centroid({"algorithm": "nearest_centroid", "unknown": 1}, 0)
    assert all(callable(getattr(est, m, None)) for m in ("fit", "predict_proba"))
    groups = {r["group"] for r in contract["registries"].values()}
    meta = (ROOT / "examples" / "plugin_example" / "pyproject.toml").read_text(encoding="utf8")
    declared = {line.split('"')[1] for line in meta.splitlines() if line.startswith('[project.entry-points."')}
    assert declared <= groups, f"example declares unknown entry-point groups {declared - groups}"
