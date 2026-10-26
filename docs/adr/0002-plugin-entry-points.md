# 0002. Plugins through Python entry points

* Status: Accepted
* Date: 2026-10-26

## Context
Researchers want to compare their own classifier, index or segmentation method on GeoCrop's spatial CV,
calibration and parcel decisions. Forking the engine to add an `elif` branch makes results hard to reproduce
and impossible to upgrade. A plugin mechanism must not add a framework, a server or configuration files.

## Decision
Use the standard Python packaging mechanism: entry points, discovered with `importlib.metadata`.
[`geocrop.plugins`](../../src/geocrop/plugins.py) defines five registries:

| Registry | Entry-point group | Contract |
|---|---|---|
| `estimators` | `geocrop.estimators` | `factory(params: dict, seed: int) -> estimator` with `fit(X, y) -> self`, `predict_proba(X) -> (n, n_classes)` and `classes_` after fit; selected with `model: {algorithm: <name>}` |
| `indices` | `geocrop.indices` | a `geocrop.indices.Index` (or zero-argument callable returning one) whose `name` equals the entry-point name |
| `segmenters` | `geocrop.segmenters` | object or class satisfying `geocrop.segmenters.Segmenter`: `name`, `segment(raster, output, **params) -> dict` with `path`, `segments` |
| `exporters` | `geocrop.exporters` | `export(source, destination, **options) -> Path \| dict` |
| `validators` | `geocrop.validators` | `validate(path, **options) -> dict` with a `problems` list |

* Built-ins register through the same registries in code (`estimators.py`, `indices.py`, `segmenters.py`;
  exporters/validators as lazy `"module:attribute"` names so core does not import adapters).
* Lookup consults the registry: `estimators.make_estimator`, `indices.get` (hence features, configuration
  validation, CBM markers, time-series features) and `benchmark.segmentation` / `benchmark.run`.
* Discovery is lazy: entry points are read only when a name is not registered yet (or a listing is requested),
  so built-in runs never scan installed packages.
* A plugin that fails to import or violates its contract emits `geocrop.plugins.PluginWarning` and is skipped.
  Entry points cannot shadow built-in names. Registration in code (`plugins.estimators.register(...)`) is
  available for notebooks and tests.
* `python -m geocrop.plugins` lists every registered name and its source.

A complete installable example is in [`examples/plugin_example`](../../examples/plugin_example).

## Consequences
* New methods are added by `pip install`; results record the algorithm name in `train_report.json` and the
  model payload. Reproducing a run requires the same plugin version, which the run manifest's package list
  only covers when the plugin is added to `provenance.PACKAGES` by the user.
* Plugins run in-process with full trust; installing one is equivalent to installing any Python package.
* Not yet wired: the CLI `--algorithm` choices list built-ins only (YAML `model.algorithm` accepts plugins),
  the pipeline forwards only the documented model keys (`model.RF_KEYS`) to estimator factories, and exporters
  and validators are not yet selectable from the CLI.
