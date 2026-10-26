# 0001. Layered architecture enforced by tests

* Status: Accepted
* Date: 2026-10-26

## Context
GeoCrop is used to produce statistics that must be reproducible years later and to replace pieces of the
method (data source, classifier, segmentation) in research. Both are only cheap if the scientific code does
not depend on where data comes from or how the engine is operated. Until now this separation was a convention;
nothing stopped a core module from importing a download client, the web server or provenance code that shells
out to `git`.

## Decision
Every module in `src/geocrop` belongs to exactly one layer, declared in one place,
[`src/geocrop/_layers.py`](../../src/geocrop/_layers.py):

```
interface    cli, web, web_jobs, web_map                         (entry points)
    │
application  pipeline, config, provenance, cache, audit,         (workflows)
    │        experiments, benchmark, demo
    │
adapters     catalog, stac, stac_import, quicklook, export,      (outside world)
    │        adapters (LPIS), snap (SNAP gpt), client
    ▼
core         indices, features, timeseries, tsfeatures, samples, model, estimators, metrics,
             uncertainty, applicability, parcels, validation, fieldwork, cbm, segmentation,
             segmentation_metrics, segmenters, unet, spatial, splits, raster, clouds, scenes,
             s2, s1, plugins, contracts, hashing
```

*Update:* the listing above records the map when this ADR was accepted. Modules added since (for example
`web_review`; `acceptance`, `preflight`, `cleanup`, `evidence`, `registry`, `public_benchmark`,
`public_transfer`; `s1_stac`, `downloads`, `public_data`, `embedding_sources`, `ftw_weights`, `openeo_adapter`,
`datacube`; and further core modules) are classified in `_layers.py`, which stays the single authoritative map;
the current picture is in [architecture](../architecture.md#layers).

Rules (a layer may always import itself and `geocrop.__version__`):
1. core imports only core, the standard library and the scientific stack listed in `CORE_THIRD_PARTY`
   (numpy, scipy, pandas, scikit-learn, rasterio, geopandas, shapely, pyproj, scikit-image, torch, ...);
   never network or process libraries (`requests`, `pystac_client`, `fastapi`, `uvicorn`, `subprocess`,
   `urllib`, ...). Core reads and writes local files only.
2. adapters import core and other adapters.
3. application imports core, adapters and application.
4. interface imports anything; web modules are imported only by `cli` (`geocrop serve`) and by each other.
5. Exceptions are listed with a reason in `_layers.EXCEPTIONS` and must be lazy (inside a function). The only
   one is `geocrop.s1.run_snap`, a backward-compatible alias of the SNAP adapter.

`tests/test_architecture.py` parses every module with `ast` and fails with the offending file, line and a fix
hint; it also fails for modules missing from the map and imports all of core in a fresh interpreter to prove no
network library is loaded transitively.

Dependencies that must point "upwards" (a core lookup that should find code living elsewhere, e.g. a plugin
classifier) are inverted through registries in `geocrop.plugins` (ADR 0002).

Violations found when the rules were introduced and how they were fixed:
* `model` (core) imported `sha256` from `provenance` (application, runs `git`/`conda` subprocesses). The
  function moved to the core module `hashing`; `geocrop.provenance.sha256` is re-exported.
* `s1` (core) ran SNAP `gpt` via `subprocess`. Execution moved to the adapter `geocrop.snap`; `s1.run_snap`
  remains as a lazy alias, so the CLI and user code are unchanged.

## Consequences
* Swapping the data provider means writing an adapter that produces scene folders (`scene.json` + bands);
  core is untouched. Replacing the classifier or adding a segmenter is a plugin (ADR 0002).
* A new module must be classified before the test suite passes; this is deliberate friction.
* The map is coarse (module level). Within-module discipline is still a review matter.
* The QGIS plugin (`integrations/qgis`) is outside the package and only talks to the CLI; it is not checked.
