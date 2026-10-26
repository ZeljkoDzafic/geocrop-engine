# Testing

| Tier | Command | Content |
|---|---|---|
| Unit and integration (default) | `pytest` | every module on synthetic data, golden-value feature regression, pipeline resumption, web/QGIS |
| Scale | `pytest -m slow` | larger rasters, bounded-memory feature stack (peak < 50 % of an in-memory stack) |
| Real data | `pytest -m network` | Sentinel-2 L2A COGs over ~4 km² in Semberija, July 2023: radiometry plausibility, compositing with cloud screening, NDVI range; Sentinel-1 RTC (Planetary Computer) backscatter plausibility; the full real-data acceptance run |
| Acceptance | `geocrop acceptance --out <dir>` (`--offline` in the default tier) | the whole chain STAC → S2 + S1 → QA → composites → features → training → prediction → segmentation → parcels → validation → export, with a pass/fail report; see [acceptance](acceptance.md) |

| Unit and integration (default) | `pytest` | every module on synthetic data, golden-value feature regression, pipeline resumption, reliability gate (below), web/QGIS |
| Scale | `pytest -m slow` | larger rasters, bounded-memory feature stack (peak < 50 % of an in-memory stack); features + predict at 1024² and 2048² with peak traced memory independent of raster size and roughly linear runtime |
| Real data | `pytest -m network` | Sentinel-2 L2A COGs over ~4 km² in Semberija, July 2023: radiometry plausibility, compositing with cloud screening, NDVI range |

The `tests` workflow runs the default tier on Python 3.10–3.12 for every push; the `integration` workflow runs the
`slow` and `network` tiers weekly and on demand. Network failures of the public STAC service skip, rather than
fail, the real-data tests.

The offline acceptance run (`tests/test_acceptance.py`, synthetic scenes, ≈ 20–30 s) is part of the default
tier; the real-data acceptance run is in the network tier and its Markdown report is published under
[docs/acceptance/](acceptance.md#published-reports).

`pytest` selects every test except `network`, so the `slow` tests also run in the default tier; `pytest -m "not slow"`
skips them for a quick local loop.

## Reliability gate (`tests/test_reliability.py`)

| Concern | What is checked |
|---|---|
| Interruption | A real `geocrop run` subprocess (features → samples → train → predict on synthetic data) is SIGKILLed inside the features, train and predict stages, with partial products on disk. The killed stage stays `running` in `stages.json` and the lock is stale. Downstream stages refuse to read the partial products. The rerun takes over the stale lock, skips completed stages, reruns the interrupted stage and its dependants, and produces the same products as an uninterrupted reference run (identical stack, samples, model file, class raster, train metrics; probabilities within 1e-6). Recorded output identities match the files and no temporary files are left. A third run skips everything. |
| Corrupt inputs | Truncated GeoTIFFs (header, middle, tail); composites with another CRS, shifted transform or shape; bands that are entirely nodata, NaN or inf; isolated NaN/±inf pixels (which must end up unclassified, never as a class); stacks without band descriptions; truncated or all-nodata stacks given to predict; corrupt or incomplete `scene.json`; SCL masks with values outside 0–11 or a float dtype; empty training layers, empty geometries, duplicate or blank ids, blank labels, missing columns and polygons in the wrong CRS. Each case must raise an error whose message names the file and the problem. |
| Reproducibility | A second run in a fresh directory, with freshly generated inputs and the same configuration, produces byte-identical stack, samples, model, class/probability/AOA rasters and an identical `train_report.json`. |
| Locks and caches | Stale locks (dead pid, empty or garbage content) are taken over, a live run's lock is respected, and a damaged `.hashcache.json` only costs re-hashing. |
| Scale (`slow`) | Features + predict at 1024² and 2048² (two windows, 43 features, 512-pixel blocks): peak traced memory stays below 25 % of the stack size and does not grow with the raster. The runtime ratio stays between 1.5 and 10 for four times the pixels. |

### Fault-injection hook
`geocrop.faultinject.point(name)` is called once in each of `features.build_stack` (after the first block), `model.train`
(after `model.joblib`, before the reports) and `model.predict` (after the first block). It does nothing unless the
environment variable `GEOCROP_FAULT_PAUSE` equals that name. Then it writes `"<name> <pid>"` to the file in
`GEOCROP_FAULT_SIGNAL` (if set) and sleeps for `GEOCROP_FAULT_SECONDS` (default 600), so the test can kill the process at
a known point. Never set these variables in production runs.

## Security tests
`tests/test_security.py` is the security gate for the local web API: authentication on every route (both
`/api` and `/api/v1`), Host allow-list (DNS rebinding), cross-origin rejection, security headers on every
response, path traversal and symbolic-link escapes in artifact downloads and output paths, blocked commands and
options, shell metacharacters, argument/body/log size limits and the job queue limit. The threat model is in
[security](security.md). `tests/test_api_v1.py` checks that the versioned and unversioned API paths share
handlers, and `tests/test_check_versions.py` the release version check.

## Coverage
CI measures coverage of the default tier on every Python version:

```bash
python -m pytest --cov=geocrop --cov-branch --cov-report=term --cov-report=xml
```

`coverage.xml` is uploaded as a workflow artifact. The build fails below the current total minus two points
(`--cov-fail-under=76`, measured on the minimal CI install `.[dev,stac,web]` where optional backends — torch,
pyarrow, matplotlib, catboost, openeo, xarray — are skipped); raise the threshold when coverage improves. Engine
commands run by web jobs execute in subprocesses and are not counted; the CLI's own tests cover them in-process.

Measured on 2026-10-26 (Python 3.12, default tier, statements + branches combined): **78.6 %** on the minimal CI
install (statements 79.7 %, branches 75.2 %) and **89.9 %** with every optional backend installed (statements
91.5 %, branches 84.8 %). The `tests/test_cov_*.py` files target error and edge paths (empty inputs, single
class, NaN, nodata-only rasters, mismatched CRS/grids, corrupt JSON, mocked HTTP for the data adapters) and add
about 11 s to the default tier. Branch coverage per module, minimal CI install:

| Module | Statements | Branches | Module | Statements | Branches |
|---|---|---|---|---|---|
| `model` | 99.5 % | 98.4 % | `pipeline` | 97.5 % | 91.1 % |
| `features` | 100 % | 97.1 % | `tsfeatures` | 97.2 % | 95.5 % |
| `parcels` | 100 % | 99.1 % | `timeseries` | 100 % | 98.0 % |
| `validation` | 100 % | 100 % | `uncertainty` | 100 % | 100 % |
| `applicability` | 100 % | 100 % | `cbm` | 99.6 % | 98.9 % |
| `metrics` | 100 % | 100 % | `splits` | 98.8 % | 97.6 % |
| `cbm_markers` | 100 % | 100 % | `temporal_quality` | 100 % | 100 % |
| `evidence` | 93.0 % | 93.2 % | `preflight` | 98.0 % | 95.2 % |
| `review` | 99.5 % | 99.1 % | `web_review` | 98.8 % | 93.9 % |
| `registry` | 98.9 % | 98.4 % | `plugins` | 97.9 % | 97.8 % |
| `web` | 97.0 % | 94.6 % | `web_jobs` | 94.4 % | 90.4 % |
| `client` | 100 % | 96.2 % | `config` | 88.1 % | 82.1 % |
| `stac_import` | 100 % | 100 % | `s1_stac` | 99.6 % | 100 % |
| `openeo_adapter` | 99.8 % | 98.7 % | `embedding_sources` | 100 % | 100 % |
| `downloads` | 100 % | 100 % | `public_data` | 100 % | 100 % |

`datacube` needs xarray (extra `cube`) and is skipped on the minimal install; with it installed its branch
coverage is 98.4 %. The total stays below 80 % on the minimal install because whole optional-backend modules
(torch encoders, CatBoost, matplotlib reports, parquet paths) are not importable there.

**Goal for 1.0:** at least 90 % *branch* coverage for the core modules that determine results — `model`,
`features`, `tsfeatures`, `pipeline`, `parcels`, `validation`, `uncertainty`, `applicability`, `metrics`,
`cbm` — and for the web security surface (`web`, `web_jobs`, `client`). **Status:** reached for all thirteen
on the minimal CI install. The closest margins are `pipeline` (91.1 %; the remaining branches are stage
combinations such as embeddings attached to time-series stacks and S1 event markers, exercised by the slow
end-to-end tier) and `web_jobs` (90.4 %; the rest are non-POSIX process handling and shutdown races). A few
remaining branches are defensive assertions that cannot fire (train/test group leakage checks in `model` and
`splits`). Not yet at 90 %: `config` (82 %; validation messages of rarely used keys). Data-access modules
that need real services (`catalog`, `stac`, `s1`) are covered by the `network` tier; `stac_import`, `s1_stac`,
`openeo_adapter`, `downloads`, `embedding_sources` and `public_data` are now also covered offline with mocked
HTTP clients.
