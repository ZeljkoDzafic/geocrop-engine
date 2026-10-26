# Testing

The suite under `tests/` has three tiers selected with pytest markers. `pytest` without options runs the
default tier (`addopts = -m 'not network'` in `pyproject.toml`).

| Tier | Command | Tests (2026-10-26) | Content |
|---|---|---|---|
| Default | `pytest` | 984 | every module on synthetic data: golden-value feature regression, pipeline resumption, the reliability gate (below), offline acceptance run, web API security gate, QGIS plugin, plugin contracts, documentation consistency |
| Scale | `pytest -m slow` | 3 (also in the default tier) | bounded-memory feature stack (peak < 50 % of an in-memory stack); features + predict at 1024² and 2048² with peak traced memory independent of raster size and roughly linear runtime |
| Real data | `pytest -m network` | 14 | public Sentinel-2 L2A COGs over ~4 km² of Semberija, July 2023 (radiometry plausibility, compositing with cloud screening, NDVI range, quicklook); Sentinel-1 RTC from Planetary Computer (backscatter plausibility); the full real-data acceptance run; odc-stac vs `stac-import` equality; AlphaEarth/TESSERA windows; Presto, Galileo, AnySat and FTW checkpoint download and loading; EuroCrops mapping table reachability; CDSE openEO graph validation (only with `CDSE_OPENEO_*` credentials) |
| Acceptance | `geocrop acceptance --out <dir>` (`--offline` runs in the default tier) | — | the whole chain STAC → S2 + S1 → QA → composites → features → training → prediction → segmentation → parcels → validation → export, with a pass/fail report; see [acceptance](acceptance.md) |

`pytest` selects every test except `network`, so the `slow` tests also run in the default tier;
`pytest -m "not slow"` skips them for a quick local loop. Run `pytest --co -q` for the current counts.

CI: the `tests` workflow runs the default tier with coverage on Python 3.10, 3.11 and 3.12 (Ubuntu 26.04) for
every push and pull request; the `integration` workflow runs `pytest -m "network or slow"` weekly and on demand
on Python 3.11 with the `dev,stac` extras only, so network tests that need PyTorch, odc-stac or credentials
skip there. Failures of the public STAC services skip, rather than fail, the real-data tests. The real-data
acceptance report is published under [docs/acceptance/](acceptance.md#published-reports).

## What the tests establish, and what they do not

* The default tier proves **software behaviour** on synthetic data with known answers (leakage-free folds,
  hand-computed estimators, resumption, error messages, security). Its accuracy numbers mean nothing for real
  crops.
* The network tier and the acceptance run prove that real Sentinel-1/2 data flow through the chain with
  physically plausible values on one small area and season. They use k-means **pseudo-labels**, so they do not
  measure crop-map accuracy.
* Classification accuracy, transfer and conformal coverage on real labels come from the
  [public benchmarks](benchmarks/README.md) (declared labels, Slovenia), which are reproducible commands rather
  than tests. Field-verified accuracy in the study region is still open; see [limitations](limitations.md).

## Documentation consistency (`tests/test_check_docs.py`)

`python tools/check_docs.py` (also run by the default tier) checks that every relative link and `#anchor` in the
Markdown documents and in `website/index.html` resolves, that every `geocrop <command>` and `--option` written in
code in the living documents exists in the CLI parser, that `docs/configuration.md` lists exactly the keys of
`geocrop.config.DEFAULTS` (marker/rule groups are documented on their own pages), and that
`docs/architecture.md` names every module of the layer map. A second test fails when `docs/cli.md` differs from
`python tools/generate_cli_docs.py` output. Dated records (CHANGELOG, reviews, release-gate and acceptance
reports) are checked for links only.

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

`coverage.xml` is uploaded as a workflow artifact. The build fails below `--cov-fail-under=71`: the total measured on
the minimal CI install `.[dev,stac,web]` (72.9 %, optional backends such as PyTorch skipped) minus two points; raise
the threshold when coverage improves. Engine commands run by web jobs execute in subprocesses and are not counted;
the CLI's own tests cover them in-process.

Measured on 2026-10-26 (Python 3.12, default tier, local environment with the optional extras including
PyTorch, so slightly higher than a CI job without them; statements + branches combined: **83.8 %**; statements
86.6 %, branches 75.2 %):

| Module | Statements | Branches | Module | Statements | Branches |
|---|---|---|---|---|---|
| `model` | 96.2 % | 90.4 % | `pipeline` | 93.8 % | 82.1 % |
| `features` | 96.4 % | 87.3 % | `tsfeatures` | 92.8 % | 86.4 % |
| `parcels` | 94.8 % | 84.5 % | `timeseries` | 96.6 % | 86.0 % |
| `validation` | 92.7 % | 64.7 % | `uncertainty` | 94.9 % | 71.4 % |
| `applicability` | 95.8 % | 70.0 % | `cbm` | 96.7 % | 88.6 % |
| `metrics` | 96.6 % | 83.3 % | `audit` | 97.7 % | 98.1 % |
| `web` | 97.0 % | 94.6 % | `web_jobs` | 91.8 % | 84.0 % |
| `client` | 91.7 % | 88.5 % | `config` | 88.5 % | 82.7 % |

**Goal for 1.0:** at least 90 % *branch* coverage for the core modules that determine results — `model`,
`features`, `tsfeatures`, `pipeline`, `parcels`, `validation`, `uncertainty`, `applicability`, `metrics`,
`cbm` — and for the web security surface (`web`, `web_jobs`, `client`). Data-access modules that need real
services (`catalog`, `stac`, `stac_import`, `s1`) are covered by the `network` tier instead.
