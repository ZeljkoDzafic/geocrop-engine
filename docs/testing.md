# Testing

| Tier | Command | Content |
|---|---|---|
| Unit and integration (default) | `pytest` | every module on synthetic data, golden-value feature regression, pipeline resumption, reliability gate (below), web/QGIS |
| Scale | `pytest -m slow` | larger rasters, bounded-memory feature stack (peak < 50 % of an in-memory stack); features + predict at 1024² and 2048² with peak traced memory independent of raster size and roughly linear runtime |
| Real data | `pytest -m network` | Sentinel-2 L2A COGs over ~4 km² in Semberija, July 2023: radiometry plausibility, compositing with cloud screening, NDVI range |

The `tests` workflow runs the default tier on Python 3.10–3.12 for every push; the `integration` workflow runs the
`slow` and `network` tiers weekly and on demand. Network failures of the public STAC service skip, rather than
fail, the real-data tests.

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
