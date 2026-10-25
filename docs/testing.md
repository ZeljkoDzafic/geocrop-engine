# Testing

| Tier | Command | Content |
|---|---|---|
| Unit and integration (default) | `pytest` | every module on synthetic data, golden-value feature regression, pipeline resumption, web/QGIS |
| Scale | `pytest -m slow` | larger rasters, bounded-memory feature stack (peak < 50 % of an in-memory stack) |
| Real data | `pytest -m network` | Sentinel-2 L2A COGs over ~4 km² in Semberija, July 2023: radiometry plausibility, compositing with cloud screening, NDVI range |

The `tests` workflow runs the default tier on Python 3.10–3.12 for every push; the `integration` workflow runs the
`slow` and `network` tiers weekly and on demand. Network failures of the public STAC service skip, rather than
fail, the real-data tests.
