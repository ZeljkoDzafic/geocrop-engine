# Testing

| Tier | Command | Content |
|---|---|---|
| Unit and integration (default) | `pytest` | every module on synthetic data, golden-value feature regression, pipeline resumption, web/QGIS |
| Scale | `pytest -m slow` | larger rasters, bounded-memory feature stack (peak < 50 % of an in-memory stack) |
| Real data | `pytest -m network` | Sentinel-2 L2A COGs over ~4 km² in Semberija, July 2023: radiometry plausibility, compositing with cloud screening, NDVI range; Sentinel-1 RTC (Planetary Computer) backscatter plausibility; the full real-data acceptance run |
| Acceptance | `geocrop acceptance --out <dir>` (`--offline` in the default tier) | the whole chain STAC → S2 + S1 → QA → composites → features → training → prediction → segmentation → parcels → validation → export, with a pass/fail report; see [acceptance](acceptance.md) |

The `tests` workflow runs the default tier on Python 3.10–3.12 for every push; the `integration` workflow runs the
`slow` and `network` tiers weekly and on demand. Network failures of the public STAC service skip, rather than
fail, the real-data tests.

The offline acceptance run (`tests/test_acceptance.py`, synthetic scenes, ≈ 20–30 s) is part of the default
tier; the real-data acceptance run is in the network tier and its Markdown report is published under
[docs/acceptance/](acceptance.md#published-reports).
