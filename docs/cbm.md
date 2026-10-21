# Checks-by-Monitoring: signatures, markers, traffic light

The EU Area Monitoring System (Reg. (EU) 2021/2116; Implementing Reg. (EU) 2022/1173) and operational tools
such as Sen4CAP and the JRC CbM toolkit assess parcels with **time series of parcel means** and **markers** —
evidence of agricultural events — summarised as a traffic light. GeoCrop Engine provides the same building blocks.

## Parcel signatures
`geocrop signatures --scenes data/interim/s2 --parcels lpis.gpkg --reference grid.tif --start 2023-03-15 --end 2023-10-31 --out sig.parquet`

For every parcel and Sentinel-2 acquisition: mean and standard deviation of each index over **interior pixels**
(inward buffer), number of clear pixels and parcel pixels. Rasterisation and accumulation are block-wise, so the
cost does not depend on the number of parcels. Output: a long table (CSV or Parquet) suitable for GIS joins,
dashboards and parcel-level classifiers.

## Markers
`geocrop markers --signatures sig.parquet --start 2023-03-15 --end 2023-10-31 --out markers.csv`

Observations covering < 50 % of a parcel are ignored; the remaining parcel series is Whittaker-smoothed on a
5-day grid and summarised as:

| Marker | Definition |
|---|---|
| `sos_doy`, `peak_doy`, `peak_value`, `eos_doy` | phenology of the parcel series |
| `harvest_doy` | first date after the peak with an NDVI decline > 0.2 within 30 days (harvest or rapid senescence) |
| `bare_soil_days` | days (April–September) with smoothed NDVI < 0.25 |
| `clear_observations` | number of usable acquisitions (markers are NaN below 4) |

## Crop calendar
A YAML calendar (`configs/crop_calendar.example.yaml`) gives expected day-of-year ranges of peak and harvest per
class. `calendar_consistent` is false when a parcel's markers contradict the calendar of the *declared* crop,
e.g. a parcel declared as soybean that peaks in May. Calendars are regional knowledge and must be set by
agronomists for each study area and year.

## Traffic light
`traffic_light` in `parcels.gpkg` combines all evidence:

| Colour | Rule |
|---|---|
| green | declaration agrees with the accepted class, the conformal set is exactly the declared crop (if available), calendar not contradicted |
| red | declared crop outside a non-empty conformal set, **or** markers contradict the declared crop's calendar, **or** (without conformal sets) a discrepancy |
| yellow | everything else with data: ambiguous sets, review decisions, outside the AOA |
| grey | no declaration or insufficient resolution |

Red and yellow parcels are inspection candidates — never automatic findings of non-compliance.

## Planning the next field campaign
`geocrop fieldwork --parcels parcels.gpkg --out visits.gpkg --n 60 --min-spacing-m 2000 --per-class 10`
ranks parcels by normalised entropy, ambiguous conformal sets and extrapolation (AOA) and selects them greedily
with a minimum spacing (use the variogram range) and optional per-class quotas — an uncertainty-sampling
active-learning loop (Tuia et al., 2011) that directs field effort to where the model is weakest.

## References
Sitokonstantinou et al. (2018) Remote Sens. 10:911 · Defourny et al. (2019) RSE 221:551–568 ·
Tuia et al. (2011) IEEE JSTSP 5:606–617 · Settles (2009) Active Learning Literature Survey.
