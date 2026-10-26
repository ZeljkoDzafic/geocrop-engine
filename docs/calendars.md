# Crop calendars and phenological windows

GeoCrop uses crop calendars in two places: the **compositing windows** of a study (`windows:` in the
configuration) and the **calendar-consistency check** of Checks-by-Monitoring markers
(`cbm.calendar` → `geocrop.cbm.calendar_consistency`). Both used to be written by hand
(`configs/crop_calendar.example.yaml`). `geocrop calendar` makes them reproducible:

| Source | Command | Output |
|---|---|---|
| Your labelled training parcels (data-driven) | `geocrop calendar derive` | expected day-of-year ranges per class + suggested windows YAML |
| WorldCereal global calendars (external prior) | `geocrop calendar worldcereal` | SOS/EOS day of year for maize and winter wheat at a location |

## Data-driven calendar from training parcels

1. Parcel signatures and markers (see [Checks-by-Monitoring](cbm.md)):

   ```bash
   geocrop markers --signatures outputs/run/cbm/signatures.parquet --start 2023-03-01 --end 2023-11-15 \
       --out outputs/run/cbm/markers.csv
   ```

2. Per-class percentiles of SOS, peak, EOS and harvest/rapid senescence for the labelled parcels:

   ```bash
   geocrop calendar derive --markers outputs/run/cbm/markers.csv \
       --labels data/reference/training_2023.gpkg --id-field parcel_id --label-field class_name \
       --percentiles 10 50 90 --min-parcels 5 \
       --out configs/crop_calendar.derived_2023.yaml \
       --year 2023 --windows-out configs/windows.suggested_2023.yaml
   ```

   `--signatures ... --start ... --end ...` can replace `--markers` (markers are then computed on the fly).

* **Calendar** (`--out`): `{class: {sos: [p10, p90], peak: [...], eos: [...], harvest: [...]}}`, rounded outward to
  whole days. `calendar_consistency` uses `peak` and `harvest`; `sos`/`eos` are informative. Events with fewer than
  `--min-parcels` finite values for a class are omitted. Use it as `cbm.calendar` in the configuration.
* **Suggested windows** (`--windows-out`): a commented `windows:` block. The season runs from the earliest
  low-percentile SOS minus 15 days to the latest high-percentile EOS plus 15 days; window boundaries are placed
  at the class-median SOS / peak / EOS dates, merging boundaries until every window lasts at least `--min-days`
  (default 30) and there are at most `--max-windows` (default 6). The comments list the class medians that
  produced the boundaries.

Limitations. A derived calendar summarises *these* parcels, *this* season and *this* marker definition: label
errors,
NDVI-only markers (harvest vs. drought senescence), cloud gaps and the Whittaker smoothing all propagate into it.
A check on one EuroCrops tile of the public benchmark (Pomurje, 2021, ~17 Sentinel-2 acquisitions,
338 parcels) gave plausible maize and oil-pumpkin ranges (median peak DOY 220 and 210) but a p90 peak of
DOY 300 for winter cereals: with sparse series the global NDVI maximum of some parcels falls on autumn
regrowth or cover crops after harvest. Review the derived calendar against agronomic knowledge and the scene inventory before freezing it into a study configuration,
and record the command in the study's provenance. A calendar derived from parcels of a CC-BY-SA dataset
(e.g. EuroCrops) is an adaptation of that dataset and inherits its ShareAlike terms.

## WorldCereal crop-calendar priors

```bash
geocrop calendar worldcereal --lon 16.17 --lat 46.62 --out outputs/calendars/worldcereal_pomurje.json
```

Downloads (once, into `$GEOCROP_CACHE/public`, default `~/.cache/geocrop/public`) the four 0.5° GeoTIFFs
`M1_SOS/M1_EOS` (maize, first season) and `WW_SOS/WW_EOS` (winter wheat) and samples them at the location. The files
are SHA-256-pinned in `geocrop.public_data.WORLDCEREAL_CHECKSUMS`. PANGAEA keeps them on tape: the first request
answers HTTP 503 while the file is staged, and the download waits (up to 15 minutes) until it is available.

Run on 2026-10-26 at 16.17 E, 46.62 N (Pomurje, Slovenia): maize SOS 113 / EOS 304, winter wheat SOS 48 /
EOS 212 (day of year, as published; the winter-wheat SOS refers to the definition of the source product, see
Franch et al., 2022). The product's own evaluation reports RMSE of roughly 18-35 days against Sentinel-2/Landsat
phenology, so treat the values as a coarse prior for windows, not as a parcel-level check.

Source and licence: Cintas, J., Franch, B., Becker-Reshef, I., et al. (2022). *Global crop calendars of maize and
wheat in the framework of the WorldCereal project* [dataset]. PANGAEA, <https://doi.org/10.1594/PANGAEA.946550>,
CC-BY-SA-4.0; method: Franch, B., et al. (2022), *GIScience & Remote Sensing* 59(1), 885-913,
<https://doi.org/10.1080/15481603.2022.2079273>. The rasters are read at run time and never bundled; values or
calendars derived from them carry the same attribution and ShareAlike licence.

## Python API

```python
from geocrop import calendars
derived = calendars.derive(markers_df, labels_df)          # labels: parcel_id, class_name
calendar = calendars.to_calendar(derived)                   # for cbm.calendar_consistency
windows = calendars.suggest_windows(derived, year=2023)     # {"P1": {"start": ..., "end": ...}, ...}
print(calendars.windows_yaml(windows, derived))
```
