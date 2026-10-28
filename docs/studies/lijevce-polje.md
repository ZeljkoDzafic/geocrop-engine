# Case study: Lijevče polje, 2023

Lijevče polje (Lijevče Polje) is the alluvial plain of the lower Vrbas in the northern Republic of Srpska,
Bosnia and Herzegovina: bounded by the Sava (north, border with Croatia), Prosara (west), Motajica (east) and
Kozara (south-west), in the municipalities of Gradiška, Laktaši and Srbac
([hr.wikipedia](https://hr.wikipedia.org/wiki/Lijev%C4%8De_polje),
[sr.wikipedia](https://sr.wikipedia.org/wiki/%D0%9B%D0%B8%D1%98%D0%B5%D0%B2%D1%87%D0%B5_%D0%BF%D0%BE%D1%99%D0%B5)).
Its lowest point (~90 m) is at the Vrbas–Sava confluence; the plain consists of alluvial terraces of the Sava and
Vrbas with higher terraces at the foot of Kozara, and has been embanked and drained mainly between 1901 and 1980
(Koljanin et al. 2025, *Tuexenia* 45: 9–36, https://doi.org/10.14471/2025.45.002). The Bardača wetland (Ramsar
site 1658, 3500 ha) lies in its north-east (https://rsis.ramsar.org/ris/1658).

| File | Purpose |
|---|---|
| `configs/studies/lijevce_polje/aoi.geojson` | study AOI (EPSG:4326) with sources, checksums, threshold and area |
| `configs/studies/lijevce_polje/study_2023.yaml` | engine configuration for the 2023 season (EPSG:32633) |
| `tools/make_study_aoi.py` | reproducible AOI construction |
| `tools/study_data_check.py` | Sentinel-2 availability and clear-observation planning check |

## AOI definition

There is no official polygon of the plain and published delimitations differ (Koljanin et al. use ~465 km² between
the Sava, the Vrbas and Kozara). The AOI is therefore an explicit, reproducible study-design choice:

```
AOI = (Gradiška ∪ Laktaši ∪ Srbac) ∩ Bosnia and Herzegovina ∩ {Copernicus DEM GLO-30 < 135 m}
      -> opening (disk radius 3 px ≈ 90 m) -> drop patches < 5 km² -> fill enclosed gaps < 5 km²
      -> vectorise -> simplify 30 m (EPSG:32633) -> clip to the municipality union
```

**Threshold.** The elevation histogram (5 m bins) of the DEM inside the three municipalities has its mode at
90–95 m (floodplain) and decreases steadily up to ~130 m, after which it flattens into the continuum of the Kozara,
Prosara and Motajica foothill slopes (bin counts 125–130 m: 53 146; 130–135 m: 49 209; 135–140 m: 50 868). The
rule "first local minimum above the mode" (`--threshold auto`, the default) therefore selects **135 m**. Lower fixed
values cut off the southern apex of the plain near Laktaši (the plain rises southwards along the Vrbas), higher ones
add the Banja Luka valley and side valleys reaching into the hills. Sensitivity (same script and cleaning,
`--threshold N`): 115 m → 563.0 km², 125 m → 651.1 km², **135 m → 719.4 km²**, 145 m → 784.9 km², 155 m → 842.4 km².

**Cleaning.** Copernicus DEM is a *surface* model: forest stands, settlements and embankments stand above the
terrain. The opening removes 1–2-pixel connections along roads and narrow valley fingers, patches < 5 km² are
dropped, and enclosed gaps < 5 km² are filled.

**Result** (`configs/studies/lijevce_polje/aoi.geojson`, SHA-256 `005235e35b9a8afd454d9397d89495800c3d7e300155a0fe8c6f05a5ebefdf13`):

| Property | Value |
|---|---|
| area (EPSG:3035, ETRS89-LAEA) | **719.4 km²** |
| by municipality | Gradiška 405.6 km², Srbac 214.7 km², Laktaši 99.0 km² |
| bounding box (WGS 84) | 16.99400–17.80011 °E, 44.85667–45.23361 °N (~2640 km², larger than the 2500 km² quicklook limit) |
| UTM zone | 33 only (12–18 °E) → `grid.crs: EPSG:32633` |
| DEM range in the three municipalities | 83.4–883.0 m |

The AOI includes the lowland of Srbac east of the Vrbas up to the foot of Motajica (consistent with the
Wikipedia delimitation "Motajica in the east") and the Sava lowland of Gradiška west to Prosara, so it is larger
than the area of Koljanin et al.

## Sources and licences

| Data | Version / access | Licence and attribution |
|---|---|---|
| geoBoundaries gbOpen BIH ADM0 + ADM3 (municipalities) | release commit `f549eab` (raw GitHub URLs in `tools/make_study_aoi.py`), SHA-256 pinned | `boundaryLicense: Public Domain` (source: Wikimedia Commons) per the geoBoundaries API; cite Runfola et al. 2020, *PLoS ONE* 15(4): e0231866, https://doi.org/10.1371/journal.pone.0231866 |
| Copernicus DEM GLO-30 Public (DSM, 2021 release) | AWS Open Data `copernicus-dem-30m` COGs, tiles N44/N45 × E016/E017 (windowed HTTPS reads), SHA-256 of the mosaicked array recorded | *Licence for the use of the Copernicus WorldDEM-30*: reproduction, distribution and adaptation allowed free of charge with the notice "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" and the liability sentence (both stored in the GeoJSON properties). https://doi.org/10.5270/ESA-c5d3d65 |
| Sentinel-2 L2A (data check, quicklooks) | Element 84 Earth Search v1 (`sentinel-2-l2a`) | "Contains modified Copernicus Sentinel data 2023" |

The AOI is a derived product of public-domain boundaries and an adapted Copernicus DEM, so committing it with the
notices above is permitted.

## Reproduce

```bash
# AOI (network: geoBoundaries on GitHub, Copernicus DEM on AWS); byte-identical output on re-run
python tools/make_study_aoi.py --out configs/studies/lijevce_polje/aoi.geojson
# study grid and configuration check
geocrop grid --aoi configs/studies/lijevce_polje/aoi.geojson --crs EPSG:32633 \
    --out data/lijevce_polje/reference/grid_10m.tif
geocrop config-check --config configs/studies/lijevce_polje/study_2023.yaml
# scene inventory over the AOI bounding box and the clear-observation planning check
geocrop stac-search --endpoint https://earth-search.aws.element84.com/v1 --collection sentinel-2-l2a \
    --bbox 16.994 44.85667 17.80011 45.23361 --start 2023-04-01 --end 2023-09-30 \
    --max-items 1000 --cloud-max 100 --out outputs/lijevce_polje/inventory_s2_2023.csv
python tools/study_data_check.py --inventory outputs/lijevce_polje/inventory_s2_2023.csv \
    --aoi configs/studies/lijevce_polje/aoi.geojson --config configs/studies/lijevce_polje/study_2023.yaml \
    --end 2023-09-30 --out outputs/lijevce_polje/data_check_2023.json
geocrop preflight --config configs/studies/lijevce_polje/study_2023.yaml \
    --inventory outputs/lijevce_polje/inventory_s2_2023.csv
# quicklook of a central subset (the web workspace preset "Lijevče polje, Jun–Aug 2023")
geocrop quicklook --bbox 17.26 45.04 17.32 45.08 --start 2023-06-01 --end 2023-08-31 \
    --out outputs/lijevce_polje/quicklook_center
```

## Data availability, 2023 (engineering check — not a study result)

Inventory (Earth Search, run 2026-10-26): **298 Sentinel-2 L2A items on 73 dates**, 1 April – 30 September 2023,
four MGRS tiles (33TXK, 33TXL, 33TYK, 33TYL), all relative orbits combined. Clear-observation coverage from the SCL
layer (clear = SCL not in {0, 1, 3, 8, 9, 10, 11}, tiles of the same date combined, 100 m planning grid over the
AOI, no mask dilation or temporal screening):

| Window | Dates | Dates with ≥ 50 % of the AOI clear | Clear dates per AOI pixel (min / median / max) | AOI share with ≥ 3 clear dates |
|---|---|---|---|---|
| P1 2023-04-01 – 05-31 | 24 | 5 | 2 / 6 / 11 | 99.9 % |
| P2 2023-06-01 – 07-15 | 18 | 9 | 2 / 9 / 14 | 100 % |
| P3 2023-07-16 – 08-31 | 19 | 13 | 4 / 13 / 17 | 100 % |
| P4 2023-09-01 – 09-30 (window ends 10-15; inventory clipped) | 12 | 7 | 2 / 7 / 10 | 98.8 % |

Dates with ≥ 90 % of the AOI clear: 2023-04-02, 04-22, 05-07, 05-22, 06-16, 07-11, 07-16, 07-21, 08-15, 08-20,
08-25, 09-09, 09-29. P1 is the cloudiest window (5 of 24 dates mostly clear), as expected for spring in the northern
lowlands; every AOI pixel still has at least two clear dates in each window, so the provisional window dates of
`study_2023.yaml` are kept. SCL misclassifies some thin cloud and haze, so the composites (with mask dilation and
temporal screening) will have fewer usable observations than these counts.

`geocrop preflight` with the study config and the inventory: grid 26.44 Mpx (bounding box ~2644 km² at 10 m),
expected disk ~1.3 GB for the configured outputs (6.6 GB of free space requested incl. intermediates), peak RAM ~1 GB, prediction ~0.4 h (rough planning
figures). It is **blocked** only by the missing training polygons; parcels (LPIS) and field observations are
missing as well (warnings). These data are not public and are TODO items in the config.

Quicklook of the central subset 17.26–17.32 °E, 45.04–45.08 °N (21.0 km², EPSG:32633; the web-workspace preset),
June–August 2023, default filters (tile cloud cover < 60 %, ≥ 30 % of the subset clear): 50 acquisitions found
(tiles 33TXK and 33TXL overlap here), 41 imported, monthly composites from 10 (June), 15 (July) and 16 (August)
acquisitions; median clear observations per pixel (`valid_count`) 5, 7 and 7. The whole AOI bounding box exceeds
the 2500 km² quicklook limit; the study-area figure of the paper uses two quicklooks (split at 17.40 °E) of the
fully clear date 2023-07-16.

## Before the final run: intake must PASS

The training polygons, LPIS/RPG declarations and field observations are restricted data and are never committed.
When they arrive, check them with `geocrop intake` before any multi-hour run; the final run starts only when the
verdict is **pass** (or **warn** with every warning reviewed and recorded):

```bash
geocrop intake --config configs/studies/lijevce_polje/study_2023.yaml --require-all \
    --training data/lijevce_polje/reference/training_2023.gpkg \
    --parcels data/lijevce_polje/reference/lpis_2023.gpkg \
    --observations data/lijevce_polje/reference/field_observations_2023.gpkg \
    --crop-mapping data/lijevce_polje/reference/crop_mapping_rs.csv
```

Step-by-step instructions for the data provider (Serbian): [Priprema podataka](../priprema-podataka.md).
`geocrop preflight` reminds of this step and still covers scenes, grid and resources.

## Open items

- Training polygons, LPIS declarations and field observations for 2023 inside the AOI (the 366 polygons of
  `configs/rs_north_2023.yaml` cover 25 municipalities; the subset inside Lijevče polje must be recounted).
- Transfer validation: leave-one-municipality-out has only three groups of unequal size (Laktaši 99 km²); decide
  between it and larger spatial blocks once class counts per group are known.
- The crop-code mapping (`parcels.crop_mapping`) still points to the example CSV.
