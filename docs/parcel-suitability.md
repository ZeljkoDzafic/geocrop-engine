# Parcel geometry suitability for 10 m analysis

`geocrop parcel-suitability` answers one question **before** a declaration is analysed: is the parcel technically
suitable for reliable Sentinel-2 (10 m) classification? A narrow strip or a sliver has few or no pixels that are not
mixed with its surroundings, so any class decision on it is unreliable whatever the declaration says.

**Semantics.** A parcel that fails the criteria is *technically unsuitable for reliable satellite analysis*. This is not a
finding about the declaration: it is neither supported nor unsupported by it, and such parcels should be reported as a
separate group (or checked by other means), never counted as unsupported declarations. The output is a recommendation
for human review, not a conclusion.

## Metrics per parcel

Parcels must be in a projected CRS in metres (they are reprojected to the raster's CRS when a raster is given).

| Column | Meaning |
|---|---|
| `area_ha` | planimetric area |
| `width_m` | representative width: `2 x` the radius of the largest inscribed circle (bisection on the negative buffer, tolerance 0.25 m); **this is the default criterion** (`--width-method inscribed`), robust for L and U shapes |
| `mean_width_m` | `2 A / P` (area over half the perimeter); equals the width of a long strip but is `A`-based and smaller than the inscribed width on compact shapes (a 100 m square gives 50 m instead of 100 m), so `--width-method mean` is the stricter variant |
| `compactness` | Polsby-Popper `4 pi A / P^2` (1 for a circle, towards 0 for slivers) |
| `elongation` | long over short side of the minimum rotated rectangle |
| `n_pixels` | pixel centres inside the parcel |
| `n_inner_pixels` | pixel centres inside the parcel shrunk by `--inner-buffer-m` (default 10) |
| `n_reliable_inner`, `share_reliable_inner` | inner pixels that are valid (finite and not nodata in every band, `raster.valid_mask`) and, with `--mask`, not 0 in the mask raster |

With `--raster` (the study grid or any raster of it) the pixels are those of that raster, counted in its extent;
without it, a virtual grid of `--resolution` metres (default 10) aligned to multiples of the resolution is used and
every pixel counts as reliable. The mask raster must be on exactly the same grid.

## Rule and reason codes

`suitability` is `suitable` or `technically_unsuitable`; `suitability_reasons` lists the codes joined by `;`:

| Code | Condition (defaults) |
|---|---|
| `area_below_min` | `area_ha < --min-area-ha` (0.2) |
| `width_below_min` | width `< --min-width-m` (50; variants 40 and 60) |
| `too_few_inner_pixels` | `n_inner_pixels < --min-inner-pixels` (4) |
| `too_few_reliable_inner_pixels` | enough inner pixels, but `n_reliable_inner < --min-reliable-inner-pixels` (default: `--min-inner-pixels`) |

**The thresholds are not validated** (`thresholds_validated: false`). Do not adopt them without the sensitivity
analysis below, and freeze them before an independent validation.

## Sensitivity analysis

`sensitivity_grid` evaluates every combination of `--grid-areas` (0.2 ha), `--grid-widths` (40, 50, 60 m) and
`--grid-inner-pixels` (1, 4, 9, 16) and reports, per configuration, the number and share of excluded parcels and the
excluded area (ha), for all parcels and by declared crop and by size class (`<0.2`, `0.2-0.5`, `0.5-1`, `1-2`, `2-5`,
`>=5 ha`). A stricter threshold never excludes fewer parcels (tested). It is written as
`suitability_sensitivity.csv` and `suitability_sensitivity.md`. Read it for the size of the effect on the data at hand
(for instance how much area is lost, and whether one crop loses disproportionately), not as a measure of correctness.

## Comparing P1, P2 and P3 on suitable parcels

With `--crop-presence crop_presence.csv` (output of [`geocrop crop-presence`](crop-presence.md)) the table is joined on
`parcel_id` and the three methods are repeated on (a) all declarations and (b) suitable parcels only
(`compare_subsets`): P1 `p1_match_whole` and `p1_match_agricultural`, P2 `soft_vote_agricultural >= --p2-threshold`
(0.5) and P3 `preliminary_status == supported`. Per subset and method it reports the declarations, those with evidence,
the number supported and the share supported; further the number and area of the excluded declarations, overall and by
declared crop (`subset_comparison.csv/.md`, `subset_excluded.csv`, `subset_excluded_by_crop.csv`). Declarations with no
suitability row are counted as not assessed and left out of the suitable subset. A different share on the suitable subset
is not an accuracy: without field-verified parcels none of the methods can be scored.

## Usage

```bash
geocrop parcel-suitability --parcels parcels.gpkg --out suitability/ [--raster grid.tif] [--mask validity.tif] \
    [--crop-presence crop_presence/crop_presence.csv]
```

Output: `parcel_suitability.csv`, `suitability_sensitivity.csv/.md`, with `--crop-presence` the `subset_*` tables and
`subset_comparison.md`, and `parcel_suitability_manifest.json` (configuration, input hashes, engine version). Options:
[CLI reference](cli.md#geocrop-parcel-suitability). Python: `geocrop.suitability.assess`, `classify`,
`sensitivity_grid`, `compare_subsets`.

## Limits

Counts depend on the pixel grid (a shift of the grid changes the pixel centres inside a small parcel); the widths
are geometric and ignore the point spread of the sensor and the registration error of the parcel boundary; validity
reflects only nodata and the optional mask, not cloud or shadow unless the mask encodes them.
