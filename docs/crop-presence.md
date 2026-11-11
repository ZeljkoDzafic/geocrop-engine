# Presence of a declared crop in heterogeneous parcels

`geocrop crop-presence` compares **three methods** (P1, P2, P3; the labels A/B/C of `geocrop parcels` are the strategies that assign a *class* to a parcel and are a different thing) on identical parcels and the same probability raster. It was
designed for cadastral parcels, which can hold several land-cover classes (forest, grass, built-up, maize, soybean ...):
a declaration is not wrong because the declared crop is not the majority of the *whole* parcel.

It separates two questions that are reported side by side and never merged into one decision:

1. Is the declared crop present on the agricultural part of the parcel?
2. Does the mapped area of that crop agree with the declared area (`--declared-area-field`, optional)?

## Methods

| | Method | What is computed |
|---|---|---|
| P1 | Pixel majority (reference) | dominant class of all pixels and of the agricultural part (`pixel_majority`, `pixel_majority_agricultural`, `p1_match_*`) |
| P2 | Area-weighted soft vote | `S(c) = sum_i A_i P_i(c) / sum_i A_i` for the declared crop, over the whole parcel (`soft_vote_whole`) and the agricultural part (`soft_vote_agricultural`); on a regular grid all `A_i` are equal |
| P3 | Segment-based evidence fusion | segments clipped to the parcel (OBIA polygons via `--segments`, otherwise connected same-class pixels): mean probability of the declared crop, dominant class, area, homogeneity (`segments` layer); supported segments (`--support-prob`, `--min-segment-ha`), `supported_share_agricultural`, `connected_support_ha` (largest 8-connected area of pixels supporting the declared crop), `segment_majority`, `pixel_segment_agree` |

The agricultural part is the set of pixels mapped as one of `--agricultural` (default maize, soybean, sunflower,
vegetables, winter crops); the declared class is always included, so a declared meadow keeps its grass. Forest,
built-up land and water are reported separately (`non_agricultural_area_ha`).

## Output

`crop_presence.csv`, `crop_presence.gpkg` (layers `parcels` and `segments`, for QGIS) and
`crop_presence_manifest.json` (configuration, input hashes, engine version). Columns per parcel include
`parcel_area_ha`, `declared_area_ha`, `agricultural_area_ha`, `crop_area_ha`, `crop_share_agricultural`,
`share_<class>` / `area_ha_<class>`, `pixel_majority`, `segment_majority`, `soft_vote_*`, `connected_support_ha`,
`mean_confidence`, `mean_margin`, `area_difference_ha`, `area_difference_pct`, `preliminary_status`, `review_reason`,
and the columns of the presence criterion and the mismatch taxonomy below (`presence_status`, `presence_reason`,
`connected_mean_p`, `connected_share_declared`, `expected_pixels`, `valid_pixel_fraction`, `agricultural_share`,
`dominant_nonagricultural_class`, `mismatch_category`, `flag_*`, `cmp_*`). The manifest (schema `geocrop-crop-presence/2`)
adds `presence_status_counts` and `mismatch_category_counts`.

## Preliminary status

`supported`, `disputed_review` or `not_supported`, from explicit thresholds (`--min-connected-ha`,
`--min-supported-share`, `--low-score`, `--min-confidence`, `--area-tolerance`). **The thresholds are not validated**
(`thresholds_validated: false`): choose them on parcels whose crop was verified in the field and freeze them before
independent validation. The status is satellite support for a declaration, never an administrative finding, and a
deviation of the mapped area is an indicator for review, not a measurement. Approval of a declaration is an
administrative decision, not a field verification; without field-verified parcels none of the methods can be scored.

A synthetic example with known truth is in the reproducibility package of the paper; the unit tests in
`tests/test_crop_presence.py` use a small synthetic raster.

## Relation to the traffic light and what the numbers mean

`preliminary_status` is a separate diagnostic output of this command; it does not feed the traffic light of
`geocrop parcels`. Presence of the crop somewhere on the parcel, the area it occupies and agreement with the declared
area are three different questions: a small connected supported part can support the claim that the crop exists without
confirming that the whole declared area is under that crop. `soft_vote_*` is computed on the valid pixels whose centres
fall inside the parcel (after the optional inward buffer); pixels with missing or invalid probabilities are excluded, not
counted as zero; every pixel has the same area on the regular grid; no minimum area enters the soft vote (`--min-segment-ha` applies only to
the segments counted as supported; `--min-connected-ha` is a threshold of the preliminary status rule, and
`connected_support_ha` itself is computed without a minimum). A mean probability is not a validated
estimate of the crop's area.

Whether a parcel is geometrically suitable for 10 m analysis at all (area, width, inner pixels) is a separate, earlier question: see [parcel suitability](parcel-suitability.md), whose `compare_subsets` repeats the P1/P2/P3 comparison on suitable parcels only.
## Audit of P3: why its supported set is a subset of P1 on the agricultural part

*A recommendation for human review, derived from the code and from synthetic parcels; not a result on field-verified data.*

`preliminary_status` returns `supported` only if `connected_support_ha >= --min-connected-ha` **and**
`supported_share_agricultural >= --min-supported-share` (default 0.5), where the share is the part of the agricultural part
covered by segments with mean P(declared) >= `--support-prob` and area >= `--min-segment-ha`. Without OBIA segments
a segment is a connected group of pixels of one predicted class, and a segment whose mean P(declared) is at least 0.5
can only consist of pixels whose argmax is the declared crop (apart from the exact tie P = 0.5), so the supported pixels
are a subset of the declared-crop pixels. A share of at least 0.5 of the agricultural part therefore means the declared
crop is at least half of it, which is the pixel majority of the agricultural part (P1) except for an exact tie at 50 %,
which `argmax` resolves by class order. Hence, with the default threshold, **P3 `supported` implies P1 agricultural
match** (up to exact ties), and P3 can only be stricter: the minimum segment area, the connected-area minimum, the low
confidence and area-difference reasons and `pixel_segment_agree is False` (forces `disputed_review`) only remove
parcels. In 600 random blocky synthetic parcels with noisy probabilities (`tests/test_crop_presence.py`) every
violation of the subset was such a tie. Consequences:

- A crop that covers 15-45 % of a heterogeneous agricultural part (for example a connected 1.05 ha soybean block on a
  3 ha parcel with 45 % maize and 20 % grass: share of the agricultural part 0.44) is `disputed_review` however large,
  connected and reliably classified it is. Lowering `--min-supported-share` below 0.5 is the only way to change this and
  it then also admits parcels where the crop is a minority of the agricultural part.
- `not_supported` requires **zero** pixels of the declared crop (and a soft vote below `--low-score`); a parcel with a
  0.04 ha speckle is always `disputed_review`, never `not_supported`.
- P3 therefore adds no support beyond P1 on the agricultural part; its added value is the segment-level and connected-area
  evidence and the reason codes, not a different decision. The columns and behaviour of P3 are unchanged for
  reproducibility (no behaviour change; no bug was found, the subset is a property of the rule).

## Presence criterion (`presence_status`)

A separate decision that does not use the share of the agricultural part. With `support_prob` the largest 8-connected
area of pixels with P(declared) >= `support_prob` (`connected_support_ha`) is evaluated:

| `presence_status` | Rule (all thresholds configurable, **not validated**) |
|---|---|
| `present` | `connected_support_ha >= --min-connected-ha` and `connected_mean_p >= --presence-min-mean-p` (0.6) and mean confidence in that area >= `--min-confidence` and, if a declared area is given, `connected_support_ha / declared_area >= --presence-min-share` (0.5) |
| `partial_presence` | connected support exists but is unreliable (`connected_support_unreliable`) or covers less than `--presence-min-share` of the declared area (`connected_area_below_declared_share`); or the declared crop is mapped on at least `--min-connected-ha` in total but not connected (`fragmented_support`) |
| `absent` | none of the above (`no_connected_support`) |
| `indeterminate` | fewer than `--min-pixels` valid pixels, declared crop not modelled, parcel smaller than `--min-expected-pixels`, or `valid_pixel_fraction < --min-valid-fraction` |

It answers "is the declared crop present on a sufficiently large, connected, reliably classified part", also when it is
not the majority. It does not say the declared area is correct (see `area_difference_*`), and the largest connected
area is a lower bound of the mapped area of the crop. `--min-connected-ha` is shared with the P3 status.

### Sensitivity

`geocrop crop-presence ... --sensitivity` also writes `presence_sensitivity.csv`: one row per cell of
`min_connected_ha` (0.03, 0.05, 0.1, 0.2) x `support_prob` (0.4, 0.5, 0.6) with the counts per `presence_status`, the
cross-tab with P1 on the agricultural part and with P2 (`present_and_p1_agri`, `present_not_p1_agri` = support beyond P1,
`p1_agri_not_present`, same for `p2`; P2 counts as supporting when `soft_vote_agricultural >= 0.5`, a convention of this
table only), the P3 `supported` count and `p3_supported_not_p1_agri` (the subset check, 0 with default settings). It
describes how decisions move with the thresholds and how they overlap between methods; with no field truth it measures
**consistency between satellite-based rules, not accuracy**. Python: `crop_presence.sensitivity(...)`.

## Mismatch taxonomy

For every parcel `mismatch_category` is the primary reason, and the boolean columns `flag_<reason>` are evaluated
independently (non-exclusive). The primary reason is chosen in the order below; `none` means present, area within
tolerance and models in agreement. All are indicators for review: they do not state what is wrong with the declaration.

| Reason | Flag true when |
|---|---|
| `too_few_pixels` | `expected_pixels` (parcel area / pixel area, after the buffer) `< --min-expected-pixels` (8), or the parcel lies in the raster but fewer than `--min-pixels` pixel centres fall in it (thin) |
| `no_valid_observations` | (not too small) `n_valid_pixels < --min-pixels` or `valid_pixel_fraction = n_valid_pixels / expected_pixels < --min-valid-fraction` (0.5); includes parcels outside the raster |
| `other_crop_present` | declared crop `absent` or `partial_presence` while another agricultural class has the pixel majority of the agricultural part |
| `declared_crop_partial` | `presence_status == partial_presence` |
| `area_disagreement` | `present` but `abs(crop_area_ha - declared_area_ha) / declared_area_ha > --area-tolerance` |
| `model_disagreement` | with `--compare-probabilities` a second model's `presence_status` or `p1_match_agricultural` differs |

Two residual primary values are not part of the six: `non_agricultural_dominant` (absent, and no other agricultural
class is present: see `dominant_nonagricultural_class`) and `declared_crop_not_modelled`. `dominant_nonagricultural_class`
is the largest non-agricultural class (grass, forest, built-up ...) of a parcel whose `agricultural_share` is at most
`--nonagri-max-share` (0.10), so analysts can see what dominates parcels without a crop.

`--compare-probabilities second.tif` analyses the same parcels on a second probability raster (for example E0 versus E8)
with the same parameters through the same code; the `cmp_*` columns hold its `presence_status`, `preliminary_status`,
`pixel_majority_agricultural`, `p1_match_agricultural`, `connected_support_ha`, `soft_vote_agricultural`. The model
disagreement is the primary reason only when nothing else applies, otherwise it stays a flag.

### Review sample

`--review-sample N` writes `review_sample.csv` and `review_sample.gpkg` (layer `sample`) with up to `N` parcels per
`mismatch_category`, spread over the range of parcel area (one draw per area-quantile bin, seeded by `--seed`), for visual
control in QGIS (`crop_presence.review_sample(table, n, output)` in Python).

```bash
geocrop crop-presence --probabilities probabilities_e8.tif --compare-probabilities probabilities_e0.tif \
  --parcels parcels.gpkg --declared-area-field declared_ha --min-connected-ha 0.05 --presence-min-share 0.5 \
  --review-sample 20 --sensitivity --out crop_presence/
```
