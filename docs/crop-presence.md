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
`mean_confidence`, `mean_margin`, `area_difference_ha`, `area_difference_pct`, `preliminary_status`, `review_reason`.

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
