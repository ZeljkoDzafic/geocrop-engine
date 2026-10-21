# Parcels and declaration verification

## Aggregation
For each parcel the probability raster is read inside the parcel's window. Pixels whose centres fall inside the
parcel **after an inward buffer** of `interior_buffer_px` pixels are used; this excludes mixed boundary pixels,
which dominate small fields (a 0.5 ha square field at 10 m has ~50 pixels, ~24 of them on its outer ring). If the
buffer removes the whole parcel the full geometry is used and `interior_buffer_applied = false`.

| Attribute | Definition |
|---|---|
| `n_valid` | number of valid pixels used |
| `majority_class`, `majority_share` | most frequent per-pixel argmax class and its share (strategy A) |
| `mean_class`, `confidence` | argmax of the mean probability vector and its value (strategy B) |
| `median_probability`, `prob_std` | median and standard deviation of the chosen class's pixel probabilities (homogeneity) |
| `margin`, `entropy` | top-2 margin and normalised entropy of the mean probability vector |
| `p_<class>`, `f_<class>` | mean probability and pixel fraction of every class |
| `strategies_agree` | A and B give the same class |
| `area_m2`, `shape_index` | area and perimeter / (2√(πA)) (1 = circle; elongated parcels have more edge pixels) |

## Decision strategies
| `strategy` | Accepted class | Review when |
|---|---|---|
| `majority` | A | any configured threshold fails |
| `mean_probability` | B | any configured threshold fails |
| `combined` | B if A = B | A ≠ B, or any configured threshold fails |

`min_valid_pixels` (default 4) produces `insufficient_resolution` instead of a forced class. Thresholds
(`min_confidence`, `min_majority_share`, `min_margin`) are **off by default**: select them on validation data with
the reject curve (target accuracy vs. review workload), never on the final test data.

## Declaration status
| Status | Meaning |
|---|---|
| `agreement` | accepted class equals the declared class |
| `discrepancy` | accepted class differs — candidate for inspection |
| `uncertain` | evidence insufficient (review decision) |
| `insufficient_resolution` | too few valid pixels for a satellite-based statement |
| `not_declared` | no declaration available |

Declaration codes are mapped to model classes with `crop_mapping` (CSV). Satellite prediction is always made
**before and independently of** the declaration; declarations are never used as training labels by default.
Statuses prioritise field inspections; they are **not** findings of non-compliance.

`parcels.summary.json` contains decision and status counts, also per declared class — the basis of the
agreement table in a report.

## Second-stage parcel model
`geocrop parcel-model --parcels parcels.gpkg --label-field verified_class` trains a Random Forest on the
aggregated descriptors (`p_*`, `f_*`, confidence, margin, entropy, homogeneity, area, shape) with spatially
blocked CV and reports it next to the direct majority and mean-probability baselines. Labels must come from
independently verified parcels.
