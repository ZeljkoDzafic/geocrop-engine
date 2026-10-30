# Reference-data intake

`geocrop intake` checks restricted reference data before a run, so that problems surface with a clear message at
hand-over and not hours into a training run. It is read-only: it never changes an input file and refuses to
overwrite one. The verdict is `pass`, `warn` or `error`; the exit status is 2 on error (and on warnings with
`--strict`). Every result is a recommendation for human review, and the thresholds are working rules, not
definitive judgements about data quality.

```bash
geocrop intake --config configs/studies/lijevce_polje/study_2023.yaml \
    --training data/reference/training_2023.gpkg --parcels data/reference/lpis_2023.gpkg \
    --observations data/reference/field_observations_2023.gpkg --crop-mapping configs/crop_mapping.csv
```

Without `--training/--parcels/--observations` the files named in `paths.*` of the configuration are used. The
report (`intake_report.json` and `.md`, by default under `<paths.output>/intake`) contains counts, class names,
declaration codes and at most `--list-offenders N` IDs or row numbers (default 20; 0 disables). It never contains
geometries or other attributes, so it can be shared.

## Inputs

* **Training polygons:** `polygon_id` (unique), `class_name`.
* **Declared parcels (LPIS/IACS):** `parcel_id` (unique) and either `declared_crop` or authority codes that a
  crop-mapping CSV (`source,class_name`) maps to model classes.
* **Field observations:** points with `class_name` and `observed_at` (ISO date); optional group field.
* **Formats:** GeoPackage, GeoJSON, Shapefile, FlatGeobuf, GeoParquet, and CSV with a WKT column
  (`--wkt-field`) or x/y columns (`--x-field --y-field --crs`); UTF-8 or cp1250, delimiters `;` `,` tab `|`,
  decimal commas. `normalize-parcels` and `normalize-observations` convert such tables to the neutral schema
  (day.month.year dates become ISO dates; other formats are refused).

## What is checked

* **Every layer:** readability (with hints for Shapefile and CSV), CRS, null or invalid geometry (and how many
  `make_valid` can repair), empty or duplicate IDs, required fields, class-name spelling (mojibake, whitespace,
  case and diacritic variants, with a suggested normalisation map), overlap with the AOI (share inside, features
  outside; 0 % inside suggests a wrong CRS or swapped x/y) and, when a `municipality` field or boundary file is
  available, counts per municipality.
* **Training:** polygons, hectares, pixels and expected samples per class; polygons smaller than one pixel after
  the interior buffer; class balance; missing target classes; support for spatial-block cross-validation and a
  locked holdout (the logic of [preflight](preflight.md)); the region field when region cross-validation is
  configured; training polygons that lie mostly outside the parcels.
* **Parcels:** declaration codes that no mapping row covers (an error from 5 %), conflicting mapping rows, target
  classes that no mapping row produces, blank or float-formatted codes, duplicate geometries and parcels below
  `min_valid_pixels` after the interior buffer.
* **Observations:** date format, dates against the season and `validation.year`, duplicates, classes absent from
  training, target classes without observations, and observations inside training polygons or their exclusion
  buffer (leakage between training and independent validation).

## Limitations

Class-support checks use polygon centroids, as preflight does, and pixel counts use polygon area divided by pixel
area. Per-municipality counts need the attribute or `--municipalities`. One `--crs` applies to all CSV tables of a
single call (convert with `normalize-*` first if they differ). Encoding detection is a heuristic (UTF-8, else
cp1250); `--encoding` overrides it. A synthetic example set that passes, a deliberately broken one and a tabular
one (cp1250, `;`, decimal comma, diacritics) are in `examples/intake_example/`.
