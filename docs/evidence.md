# Parcel evidence packages (audit trail)

`geocrop evidence` writes one self-contained, versioned package per parcel decision. A package lets
an analyst, an auditor or a farmer see *why* a parcel is green, yellow, red or grey: which
classification, which rules with which inputs and thresholds, and which satellite acquisitions were
actually observed and which values were interpolated. Review flags support inspection
prioritisation. They are not findings of non-compliance.

```bash
geocrop evidence --parcels parcels/parcels.gpkg --out evidence \
  --signatures cbm/signatures.csv --start 2023-03-15 --end 2023-10-31 \
  --markers cbm/markers.csv --config config.yaml \
  --model model --run-manifest run_manifest.json --scenes scenes/s2 \
  [--ids P001 P017] [--temporal-support cbm/temporal_support.csv] [--no-png]
```

| Input | Use |
|---|---|
| `--parcels` | decision table from `geocrop parcels` or `geocrop parcel-classify` (`.gpkg`, `.geojson` or `.csv`). The `<parcels>.summary.json` next to it (or `--summary`) supplies the decision thresholds, the conformal alpha and `min_aoa_fraction`. |
| `--signatures`, `--start`, `--end` | parcel series. They separate observed from rejected (cloud/coverage) acquisitions, give the interpolated grid and, when `--temporal-support` is not given, compute the temporal support ([temporal quality](temporal-quality.md)). `--start`/`--end` default to `timeseries.start/end` of `--config`. |
| `--markers` | `geocrop markers` or `geocrop event-markers` table |
| `--config`, `--marker-rules`, `--temporal-rules` | `cbm.rules` and `cbm.temporal_support`. They change the outcome only when `enabled`. Temporal-support checks are always listed for information when support is available. |
| `--model` | model directory or `model.joblib`. Its sha256 is recorded and compared with the table summary. |
| `--run-manifest` | `run_manifest.json` from `geocrop manifest` or the pipeline: path, sha256, schema version, config fingerprint and git commit |
| `--scenes` | scene root. Acquisition dates are resolved to scene ids (`source` product name or folder). |

## Output

```
evidence/
  evidence_manifest.json        inputs (sha256), rule settings, package list with evidence sha256
  index.csv, index.gpkg         one row per parcel (geometry from the parcel layer)
  <parcel>/evidence.json        the record (below)
  <parcel>/evidence.html        human-readable page
  <parcel>/timeseries.png       observed points vs interpolated curve, phenology/harvest dates, outcome
```

Folder names are the parcel ids with unsafe characters replaced. When two ids collide after
sanitising, a short hash is appended.

### `evidence.json` (contract `evidence`, `schema_version` 1)

| Key | Content |
|---|---|
| `schema`, `schema_version`, `generator` | `geocrop-parcel-evidence`, contract version, engine version |
| `parcel_id`, `declared_class` | |
| `classification` | predicted class, decision and decision reasons, declaration status, confidence, margin, entropy, class probabilities, prediction set and size, `cp_status`, conformal alpha |
| `applicability` | `aoa_fraction` (plus coverage or dissimilarity), threshold, inside/outside |
| `markers` | phenology, harvest, bare-soil and event-marker fields that were supplied |
| `temporal_support` | the `tq_*` metrics (without prefix), tolerance, SOS/peak/EOS dates, observed dates |
| `rules` | every evaluated rule as `{rule, stage, inputs, threshold, triggered, outcome, reason_code}`. Stages: decision thresholds and AOA, declaration comparison, conformal set, crop calendar, base traffic light, each marker-rule family, each temporal-support check. |
| `rule_settings` | resolved `cbm.rules` and `cbm.temporal_support` |
| `outcome` | recomputed `traffic_light` and `traffic_light_reasons`, the stored value of the table, and `consistent_with_table` |
| `data_sources` | `observed_acquisitions` (date, index, parcel mean, valid fraction, scene ids), `rejected_acquisitions` (with reason), `interpolated_grid` (each grid date with curve value, distance to the nearest observation and `interpolated` flag), season settings |
| `model`, `run_manifest`, `inputs` | sha256 references of the model, run manifest and every input file |
| `notes` | caveats, for example when the stored and recomputed traffic lights differ |

`geocrop.evidence.load(path)` reads a package, migrates older versions and refuses newer ones
(`contracts.SchemaVersionError`). `geocrop.evidence.validate(body)` lists the problems of a body.

### HTML

The page contains no scripts, no external fonts, images or styles, and no inline `style` attributes.
It declares its own policy, `Content-Security-Policy: default-src 'none'; style-src 'sha256-…';
img-src 'none'`, so it renders unchanged when it is served under a strict CSP or opened from disk.
The chart is inline SVG (presentation attributes only). The PNG is a separate file for reports.

### Determinism

The packages carry no timestamps; the referenced run manifest holds the run time. The same inputs
give byte-identical JSON, HTML, PNG and index files. Floats are rounded to six decimals and the PNG
is written without software metadata. A test enforces this.

## Synthetic check (software verification, not accuracy)

The run below packaged all 144 parcels of the synthetic demo after the
[temporal-quality](temporal-quality.md) example. It took 13 s and peaked at about 256 MB RSS on a
laptop, with 13 MB of output (9 yellow, 11 red, 124 green):

```bash
geocrop evidence --parcels routed.gpkg --out ev --signatures demo/signatures.csv \
  --start 2023-03-15 --end 2023-10-31 --markers markers.csv --temporal-rules '{"enabled": true}' \
  --model pc/model --summary pc/parcels.summary.json
```

## Limitations

* Decision rules are reconstructed from the table columns and the summary thresholds. The traffic
  light is recomputed with the supplied rule settings. When the stored table was produced with other
  settings, the package says so (`consistent_with_table: false` and a note) instead of hiding the
  difference.
* Image chips per date are not part of the package (that is the analyst-review interface, gap G7).
* Packages reference inputs by path and hash. They do not copy the parcel layer or imagery. Treat
  them as personal data where parcel ids or declarations identify farmers.
