# Model and benchmark registry

`geocrop.registry` keeps trained models and benchmark results in a local, file-based registry, so that a map can
always be traced back to an exact model file, and model or benchmark comparisons only happen between results that
were scored on identical data splits. It needs no server and no database. The design takes ideas from the model
registries of WorldCereal and Fields of The World and from benchmark tables such as those of TorchGeo and FTW.
No code from those projects is used.

## Location

The default location is `./registry` in the working directory, which is the workspace for `geocrop serve` jobs. It
can be changed with `$GEOCROP_REGISTRY=/path/to/registry` or `--registry` on every `geocrop models` and
`geocrop benchmarks` action. To switch off automatic benchmark recording, set `GEOCROP_REGISTRY=off`.

```
registry/
  models/<name>/<version>/model.joblib     read-only copy (or hard link with --link)
  models/<name>/<version>/aoa.json, conformal.json, train_report.json, ...   companions, if present
  models/<name>/<version>/entry.json       immutable metadata, sealed by entry_sha256
  models/<name>/stages.jsonl               append-only log of stage changes
  benchmarks/<run id>.json                 one immutable record per benchmark run
```

Entries are written with exclusive create and set to read-only. They are never rewritten. Stage changes go into an
append-only log, so the full promotion history survives.

## Models

```bash
geocrop train --samples samples.npz --out model --block-m 500 --folds 3 --holdout 0.25
geocrop models register model/model.joblib --name semberija-crops \
    --samples samples.npz --features features/stack.tif --hcat hcat.csv --tags 2024 sentinel2 \
    --licence presto=MIT
geocrop models list                       # model versions and pinned pretrained checkpoints
geocrop models show semberija-crops@1
geocrop models compare semberija-crops@1 semberija-crops@2
geocrop models promote semberija-crops@2 --stage production   # the previous production version is archived
geocrop models verify                     # recompute every hash; exit 2 on mismatch
geocrop predict --features features/stack.tif --model registry:semberija-crops@production --out prediction
```

### What an entry records

| Field | Content |
|---|---|
| `artifact` | content SHA-256, size, source path, copy or hard link |
| `files` | SHA-256 of every companion file (AOA, conformal thresholds, train report, ...) |
| `engine` | GeoCrop version, git commit, dirty flag |
| `algorithm`, `params`, `seed`, `temperature`, `calibration_mode` | from the model payload |
| `feature_manifest` | feature names in model order, group counts, per-feature group/sensor/window, manifest schema version |
| `input_contract` | feature order, windows, CRS, grid resolution, nodata of the stack the model was trained on |
| `classes`, `taxonomy` | class list, optional taxonomy name, HCAT mapping and unmapped classes (`--hcat`) |
| `training_data` | samples, polygons, spatial blocks, counts per class, years/regions (from the samples) |
| `validation` | spatial CV (with bootstrap CIs), nested CV, locked holdout, transfer (leave-one-region/year-out), conformal coverage check |
| `fingerprints` | CV, holdout and transfer split fingerprints, recomputed and verified at registration |
| `licences` | licences of pretrained components: every embedding provider found in the feature manifest (AlphaEarth CC-BY-4.0, TESSERA, Presto, Galileo and AnySat MIT, plugins), with the licence, code licence and pinned weights it declares as a feature provider, plus `--licence` entries |
| `provenance`, `created_utc`, `tags`, `description` | |

### Split fingerprints and refused comparisons

When it registers a model, the registry rebuilds the training design from the samples: the same locked holdout,
calibration split and spatial folds as `geocrop train`. It then hashes the exact train and test indices of every
fold. It accepts these fingerprints only if the rebuilt fold sizes and sample counts match `train_report.json`.
Otherwise the entry records `verified: false` and the reason. Without `--samples`, the registry looks for a
`samples.npz` in the model folder or one level above it.

`geocrop models compare A B` prints metric deltas (B − A) only for validation sections (CV, nested CV, holdout,
transfer) whose fingerprints are identical in both models. It lists every other section with the reason it was
not compared. When no section is comparable, the command refuses with exit code 2, for example:

```
geocrop models: Refusing to compare demo-default@1 and demo-crops@1: no validation section was scored on identical
splits ({'cv': 'different cv split fingerprints (969bfef89bd7 vs f97c71cbea04)', 'holdout': 'present in only one model'})
```

### Stages and model references

A new version starts in stage `none`. `promote --stage staging|production|archived` appends to `stages.jsonl`.
Promoting a version to production archives the previous production version. A version that fails hash
verification cannot be promoted, except to `archived`.

References have the form `name@<version>`, `name@<stage>`, `name@latest`, or `name` alone, which means the
production version. `geocrop predict --model registry:<ref>` does the following before any pixel is read:

1. It resolves the reference.
2. It verifies the artefact and entry hashes.
3. It checks the feature stack against the input contract: every model feature present, same CRS, same grid
   resolution, and the same group, sensor and window for each feature listed in the stack's feature manifest.

A mismatch stops the run with a clear message, for example
`Feature stack coarse.tif does not satisfy the input contract of crops@1: grid resolution [20.0, 20.0] differs from
the contract [10.0, 10.0]`. The resolved reference is added to `prediction_manifest.json` under `registry`.

### Pretrained checkpoints

The pinned external checkpoints are listed as entries of kind `pretrained`: every Fields of The World model
(`ftw/<name>`, from `geocrop.ftw.MODELS`), the Presto encoder (`presto/default`, from
`geocrop.embedding_sources.PRESTO_CHECKPOINT`) and the weights of every other registered feature provider
(`anysat/default`, `galileo/<file>` for its two files, plugins likewise), taken from the `weights` and `licence`
the provider declares. Their URL, size, SHA-256 and licence are the existing pins. The
registry never downloads or stores them. `geocrop models verify` checks a cached copy against its pin, and an
uncached checkpoint is reported as `not cached`. Non-commercial FTW checkpoints keep their licence label
(`Mixed Open Licenses`, `commercial_use: false`).

## Benchmarks

The following commands each write one immutable record per run into `registry/benchmarks/`:

| Command | Task | Split fingerprint covers |
|---|---|---|
| `geocrop benchmark` | `crop-classification` | sample support, fold indices per seed, locked holdout |
| `geocrop benchmark-models` | `crop-classification` | same, for every repeat seed |
| `geocrop ablation` | `crop-classification` | same (plus transfer groups with `--cv-strategy`) |
| `geocrop transfer` | `crop-transfer/<strategy>` | sample support and region/year group assignment (one record per strategy) |
| `geocrop benchmark-public` | `public/<protocol id>` | protocol id and version, parcel roles, sample fingerprint, split and evaluation settings |
| `geocrop benchmark-segmentation` | `field-segmentation` | SHA-256 of raster and reference parcels, boundary tolerance, IoU threshold |

A record contains the kind, task, protocol settings, split and dataset fingerprints, one row per candidate with
its metrics and confidence intervals where the command computes them, the runtime, the engine version and git
commit, and the output folder. The dataset fingerprint covers labels, polygon ids, pixel centres and CRS, not the
feature values. As a result, feature-set ablations and model benchmarks on the same samples and folds are ranked
together, while runs on other folds are not. The registry hooks never fail a benchmark: if recording fails, you
get a warning.

```bash
geocrop benchmarks list --task crop-classification
geocrop benchmarks show 20261026T023109-benchmark-58bd3b9828
geocrop benchmarks leaderboard --task crop-classification              # Markdown, one table per split
geocrop benchmarks leaderboard --task crop-classification --format csv --out leaderboard.csv
geocrop benchmarks leaderboard --task field-segmentation --metric boundary_f1 --split 3fa2
```

A leaderboard only ranks runs with identical split fingerprints. Each fingerprint gets its own table, or its own
`split_fingerprint` value in the CSV. `--split` selects one of them. Metrics where lower is better, such as ECE,
runtimes, the transfer gap or over-segmentation, are ranked in ascending order.

### Example (synthetic data)

These are the commands from the synthetic demo that produced the comparison and the refusal shown above:

```bash
export GEOCROP_REGISTRY=$PWD/registry
geocrop demo --root demo --quick                       # also records its ablation run
geocrop train --samples demo/features/samples.npz --out m_rf --block-m 500 --folds 3 --holdout 0.25 \
    --params '{"n_estimators": 100, "n_jobs": 1}'
geocrop train --samples demo/features/samples.npz --out m_et --block-m 500 --folds 3 --holdout 0.25 \
    --algorithm extra_trees --params '{"n_estimators": 100, "n_jobs": 1}'
geocrop models register m_rf/model.joblib --name demo-crops --samples demo/features/samples.npz \
    --features demo/features/stack.tif
geocrop models register m_et/model.joblib --name demo-crops --samples demo/features/samples.npz \
    --features demo/features/stack.tif
geocrop models compare demo-crops@1 demo-crops@2
geocrop benchmark --samples demo/features/samples.npz --out bench --algorithms rf extra_trees hist_gb \
    --block-m 500 --folds 3 --params '{"n_estimators": 100, "n_jobs": 1}'
geocrop benchmark-models --samples demo/features/samples.npz --out bench_models --models rf extra_trees \
    --block-m 500 --folds 3 --repeats 1
geocrop benchmarks leaderboard --task crop-classification
```

The `benchmark` run and the `benchmark-models` run share one split fingerprint, so they appear in one table. The
demo's ablation used other folds, so it gets a separate table. All numbers come from **synthetic** data. They
check that the software works and say nothing about accuracy.

## Web and programmatic access

`geocrop.registry.overview(root=None)` returns a read-only, JSON-serialisable snapshot of the registry: models
with their stages, pretrained pins, benchmark run summaries and tasks. It writes nothing and hashes nothing, so
the web workspace can call it directly. All other operations are plain functions: `register`, `get`,
`list_models`, `promote`, `verify`, `compare`, `resolve_for_predict`, `list_benchmarks`, `leaderboard`,
`leaderboard_markdown` and `leaderboard_csv`.

## Limitations

- The registry is a local directory and has no locking. Concurrent `promote` calls from several processes append
  to the same log, and the last line wins.
- Split fingerprints cover sampled pixel support and fold indices, not the full polygon geometries. A model
  registered without its samples cannot be compared.
- Read-only file permissions protect against accidental edits, not against a user who deliberately changes them.
  `verify` detects any such change.
- `--link` hard links share content with the original file. If the original is edited later, `verify` reports the
  mismatch.
