# HTML workspace, HTTP jobs and Jupyter

The web extra adds a local interface around the existing engine CLI. All existing CLI commands and
Python functions remain available. No frontend build service, Node runtime, external fonts or CDN is
needed. HTML/CSS/JavaScript are shipped inside the Python package.

![GeoCrop workspace with a Sentinel-2 quicklook of Semberija on the map](images/web-workspace.png)

The screenshot shows a monthly true-colour composite from `geocrop quicklook` (contains modified Copernicus
Sentinel data 2023); it is imagery for orientation, not a crop-accuracy result.

## Start the workspace

```bash
pip install -e '.[web]'
# Set a random token of at least 32 characters in your terminal/environment.
# Keep its value out of source files, notebook outputs and Git.
export GEOCROP_WEB_TOKEN=...  # enter your token locally
geocrop serve --workspace /path/to/your/study --port 8765
```

Open the local server on port 8765 in your browser and enter the same token. The command binds to
loopback only. A notebook on another machine can use an SSH-forwarded local port. This is a single
trusted owner's research tool running as that owner's account, not a public/multi-tenant service or
a security sandbox for untrusted model files. Inputs, configs and joblib models must be trusted.

## First steps: from an area and dates to imagery

The workspace opens on the **Map**. With no job selected it offers three ways in:

1. **New analysis** (or press <kbd>N</kbd>) opens a three-step sheet:
   *Area* — type a bounding box in degrees (WGS 84), use the Semberija example, or choose **Draw on map** and drag a
   rectangle (or click/tap two corners; <kbd>Esc</kbd> cancels); *Period* — start and end dates or a preset such as
   the latest growing season; *Run* — review the area size, the period and a fresh output folder. The sheet shows the
   exact `geocrop quicklook ...` command it submits, so the same run can be repeated in a shell or notebook.
2. **Try Semberija, Jun–Aug 2023** submits that quicklook with one click.
3. **Run the synthetic demo** is a software check that produces crop classes, uncertainty and parcels.

While a job runs, the map shows its progress and the output streams below the map. When it finishes, a
notification appears and the imagery is drawn on the map. The *imagery bar* at the bottom of the map switches between
monthly composites and single scenes, between true colour, false colour and NDVI, and steps through the dates
(<kbd>[</kbd> and <kbd>]</kbd>). The side panel holds the imagery switch and opacity, a switch per result layer and
the legend. The expand button (or <kbd>F</kbd>) shows the map full-window.

## Views

* **Map** — imagery and results of the selected job, its output log and its files (with type and size; click to
  download).
* **Commands** — every engine command as a form generated from the actual CLI parser, grouped by purpose and
  filterable. Required inputs come first; optional settings sit under *Advanced options*. The live *Command line*
  preview omits unchanged defaults, so it is exactly what one would type.
* **Jobs** — the persistent job history with status, relative start time, duration and exit code;
  queued/running/succeeded/failed/cancelled/interrupted states, incremental logs and cancellation.
* **Terminal** — a CLI console for typing engine commands with quoted-argument support.
* **Python & API** — Jupyter/Python and HTTP examples and an authenticated OpenAPI schema download.

The sidebar lists recent jobs; on a phone the views move to a bottom tab bar. Light and dark appearance follow the
system; the moon/sun button overrides it (remembered in this browser only).

### Keyboard shortcuts

| Keys | Action |
|---|---|
| <kbd>⌘K</kbd> / <kbd>Ctrl K</kbd> | command palette: search commands and actions |
| <kbd>N</kbd> | new analysis |
| <kbd>1</kbd>–<kbd>5</kbd> | Map, Commands, Jobs, Terminal, Python & API |
| <kbd>F</kbd> | expand or restore the map |
| <kbd>[</kbd> / <kbd>]</kbd> | previous / next imagery date |
| <kbd>J</kbd> / <kbd>K</kbd> | next / previous job |
| <kbd>?</kbd> | shortcut overview |
| <kbd>Esc</kbd> | close a sheet, cancel area drawing or leave the expanded map |

Single-key shortcuts are ignored while typing in a field or the terminal.

The console invokes `python -m geocrop.cli` with an argument list, never a shell. `serve`, direct SNAP
executable invocation and `manifest --snap-gpt` are unavailable in the web console; use the ordinary
CLI when those external programs are needed. Shell operators, arbitrary Python, environment assignments
and `cd` are not console commands. All jobs use the chosen workspace as their working directory.

One worker executes jobs sequentially. Job state/logs live in `<workspace>/.geocrop-web` by default;
`--state-dir` changes that location. On POSIX an advisory lock prevents two servers sharing it.
Queued/running records found after restart are marked interrupted rather than replayed. Normal shutdown
stops the active process group; after an abrupt machine/process failure, inspect any surviving child
processes before manually rerunning an interrupted job. Cancellation can leave partial engine outputs;
they are retained for inspection and the engine's own pipeline resume rules still apply.

Job outputs must stay beneath the workspace and outside the web state directory. A config-driven `run`
also checks `paths.output`, `paths.features` and `paths.composites`. Inputs may reference accessible local
data. Choose fresh output paths for new experiments; the underlying engine retains its documented
overwrite/resume behavior when existing output paths are explicitly chosen. No project files or existing
commands are removed by installing the interface.

`GEOCROP_WEB_THREADS` sets OpenMP/OpenBLAS threads per job (default 2); the model's `n_jobs` setting remains
independent. The server access token is removed from child-job environments. The UI holds it only in
page memory, not URLs or browser local storage. Every API request requires Bearer authentication;
cross-origin browser requests are rejected. Artifact downloads are attachments, and listings are
limited to the first 200 files within a job's declared output locations. Use local filesystem tools
for larger output inventories.

## Jupyter Notebook / JupyterLab / Python

The client uses `requests`, already a core dependency; only the server needs the web extra. Install
the engine in the notebook kernel's Python environment and set `GEOCROP_WEB_TOKEN` there securely.

```python
from geocrop.client import EngineClient

with EngineClient() as engine:  # default endpoint: local port 8765; token from environment
    job = engine.submit(["indices", "--format", "json"])
    done = engine.wait(job["id"], timeout=60)
    print(engine.logs(job["id"])["text"])
```

For a full software check, submit `['demo', '--root', 'outputs/my-new-demo', '--quick']`. This is synthetic
verification, not a measurement of real crop accuracy. A remote/fowarded endpoint can be passed as
`EngineClient(base_url=...)`. Paths in submitted commands belong to the **server workspace**, not the
notebook machine. `engine.artifacts(job_id)` lists available files; `engine.download(job_id, path,
destination)` streams into a new local file and refuses to replace an existing destination.

`wait(..., check=True)` raises for failed/cancelled/interrupted jobs. `check=False` returns their status
and exit code for inspection; a timeout does not cancel the job. `logs(job_id, offset=...)` returns text
plus a byte `next_offset` for incremental reading. `cancel(job_id)` explicitly requests cancellation.
See the runnable [example notebook](../examples/web_notebook.ipynb); it contains no token or saved results.

## HTTP API

All endpoints below require `Authorization: Bearer <token>`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | version, workspace and worker count |
| GET | `/api/commands` | actual parser command/argument catalogue |
| POST | `/api/jobs` | submit `{"argv": ["indices", "--format", "json"]}` or `{"command": "geocrop indices"}` |
| GET | `/api/jobs` | newest jobs; `limit` defaults to 50, maximum 200 |
| GET | `/api/jobs/{id}` | status, times, args, outputs and exit code |
| GET | `/api/jobs/{id}/logs` | byte `offset` and bounded `limit` (up to 262144) |
| POST | `/api/jobs/{id}/cancel` | stop a queued/running job |
| GET | `/api/jobs/{id}/artifacts` | up to 200 local output files |
| GET | `/api/jobs/{id}/artifact?path=...` | download a listed artifact |
| GET | `/api/openapi.json` | machine-readable API description |

Submission returns HTTP 202 and a job ID, not the final engine result. HTTP 400 reports invalid commands
or output locations, 401 missing/incorrect authentication, 403 cross-origin requests and 404 unknown
jobs/artifacts. Typed request validation can return 422. A job with engine exit code 0 succeeds; any
other code is recorded as failed, including audit/readiness commands that intentionally use nonzero
exit codes to report data problems. Use the logs to distinguish those outcomes.

## Results map
Selecting a succeeded job loads its spatial outputs on the **Map** (`GET /api/jobs/{id}/map`):

| Layer | Source | Default |
|---|---|---|
| Crop classes | `classes.tif` + `class_legend.json` | on |
| Parcels — traffic light | parcel results with `traffic_light` (outline colour green/yellow/red/grey, popup with declared/predicted class, status and confidence) | on |
| Max class probability, normalised entropy | `uncertainty.tif` bands 1 and 3 | off |
| Inside area of applicability | `aoa.tif` band 2 | off |
| Other vector outputs (segments, reference layers, observations) | `.gpkg` / `.geojson` | off |
| Satellite imagery: true colour (B4-B3-B2), false colour (B8-B4-B3), NDVI | every window composite and scene folder of the job | latest cloud-free composite (true colour) when the job has no classification |

Imagery is chosen with the imagery bar (composites or scenes, product, date) and the side panel (on/off, opacity), so
only one image is drawn at a time.

### Real imagery in two minutes
```bash
geocrop quicklook --bbox 19.10 44.72 19.16 44.76 --start 2023-04-01 --end 2023-09-30 --out outputs/semberija-2023
```
reads only the area of interest from public Sentinel-2 L2A COGs (no account needed), keeps acquisitions with at
least 30 % clear pixels in the area, builds monthly cloud-screened composites and shows them on the map. The same
command is available in the web workspace.

### Basemaps
None (default), Sentinel-2 cloudless 2023 / 2024 (EOX IT Services GmbH, contains modified Copernicus Sentinel data,
CC BY-NC-SA 4.0 — non-commercial use) or OpenStreetMap. Enabling a basemap sends tile requests for the viewed area to
the provider.

Rasters are warped to WGS 84 and downsampled (≤ 1024 px) into PNG previews; vectors are simplified and capped at
20,000 features. Previews are for orientation only — download the files for analysis. Leaflet is bundled with the
package (no CDN).
