# HTML workspace, HTTP jobs and Jupyter

The web extra adds a local interface around the existing engine CLI. All existing CLI commands and
Python functions remain available. No frontend build service, Node runtime, external fonts or CDN is
needed. HTML/CSS/JavaScript are shipped inside the Python package.

![GeoCrop workspace after a synthetic software-check run](images/web-workspace.png)

The screenshot shows synthetic verification data, not an observational crop-accuracy result.

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

The UI provides:

* forms generated from the actual CLI parser, including required fields and optional settings;
* an engine command console with quoted-argument support;
* persistent job history, queued/running/succeeded/failed/cancelled/interrupted states and exit codes;
* incremental stdout/stderr logs, cancellation and artifact downloads;
* Python/Jupyter examples and an authenticated OpenAPI schema download.

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
