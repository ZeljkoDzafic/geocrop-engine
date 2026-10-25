# QGIS integration

The standalone GeoCrop plugin adds three **Processing Toolbox → GeoCrop Engine** algorithms. It calls
the same authenticated HTTP API as the HTML workspace and Jupyter client. The engine runs in its own
Python environment; QGIS does not need NumPy/GDAL/ML versions matching the engine or any pip-installed
engine packages. The plugin transport uses only the Python standard library and PyQGIS.

## Install

1. Start the engine with the [web extra](web-interface.md): set a random `GEOCROP_WEB_TOKEN` securely,
   then run `geocrop serve --workspace /path/to/study --port 8765`.
2. From the engine checkout build a new plugin ZIP:

   ```bash
   python tools/build_qgis_plugin.py --out /path/to/new/geocrop-qgis.zip
   ```

3. In QGIS open **Plugins → Manage and Install Plugins → Install from ZIP**, choose that archive,
   then enable **GeoCrop Engine**. Enable QGIS's built-in **Processing** plugin if its toolbox is hidden.
4. Open **Processing Toolbox → GeoCrop Engine → Engine jobs**, or **Plugins → GeoCrop → Run GeoCrop command**.

The builder preserves an existing ZIP instead of replacing it. The plugin targets QGIS 3.28–3.x with Python 3.9+ and
is tested with actual QGIS 3.40.6. It does not modify existing layers or project data. New downloads
are placed in a fresh local job subdirectory; previously downloaded/user files are not replaced.

## Authentication

Choose one of these methods; the token value is never a Processing parameter:

* Start QGIS from an environment containing `GEOCROP_WEB_TOKEN` (or use the **Token environment variable
  name** parameter to select another variable). The name may appear in Processing history; the value does not.
* Create a **Basic** configuration in QGIS Authentication Manager. Use any username, e.g. `geocrop`, and
  put the engine token in its password field. Unlock the manager before starting the algorithm, then
  choose that configuration in **QGIS authentication config**. Its encrypted configuration ID is stored
  in Processing history; the plugin sends the password as a **Bearer** token, not HTTP Basic auth.

Do not paste the token into a command, saved model, URL or project. The default URL uses local port 8765.
For an engine on another machine use a local forwarded port; the plugin accepts loopback endpoints only
and refuses credential-bearing URLs and HTTP redirects. Server paths and QGIS download paths are distinct:
command arguments always refer to the **server workspace**, even when QGIS runs on a different computer.
There is no implicit upload of an active or in-memory QGIS layer. Export/transfer input layers into the
server workspace first, and reference those paths in the engine command/configuration.

## Algorithms

| Algorithm ID | Purpose |
|---|---|
| `geocrop:runcommand` | Run an existing engine command, including `run`, `audit`, `train`, `predict`, `parcels` or `benchmark` |
| `geocrop:demo` | Run the full synthetic software check; blank root chooses a fresh server output directory |
| `geocrop:loadjob` | Reconnect to a job ID, monitor it and/or retrieve completed results |

**Wait for completion** defaults to true. If disabled, the algorithm returns `JOB_ID`/`STATUS` promptly;
use `loadjob` to reconnect. Logs appear in QGIS Processing feedback. A monitoring timeout leaves the
engine job running and reports its ID. Cancelling a newly submitted task requests remote cancellation
by default; stopping `loadjob` monitoring does not cancel the remote job unless **Cancel remote job**
is enabled. An engine nonzero exit code raises a Processing error with the job ID and status, rather
than loading partial products as successful results.

Successful `.tif`, `.tiff`, `.gpkg`, `.geojson` and `.parquet` outputs can be downloaded and scheduled for
loading into the current QGIS project. Unsupported local GDAL/OGR formats produce a warning and retain
the downloaded file. By default, prediction/parcel products are preferred when present; otherwise all
supported GIS outputs are selected. Choose **All supported GIS outputs** to include composites and other
generated layers. Turn off **Download and load** for metadata-only commands or very large products.
The engine artifact API lists at most 200 files; larger inventories need local filesystem access.

Outputs are `JOB_ID`, `STATUS`, `RESULT_DIR` and `LAYERS`. Download cancellation reports
`STATUS=download_cancelled` and keeps completed files. Downloaded copies can be larger than the QGIS
project itself; choose an appropriate results parent directory. Layer registration is scheduled through
the Processing context, so background algorithms do not modify the project from a worker thread.

## QGIS Python console and Model Designer

After enabling the plugin (token supplied through environment/auth manager):

```python
import processing

result = processing.run("geocrop:runcommand", {
    "SERVER_URL": "http://127.0.0.1:8765",
    "TOKEN_ENV": "GEOCROP_WEB_TOKEN",
    "COMMAND": "geocrop indices --format json",
    "LOAD_LAYERS": False,
})
print(result["JOB_ID"], result["STATUS"])
```

In scripts, `processing.run` returns paths; load a returned file using `iface.addRasterLayer` /
`iface.addVectorLayer`, or use `processing.runAndLoadResults` / the normal Processing dialog to load
scheduled layers. The algorithms are also available in Model Designer. Model files contain command
paths and auth configuration IDs/environment names, not raw tokens. For non-QGIS notebooks use
`geocrop.client.EngineClient` instead. Synthetic demo results establish software behavior only.
