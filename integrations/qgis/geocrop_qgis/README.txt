GeoCrop Engine for QGIS
======================

1. Install the separate engine with its web extra: pip install -e '.[web]'.
2. Set a random GEOCROP_WEB_TOKEN (at least 32 characters) securely and start:
   geocrop serve --workspace /path/to/study --port 8765
3. In QGIS, install this ZIP from Plugins > Manage and Install Plugins > Install from ZIP.
4. Enable GeoCrop Engine and the built-in Processing plugin. Open Processing Toolbox > GeoCrop Engine.
5. Supply the token through the QGIS process environment or an unlocked QGIS Basic authentication
   configuration, storing the token in its password field. Never put the token in a command/model.

Algorithms: geocrop:runcommand, geocrop:demo, geocrop:loadjob.
Commands reference paths on the engine server, not automatically exported QGIS layers.
Downloads use a new local subdirectory and preserve existing files. Results are scheduled for loading
through the Processing context. A monitoring timeout leaves the remote job running; reconnect with JOB_ID.
Remote engines require a locally forwarded port. The plugin requires no engine/ML pip packages in QGIS Python.
Synthetic demo results are software checks, not real crop-accuracy measurements.

Full documentation: docs/qgis.md in the GeoCrop Engine repository.
Source, issues and license notices: https://github.com/ZeljkoDzafic/geocrop-engine
