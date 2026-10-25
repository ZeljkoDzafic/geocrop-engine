"""QGIS Processing algorithms: remote jobs, streamed logs and scheduled result layers."""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsAuthMethodConfig,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingOutputMultipleLayers,
    QgsProcessingOutputString,
    QgsProcessingParameterAuthConfig,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsRasterLayer,
    QgsVectorLayer,
)

from .transport import ApiClient, ApiError, Cancelled

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
LAYER_SUFFIXES = {".tif", ".tiff", ".gpkg", ".geojson", ".parquet"}


def access_token(auth_config, variable):
    if auth_config:
        manager = QgsApplication.authManager()
        if not manager.masterPasswordIsSet():
            raise QgsProcessingException("Unlock QGIS Authentication Manager before using the selected configuration")
        config = QgsAuthMethodConfig()
        if not manager.loadAuthenticationConfig(auth_config, config, True):
            raise QgsProcessingException("Could not load the QGIS authentication configuration")
        token = config.configMap().get("password")
        if not token:
            raise QgsProcessingException("The authentication configuration needs the engine token in its password field")
        return token
    token = os.environ.get(variable)
    if not token:
        raise QgsProcessingException(f"Set {variable} before starting QGIS, or choose an unlocked authentication config")
    return token


def selected_layers(artifacts, scope):
    candidates = [a for a in artifacts if Path(a["path"]).suffix.lower() in LAYER_SUFFIXES]
    if scope == 0:
        preferred = [a for a in candidates if {"prediction", "parcels"} & set(Path(a["path"]).parts)]
        return preferred or candidates
    return candidates


class EngineAlgorithm(QgsProcessingAlgorithm):
    mode = "command"

    def name(self):
        return {"command": "runcommand", "demo": "demo", "existing": "loadjob"}[self.mode]

    def displayName(self):
        return {"command": "Run engine command", "demo": "Run synthetic demo", "existing": "Monitor / load existing job"}[self.mode]

    def group(self):
        return "Engine jobs"

    def groupId(self):
        return "enginejobs"

    def createInstance(self):
        return type(self)()

    def shortHelpString(self):
        return ("Calls a separately running GeoCrop HTTP server. Command paths belong to the server workspace. "
                "Use an environment variable or unlocked QGIS auth config (password field) for the token; "
                "never put the token in command parameters. Successful GIS outputs are downloaded into a new "
                "local subdirectory and scheduled for loading. No engine packages are needed in QGIS Python. "
                "Synthetic demo metrics verify software only, not crop accuracy. A monitoring timeout leaves "
                "the remote job running; save JOB_ID to reconnect. Result listing is limited by the server to 200 files.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterString("SERVER_URL", "Engine URL (local or forwarded port)",
                                                       defaultValue=os.environ.get("GEOCROP_WEB_URL", "http://127.0.0.1:8765")))
        self.addParameter(QgsProcessingParameterString("TOKEN_ENV", "Token environment variable name",
                                                       defaultValue="GEOCROP_WEB_TOKEN"))
        self.addParameter(QgsProcessingParameterAuthConfig("AUTH_CONFIG", "QGIS authentication config (optional)", optional=True))
        if self.mode == "command":
            self.addParameter(QgsProcessingParameterString("COMMAND", "GeoCrop command (not a shell)",
                                                           defaultValue="geocrop indices --format json", multiLine=True))
        elif self.mode == "demo":
            self.addParameter(QgsProcessingParameterString("ROOT", "New demo output path on server (blank = unique)", optional=True))
            self.addParameter(QgsProcessingParameterBoolean("QUICK", "Quick synthetic software check", defaultValue=True))
        else:
            self.addParameter(QgsProcessingParameterString("JOB_ID", "Existing engine job ID"))
        self.addParameter(QgsProcessingParameterBoolean("WAIT", "Wait for completion", defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber("TIMEOUT", "Monitoring timeout (seconds)",
                                                      type=QgsProcessingParameterNumber.Integer, defaultValue=600, minValue=1))
        self.addParameter(QgsProcessingParameterBoolean("CANCEL_REMOTE", "Cancel remote job when QGIS task is cancelled",
                                                        defaultValue=self.mode != "existing"))
        self.addParameter(QgsProcessingParameterBoolean("LOAD_LAYERS", "Download and load successful GIS outputs", defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum("LAYER_SCOPE", "Output layers", options=[
            "Prefer prediction and parcel outputs", "All supported GIS outputs"], defaultValue=0))
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_DIR", "Local results parent (new subdirectory per job)", optional=True))
        self.addOutput(QgsProcessingOutputString("JOB_ID", "Engine job ID"))
        self.addOutput(QgsProcessingOutputString("STATUS", "Engine job status"))
        self.addOutput(QgsProcessingOutputString("RESULT_DIR", "New local results directory"))
        self.addOutput(QgsProcessingOutputMultipleLayers("LAYERS", "GeoCrop result layers"))

    def processAlgorithm(self, parameters, context, feedback):
        token = access_token(self.parameterAsString(parameters, "AUTH_CONFIG", context),
                             self.parameterAsString(parameters, "TOKEN_ENV", context))
        job_id = None
        paths, directory = [], ""
        try:
            client = ApiClient(self.parameterAsString(parameters, "SERVER_URL", context), token)
            client.health()
            if self.mode == "existing":
                job_id = self.parameterAsString(parameters, "JOB_ID", context)
                job = client.job(job_id)
            elif self.mode == "demo":
                root = self.parameterAsString(parameters, "ROOT", context) or f"outputs/qgis-demo-{uuid.uuid4().hex[:12]}"
                argv = ["demo", "--root", root]
                if self.parameterAsBool(parameters, "QUICK", context):
                    argv.append("--quick")
                job = client.submit(argv=argv)
            else:
                job = client.submit(command=self.parameterAsString(parameters, "COMMAND", context))
            job_id = job["id"]
            feedback.pushInfo(f"GeoCrop JOB_ID: {job_id}")
            deadline = time.monotonic() + self.parameterAsInt(parameters, "TIMEOUT", context)
            offset = 0
            while True:
                if feedback.isCanceled():
                    if self.parameterAsBool(parameters, "CANCEL_REMOTE", context):
                        job = client.cancel(job_id)
                    else:
                        job = client.job(job_id)
                    feedback.pushInfo(f"Monitoring stopped; remote job is {job['status']}. JOB_ID: {job_id}")
                    return {"JOB_ID": job_id, "STATUS": job["status"], "RESULT_DIR": "", "LAYERS": []}
                log = client.logs(job_id, offset)
                offset = log["next_offset"]
                if log["text"]:
                    feedback.pushInfo(log["text"])
                feedback.setProgressText(f"GeoCrop: {job['status']}")
                if job["status"] in TERMINAL:
                    # Drain a large final log in chunks before reporting success/failure.
                    while log["text"]:
                        log = client.logs(job_id, offset)
                        offset = log["next_offset"]
                        if log["text"]:
                            feedback.pushInfo(log["text"])
                    break
                if not self.parameterAsBool(parameters, "WAIT", context):
                    return {"JOB_ID": job_id, "STATUS": job["status"], "RESULT_DIR": "", "LAYERS": []}
                if time.monotonic() >= deadline:
                    raise QgsProcessingException(f"Monitoring timed out; remote job is still {job['status']}. Reconnect using JOB_ID: {job_id}")
                time.sleep(.25)
                job = client.job(job_id)
            if job["status"] != "succeeded":
                raise QgsProcessingException(f"GeoCrop job {job_id} ended as {job['status']} (exit {job.get('returncode')}); inspect the log")
            if self.parameterAsBool(parameters, "LOAD_LAYERS", context):
                artifacts = selected_layers(client.artifacts(job_id), self.parameterAsEnum(parameters, "LAYER_SCOPE", context))
                if artifacts:
                    parent = self.parameterAsString(parameters, "OUTPUT_DIR", context)
                    if not parent or parent == QgsProcessing.TEMPORARY_OUTPUT:
                        parent = tempfile.mkdtemp(prefix="geocrop-qgis-")
                    Path(parent).mkdir(parents=True, exist_ok=True)
                    directory = tempfile.mkdtemp(prefix=f"job-{job_id[:8]}-", dir=parent)
                    for i, artifact in enumerate(artifacts):
                        path = client.download(job_id, artifact["path"], directory, feedback.isCanceled)
                        layer = (QgsRasterLayer(str(path), path.stem) if path.suffix.lower() in {".tif", ".tiff"}
                                 else QgsVectorLayer(str(path), path.stem, "ogr"))
                        if not layer.isValid():
                            feedback.reportError(f"Downloaded but could not load GIS layer: {path}", fatalError=False)
                            continue
                        paths.append(str(path))
                        context.addLayerToLoadOnCompletion(str(path), QgsProcessingContext.LayerDetails(
                            f"GeoCrop · {path.stem}", context.project(), "LAYERS"))
                        feedback.setProgress(100 * (i + 1) / len(artifacts))
            feedback.setProgress(100)
            return {"JOB_ID": job_id, "STATUS": job["status"], "RESULT_DIR": directory, "LAYERS": paths}
        except Cancelled:
            feedback.pushInfo(f"Download cancelled; completed files are retained. Engine job: {job_id}")
            return {"JOB_ID": job_id or "", "STATUS": "download_cancelled", "RESULT_DIR": directory, "LAYERS": paths}
        except (ApiError, OSError, ValueError) as exc:
            raise QgsProcessingException(str(exc)) from exc


class RunCommandAlgorithm(EngineAlgorithm):
    mode = "command"


class DemoAlgorithm(EngineAlgorithm):
    mode = "demo"


class LoadJobAlgorithm(EngineAlgorithm):
    mode = "existing"
