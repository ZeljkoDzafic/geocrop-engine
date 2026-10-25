"""Executed by a real QGIS Python interpreter from test_qgis.py, not imported by pytest."""
import json
import os
import shlex
import sys
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsAuthMethodConfig,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtWidgets import QMainWindow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/qgis"))


def main():
    base = Path(os.environ["GEOCROP_QGIS_TEST_RESULTS"])
    profile = base / "profile"
    profile.mkdir()
    app = QgsApplication([], False, str(profile))
    app.initQgis()
    sys.path.insert(0, str(Path(QgsApplication.pkgDataPath()) / "python/plugins"))
    import processing
    from geocrop_qgis import classFactory
    from geocrop_qgis.algorithms import RunCommandAlgorithm
    from geocrop_qgis.transport import ApiClient
    from processing.core.Processing import Processing

    Processing.initialize()
    server = Path(os.environ["GEOCROP_QGIS_TEST_WORKSPACE"])
    url = os.environ["GEOCROP_QGIS_TEST_URL"]

    class Interface:
        def __init__(self):
            self.window, self.actions = QMainWindow(), []

        def mainWindow(self):
            return self.window

        def addPluginToMenu(self, menu, action):
            self.actions.append(action)

        def removePluginMenu(self, menu, action):
            self.actions.remove(action)

    class Feedback(QgsProcessingFeedback):
        def __init__(self):
            super().__init__()
            self.messages = []

        def pushInfo(self, message):
            self.messages.append(message)

    iface = Interface()
    plugin = classFactory(iface)
    plugin.initGui()
    result = {}
    try:
        context = QgsProcessingContext()
        context.setProject(QgsProject.instance())
        feedback = Feedback()
        common = {"SERVER_URL": url, "TOKEN_ENV": "GEOCROP_WEB_TOKEN", "WAIT": True,
                  "TIMEOUT": 30, "LOAD_LAYERS": True, "OUTPUT_DIR": str(base / "downloads")}
        grid = processing.run("geocrop:runcommand", {**common, "COMMAND":
            f"geocrop grid --aoi {shlex.quote(str(server / 'input.gpkg'))} --crs EPSG:32634 --out outputs/grid.tif"},
            context=context, feedback=feedback)
        assert grid["STATUS"] == "succeeded" and len(grid["LAYERS"]) == 1, grid
        assert QgsRasterLayer(grid["LAYERS"][0], "grid").isValid()
        assert grid["LAYERS"][0] in context.layersToLoadOnCompletion()
        vector = processing.run("geocrop:runcommand", {**common, "COMMAND":
            f"geocrop normalize-parcels --input {shlex.quote(str(server / 'input.gpkg'))} "
            "--id-field code --declared-field crop --out outputs/parcels.gpkg"}, context=context, feedback=feedback)
        assert vector["STATUS"] == "succeeded" and len(vector["LAYERS"]) == 1, vector
        assert QgsVectorLayer(vector["LAYERS"][0], "parcels", "ogr").isValid()
        old = Path(vector["LAYERS"][0]).read_bytes()
        reloaded = processing.run("geocrop:loadjob", {**common, "JOB_ID": vector["JOB_ID"]}, context=context, feedback=feedback)
        assert reloaded["LAYERS"][0] != vector["LAYERS"][0]
        assert Path(vector["LAYERS"][0]).read_bytes() == old
        assert not QgsProject.instance().mapLayers(), "Background algorithm must schedule, not directly add layers"
        result.update(raster=True, vector=True, reconnect=True, preserves_existing_downloads=True,
                      processing_context_layers=len(context.layersToLoadOnCompletion()), plugin_menu=True)
        auto_context = QgsProcessingContext()
        auto_context.setProject(QgsProject.instance())
        processing.runAndLoadResults("geocrop:loadjob", {**common, "JOB_ID": vector["JOB_ID"]},
                                     context=auto_context, feedback=feedback)
        assert any(layer.isValid() for layer in QgsProject.instance().mapLayers().values())
        result["automatic_project_loading"] = True

        manager = QgsApplication.authManager()
        assert manager.setMasterPassword("temporary-qgis-test-password", True)
        auth = QgsAuthMethodConfig()
        auth.setName("GeoCrop API test")
        auth.setMethod("Basic")
        auth.setConfig("username", "geocrop")
        auth.setConfig("password", os.environ["GEOCROP_WEB_TOKEN"])
        assert manager.storeAuthenticationConfig(auth)
        authenticated = processing.run("geocrop:runcommand", {**common, "TOKEN_ENV": "UNSET_QGIS_TEST_TOKEN",
            "AUTH_CONFIG": auth.id(), "COMMAND": "geocrop indices --format json", "LOAD_LAYERS": False},
            context=context, feedback=feedback)
        assert authenticated["STATUS"] == "succeeded"
        result["encrypted_auth_config"] = True

        demo = processing.run("geocrop:demo", {**common, "WAIT": False, "LOAD_LAYERS": False},
                              context=context, feedback=feedback)
        api = ApiClient(url, os.environ["GEOCROP_WEB_TOKEN"])
        demo_job = api.job(demo["JOB_ID"])
        assert demo_job["argv"][0] == "demo" and "--quick" in demo_job["argv"]
        assert demo_job["argv"][2].startswith("outputs/qgis-demo-")
        api.cancel(demo["JOB_ID"])
        result["demo_submission"] = True

        try:
            processing.run("geocrop:runcommand", {**common, "COMMAND": "geocrop config-check --config missing.yaml"},
                           context=context, feedback=feedback)
            raise AssertionError("Failed engine jobs must not report successful Processing results")
        except QgsProcessingException:
            result["failed_exit_propagates"] = True

        cancel_feedback = Feedback()
        cancel_feedback.cancel()
        algorithm = RunCommandAlgorithm()
        algorithm.initAlgorithm()
        cancelled = algorithm.processAlgorithm({**common, "COMMAND":
            "geocrop demo --root outputs/qgis-cancel --quick", "CANCEL_REMOTE": True}, context, cancel_feedback)
        assert cancelled["STATUS"] == "cancelled", cancelled
        result["cancellation"] = True

        timeout_feedback = Feedback()
        try:
            algorithm.processAlgorithm({**common, "TIMEOUT": 1, "COMMAND":
                "geocrop demo --root outputs/qgis-timeout --quick"}, context, timeout_feedback)
            raise AssertionError("Long-running job should time out its monitoring")
        except QgsProcessingException as exc:
            assert "Monitoring timed out" in str(exc), str(exc)
            job_id = next(m.split(": ", 1)[1] for m in timeout_feedback.messages if m.startswith("GeoCrop JOB_ID:"))
            client = ApiClient(url, os.environ["GEOCROP_WEB_TOKEN"])
            assert client.job(job_id)["status"] in ("queued", "running")
            client.cancel(job_id)
            result["timeout_retains_remote_job"] = True
        assert not any(os.environ["GEOCROP_WEB_TOKEN"] in m for m in feedback.messages)
        result["token_not_in_feedback"] = True
    finally:
        plugin.unload()
    assert not iface.actions and QgsApplication.processingRegistry().providerById("geocrop") is None
    assert QgsProject.instance().mapLayers(), "Unloading tools must preserve loaded result layers"
    result["clean_unload"] = True
    (base / "qgis-result.json").write_text(json.dumps(result, indent=2))
    app.exitQgis()


if __name__ == "__main__":
    main()
