"""Standalone packaging/transport checks and optional integration in an actual QGIS runtime."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/qgis"))
from geocrop_qgis.transport import ApiClient, ApiError, Cancelled  # noqa: E402


@pytest.mark.parametrize("url", ["http://example.com", "file:///etc/passwd", "http://token@localhost:8765",
                                 "http://127.0.0.1:8765?token=bad", "http://127.0.0.1:8765#token"])
def test_transport_rejects_nonlocal_or_credential_urls(url):
    with pytest.raises(ApiError, match="loopback"):
        ApiClient(url, "test-token")


@pytest.mark.parametrize("path", ["../secret", "/absolute.tif", "a/../../b", "C:/file.tif", "a\\file.tif"])
def test_download_rejects_paths_outside_result_directory(tmp_path, path):
    client = ApiClient("http://127.0.0.1:8765", "test-token")
    with pytest.raises(ApiError, match="artifact path"):
        client.download("job", path, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_never_replaces_existing_file_and_supports_cancellation(tmp_path, monkeypatch):
    client = ApiClient("http://127.0.0.1:8765", "test-token")
    monkeypatch.setattr(client, "_open", lambda *args, **kwargs: BytesIO(b"new data"))
    target = client.download("job", "outputs/result.tif", tmp_path)
    assert target.read_bytes() == b"new data"
    target.write_bytes(b"keep existing data")
    with pytest.raises(FileExistsError):
        client.download("job", "outputs/result.tif", tmp_path)
    assert target.read_bytes() == b"keep existing data"
    with pytest.raises(Cancelled):
        client.download("job", "outputs/cancel.tif", tmp_path, lambda: True)
    assert not (tmp_path / "outputs/cancel.tif").exists()
    assert not list(tmp_path.rglob(".geocrop-*"))


def test_plugin_zip_has_installable_root_and_no_cache_or_secrets(tmp_path):
    spec = importlib.util.spec_from_file_location("build_qgis_plugin", ROOT / "tools/build_qgis_plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.build(tmp_path / "plugin.zip")
    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {"geocrop_qgis/metadata.txt", "geocrop_qgis/__init__.py", "geocrop_qgis/provider.py",
                "geocrop_qgis/algorithms.py", "geocrop_qgis/transport.py", "geocrop_qgis/icon.svg"} <= names
        assert {"geocrop_qgis/LICENSE.txt", "geocrop_qgis/NOTICE.txt", "geocrop_qgis/README.txt"} <= names
        assert not any("__pycache__" in p or p.endswith((".sqlite", ".log", ".pyc")) for p in names)
        assert "hasProcessingProvider=True" in archive.read("geocrop_qgis/metadata.txt").decode()
    old = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.build(output)
    assert output.read_bytes() == old


def test_processing_provider_in_actual_qgis(tmp_path):
    python = Path("/usr/bin/python3")
    if not python.exists() or subprocess.run([str(python), "-c", "import qgis.core"],
                                            capture_output=True, timeout=15).returncode:
        pytest.skip("A system QGIS Python installation is not available")
    uvicorn = pytest.importorskip("uvicorn")
    import geopandas as gpd
    from shapely.geometry import box

    from geocrop.web import create_app

    token = "qgis-test-token-" + "x" * 32
    workspace = tmp_path / "server"
    workspace.mkdir()
    gpd.GeoDataFrame({"code": ["one"], "crop": ["maize"]}, geometry=[box(500000, 4999900, 500100, 5000000)],
                     crs="EPSG:32634").to_file(workspace / "input.gpkg", driver="GPKG")
    original = (workspace / "input.gpkg").read_bytes()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(workspace, token), log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.05)
        assert server.started
        env = os.environ.copy()
        for key, subdir in (("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data"),
                            ("QGIS_AUTH_DB_DIR_PATH", "auth"), ("XDG_RUNTIME_DIR", "runtime")):
            path = tmp_path / subdir
            path.mkdir(mode=0o700)
            env[key] = str(path)
        env.update({"QT_QPA_PLATFORM": "offscreen", "GEOCROP_WEB_TOKEN": token,
                    "GEOCROP_QGIS_TEST_RESULTS": str(tmp_path), "GEOCROP_QGIS_TEST_WORKSPACE": str(workspace),
                    "GEOCROP_QGIS_TEST_URL": f"http://127.0.0.1:{port}"})
        run = subprocess.run([str(python), str(ROOT / "tests/qgis_runner.py")], env=env,
                             capture_output=True, text=True, timeout=90)
        assert run.returncode == 0, run.stdout + run.stderr
        result = json.loads((tmp_path / "qgis-result.json").read_text())
        assert result["raster"] and result["vector"] and result["reconnect"]
        assert result["automatic_project_loading"]
        assert result["encrypted_auth_config"] and result["cancellation"]
        assert result["demo_submission"]
        assert result["timeout_retains_remote_job"] and result["token_not_in_feedback"]
        assert result["processing_context_layers"] == 3
        assert result["preserves_existing_downloads"] and result["clean_unload"]
        assert (workspace / "input.gpkg").read_bytes() == original
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
