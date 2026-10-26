import json
from pathlib import Path

import pytest
import requests

from geocrop.client import EngineClient


class Response:
    def __init__(self, value=None, data=b"example"):
        self.value, self.data = value, data

    def raise_for_status(self):
        pass

    def json(self):
        return self.value

    def iter_content(self, size):
        yield self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_client_submits_typed_arguments_without_token_in_payload(monkeypatch):
    client = EngineClient(token="test-token")
    captured = []

    def request(method, url, **kwargs):
        captured.append((method, url, kwargs))
        return Response({"id": "job", "status": "queued"})

    monkeypatch.setattr(client.session, "request", request)
    assert client.submit(["indices"])["id"] == "job"
    assert captured[0][2]["json"] == {"argv": ["indices"], "command": None}
    assert "test-token" not in json.dumps(captured)
    with pytest.raises(ValueError, match="exactly one"):
        client.submit(["indices"], command="indices")
    client.close()


def test_wait_distinguishes_failure_timeout_and_success(monkeypatch):
    client = EngineClient(token="test-token")
    monkeypatch.setattr(client, "job", lambda _: {"status": "failed", "returncode": 2})
    assert client.wait("job", check=False)["returncode"] == 2
    with pytest.raises(RuntimeError, match="failed"):
        client.wait("job")
    monkeypatch.setattr(client, "job", lambda _: {"status": "succeeded", "returncode": 0})
    assert client.wait("job")["status"] == "succeeded"
    monkeypatch.setattr(client, "job", lambda _: {"status": "running"})
    with pytest.raises(TimeoutError):
        client.wait("job", timeout=.01, poll_interval=.005)
    with pytest.raises(ValueError):
        client.wait("job", poll_interval=0)
    client.close()


def test_download_preserves_existing_destination_and_streams_new_file(tmp_path, monkeypatch):
    client = EngineClient(token="test-token")
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: Response(data=b"artifact"))
    destination = tmp_path / "new" / "result.txt"
    assert client.download("job", "outputs/result.txt", destination) == destination
    assert destination.read_bytes() == b"artifact"
    destination.write_text("keep existing data")
    with pytest.raises(FileExistsError):
        client.download("job", "outputs/result.txt", destination)
    assert destination.read_text() == "keep existing data"
    assert not list(destination.parent.glob(".geocrop-download-*"))
    client.close()


def test_http_errors_propagate(monkeypatch):
    client = EngineClient(token="test-token")

    def fail(*args, **kwargs):
        raise requests.HTTPError("401 Unauthorized")

    monkeypatch.setattr(client.session, "request", fail)
    with pytest.raises(requests.HTTPError, match="401"):
        client.health()
    client.close()


def test_example_notebook_is_clean_and_has_executable_python():
    path = Path(__file__).resolve().parents[1] / "examples/web_notebook.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(path), "exec")


class Routed(Response):
    def __init__(self, status, value):
        super().__init__(value)
        self.status_code = status


def test_client_prefers_v1_and_pins_the_detected_prefix(monkeypatch):
    client = EngineClient(token="test-token")
    urls = []

    def request(method, url, **kwargs):
        urls.append(url)
        return Routed(200, {"status": "ok", "jobs": []})

    monkeypatch.setattr(client.session, "request", request)
    client.health()
    client.jobs()
    assert urls == ["http://127.0.0.1:8765/api/v1/health", "http://127.0.0.1:8765/api/v1/jobs"]
    assert client.api_prefix == "/api/v1"
    client.close()


def test_client_falls_back_only_for_missing_routes(monkeypatch):
    client = EngineClient(token="test-token")
    urls = []

    def legacy_server(method, url, **kwargs):
        urls.append(url)
        if "/api/v1/" in url:
            return Routed(404, {"detail": "Not Found"})
        return Routed(200, {"id": "a/b", "status": "running"})

    monkeypatch.setattr(client.session, "request", legacy_server)
    assert client.job("a/b")["status"] == "running"
    assert urls == ["http://127.0.0.1:8765/api/v1/jobs/a%2Fb", "http://127.0.0.1:8765/api/jobs/a%2Fb"]
    client.job("a/b")
    assert urls[-1] == "http://127.0.0.1:8765/api/jobs/a%2Fb" and client.api_prefix == "/api"

    fresh = EngineClient(token="test-token")
    monkeypatch.setattr(fresh.session, "request", lambda method, url, **kw: urls.append(url) or Routed(
        404, {"detail": "Job or artifact not found"}))
    fresh.job("missing")  # Response stub does not raise; a real 404 would raise HTTPError
    assert urls[-1].endswith("/api/v1/jobs/missing") and fresh.api_prefix == "/api/v1"
    pinned = EngineClient(token="test-token", api_prefix="/api")
    monkeypatch.setattr(pinned.session, "request", lambda method, url, **kw: urls.append(url) or Routed(200, {}))
    pinned.health()
    assert urls[-1].endswith("/api/health")
    for c in (client, fresh, pinned):
        c.close()
