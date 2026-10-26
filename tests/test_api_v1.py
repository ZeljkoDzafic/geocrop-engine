"""The versioned REST contract (/api/v1) and the unversioned compatibility aliases (/api)."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from geocrop.client import EngineClient  # noqa: E402
from geocrop.web import API_V1_PREFIX, add_versioned_aliases, create_app  # noqa: E402

TOKEN = "version-token-" + "v" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, TOKEN)


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://127.0.0.1") as test:
        yield test


def route_table(app):
    return {(r.path, m) for r in app.routes if isinstance(r, APIRoute) for m in r.methods}


def test_every_unversioned_route_has_a_v1_twin_with_the_same_handler(app):
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    legacy = [r for r in routes if r.path.startswith("/api/") and not r.path.startswith("/api/v1/")]
    assert len(legacy) >= 10
    for route in legacy:
        twin = next(r for r in routes if r.path == "/api/v1" + route.path[4:] and r.methods == route.methods)
        assert twin.endpoint is route.endpoint
        assert twin.status_code == route.status_code
        assert [d.dependency for d in twin.dependencies] == [d.dependency for d in route.dependencies]


def test_openapi_documents_only_the_versioned_contract(client):
    for prefix in ("/api", "/api/v1"):
        paths = client.get(f"{prefix}/openapi.json", headers=AUTH).json()["paths"]
        assert paths and all(p.startswith("/api/v1/") for p in paths)
        assert {"/api/v1/health", "/api/v1/jobs", "/api/v1/jobs/{job_id}/artifact"} <= set(paths)


def test_both_prefixes_serve_identical_results(client):
    for path in ("/health", "/commands", "/jobs"):
        assert client.get("/api" + path, headers=AUTH).json() == client.get("/api/v1" + path, headers=AUTH).json()
    job = client.post("/api/v1/jobs", json={"argv": ["indices", "--format", "json"]}, headers=AUTH)
    assert job.status_code == 202
    job_id = job.json()["id"]
    deadline = time.monotonic() + 30
    while client.get(f"/api/jobs/{job_id}", headers=AUTH).json()["status"] not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(.05)
    assert client.get(f"/api/jobs/{job_id}", headers=AUTH).json() == client.get(f"/api/v1/jobs/{job_id}",
                                                                                  headers=AUTH).json()
    assert client.get(f"/api/v1/jobs/{job_id}/logs", headers=AUTH).json()["text"]
    assert client.post(f"/api/v1/jobs/{job_id}/cancel", headers=AUTH).json()["status"] == "succeeded"
    assert client.get("/api/v1/jobs/missing", headers=AUTH).json() == {"detail": "Job or artifact not found"}


def test_aliasing_is_generic_idempotent_and_keeps_dependencies():
    app = FastAPI()
    calls = []

    def guard():
        calls.append("guard")

    @app.get("/api/files", dependencies=[Depends(guard)])
    def files(prefix: str = ""):
        return {"prefix": prefix}

    @app.post("/api/v1/only-versioned")
    def only():
        return {}

    @app.get("/other")
    def other():
        return {}

    add_versioned_aliases(app)
    before = route_table(app)
    add_versioned_aliases(app)
    assert route_table(app) == before
    assert ("/api/v1/files", "GET") in before and ("/api/v1/v1/only-versioned", "POST") not in before
    assert not any(path.startswith(API_V1_PREFIX + "/other") or path == "/api/v1" for path, _ in before)
    with TestClient(app) as test:
        assert test.get("/api/v1/files", params={"prefix": "x"}).json() == {"prefix": "x"}
    assert calls == ["guard"]


def test_client_speaks_v1_and_falls_back_to_legacy_servers(app, tmp_path):
    def bind(engine, test):
        headers = dict(engine.session.headers)
        engine.session.request = lambda method, url, timeout=None, **kw: test.request(method, url, headers=headers, **kw)

    with TestClient(app, base_url="http://127.0.0.1") as test:
        engine = EngineClient("http://127.0.0.1", token=TOKEN)
        bind(engine, test)
        assert engine.health()["status"] == "ok"
        assert engine.api_prefix == "/api/v1"

    (tmp_path / "legacy").mkdir()
    legacy = create_app(tmp_path / "legacy", TOKEN)
    legacy.router.routes[:] = [r for r in legacy.router.routes
                               if not (isinstance(r, APIRoute) and r.path.startswith("/api/v1/"))]
    with TestClient(legacy, base_url="http://127.0.0.1") as test:
        engine = EngineClient("http://127.0.0.1", token=TOKEN)
        bind(engine, test)
        assert {c["name"] for c in engine.commands()} >= {"indices"}
        assert engine.api_prefix == "/api"
        assert engine.jobs() == []
