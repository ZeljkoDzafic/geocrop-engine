"""Security gate for the local web API: authentication, host/origin checks, headers, path confinement,
command restrictions and resource bounds. See docs/security.md for the threat model."""
import json
import os
import re
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from geocrop import web_jobs  # noqa: E402
from geocrop.web import MAX_BODY_BYTES, SECURITY_HEADERS, create_app, host_allowed  # noqa: E402
from geocrop.web_jobs import JobManager, validated_command  # noqa: E402

TOKEN = "security-token-" + "s" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PREFIXES = ["/api", "/api/v1"]


@pytest.fixture
def paused(monkeypatch):
    """Jobs are recorded but never executed, so path/limit checks need no subprocess."""
    monkeypatch.setattr(JobManager, "_worker", lambda self: self._stopping.wait())


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def app(workspace):
    return create_app(workspace, TOKEN)


@pytest.fixture
def client(app, paused):
    with TestClient(app, base_url="http://127.0.0.1") as test:
        yield test


@pytest.fixture
def live(app):
    with TestClient(app, base_url="http://127.0.0.1") as test:
        yield test


def api_routes(app):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api"):
            for method in route.methods:
                yield method, re.sub(r"\{[^}]+\}", "0123abc", route.path)


def submit(client, prefix="/api", **body):
    return client.post(f"{prefix}/jobs", json=body, headers=AUTH)


def outside_file(workspace):
    secret = workspace.parent / "outside-secret.txt"
    secret.write_text("outside the workspace")
    return secret


# --- authentication -------------------------------------------------------------------------------------

def test_every_api_route_requires_the_bearer_token(app, client):
    routes = list(api_routes(app))
    assert len(routes) >= 2 * 10 and {p for _, p in routes if p.startswith("/api/v1/")}
    wrong = [{}, {"Authorization": "Bearer " + "x" * len(TOKEN)}, {"Authorization": TOKEN},
             {"Authorization": f"bearer {TOKEN}"}, {"Authorization": f"Basic {TOKEN}"},
             {"Authorization": f"Bearer {TOKEN}x"}, {"Authorization": f"Bearer {TOKEN[:-1]}"},
             {"Authorization": "Bearer "}]
    for method, path in routes:
        for headers in wrong:
            response = client.request(method, path, headers=headers, json={"argv": ["indices"]})
            assert response.status_code == 401, (method, path, headers)
            assert response.headers["www-authenticate"] == "Bearer"
        assert client.request(method, path, params={"token": TOKEN}).status_code == 401
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []


@pytest.mark.parametrize("path", ["/api", "/api/", "/api/unknown", "/api/v1/unknown", "/api/v2/health",
                                  "/api/filez", "/api/v1/filez"])
def test_unknown_api_paths_need_the_token_before_revealing_anything(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 404


@pytest.mark.parametrize("path", ["/api/files", "/api/v1/files"])
def test_workspace_listing_requires_the_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 200
    assert client.get(path, params={"prefix": "../"}, headers=AUTH).status_code == 400


def test_static_ui_is_public_but_carries_no_secrets(client):
    page = client.get("/")
    assert page.status_code == 200 and TOKEN not in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/../web.py").status_code == 404
    assert client.get("/static/%2e%2e/web.py").status_code == 404


def test_short_or_missing_tokens_refuse_to_start(workspace):
    for token in (None, "", "short", "x" * 31):
        with pytest.raises(ValueError, match="32 characters"):
            create_app(workspace, token)


# --- host and origin ------------------------------------------------------------------------------------

@pytest.mark.parametrize("host,allowed", [
    ("127.0.0.1", True), ("127.0.0.1:8765", True), ("localhost:8765", True), ("LOCALHOST", True),
    ("[::1]", True), ("[::1]:8765", True),
    ("evil.example", False), ("127.0.0.1.evil.example", False), ("localhost.evil.example:8765", False),
    ("127.0.0.1:abc", False), ("127.0.0.1:8765:1", False), ("127.0.0.1:1234567", False), ("::1", False),
    ("[::1]evil", False), ("", False), (None, False), ("0.0.0.0:8765", False), ("192.168.1.10", False),
])
def test_host_header_allow_list(host, allowed):
    assert host_allowed(host) is allowed


@pytest.mark.parametrize("host", ["evil.example", "attacker.test:8765", "127.0.0.1.nip.io", "0.0.0.0"])
def test_dns_rebinding_hosts_are_rejected_with_headers(client, host):
    response = client.get("/api/health", headers={**AUTH, "Host": host})
    assert response.status_code == 400
    assert response.headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]
    assert client.get("/", headers={"Host": host}).status_code == 400


@pytest.mark.parametrize("host", ["localhost:8765", "[::1]:8765", "127.0.0.1:8765"])
def test_loopback_hosts_with_ports_are_accepted(client, host):
    assert client.get("/api/health", headers={**AUTH, "Host": host}).status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "http://evil.example", "null", "https://127.0.0.1",
                                    "http://127.0.0.1:9999", "http://localhost", "file://"])
def test_cross_origin_requests_are_rejected_even_with_a_token(client, origin):
    for prefix in PREFIXES:
        assert client.get(f"{prefix}/jobs", headers={**AUTH, "Origin": origin}).status_code == 403
        response = client.post(f"{prefix}/jobs", json={"argv": ["indices"]}, headers={**AUTH, "Origin": origin})
        assert response.status_code == 403
        assert "content-security-policy" in response.headers
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []


def test_same_origin_and_fetch_metadata(client):
    assert client.get("/api/health", headers={**AUTH, "Origin": "http://127.0.0.1"}).status_code == 200
    assert client.get("/api/health", headers={**AUTH, "Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert client.get("/api/health", headers={**AUTH, "Sec-Fetch-Site": "cross-site"}).status_code == 403
    # A cross-site navigation to the UI itself is fine; it holds no data without the token.
    assert client.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


# --- security headers -----------------------------------------------------------------------------------

def test_security_headers_on_every_kind_of_response(client, monkeypatch):
    responses = {
        "ui": client.get("/"),
        "static": client.get("/static/app.js"),
        "ok": client.get("/api/health", headers=AUTH),
        "v1": client.get("/api/v1/health", headers=AUTH),
        "unauthorised": client.get("/api/health"),
        "forbidden": client.get("/api/health", headers={**AUTH, "Origin": "https://evil.example"}),
        "bad host": client.get("/", headers={"Host": "evil.example"}),
        "not found": client.get("/api/jobs/missing", headers=AUTH),
        "static missing": client.get("/static/missing.js"),
        "invalid": client.get("/api/jobs?limit=0", headers=AUTH),
        "rejected": submit(client, argv=["serve"]),
        "too large": client.post("/api/jobs", content=b"x" * (MAX_BODY_BYTES + 1),
                                 headers={**AUTH, "Content-Type": "application/json"}),
    }
    for name, response in responses.items():
        for header, value in SECURITY_HEADERS.items():
            assert response.headers.get(header) == value, (name, header)
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    for directive in ("default-src 'self'", "script-src 'self'", "frame-ancestors 'none'", "object-src 'none'",
                      "base-uri 'none'", "connect-src 'self'"):
        assert directive in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


def test_unexpected_errors_hide_details_and_keep_headers(client, monkeypatch):
    def broken(self, limit=50):
        raise RuntimeError("internal secret detail")
    monkeypatch.setattr(JobManager, "list", broken)
    response = client.get("/api/v1/jobs", headers=AUTH)
    assert response.status_code == 500
    assert "secret" not in response.text and "Traceback" not in response.text
    assert response.headers["x-frame-options"] == "DENY"


# --- artifact downloads ---------------------------------------------------------------------------------

@pytest.fixture
def job_with_output(client, workspace):
    job = submit(client, argv=["manifest", "--out", "outputs/a"]).json()
    out = workspace / "outputs/a"
    out.mkdir(parents=True)
    (out / "result.txt").write_text("result")
    (out / "nested").mkdir()
    (out / "nested/deep.json").write_text("{}")
    return job["id"]


TRAVERSALS = [
    "../outside-secret.txt", "outputs/a/../../../outside-secret.txt", "/etc/passwd", "//etc/passwd",
    "%2e%2e/outside-secret.txt", "..%2Foutside-secret.txt", "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..\\outside-secret.txt", "outputs\\a\\..\\..\\..\\outside-secret.txt", "C:\\Windows\\win.ini",
    "outputs/a/./result.txt", "outputs//a/result.txt", "outputs/a/result.txt\x00.png", "~/.ssh/id_rsa",
    ".geocrop-web/jobs.sqlite3", ".geocrop-web/worker.lock", "outputs/a", "", " outputs/a/result.txt",
]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_artifact_download_accepts_only_listed_files(client, workspace, job_with_output, prefix):
    outside_file(workspace)
    url = f"{prefix}/jobs/{job_with_output}/artifact"
    listed = {f["path"] for f in client.get(f"{prefix}/jobs/{job_with_output}/artifacts", headers=AUTH).json()["files"]}
    assert listed == {"outputs/a/result.txt", "outputs/a/nested/deep.json"}
    ok = client.get(url, params={"path": "outputs/a/result.txt"}, headers=AUTH)
    assert ok.status_code == 200 and ok.content == b"result"
    assert "attachment" in ok.headers["content-disposition"]
    assert ok.headers["content-type"] == "application/octet-stream"
    for path in TRAVERSALS + [str(workspace / "outputs/a/result.txt")]:
        response = client.get(url, params={"path": path}, headers=AUTH)
        assert response.status_code == 404, path
        assert b"outside the workspace" not in response.content
    assert client.get(url, headers=AUTH).status_code == 422


def test_artifact_path_encoding_in_the_raw_url(client, workspace, job_with_output):
    outside_file(workspace)
    for raw in ("..%2F..%2F..%2Foutside-secret.txt", "%2e%2e%2f%2e%2e%2foutside-secret.txt",
                "%252e%252e%252foutside-secret.txt", "..%5C..%5Coutside-secret.txt", "%2Fetc%2Fpasswd"):
        response = client.get(f"/api/v1/jobs/{job_with_output}/artifact?path={raw}", headers=AUTH)
        assert response.status_code == 404, raw
    for job in ("..%2F..%2Fetc%2Fpasswd", "%2e%2e", "..", "0' OR '1'='1"):
        for tail in ("", "/logs", "/artifacts", "/artifact?path=outputs/a/result.txt", "/map"):
            assert client.get(f"/api/jobs/{job}{tail}", headers=AUTH).status_code in (404, 405), (job, tail)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symbolic links")
def test_symlinks_cannot_escape_through_listing_download_or_map(client, workspace, job_with_output):
    out = workspace / "outputs/a"
    secret = outside_file(workspace)
    (out / "secret.txt").symlink_to(secret)
    (out / "passwd").symlink_to("/etc/passwd")
    (out / "etc").symlink_to("/etc", target_is_directory=True)
    (out / "state.sqlite3").symlink_to(workspace / ".geocrop-web/jobs.sqlite3")
    (out / "inside.txt").symlink_to(out / "result.txt")
    listed = {f["path"] for f in client.get(f"/api/jobs/{job_with_output}/artifacts", headers=AUTH).json()["files"]}
    assert not {"outputs/a/secret.txt", "outputs/a/passwd", "outputs/a/state.sqlite3"} & listed
    assert not any(p.startswith("outputs/a/etc/") for p in listed)
    for path in ("outputs/a/secret.txt", "outputs/a/passwd", "outputs/a/etc/passwd", "outputs/a/etc/hosts",
                 "outputs/a/state.sqlite3"):
        assert client.get(f"/api/jobs/{job_with_output}/artifact", params={"path": path}, headers=AUTH).status_code == 404
    manager = client.app.state.jobs
    resolved = manager.output_files(job_with_output)
    assert all(p.is_relative_to(workspace) and not p.is_relative_to(manager.state_dir) for p in resolved)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symbolic links")
def test_file_swapped_for_symlink_after_listing_is_refused(client, workspace, job_with_output):
    secret = outside_file(workspace)
    url = f"/api/jobs/{job_with_output}/artifact"
    assert client.get(url, params={"path": "outputs/a/result.txt"}, headers=AUTH).status_code == 200
    target = workspace / "outputs/a/result.txt"
    target.unlink()
    target.symlink_to(secret)
    response = client.get(url, params={"path": "outputs/a/result.txt"}, headers=AUTH)
    assert response.status_code == 404 and b"outside" not in response.content


@pytest.mark.skipif(os.name != "posix", reason="POSIX symbolic links")
def test_output_root_replaced_by_symlink_lists_nothing_outside(client, workspace, job_with_output):
    import shutil
    shutil.rmtree(workspace / "outputs/a")
    (workspace / "outputs/a").symlink_to("/etc", target_is_directory=True)
    assert client.get(f"/api/jobs/{job_with_output}/artifacts", headers=AUTH).json()["files"] == []
    assert client.get(f"/api/jobs/{job_with_output}/artifact", params={"path": "outputs/a/passwd"},
                      headers=AUTH).status_code == 404


# --- job output confinement -----------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["manifest", "--out", "../outside"], ["manifest", "--out=../outside"], ["manifest", "--ou", "../outside"],
    ["manifest", "--out", "outputs/../../outside"], ["manifest", "--out", "/tmp/geocrop-outside"],
    ["manifest", "--out", "/etc/geocrop"], ["manifest", "--out", "//etc/geocrop"],
    ["manifest", "--out", "."], ["manifest", "--out", ""], ["manifest", "--out", "outputs/.."],
    ["manifest", "--out", ".geocrop-web"], ["manifest", "--out", ".geocrop-web/jobs.sqlite3"],
    ["manifest", "--out", "./.geocrop-web/../.geocrop-web/x"], ["demo", "--root", "../demo"],
    ["demo", "--root", "/tmp/geocrop-demo-outside", "--quick"],
])
def test_outputs_outside_the_workspace_or_in_job_state_are_rejected(client, argv):
    response = submit(client, argv=argv)
    assert response.status_code == 400, argv
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []


def test_relative_lookalikes_stay_inside_the_workspace(client, workspace):
    for name in ("..\\outside", "%2e%2e/outside", "~/outside", "outputs/..hidden", "C:\\outside"):
        response = submit(client, "/api/v1", argv=["manifest", "--out", name])
        assert response.status_code == 202, name
        assert all(Path(p).is_relative_to(workspace) for p in response.json()["outputs"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX symbolic links")
def test_symlinked_directories_cannot_redirect_outputs(client, workspace, tmp_path):
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / "etc").symlink_to("/etc", target_is_directory=True)
    (workspace / "state").symlink_to(workspace / ".geocrop-web", target_is_directory=True)
    for out in ("escape/run", "escape", "etc/geocrop", "state/run"):
        assert submit(client, argv=["manifest", "--out", out]).status_code == 400, out


def test_run_config_outputs_are_confined_and_parse_errors_do_not_leak(client, workspace):
    (workspace / "evil.yaml").write_text("paths:\n  output: /tmp/geocrop-outside\n")
    assert submit(client, argv=["run", "--config", "evil.yaml"]).status_code == 400
    (workspace / "broken.yaml").write_text("top-secret-value: [unclosed\n")
    response = submit(client, argv=["run", "--config", "broken.yaml"])
    assert response.status_code == 400 and "top-secret-value" not in response.text
    response = submit(client, argv=["run", "--config", "/etc/passwd"])
    assert response.status_code == 400 and "root" not in response.text
    assert submit(client, argv=["run", "--config", "missing.yaml"]).status_code == 400


# --- command restrictions -------------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {"argv": ["serve"]}, {"argv": ["serve", "--workspace", "/"]}, {"command": "geocrop serve --port 9000"},
    {"argv": ["snap", "--gpt", "/bin/sh", "--graph", "g", "--input", "i", "--output", "o"]},
    {"command": "snap --gpt /bin/sh --graph g --input i --output o"},
    {"argv": ["manifest", "--out", "outputs/x", "--snap-gpt", "/bin/sh"]},
    {"argv": ["manifest", "--out", "outputs/x", "--snap-gpt=/bin/sh"]},
    {"argv": ["manifest", "--out", "outputs/x", "--snap-gp=/bin/sh"]},
    {"argv": ["manifest", "--out", "outputs/x", "--snap", "/bin/sh"]},
    {"command": "manifest --out outputs/x --snap-g /bin/sh"},
    {"argv": ["SERVE"]}, {"argv": ["--version"]}, {"argv": ["-h"]}, {"argv": ["indices", "--help"]},
    {"argv": ["geocrop"]}, {"argv": []}, {"command": ""}, {"command": "geocrop"},
])
def test_blocked_commands_and_options(client, body):
    assert client.post("/api/v1/jobs", json=body, headers=AUTH).status_code == 400
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []


@pytest.mark.parametrize("command", [
    "indices; touch pwned", "indices && touch pwned", "indices || touch pwned", "indices | sh",
    "indices `touch pwned`", "indices $(touch pwned)", "indices > pwned", "indices < /etc/passwd",
    "indices & touch pwned", "touch pwned", "sh -c 'touch pwned'", "bash -c 'touch pwned'",
    "python -c 'open(\"pwned\",\"w\")'", "indices\ntouch pwned", "PWNED=1 indices", "cd / && indices",
    "indices '", 'indices "unterminated', "indices --format json; rm -rf /", "geocrop indices --format `id`",
    "/bin/sh", "../../bin/sh", "indices --format $(id)",
])
def test_shell_operators_are_never_interpreted(client, workspace, command):
    assert submit(client, command=command).status_code == 400
    assert not (workspace / "pwned").exists()
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []


def test_accepted_metacharacters_reach_the_engine_as_literal_arguments(live, workspace, monkeypatch):
    launched = []
    original = web_jobs.spawn

    def spy(args, cwd, env, stdout):
        launched.append((args, False))     # spawn has no shell: argv goes to posix_spawn / Popen as a list
        return original(args, cwd, env, stdout)

    monkeypatch.setattr(web_jobs, "spawn", spy)
    job = submit(live, command="geocrop manifest --out 'outputs/$(touch pwned);x'").json()
    deadline = time.monotonic() + 30
    while live.get(f"/api/jobs/{job['id']}", headers=AUTH).json()["status"] not in web_jobs.TERMINAL:
        assert time.monotonic() < deadline
        time.sleep(.05)
    (args, shell), = launched
    assert shell is False and isinstance(args, list)
    assert args[-1] == "outputs/$(touch pwned);x"
    assert not (workspace / "pwned").exists()
    assert (workspace / "outputs/$(touch pwned);x/run_manifest.json").exists()


# --- size limits ----------------------------------------------------------------------------------------

def test_argv_size_limits(client):
    limit = web_jobs.MAX_ARG_BYTES
    assert submit(client, argv=["indices"] + ["--format"] * 256).status_code == 422  # > 256 items
    assert submit(client, argv=["manifest", "--out", "o" * (limit + 1)]).status_code == 400
    assert submit(client, argv=["manifest", "--out", "ž" * (limit // 2 + 1)]).status_code == 400  # bytes, not chars
    many = ["manifest", "--out", "outputs/x", "--inputs"] + ["i" * 60000] * 3
    assert submit(client, argv=many).status_code == 400  # total > MAX_ARGV_BYTES
    assert submit(client, command="indices " + "x" * 65536).status_code == 422
    assert submit(client, argv=["indices", "--format\x00json"]).status_code == 400
    assert submit(client, argv=["indices", 1]).status_code == 422
    assert submit(client, argv="indices").status_code == 422
    assert client.post("/api/jobs", json={"argv": ["indices"], "extra": 1}, headers=AUTH).status_code == 422
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []
    with pytest.raises(ValueError):
        validated_command(argv=["x"] * (web_jobs.MAX_ARGS + 1))


def test_request_body_size_is_bounded(client):
    big = json.dumps({"command": "indices", "pad": "x" * MAX_BODY_BYTES}).encode()
    for prefix in PREFIXES:
        response = client.post(f"{prefix}/jobs", content=big, headers={**AUTH, "Content-Type": "application/json"})
        assert response.status_code == 413
    response = client.post("/api/jobs", content=b"{}", headers={**AUTH, "Content-Length": "abc"})
    assert response.status_code == 400

    def chunks():
        yield b'{"argv": ["indices"]}'
    response = client.post("/api/jobs", content=chunks(), headers={**AUTH, "Content-Type": "application/json"})
    assert response.status_code == 411


def test_oversized_logs_are_served_in_bounded_chunks(client):
    job = submit(client, argv=["indices"]).json()
    log = client.app.state.jobs.state_dir / f"{job['id']}.log"
    log.write_bytes(b"x" * (3 * 1024 * 1024))
    url = f"/api/v1/jobs/{job['id']}/logs"
    default = client.get(url, headers=AUTH).json()
    assert len(default["text"]) == 65536 and default["next_offset"] == 65536
    biggest = client.get(url, params={"limit": 262144}, headers=AUTH).json()
    assert len(biggest["text"]) == 262144
    for params in ({"limit": 262145}, {"limit": 0}, {"offset": -1}, {"limit": "all"}):
        assert client.get(url, params=params, headers=AUTH).status_code == 422
    past = client.get(url, params={"offset": 10 ** 12}, headers=AUTH).json()
    assert past == {"text": "", "next_offset": 10 ** 12}


def test_list_limit_bounds(client):
    for limit in (0, 201, -5):
        assert client.get("/api/jobs", params={"limit": limit}, headers=AUTH).status_code == 422
    assert client.get("/api/jobs/x/map", params={"max_px": 100000}, headers=AUTH).status_code == 422


def test_queue_limit_and_single_worker(client):
    assert client.get("/api/v1/health", headers=AUTH).json()["workers"] == 1
    for _ in range(web_jobs.MAX_QUEUED_JOBS):
        assert submit(client, argv=["indices"]).status_code == 202
    response = submit(client, "/api/v1", argv=["indices"])
    assert response.status_code == 400 and "queue is full" in response.json()["detail"]
    jobs = client.get("/api/jobs", params={"limit": 200}, headers=AUTH).json()["jobs"]
    assert len(jobs) == web_jobs.MAX_QUEUED_JOBS and {j["status"] for j in jobs} == {"queued"}


def test_one_engine_process_at_a_time(live, monkeypatch):
    running, peak = [0], [0]
    original = web_jobs.spawn

    class Tracked:
        def __init__(self, *args):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            self._p = original(*args)
            self.pid = self._p.pid

        def wait(self, timeout=None):
            try:
                return self._p.wait(timeout)
            finally:
                if self._p.returncode is not None and running[0]:
                    running[0] -= 1

        def __getattr__(self, name):
            return getattr(self._p, name)

    monkeypatch.setattr(web_jobs, "spawn", Tracked)
    ids = [submit(live, argv=["indices"]).json()["id"] for _ in range(3)]
    deadline = time.monotonic() + 60
    while any(live.get(f"/api/jobs/{i}", headers=AUTH).json()["status"] not in web_jobs.TERMINAL for i in ids):
        assert time.monotonic() < deadline
        time.sleep(.05)
    assert peak[0] == 1
