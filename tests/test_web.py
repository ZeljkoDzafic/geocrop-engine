"""Web jobs exercise real CLI subprocesses; the HTTP layer never invokes a shell."""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from geocrop import web_jobs  # noqa: E402
from geocrop.web import create_app  # noqa: E402
from geocrop.web_jobs import JobManager, command_catalogue, validated_command  # noqa: E402

TOKEN = "test-token-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def wait_for(client, job_id, statuses, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in statuses:
            return job
        time.sleep(.03)
    pytest.fail(f"Job {job_id} did not reach {statuses}: {job}")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, TOKEN), base_url="http://127.0.0.1") as test:
        yield test


def test_authentication_static_ui_origin_and_schema(client):
    page = client.get("/")
    assert page.status_code == 200 and "CLI console" in page.text and "id=\"map\"" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/theme.js").status_code == 200
    # strict CSP: no inline scripts or handlers; the terminal component mounts into an empty root
    assert '<div id="terminal-root"></div>' in page.text and "<script>" not in page.text and " onclick=" not in page.text
    assert client.get("/api/health").status_code == 401
    assert client.post("/api/jobs", json={"argv": ["indices"]}).status_code == 401
    assert client.get("/api/health", headers=AUTH).json()["workers"] == 1
    assert client.get("/api/jobs", headers={**AUTH, "Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/jobs", headers={**AUTH, "Host": "evil.example"}).status_code == 400
    assert client.get("/api/openapi.json", headers=AUTH).json()["openapi"].startswith("3.")
    assert client.get("/api/openapi.json").status_code == 401


def test_catalogue_matches_available_engine_commands(client):
    commands = client.get("/api/commands", headers=AUTH).json()["commands"]
    assert {"run", "train", "predict", "demo", "audit", "benchmark", "ts-features"} <= {c["name"] for c in commands}
    assert not {"serve", "snap"} & {c["name"] for c in commands}
    train = next(c for c in commands if c["name"] == "train")
    assert any(f["flag"] == "--calibration-mode" for f in train["fields"])
    assert validated_command(command='geocrop train --samples "samples with spaces.npz" --out outputs/m')[2] == "samples with spaces.npz"
    assert command_catalogue() == commands


@pytest.mark.parametrize("body", [
    {"command": "geocrop indices && touch injected"}, {"command": "python -c pass"},
    {"argv": ["serve"]}, {"argv": ["snap", "--gpt", "/bin/sh"]},
    {"argv": ["manifest", "--out", "outputs/x", "--snap-gpt=/bin/sh"]},
    {"argv": ["indices", "--bad"]}, {"argv": ["indices"], "command": "indices"}, {},
])
def test_invalid_commands_are_rejected_without_jobs(client, tmp_path, body):
    assert client.post("/api/jobs", json=body, headers=AUTH).status_code == 400
    assert client.get("/api/jobs", headers=AUTH).json()["jobs"] == []
    assert not (tmp_path / "injected").exists()


def test_output_paths_cannot_escape_workspace_or_write_job_state(client, tmp_path):
    for output in (str(tmp_path.parent / "outside"), ".geocrop-web/overwrite", "."):
        response = client.post("/api/jobs", json={"argv": ["manifest", "--out", output]}, headers=AUTH)
        assert response.status_code == 400


def test_real_cli_success_failure_logs_and_exit_codes(client):
    good = client.post("/api/jobs", json={"command": "geocrop indices --format json"}, headers=AUTH)
    assert good.status_code == 202
    job_id = good.json()["id"]
    job = wait_for(client, job_id, {"succeeded", "failed"})
    assert job["status"] == "succeeded" and job["returncode"] == 0
    logs = client.get(f"/api/jobs/{job_id}/logs", headers=AUTH).json()
    assert any(row["name"] == "NDVI" for row in json.loads(logs["text"]))
    assert client.get(f"/api/jobs/{job_id}/logs?offset={logs['next_offset']}", headers=AUTH).json()["text"] == ""
    assert client.get(f"/api/jobs/{job_id}/logs?offset=-1", headers=AUTH).status_code == 422
    bad = client.post("/api/jobs", json={"argv": ["config-check", "--config", "missing.yaml"]}, headers=AUTH).json()
    job = wait_for(client, bad["id"], {"failed"})
    assert job["returncode"] != 0
    assert "FileNotFoundError" in client.get(f"/api/jobs/{bad['id']}/logs", headers=AUTH).json()["text"]


def test_job_artifacts_download_and_path_confinement(client, tmp_path):
    response = client.post("/api/jobs", json={"argv": ["manifest", "--out", "outputs/test"]}, headers=AUTH)
    job_id = response.json()["id"]
    assert wait_for(client, job_id, {"succeeded", "failed"})["status"] == "succeeded"
    path = "outputs/test/run_manifest.json"
    files = client.get(f"/api/jobs/{job_id}/artifacts", headers=AUTH).json()["files"]
    assert path in {f["path"] for f in files}
    downloaded = client.get(f"/api/jobs/{job_id}/artifact", params={"path": path}, headers=AUTH)
    assert downloaded.json()["schema"] == "geocrop-run/1"
    assert "attachment" in downloaded.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job_id}/artifact", params={"path": "../outside"}, headers=AUTH).status_code == 404
    secret = tmp_path.parent / (tmp_path.name + "-outside.txt")
    secret.write_text("not an output")
    (tmp_path / "outputs/test/link.txt").symlink_to(secret)
    assert "outputs/test/link.txt" not in {f["path"] for f in client.get(f"/api/jobs/{job_id}/artifacts", headers=AUTH).json()["files"]}


def test_running_job_can_be_cancelled(client):
    job = client.post("/api/jobs", json={"argv": ["demo", "--root", "outputs/cancelled-demo", "--quick"]}, headers=AUTH).json()
    wait_for(client, job["id"], {"running"})
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel", headers=AUTH).json()
    assert cancelled["status"] == "cancelled"
    assert client.get("/api/jobs/not-a-job", headers=AUTH).status_code == 404


def test_queued_cancellation_and_restart_recovery(tmp_path, monkeypatch):
    def paused_worker(self):
        self._stopping.wait()
    monkeypatch.setattr(JobManager, "_worker", paused_worker)
    worker = JobManager(tmp_path)
    try:
        job = worker.submit(argv=["indices"])
        assert worker.cancel(job["id"])["status"] == "cancelled"
        pending = worker.submit(argv=["indices"])
    finally:
        worker.close()
    with sqlite3.connect(worker.db_path) as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (pending["id"],))
    recovered = JobManager(tmp_path)
    try:
        assert recovered.get(pending["id"])["status"] == "interrupted"
        assert recovered.get(job["id"])["status"] == "cancelled"
    finally:
        recovered.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX advisory worker lock")
def test_second_worker_cannot_recover_an_active_workers_jobs(tmp_path):
    first = JobManager(tmp_path)
    try:
        with pytest.raises(ValueError, match="already uses"):
            JobManager(tmp_path)
    finally:
        first.close()


def test_server_token_is_not_inherited_by_child_job(client, monkeypatch):
    monkeypatch.setenv("GEOCROP_WEB_TOKEN", TOKEN)
    original = web_jobs.spawn
    observations = []

    def launch(args, cwd, env, stdout):
        observations.append("GEOCROP_WEB_TOKEN" in env)
        assert isinstance(args, list)
        return original(args, cwd, env, stdout)

    monkeypatch.setattr(web_jobs, "spawn", launch)
    job = client.post("/api/jobs", json={"argv": ["indices"]}, headers=AUTH).json()
    assert wait_for(client, job["id"], {"succeeded", "failed"})["status"] == "succeeded"
    assert observations == [False]


def test_incremental_logs_preserve_multibyte_text(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "_worker", lambda self: self._stopping.wait())
    worker = JobManager(tmp_path)
    try:
        job = worker.submit(argv=["indices"])
        expected = "Željko 🌱 κ\n"
        (worker.state_dir / f"{job['id']}.log").write_bytes(expected.encode())
        chunks, offset = [], 0
        for _ in range(30):
            part = worker.logs(job["id"], offset, limit=1)
            if not part["text"]:
                break
            chunks.append(part["text"])
            assert part["next_offset"] > offset
            offset = part["next_offset"]
        assert "".join(chunks) == expected
        assert offset == len(expected.encode())
    finally:
        worker.close()


PROJ_FORK_SCRIPT = """
import os, shutil, sys, tempfile, time
from pathlib import Path
import pyproj
data = Path(tempfile.mkdtemp()) / "proj"
shutil.copytree(pyproj.datadir.get_data_dir(), data)
pyproj.datadir.set_data_dir(str(data))
# a map preview uses PROJ: its proj.db handle is cached in this process
pyproj.Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True).transform(500000, 4970000)
db = data / "proj.db"                      # proj.db replaced on disk (e.g. pyproj reinstalled)
shutil.copy(db, db.with_suffix(".tmp"))
os.replace(db.with_suffix(".tmp"), db)
from geocrop.web_jobs import JobManager
manager = JobManager(Path(sys.argv[1]))
codes = []
for _ in range(2):
    job = manager.submit(argv=["indices"])
    while manager.get(job["id"])["status"] not in ("succeeded", "failed"):
        time.sleep(0.05)
    codes.append(manager.get(job["id"])["returncode"])
manager.close()
print(codes)
"""


def test_jobs_survive_proj_state_from_map_previews(tmp_path):
    """Regression: after a map preview had used PROJ, every job child crashed with -11 (SIGSEGV) on macOS.

    PROJ's pthread_atfork child handler closed its cached proj.db handle inside the forked child and SQLite
    logged (proj.db had changed on disk) before exec. Jobs are now started with posix_spawn, which runs no
    fork handlers. The scenario runs in a separate interpreter so this test process keeps a clean PROJ state.
    """
    pytest.importorskip("pyproj")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(Path(web_jobs.__file__).resolve().parents[1]), *filter(None, [os.environ.get("PYTHONPATH")])])}
    proc = subprocess.run([sys.executable, "-c", PROJ_FORK_SCRIPT, str(tmp_path)], capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "[0, 0]"


@pytest.mark.skipif(not hasattr(os, "posix_spawn"), reason="POSIX only")
def test_jobs_are_spawned_without_fork(tmp_path, monkeypatch):
    def no_fork(*args, **kwargs):
        raise AssertionError("job children must not be started with fork()")

    monkeypatch.setattr(subprocess, "Popen", no_fork)
    monkeypatch.setattr(os, "fork", no_fork)
    log = tmp_path / "job.log"
    with log.open("ab") as out:
        child = web_jobs.spawn([sys.executable, "-m", "geocrop.cli", "manifest", "--out", "relative"], tmp_path,
                               {**os.environ, "PYTHONPATH": str(Path(web_jobs.__file__).resolve().parents[1])}, out)
    assert child.wait(timeout=60) == 0, log.read_text()
    assert (tmp_path / "relative" / "run_manifest.json").exists()      # ran in the job's working directory
    assert child.poll() == 0                                           # reaped once, status kept
    with log.open("ab") as out, pytest.raises(ValueError):
        web_jobs.spawn([sys.executable, "-c", "print(1)"], tmp_path, dict(os.environ), out)
