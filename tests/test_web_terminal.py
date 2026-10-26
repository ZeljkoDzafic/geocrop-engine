"""Workspace path completion for the web terminal stays inside the workspace."""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from geocrop.web import FILE_LIMIT, create_app  # noqa: E402

TOKEN = "test-token-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "study"
    (root / "outputs" / "demo").mkdir(parents=True)
    (root / "outputs" / "demo" / "classes.tif").write_bytes(b"x")
    (root / "outputs" / "model.joblib").write_bytes(b"x")
    (root / "configs").mkdir()
    (root / "configs" / "study.yaml").write_text("paths: {}\n")
    (root / "Data with spaces").mkdir()
    (root / ".hidden").mkdir()
    (root / ".env").write_text("SECRET=1\n")
    return root


@pytest.fixture
def client(workspace):
    with TestClient(create_app(workspace, TOKEN), base_url="http://127.0.0.1") as test:
        yield test


def entries(client, prefix="", status=200):
    response = client.get("/api/files", params={"prefix": prefix}, headers=AUTH)
    assert response.status_code == status, response.text
    return response.json()["entries"] if status == 200 else response.json()


def names(items):
    return [item["name"] for item in items]


def test_terminal_assets_are_served_and_mounted(client):
    page = client.get("/").text
    assert 'id="terminal-root"' in page
    assert "/static/terminal.js" in page and "/static/terminal.css" in page
    script = client.get("/static/terminal.js")
    assert script.status_code == 200 and "gc-terminal" in script.text
    style = client.get("/static/terminal.css")
    assert style.status_code == 200 and ".gc-terminal" in style.text


def test_files_requires_authentication(client):
    assert client.get("/api/files").status_code == 401
    assert client.get("/api/files", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_listing_hides_dotfiles_and_state_dir_and_sorts_directories_first(client, workspace):
    items = entries(client)
    assert names(items) == ["configs", "Data with spaces", "outputs"]
    assert all(item["is_dir"] for item in items)
    assert (workspace / ".geocrop-web").is_dir()  # state dir exists but is never listed
    nested = entries(client, "outputs/")
    assert nested == [{"name": "demo", "path": "outputs/demo", "is_dir": True},
                      {"name": "model.joblib", "path": "outputs/model.joblib", "is_dir": False}]


def test_prefix_filtering_is_case_insensitive_and_nested(client):
    assert names(entries(client, "out")) == ["outputs"]
    assert names(entries(client, "d")) == ["Data with spaces"]
    assert names(entries(client, "outputs/de")) == ["demo"]
    assert entries(client, "outputs/demo/cl") == [
        {"name": "classes.tif", "path": "outputs/demo/classes.tif", "is_dir": False}]
    assert entries(client, "./configs/st")[0]["path"] == "configs/study.yaml"
    assert entries(client, "missing/dir/") == []
    assert entries(client, "configs/study.yaml/") == []


@pytest.mark.parametrize("prefix", ["../", "..", "outputs/../../", "outputs/../..", "/etc/", "/",
                                    "\\etc", "C:/Windows", "..\\..\\", ".hidden/", ".geocrop-web/"])
def test_traversal_absolute_and_hidden_prefixes_are_rejected(client, prefix):
    assert client.get("/api/files", params={"prefix": prefix}, headers=AUTH).status_code == 400


def test_url_encoded_traversal_is_rejected(client):
    for raw in ("%2e%2e%2f", "..%2f..%2f", "%2Fetc%2F", "outputs%2F%2E%2E%2F%2E%2E%2F"):
        assert client.get(f"/api/files?prefix={raw}", headers=AUTH).status_code == 400
    # Double encoding stays a literal (non-existent) name inside the workspace.
    assert client.get("/api/files?prefix=%252e%252e%252f", headers=AUTH).json()["entries"] == []


def test_symlinks_escaping_the_workspace_are_not_listed_or_followed(client, workspace):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / "secret-link.txt").symlink_to(outside / "secret.txt")
    (workspace / "inside-link").symlink_to(workspace / "outputs", target_is_directory=True)
    listed = names(entries(client))
    assert "escape" not in listed and "secret-link.txt" not in listed
    assert "inside-link" in listed
    assert entries(client, "escape/") == []
    assert names(entries(client, "inside-link/")) == ["demo", "model.joblib"]


def test_custom_state_dir_inside_workspace_is_hidden(workspace):
    with TestClient(create_app(workspace, TOKEN, state_dir=workspace / "jobstate"),
                    base_url="http://127.0.0.1") as client:
        assert "jobstate" not in names(entries(client))
        assert entries(client, "jobstate/") == []
        (workspace / "state-link").symlink_to(workspace / "jobstate", target_is_directory=True)
        assert "state-link" not in names(entries(client))
        assert entries(client, "state-link/") == []


def test_listing_is_limited(client, workspace):
    bulk = workspace / "bulk"
    bulk.mkdir()
    for index in range(FILE_LIMIT + 50):
        (bulk / f"f{index:04d}.txt").write_bytes(b"")
    items = entries(client, "bulk/")
    assert len(items) == FILE_LIMIT
    assert items[0]["name"] == "f0000.txt"
    assert client.get("/api/files", params={"prefix": "x" * 5000}, headers=AUTH).status_code == 422


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_unreadable_directory_returns_empty(client, workspace):
    locked = workspace / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert entries(client, "locked/") == []
    finally:
        locked.chmod(0o755)
