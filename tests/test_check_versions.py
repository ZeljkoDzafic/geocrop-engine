"""tools/check_versions.py keeps pyproject, CITATION.cff, geocrop.__version__ and CHANGELOG in step."""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_versions", ROOT / "tools/check_versions.py")
check_versions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_versions)


@pytest.fixture
def tree(tmp_path):
    for name in ("pyproject.toml", "CITATION.cff", "CHANGELOG.md", "src/geocrop/__init__.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    return tmp_path


def test_repository_versions_are_consistent(capsys):
    assert check_versions.main([]) == 0
    versions = check_versions.collect()
    assert len(set(versions.values()) - {None}) == 1
    assert "versions consistent" in capsys.readouterr().out


@pytest.mark.parametrize("name,old,new", [
    ("pyproject.toml", 'version = "{v}"', 'version = "{v}9"'),
    ("CITATION.cff", "version: {v}", "version: {v}9"),
    ("src/geocrop/__init__.py", '__version__ = "{v}"', '__version__ = "{v}9"'),
])
def test_any_mismatch_fails(tree, name, old, new):
    version = check_versions.pyproject_version(tree)
    path = tree / name
    text = path.read_text(encoding="utf8")
    assert old.format(v=version) in text
    path.write_text(text.replace(old.format(v=version), new.format(v=version), 1), encoding="utf8")
    assert check_versions.main(["--root", str(tree)]) == 1


def test_unreleased_is_skipped_and_new_release_heading_must_match(tree):
    version = check_versions.pyproject_version(tree)
    assert check_versions.changelog_release(tree) == version
    changelog = tree / "CHANGELOG.md"
    changelog.write_text(changelog.read_text(encoding="utf8").replace(
        "## Unreleased", "## Unreleased\n\n## 99.0.0 — 2099-01-01\n### Added\n- Future.\n", 1), encoding="utf8")
    assert check_versions.changelog_release(tree) == "99.0.0"
    assert check_versions.main(["--root", str(tree)]) == 1


def test_tag_check_and_release_notes(tree, tmp_path):
    version = check_versions.pyproject_version(tree)
    notes = tmp_path / "notes.md"
    assert check_versions.main(["--root", str(tree), "--tag", f"v{version}", "--notes", str(notes)]) == 0
    body = notes.read_text(encoding="utf8")
    assert body.strip() and not body.startswith("## ") and "\n## " not in body
    assert check_versions.main(["--root", str(tree), "--tag", "v0.0.1"]) == 1
    assert check_versions.main(["--root", str(tree), "--tag", f"refs/tags/v{version}"]) == 0
    with pytest.raises(ValueError):
        check_versions.changelog_section(tree, "0.0.1")
