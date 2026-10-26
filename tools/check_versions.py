#!/usr/bin/env python3
"""Check that every place that states the release version agrees.

Compared: ``pyproject.toml`` ``[project] version``, ``CITATION.cff`` ``version``, the source fallback and the
importable ``geocrop.__version__``, and the latest released heading in ``CHANGELOG.md`` (``## X.Y.Z``; the
``## Unreleased`` section is skipped). With ``--tag vX.Y.Z`` the Git tag must match too. With
``--notes FILE`` the matching CHANGELOG section is written to FILE (used for GitHub Release notes).

Exit status 0 when consistent, 1 otherwise. Standard library only.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = r"\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|dev|post)\.?\d+)?"
HEADING = re.compile(rf"^## \[?v?({SEMVER})\]?(?:\s|$)", re.MULTILINE)


def pyproject_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf8")
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    match = re.search(r'^version\s*=\s*"([^"]+)"', project.group(1) if project else "", re.MULTILINE)
    if not match:
        raise ValueError("pyproject.toml has no [project] version")
    return match.group(1)


def citation_version(root: Path) -> str:
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", (root / "CITATION.cff").read_text(encoding="utf8"),
                      re.MULTILINE)
    if not match:
        raise ValueError("CITATION.cff has no version")
    return match.group(1)


def source_fallback_version(root: Path) -> str:
    text = (root / "src/geocrop/__init__.py").read_text(encoding="utf8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError("src/geocrop/__init__.py has no fallback __version__")
    return match.group(1)


def changelog_release(root: Path) -> str:
    match = HEADING.search((root / "CHANGELOG.md").read_text(encoding="utf8"))
    if not match:
        raise ValueError("CHANGELOG.md has no released '## X.Y.Z' heading")
    return match.group(1)


def changelog_section(root: Path, version: str) -> str:
    """Body of the ``## <version>`` section, without its heading."""
    text = (root / "CHANGELOG.md").read_text(encoding="utf8")
    for match in HEADING.finditer(text):
        if match.group(1) == version:
            start = text.index("\n", match.start()) + 1 if "\n" in text[match.start():] else len(text)
            following = re.compile(r"^## ", re.MULTILINE).search(text, start)
            return text[start:following.start() if following else len(text)].strip() + "\n"
    raise ValueError(f"CHANGELOG.md has no section for {version}")


def imported_version(root: Path) -> str | None:
    sys.path.insert(0, str(root / "src"))
    try:
        module = importlib.import_module("geocrop")
        return getattr(module, "__version__", None)
    except ImportError:
        return None
    finally:
        sys.path.pop(0)


def collect(root: Path = ROOT) -> dict[str, str | None]:
    return {
        "pyproject.toml": pyproject_version(root),
        "CITATION.cff": citation_version(root),
        "geocrop/__init__.py fallback": source_fallback_version(root),
        "geocrop.__version__": imported_version(root),
        "CHANGELOG.md latest release": changelog_release(root),
    }


def problems(versions: dict[str, str | None], tag: str | None = None) -> list[str]:
    expected = versions["pyproject.toml"]
    found = [f"{name} is {value} (expected {expected})" for name, value in versions.items()
             if value is not None and value != expected]
    if tag is not None and tag.removeprefix("refs/tags/").removeprefix("v") != expected:
        found.append(f"tag {tag} does not match version {expected}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="Git tag that must equal v<version>, e.g. v1.0.0")
    parser.add_argument("--notes", type=Path, help="write the CHANGELOG section of this version here")
    args = parser.parse_args(argv)
    try:
        versions = collect(args.root)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, value in versions.items():
        print(f"{name:32s} {value if value is not None else '(not importable, skipped)'}")
    errors = problems(versions, args.tag)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    if args.notes:
        try:
            args.notes.write_text(changelog_section(args.root, versions["pyproject.toml"]), encoding="utf8")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    print("versions consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
