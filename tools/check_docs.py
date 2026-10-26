#!/usr/bin/env python3
"""Check that the documentation agrees with the repository and the code.

Checks:

* **links** - every relative link in the Markdown documents (and ``href``/``src`` in ``website/index.html``)
  points to an existing file; ``#anchors`` into Markdown files must match a heading (GitHub slug rules).
* **commands** - every ``geocrop <command> [<subcommand>]`` written in code (fenced blocks and inline code) of
  the living documents exists in the CLI parser, and every ``--option`` on the same command line is accepted
  by that command.
* **modules** - every module classified in ``geocrop._layers.LAYERS`` is named in ``docs/architecture.md``.
* **config keys** - every key listed in the table of ``docs/configuration.md`` exists in
  ``geocrop.config.DEFAULTS``, and every default key is listed there (the nested marker/rule groups in
  ``CONFIG_DOCUMENTED_ELSEWHERE`` are documented in their own pages).

Historical records (CHANGELOG, dated reviews, release-gate and acceptance reports) are checked for links only:
they describe the code as it was when they were written.

Usage: ``python tools/check_docs.py [--root DIR]``; exit status 0 when consistent, 1 otherwise.
Run with the package importable (``pip install -e .`` or ``PYTHONPATH=src``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Markdown documents checked for links (glob patterns relative to the root).
LINK_DOCS = ("*.md", "docs/**/*.md", "deploy/*.md", "website/*.md", "examples/**/*.md", "integrations/**/*.md",
             "configs/**/*.md")
#: Documents that are dated records; command/option names in them are not checked.
HISTORICAL = ("CHANGELOG.md", "docs/reviews/", "docs/release-gates/", "docs/acceptance/")
#: HTML pages whose relative href/src targets must exist.
HTML_PAGES = ("website/index.html",)

#: Config sections whose individual keys are documented on their own pages (markers.md, temporal-quality.md),
#: or which hold user-chosen names (windows, experiments).
CONFIG_DOCUMENTED_ELSEWHERE = ("cbm.events", "cbm.rules", "cbm.temporal_support", "windows", "experiments")

FENCE = re.compile(r"^(```|~~~)")
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
HTML_REF = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""")
INLINE_CODE = re.compile(r"`([^`]+)`")
COMMAND = re.compile(r"(?:^|(?<=[\s(`'\"]))(?<!from )(?<!import )geocrop\s+([a-z][a-z0-9-]*)"
                     r"((?:\s+(?!geocrop\b)[^\s|;&#)`+]+)*)")
OPTION = re.compile(r"(?:^|\s)(--[a-z][a-z0-9-]*)")


def _slug(heading: str) -> str:
    text = re.sub(r"`|\*\*|__|\[|\]\([^)]*\)", "", heading.strip()).lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf8").splitlines():
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            slug = _slug(match.group(1))
            n = seen.get(slug, 0)
            out.add(slug if n == 0 else f"{slug}-{n}")
            seen[slug] = n + 1
    out.update(re.findall(r"""<a\s+(?:name|id)=["']([^"']+)["']""", path.read_text(encoding="utf8")))
    return out


def _prose_and_code(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split a Markdown text into prose lines and code snippets (fenced lines joined over ``\\``, inline code)."""
    prose, code = [], []
    in_fence = False
    pending = ""
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            pending = ""
            continue
        if in_fence:
            stripped = re.sub(r"(^|\s)#.*$", "", line).rstrip()      # shell comments are prose
            if stripped.endswith("\\"):
                pending += stripped[:-1] + " "
                continue
            code.append((number, pending + stripped))
            pending = ""
            continue
        prose.append((number, INLINE_CODE.sub(" ", line)))
        code.extend((number, span) for span in INLINE_CODE.findall(line))
    return prose, code


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _doc_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in LINK_DOCS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def check_link(source: Path, target: str, root: Path) -> str | None:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("//"):
        return None                                   # absolute URL, mailto:, ...
    target = target.split("?", 1)[0]
    path_part, _, anchor = target.partition("#")
    dest = source if not path_part else (source.parent / path_part).resolve()
    if path_part.startswith("/"):
        return f"root-relative link {target!r} does not work on GitHub"
    if not dest.exists():
        return f"broken link {target!r}"
    try:
        dest.relative_to(root.resolve() if dest.is_absolute() else root)
    except ValueError:
        return f"link {target!r} leaves the repository"
    if anchor and dest.suffix == ".md" and anchor.lower() not in _anchors(dest):
        return f"missing anchor {target!r}"
    return None


def check_links(root: Path) -> list[str]:
    errors = []
    for doc in _doc_files(root):
        prose, _ = _prose_and_code(doc.read_text(encoding="utf8"))
        for number, line in prose:
            for match in LINK.finditer(line):
                target = match.group(1) or match.group(2)
                problem = check_link(doc, target, root)
                if problem:
                    errors.append(f"{_rel(doc, root)}:{number}: {problem}")
    for page in HTML_PAGES:
        path = root / page
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            for target in HTML_REF.findall(line):
                if target.startswith("#") or target.startswith("data:"):
                    continue
                problem = check_link(path, target, root)
                if problem:
                    errors.append(f"{page}:{number}: {problem}")
    return errors


def cli_commands() -> dict[str, tuple[set[str], dict[str, set[str]]]]:
    """``{command: (options, {subcommand: options})}`` from the argparse definitions."""
    from geocrop.cli import build_parser

    def options(parser: argparse.ArgumentParser) -> set[str]:
        return {s for a in parser._actions for s in a.option_strings}

    def subparsers(parser: argparse.ArgumentParser):
        return next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)

    out = {}
    for name, parser in subparsers(build_parser()).choices.items():
        nested = subparsers(parser)
        out[name] = (options(parser), {k: options(v) for k, v in (nested.choices.items() if nested else ())})
    return out


def check_commands(root: Path, commands: dict | None = None) -> list[str]:
    commands = cli_commands() if commands is None else commands
    errors = []
    for doc in _doc_files(root):
        rel = _rel(doc, root)
        if any(rel == h or rel.startswith(h) for h in HISTORICAL):
            continue
        _, code = _prose_and_code(doc.read_text(encoding="utf8"))
        for number, snippet in code:
            for match in COMMAND.finditer(snippet):
                name, rest = match.group(1), match.group(2)
                if name not in commands:
                    errors.append(f"{rel}:{number}: unknown command 'geocrop {name}'")
                    continue
                allowed, nested = commands[name]
                tokens = rest.split()
                if nested and tokens and tokens[0] in nested:
                    allowed = allowed | nested[tokens[0]]
                elif nested and tokens and re.fullmatch(r"[a-z][a-z0-9-]*", tokens[0]):
                    errors.append(f"{rel}:{number}: unknown command 'geocrop {name} {tokens[0]}'")
                    continue
                for option in OPTION.findall(rest):
                    if option not in allowed:
                        errors.append(f"{rel}:{number}: 'geocrop {name}' has no option {option}")
    return errors


def _config_keys_from_table(text: str) -> list[tuple[int, str]]:
    """``(line, dotted key)`` for every key of the configuration table in docs/configuration.md."""
    keys = []
    section = ""
    for number, line in enumerate(text.splitlines(), 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) < 4 or set(cells[0]) <= {"-"} and cells[0] or cells[0] == "Section":
            continue
        if cells[0]:
            section = "" if cells[0] == "(top)" else cells[0].strip("`")
        key_cell = cells[1]
        for name in re.findall(r"`([^`]+)`", key_cell):
            if name.startswith("<"):
                continue                               # free-form names such as windows.<name>
            if ":" in name:                           # `smoothing: {enabled, window, smoothness}`
                head, _, inner = name.partition(":")
                for sub in re.findall(r"[a-z_0-9]+", inner):
                    keys.append((number, ".".join(p for p in (section, head.strip(), sub) if p)))
                continue
            keys.append((number, ".".join(p for p in (section, name) if p)))
    return keys


def check_config_keys(root: Path) -> list[str]:
    from geocrop import config

    errors = []
    doc = root / "docs/configuration.md"
    documented = _config_keys_from_table(doc.read_text(encoding="utf8"))
    for number, dotted in documented:
        node = config.DEFAULTS
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                errors.append(f"docs/configuration.md:{number}: config key '{dotted}' is not in geocrop.config.DEFAULTS")
                break
            node = node[part]
    names = {dotted for _, dotted in documented}

    def leaves(node: dict, prefix: str = ""):
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                yield from leaves(value, dotted + ".")
            else:
                yield dotted

    for dotted in leaves(config.DEFAULTS):
        if dotted in names or any(dotted == g or dotted.startswith(g + ".") for g in CONFIG_DOCUMENTED_ELSEWHERE):
            continue
        errors.append(f"docs/configuration.md: config key '{dotted}' (geocrop.config.DEFAULTS) is not documented")
    return errors


def check_modules(root: Path) -> list[str]:
    from geocrop import _layers

    text = (root / "docs/architecture.md").read_text(encoding="utf8")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return [f"docs/architecture.md: module '{module}' ({layer} layer in geocrop._layers) is not mentioned"
            for layer, modules in _layers.LAYERS.items() for module in sorted(modules) if module not in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = check_links(root) + check_commands(root) + check_config_keys(root) + check_modules(root)
    for error in errors:
        print(error)
    print(f"docs check: {len(errors)} problem(s)" if errors else "docs consistent")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
