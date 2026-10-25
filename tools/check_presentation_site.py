"""Check presentation-site assets, local links and captured screenshot provenance (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1] / "website"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references = set(), []
        self.errors = []
        self.english = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "html":
            self.english = values.get("lang") == "en"
        if "id" in values:
            if values["id"] in self.ids:
                self.errors.append(f"Duplicate id: {values['id']}")
            self.ids.add(values["id"])
        if tag == "img" and "alt" not in values:
            self.errors.append("Image without alternative text")
        for key in ("href", "src", "data-image"):
            if key in values:
                self.references.append(values[key])
        for key in ("aria-controls", "aria-labelledby"):
            for target in values.get(key, "").split():
                self.references.append("#" + target)


def check() -> list[str]:
    parser = Page()
    parser.feed((ROOT / "index.html").read_text(encoding="utf8"))
    errors = list(parser.errors)
    if not parser.english:
        errors.append("Presentation page must declare English")
    for ref in parser.references:
        parsed = urlsplit(ref)
        if parsed.scheme or parsed.netloc:
            continue
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in parser.ids:
                errors.append(f"Unknown anchor: {ref}")
            continue
        path = (ROOT / unquote(parsed.path)).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append(f"Missing/escaping local asset: {ref}")
    evidence = json.loads((ROOT / "assets/screenshots.json").read_text())
    if {item["interface"] for item in evidence["screenshots"]} != {"web", "cli", "qgis"}:
        errors.append("Web, CLI and QGIS screenshot evidence is required")
    for item in evidence["screenshots"]:
        path = (ROOT / item["file"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            errors.append(f"Missing screenshot: {item['interface']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Screenshot provenance hash mismatch: {item['interface']}")
        if not item.get("capture_method") or not item.get("source_commit"):
            errors.append(f"Missing capture/source evidence: {item['interface']}")
    with ZipFile(ROOT / "assets/downloads/geocrop-qgis-0.1.0.zip") as plugin:
        if plugin.testzip() is not None or "geocrop_qgis/metadata.txt" not in plugin.namelist():
            errors.append("Invalid QGIS plugin archive")
    return errors


if __name__ == "__main__":
    issues = check()
    for issue in issues:
        print(issue)
    if issues:
        raise SystemExit(1)
    print("Presentation page, local links, screenshots and QGIS download passed.")
