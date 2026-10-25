"""Build a standalone QGIS 'Install from ZIP' package without requiring QGIS or the engine."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PLUGIN = Path(__file__).resolve().parents[1] / "integrations/qgis/geocrop_qgis"


def build(destination: Path) -> Path:
    destination = Path(destination)
    files = [PLUGIN / name for name in ("__init__.py", "plugin.py", "provider.py", "algorithms.py",
                                       "transport.py", "metadata.txt", "icon.svg", "README.txt")]
    if not (PLUGIN / "metadata.txt").is_file() or not (PLUGIN / "__init__.py").is_file():
        raise ValueError("QGIS plugin source/metadata is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"geocrop_qgis/{path.name}")
        for name in ("LICENSE", "NOTICE"):
            archive.write(PLUGIN.parents[2] / name, f"geocrop_qgis/{name}.txt")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New ZIP path; existing files are preserved")
    print(build(parser.parse_args().out))


if __name__ == "__main__":
    main()
