"""Copernicus Sentinel attribution carried by every raster derived from Sentinel-1/2 data.

The Copernicus Sentinel data legal notice asks users of modified Sentinel data to state
"Contains modified Copernicus Sentinel data <year>". The engine writes it as the GeoTIFF tag
``GEOCROP_ATTRIBUTION`` (plus ``GEOCROP_DATA_LICENCE``) where Sentinel data enter (scene import from CDSE SAFE,
Earth Search STAC, Planetary Computer, openEO; SNAP outputs), carries it forward through composites, feature
stacks, time-series stacks and predictions (`propagate`), and exports it into COG tags and STAC items
(``license``/``providers``/``attribution``).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

TAG = "GEOCROP_ATTRIBUTION"
LICENCE_TAG = "GEOCROP_DATA_LICENCE"
COPERNICUS_LICENCE = ("Copernicus Sentinel data terms (free, full and open access): "
                      "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice")
_PREFIX = "Contains modified Copernicus Sentinel data"
_YEARS = re.compile(rf"{_PREFIX} ([0-9, \-]+)")


def _year_list(years) -> list[int]:
    out = set()
    for y in years:
        if y is None:
            continue
        if hasattr(y, "year"):
            out.add(int(y.year))
        else:
            text = str(y).strip()
            if text[:4].isdigit():
                out.add(int(text[:4]))
    return sorted(out)


def copernicus(years) -> str:
    """``"Contains modified Copernicus Sentinel data 2023"`` (several years: ``"2023, 2024"``).

    `years`: ints, ``date``/``datetime`` objects or ISO strings; an empty input gives the text without a year."""
    ys = _year_list(years if isinstance(years, Iterable) and not isinstance(years, (str, bytes)) else [years])
    return f"{_PREFIX} {', '.join(map(str, ys))}" if ys else _PREFIX


def years_of(text: str | None) -> list[int]:
    """Years named in an attribution text written by `copernicus`."""
    if not text:
        return []
    out = set()
    for match in _YEARS.finditer(text):
        out.update(int(t) for t in re.findall(r"\d{4}", match.group(1)))
    return sorted(out)


def tags(years) -> dict[str, str]:
    """GeoTIFF tags for Sentinel-derived data of `years`."""
    return {TAG: copernicus(years), LICENCE_TAG: COPERNICUS_LICENCE}


def read(paths) -> str | None:
    """Merged Copernicus attribution of existing rasters (None when none of them carries one)."""
    import rasterio

    found, years = False, set()
    for p in paths:
        if p is None or not Path(p).is_file():
            continue
        try:
            with rasterio.open(p) as ds:
                text = ds.tags().get(TAG)
        except Exception:      # unreadable side files never block a run because of attribution
            continue
        if text:
            found = True
            years.update(years_of(text))
    return copernicus(sorted(years)) if found else None


def propagate(sources, dst) -> str | None:
    """Copy the merged attribution of `sources` (raster paths) onto the open dataset `dst`; returns it."""
    text = read(sources)
    if text:
        dst.update_tags(**{TAG: text, LICENCE_TAG: COPERNICUS_LICENCE})
    return text


def folder_rasters(folder: str | Path, limit: int = 8) -> list[Path]:
    """A few GeoTIFFs of a scene/composite folder to read the attribution from."""
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(folder.glob("*.tif"))[:limit] if folder.is_dir() else []
