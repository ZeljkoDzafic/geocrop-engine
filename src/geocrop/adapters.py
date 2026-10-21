"""Neutral schema adapters: map any provider's parcel / reference fields to the engine contract."""
from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd


def read_mapping(path: str | Path | None) -> dict:
    """Two-column CSV (source_code,class_name) mapping declaration or field codes to model classes."""
    if not path:
        return {}
    with Path(path).open(encoding="utf8") as fh:
        rows = list(csv.reader(fh))
    if rows and rows[0][:2] == ["source", "class_name"]:
        rows = rows[1:]
    return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2 and r[0].strip()}


def normalize_parcels(source: str | Path, output: str | Path, id_field: str, declared_field: str | None = None,
                      mapping: dict | None = None, pseudonymise: bool = False, make_valid: bool = True) -> dict:
    """Write parcels as `parcel_id` (+ `declared_crop`) GeoPackage.

    pseudonymise=True replaces IDs with sequential codes and writes the lookup next to the output
    (keep that lookup private); geometries are repaired with make_valid when requested.
    """
    src = gpd.read_file(source)
    if id_field not in src:
        raise ValueError(f"Missing parcel ID field {id_field!r}")
    if src.crs is None:
        raise ValueError("Parcels need a CRS")
    out = gpd.GeoDataFrame({"parcel_id": src[id_field].astype(str)}, geometry=src.geometry, crs=src.crs)
    if out.parcel_id.duplicated().any():
        raise ValueError("Duplicate parcel IDs")
    if declared_field:
        if declared_field not in src:
            raise ValueError(f"Missing declaration field {declared_field!r}")
        codes = src[declared_field].astype(str)
        out["declared_crop"] = codes.map(lambda c: (mapping or {}).get(c, c))
    repaired = 0
    if make_valid:
        bad = ~out.geometry.is_valid
        repaired = int(bad.sum())
        if repaired:
            out.loc[bad, "geometry"] = out.loc[bad, "geometry"].make_valid()
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if pseudonymise:
        lookup = dest.with_name(dest.stem + "_id_lookup.csv")
        codes = [f"P{i:07d}" for i in range(len(out))]
        out[["parcel_id"]].assign(pseudonym=codes).to_csv(lookup, index=False)
        out["parcel_id"] = codes
    dest.unlink(missing_ok=True)
    out.to_file(dest, driver="GPKG" if dest.suffix.lower() == ".gpkg" else "GeoJSON")
    return {"path": str(dest), "parcels": int(len(out)), "repaired_geometries": repaired,
            "pseudonymised": pseudonymise}
