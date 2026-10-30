"""Generate the SYNTHETIC intake example (no real data): ``good/`` passes `geocrop intake`, ``broken/`` does not.

All coordinates, IDs, classes and dates are invented on a regular grid near Gradiska; nothing derives from any
authority's data. Run from the repository root: ``python examples/intake_example/make_example.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, box, mapping

HERE = Path(__file__).resolve().parent
X0, Y0, SIDE, CELL = 676000.0, 4984000.0, 8000.0, 400.0      # EPSG:32633, 8 x 8 km, 400 m cells
MUNICIPALITIES = [("Gradiška", 0), ("Laktaši", 1), ("Srbac", 2)]


def feature_collection(rows, crs="EPSG:32633"):
    return {"type": "FeatureCollection", "name": "synthetic", "crs": {"type": "name", "properties": {
        "name": f"urn:ogc:def:crs:EPSG::{crs.split(':')[1]}"}},
        "features": [{"type": "Feature", "properties": p, "geometry": mapping(g)} for p, g in rows]}


def _rounded(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, tuple):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    return value


def dump(path: Path, collection):
    path.parent.mkdir(parents=True, exist_ok=True)
    # coordinates rounded to 1e-6 (about 0.1 m): the last floating-point digits of reprojected values differ
    # between platforms and GEOS/PROJ builds, and the example files are compared byte for byte
    path.write_text(json.dumps(_rounded(collection), ensure_ascii=False, indent=0), encoding="utf8")


def town(x):
    return MUNICIPALITIES[min(int((x - X0) / (SIDE / 3)), 2)][0]


def build():
    rng = np.random.default_rng(7)
    cells = [(X0 + i * CELL, Y0 + j * CELL) for i in range(int(SIDE / CELL)) for j in range(int(SIDE / CELL))]
    order = rng.permutation(len(cells))
    take = iter(order)

    def cell():
        return cells[next(take)]

    training = []
    for cls in ("soybean", "sunflower", "maize", "grassland"):
        for n in range(15):
            x, y = cell()
            training.append(({"polygon_id": f"T{cls[:3].upper()}{n:02d}", "class_name": cls, "municipality": town(x)},
                             box(x + 100, y + 100, x + 200, y + 200)))
    parcels, observations = [], []
    declared = {"soybean": "SOJA", "sunflower": "SUNCOKRET", "maize": "KUKURUZ", "grassland": "TRAVA"}
    for cls, code in declared.items():
        for n in range(27):
            x, y = cell()
            pid = f"P{cls[:3].upper()}{n:02d}"
            parcels.append(({"parcel_id": pid, "declared_crop": code}, box(x + 80, y + 80, x + 320, y + 320)))
            if cls != "grassland":
                observations.append(({"class_name": cls, "observed_at": f"2023-{6 + n % 3:02d}-{10 + n % 15:02d}",
                                      "group_id": pid}, Point(x + 200, y + 200)))
    return training, parcels, observations


def write_aoi():
    to_ll = Transformer.from_crs(32633, 4326, always_xy=True)
    ring = [to_ll.transform(x, y) for x, y in [(X0 - 100, Y0 - 100), (X0 + SIDE + 100, Y0 - 100),
                                               (X0 + SIDE + 100, Y0 + SIDE + 100), (X0 - 100, Y0 + SIDE + 100),
                                               (X0 - 100, Y0 - 100)]]
    props = {"name": "SYNTHETIC example AOI", "municipalities": [m for m, _ in MUNICIPALITIES]}
    from shapely.geometry import Polygon
    dump(HERE / "aoi.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props, "geometry": mapping(Polygon(ring))}]})


def main():
    training, parcels, observations = build()
    write_aoi()
    mapping_csv = "source,class_name\nSOJA,soybean\nSUNCOKRET,sunflower\nKUKURUZ,maize\nTRAVA,grassland\n"
    good = HERE / "good"
    dump(good / "training.geojson", feature_collection(training))
    dump(good / "parcels.geojson", feature_collection(parcels))
    dump(good / "observations.geojson", feature_collection(observations))
    (good / "crop_mapping.csv").write_text(mapping_csv, encoding="utf8")

    # broken: every fault below is found by `geocrop intake` (see tests/test_intake.py)
    bad = HERE / "broken"
    t = [(dict(p), g) for p, g in training]
    t[0][0]["class_name"] = "Soybean"                       # case variant
    t[1][0]["class_name"] = "soybean "                      # trailing space
    t[2][0]["class_name"] = "Soja"                          # not a configured class
    t[3][0]["polygon_id"] = t[4][0]["polygon_id"]           # duplicate ID
    t[5] = (t[5][0], box(*t[5][1].bounds[:2], t[5][1].bounds[0] + 2, t[5][1].bounds[1] + 2))  # 2 m square: < 1 pixel
    x0, y0 = t[6][1].bounds[:2]
    from shapely.geometry import Polygon
    t[6] = (t[6][0], Polygon([(x0, y0), (x0 + 100, y0 + 100), (x0 + 100, y0), (x0, y0 + 100), (x0, y0)]))  # bow-tie
    t[7] = (t[7][0], box(X0 + 30000, Y0, X0 + 30100, Y0 + 100))   # far outside the AOI
    dump(bad / "training.geojson", feature_collection(t))
    p = [(dict(q), g) for q, g in parcels]
    for i in range(6):
        p[i][0]["declared_crop"] = "NEPOZNATO"              # code missing from the mapping
    dump(bad / "parcels.geojson", feature_collection(p))
    o = [(dict(q), g) for q, g in observations]
    o[0][0]["observed_at"] = "15.06.2023."                  # not ISO
    o[1][0]["observed_at"] = "2022-05-01"                   # outside the season
    o[2] = (o[2][0], Point(training[0][1].centroid.x, training[0][1].centroid.y))  # inside a training polygon
    dump(bad / "observations.geojson", feature_collection(o))
    (bad / "crop_mapping.csv").write_text("source,class_name\nSOJA,soybean\nSUNCOKRET,sunflower\nKUKURUZ,maize\n",
                                          encoding="utf8")

    # tabular exports as a Serbian Windows tool would write them: cp1250, semicolon, WKT / decimal-comma x,y
    tab = HERE / "tabular"
    tab.mkdir(exist_ok=True)
    names = {"SOJA": "SOJA", "SUNCOKRET": "SUNCOKRET", "KUKURUZ": "KUKURUZ", "TRAVA": "TRAVA"}
    rows = ["parcel_id;declared_crop;opština;wkt"]
    for q, g in parcels[:12]:
        rows.append(f"{q['parcel_id']};{names[q['declared_crop']]};Gradiška;{g.wkt}")
    rows.append("PXSEC;ŠEĆERNA REPA;Laktaši;" + parcels[12][1].wkt)
    (tab / "parcels_cp1250.csv").write_bytes("\n".join(rows).encode("cp1250"))
    (tab / "crop_mapping_cp1250.csv").write_bytes(
        (mapping_csv + "ŠEĆERNA REPA,sugar_beet\nŽITARICE,winter_crops\n").encode("cp1250"))
    obs = ["klasa;datum;grupa;x;y"]
    for q, g in observations[:10]:
        xs, ys = f"{g.x:.1f}".replace(".", ","), f"{g.y:.1f}".replace(".", ",")
        obs.append(f"{q['class_name']};{q['observed_at'][8:]}.{q['observed_at'][5:7]}.2023.;{q['group_id']};{xs};{ys}")
    (tab / "observations.csv").write_bytes("\n".join(obs).encode("cp1250"))


if __name__ == "__main__":
    main()
