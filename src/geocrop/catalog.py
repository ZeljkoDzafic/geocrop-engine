"""Copernicus Data Space Ecosystem (CDSE) catalogue inventory and authenticated product download.

The inventory is the first experimental step: it records every candidate product for the AOI and
season (date, tile / relative orbit, orbit direction, cloud cover, size) so that window boundaries
and storage needs are decided from real availability, not assumptions.

Network access happens only when these functions are called explicitly. Credentials come from the
environment (CDSE_USERNAME / CDSE_PASSWORD, or a ready CDSE_ACCESS_TOKEN) and are never written to disk.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import requests

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC = "https://stac.dataspace.copernicus.eu/v1"

PRODUCT_TYPES = {"SENTINEL-2": "S2MSI2A", "SENTINEL-1": "IW_GRDH_1S"}
FIELDS = ["id", "name", "collection", "product_type", "sensing_start", "tile", "relative_orbit",
          "orbit_direction", "cloud_cover", "bytes", "online"]


def _attr(product: dict, name: str):
    for a in product.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return None


def build_filter(aoi_wkt: str, start: str, end: str, collection: str, product_type: str | None = None,
                 cloud_max: float | None = None) -> str:
    wkt = aoi_wkt.replace("'", "''")
    parts = [f"Collection/Name eq '{collection}'",
             f"ContentDate/Start ge {start}T00:00:00.000Z", f"ContentDate/Start le {end}T23:59:59.999Z",
             f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"]
    product_type = product_type if product_type is not None else PRODUCT_TYPES.get(collection)
    if product_type:
        parts.append("Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
                     f"att/OData.CSC.StringAttribute/Value eq '{product_type}')")
    if cloud_max is not None:
        parts.append("Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
                     f"att/OData.CSC.DoubleAttribute/Value le {float(cloud_max)})")
    return " and ".join(parts)


def inventory(aoi_wkt: str, start: str, end: str, output: str | Path, collection: str = "SENTINEL-2",
              product_type: str | None = None, cloud_max: float | None = None, max_products: int = 5000,
              session: requests.Session | None = None) -> dict:
    """Write a product manifest CSV (+ JSON summary with size forecast) for an AOI and date range.

    Tile-level cloud cover is only metadata: do not discard scenes on it alone, the AOI part
    of a cloudy tile may be clear (use cloud_max only for coarse pre-filtering).
    """
    http = session or requests.Session()
    flt = build_filter(aoi_wkt, start, end, collection, product_type, cloud_max)
    rows, skip, page = [], 0, 100
    while len(rows) < max_products:
        res = http.get(ODATA, params={"$filter": flt, "$top": min(page, max_products - len(rows)), "$skip": skip,
                                      "$orderby": "ContentDate/Start asc", "$expand": "Attributes"}, timeout=120)
        res.raise_for_status()
        batch = res.json().get("value", [])
        for p in batch:
            rows.append({"id": p.get("Id"), "name": p.get("Name"), "collection": collection,
                         "product_type": _attr(p, "productType"),
                         "sensing_start": (p.get("ContentDate") or {}).get("Start"),
                         "tile": _attr(p, "tileId"), "relative_orbit": _attr(p, "relativeOrbitNumber"),
                         "orbit_direction": _attr(p, "orbitDirection"), "cloud_cover": _attr(p, "cloudCover"),
                         "bytes": p.get("ContentLength"), "online": p.get("Online")})
        skip += len(batch)
        if len(batch) < page:
            break
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(int(r["bytes"] or 0) for r in rows)
    summary = {"collection": collection, "start": start, "end": end, "products": len(rows),
               "total_gb": round(total / 1e9, 2),
               "tiles_or_orbits": sorted({str(r["tile"] or r["relative_orbit"]) for r in rows}),
               "orbit_directions": sorted({str(r["orbit_direction"]) for r in rows if r["orbit_direction"]}),
               "truncated": len(rows) >= max_products}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


class TokenProvider:
    """CDSE OAuth2 access token with automatic refresh (access tokens expire after ~10 minutes)."""

    def __init__(self, username: str | None = None, password: str | None = None, token: str | None = None,
                 session: requests.Session | None = None):
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")
        self.static = token or os.environ.get("CDSE_ACCESS_TOKEN")
        self.http = session or requests.Session()
        self._token, self._refresh, self._expires = None, None, 0.0
        if not self.static and not (self.username and self.password):
            raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD (or CDSE_ACCESS_TOKEN)")

    def _request(self, data: dict) -> None:
        res = self.http.post(TOKEN_URL, data={"client_id": "cdse-public", **data}, timeout=60)
        res.raise_for_status()
        body = res.json()
        self._token, self._refresh = body["access_token"], body.get("refresh_token")
        self._expires = time.time() + float(body.get("expires_in", 600)) - 60

    def __call__(self, force: bool = False) -> str:
        if self.static and not (self.username and self.password):
            return self.static
        if force or self._token is None or time.time() >= self._expires:
            if self._refresh and not force:
                try:
                    self._request({"grant_type": "refresh_token", "refresh_token": self._refresh})
                    return self._token
                except requests.HTTPError:
                    pass
            self._request({"grant_type": "password", "username": self.username, "password": self.password})
        return self._token


def product_id(value) -> str:
    """Validate a CDSE product UUID (also prevents path/URL injection from manifests)."""
    return str(uuid.UUID(str(value)))


def download_product(pid: str, dest: str | Path, tokens: TokenProvider, chunk: int = 8 * 1024 * 1024,
                     retries: int = 5, session: requests.Session | None = None) -> dict:
    """Resumable download (HTTP Range) into `<dest>.part`, atomically renamed when complete."""
    pid = product_id(pid)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": True}
    part = dest.with_name(dest.name + ".part")
    http = session or requests.Session()
    url = f"{DOWNLOAD}({pid})/$value"
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Authorization": f"Bearer {tokens()}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with http.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as res:
                if res.status_code == 401:
                    tokens(force=True)
                    raise requests.HTTPError("401 Unauthorized (token refreshed)")
                res.raise_for_status()
                if offset and res.status_code == 200:      # server ignored Range: restart
                    offset = 0
                    part.unlink(missing_ok=True)
                length = res.headers.get("Content-Length")
                expected = int(length) + offset if length and length.isdigit() else None
                with part.open("ab" if offset else "wb") as fh:
                    for block in res.iter_content(chunk_size=chunk):
                        if block:
                            fh.write(block)
            if expected is not None and part.stat().st_size != expected:
                raise OSError(f"Truncated transfer: {part.stat().st_size} != {expected}")
            part.replace(dest)
            return {"path": str(dest), "bytes": dest.stat().st_size, "cached": False}
        except (requests.RequestException, OSError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("unreachable")


def checksum(path: str | Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_manifest(manifest: str | Path, directory: str | Path, limit: int | None = None,
                      tokens: TokenProvider | None = None) -> list[dict]:
    """Download every product of an inventory CSV; status is checkpointed after each product."""
    rows = list(csv.DictReader(Path(manifest).open(encoding="utf8")))[: limit or None]
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    tokens = tokens or TokenProvider()
    status = []
    for row in rows:
        pid = product_id(row["id"])
        name = Path(str(row.get("name") or pid)).name
        target = folder / (name if name.endswith(".zip") else name + ".zip")
        try:
            state = {"id": pid, "status": "completed", **download_product(pid, target, tokens)}
            state["sha256"] = checksum(target)
        except Exception as exc:  # noqa: BLE001 - recorded, the batch continues
            state = {"id": pid, "status": "failed", "error": str(exc)}
        status.append(state)
        (folder / "download_status.json").write_text(json.dumps(status, indent=2), encoding="utf8")
    return status
