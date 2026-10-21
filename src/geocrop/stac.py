"""Provider-neutral STAC discovery (CDSE, Earth Search, Planetary Computer, ...).

Network access only on explicit call. The inventory records asset links; it does not download
imagery or persist credentials. Do not publish signed asset URLs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = ["item_id", "collection", "datetime", "cloud_cover", "platform", "orbit_state", "relative_orbit",
          "tile", "asset_key", "href", "media_type", "roles"]


def discover(endpoint: str, collection: str, bbox, start: str, end: str, output: str | Path,
             max_items: int = 1000, cloud_max: float | None = None) -> dict:
    if not endpoint.startswith("https://"):
        raise ValueError("STAC endpoint must use HTTPS")
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Expected bbox: west south east north")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError('Install optional STAC support: pip install "geocrop-engine[stac]"') from exc
    search = Client.open(endpoint).search(collections=[collection], bbox=list(bbox), datetime=f"{start}/{end}",
                                          max_items=max_items)
    rows = []
    for item in search.items():
        props = item.properties
        cloud = props.get("eo:cloud_cover")
        if cloud_max is not None and cloud is not None and float(cloud) > cloud_max:
            continue
        for key, asset in item.assets.items():
            rows.append({"item_id": item.id, "collection": item.collection_id, "datetime": props.get("datetime") or "",
                         "cloud_cover": "" if cloud is None else cloud, "platform": props.get("platform", ""),
                         "orbit_state": props.get("sat:orbit_state", ""),
                         "relative_orbit": props.get("sat:relative_orbit", ""),
                         "tile": props.get("s2:mgrs_tile", props.get("grid:code", "")),
                         "asset_key": key, "href": asset.href, "media_type": asset.media_type or "",
                         "roles": ",".join(asset.roles or [])})
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    meta = {"endpoint": endpoint, "collection": collection, "bbox": list(bbox), "start": start, "end": end,
            "items": len({r["item_id"] for r in rows}), "assets": len(rows), "max_items": max_items,
            "note": "Asset URLs may require authentication; this is an inventory, not a download."}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    return meta


