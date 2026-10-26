"""Single source of truth for the package's layered architecture (see docs/adr/0001-layered-architecture.md).

Every module in ``src/geocrop`` belongs to exactly one layer. ``tests/test_architecture.py`` parses all
modules with ``ast`` and fails when an import crosses a layer boundary in the wrong direction or when a
core module imports a network/process library. A new module must be added here before the suite passes.

    interface    ──►  application  ──►  adapters  ──►  core
    (cli, web*)       (workflows)       (outside world)  (pure geospatial/ML on local files)

An arrow means "may import". Every layer may import modules of its own layer and the package root
(``geocrop.__version__``).
"""
from __future__ import annotations

CORE = "core"
ADAPTERS = "adapters"
APPLICATION = "application"
INTERFACE = "interface"

#: Which layers each layer may import from (besides itself).
ALLOWED: dict[str, frozenset[str]] = {
    CORE: frozenset({CORE}),
    ADAPTERS: frozenset({CORE, ADAPTERS}),
    APPLICATION: frozenset({CORE, ADAPTERS, APPLICATION}),
    INTERFACE: frozenset({CORE, ADAPTERS, APPLICATION, INTERFACE}),
}

LAYERS: dict[str, frozenset[str]] = {
    # Pure geospatial/ML logic. Reads and writes local files only; no network, no subprocesses.
    CORE: frozenset({
        "applicability", "cbm", "clouds", "contracts", "embeddings", "estimators", "features", "fieldwork", "hashing",
        "indices", "metrics", "model", "parcels", "plugins", "presto_encoder", "raster", "s1", "s2", "samples", "scenes",
        "segmentation", "segmentation_metrics", "segmenters", "spatial", "splits", "timeseries", "tsfeatures",
        "uncertainty", "unet", "validation", "_layers",
    }),
    # The outside world: data providers, external tools, exchange formats, remote clients.
    ADAPTERS: frozenset({
        "adapters",      # LPIS / national schema mapping
        "catalog",       # Copernicus Data Space (OData, HTTP)
        "client",        # HTTP client for the web workspace
        "embedding_sources",  # AlphaEarth / TESSERA windowed reads, pinned Presto checkpoint
        "export",        # COG, GeoParquet, STAC items, HCAT
        "quicklook",     # STAC COG quicklooks
        "snap",          # ESA SNAP gpt runner (subprocess)
        "stac",          # STAC inventory (pystac-client)
        "stac_import",   # STAC COG import
    }),
    # Workflows that compose core and adapters: configuration, orchestration, provenance, studies.
    APPLICATION: frozenset({
        "audit", "benchmark", "cache", "config", "demo", "experiments", "pipeline", "provenance",
    }),
    # Entry points. Nothing below this layer may import it.
    INTERFACE: frozenset({"cli", "web", "web_jobs", "web_map"}),
}

#: Web modules may only be imported by the CLI (``geocrop serve``) and by each other.
WEB_MODULES = frozenset({"web", "web_jobs", "web_map"})
WEB_IMPORTERS = frozenset({"cli"}) | WEB_MODULES

#: Third-party libraries core modules may use: the scientific/geospatial stack and optional ML backends.
CORE_THIRD_PARTY = frozenset({
    "numpy", "scipy", "pandas", "sklearn", "joblib", "rasterio", "geopandas", "shapely", "pyproj", "affine",
    "skimage", "torch", "lightgbm", "catboost", "omnicloudmask",
})

#: Standard-library modules core must not use: network access and process execution belong in adapters.
CORE_FORBIDDEN_STDLIB = frozenset({
    "subprocess", "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "xmlrpc", "webbrowser",
})

#: Network/web libraries named explicitly so that failure messages are specific.
NETWORK_LIBRARIES = frozenset({
    "requests", "httpx", "aiohttp", "urllib3", "pystac_client", "fastapi", "starlette", "uvicorn", "pydantic",
})

#: Deliberate exceptions, each with a reason. Keys are (importer, imported) and the import must be lazy
#: (inside a function), so the importer's module still loads without the imported dependency.
EXCEPTIONS: dict[tuple[str, str], str] = {
    ("s1", "snap"): "backward-compatible `geocrop.s1.run_snap` alias; SNAP execution lives in the snap adapter",
}


def layer_of(module: str) -> str | None:
    """Layer of a top-level ``geocrop`` module name, or None when it is unclassified."""
    for layer, modules in LAYERS.items():
        if module in modules:
            return layer
    return None
