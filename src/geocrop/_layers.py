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
        "active_learning", "applicability", "cbm", "cbm_markers", "clouds", "contracts", "embeddings", "estimators",
        "faultinject", "features", "fieldwork", "ftw", "hashing",
        "indices", "metrics", "model", "parcels", "plugins", "presto_encoder", "raster", "s1", "s2", "samples", "scenes",
        "segmentation", "segmentation_metrics", "segmenters", "spatial", "splits", "timeseries", "tsfeatures",
        "uncertainty", "unet", "validation", "_layers",
        "calendars",     # data-driven crop calendars and phenological windows
        "hcat",          # EuroCrops HCAT class harmonisation
        "parcel_classify", "temporal_models",
        "review",        # analyst parcel review: queue, evidence, append-only decisions, label export
        "feature_providers",  # foundation-model feature-provider protocol + built-in providers (no network)
        "galileo_encoder",    # Galileo encoder port (torch, optional)
        "anysat_encoder",     # AnySat encoder port (torch, optional)
        "temporal_quality",  # observed vs interpolated support, temporal-support routing rules
    }),
    # The outside world: data providers, external tools, exchange formats, remote clients.
    ADAPTERS: frozenset({
        "adapters",      # LPIS / national schema mapping
        "catalog",       # Copernicus Data Space (OData, HTTP)
        "client",        # HTTP client for the web workspace
        "embedding_sources",  # AlphaEarth / TESSERA windowed reads, pinned Presto checkpoint
        "export",        # COG, GeoParquet, STAC items, HCAT
        "ftw_weights",   # Fields of The World checkpoint download (HTTPS, sha256-pinned)
        "quicklook",     # STAC COG quicklooks
        "s1_stac",       # Sentinel-1 RTC from Planetary Computer STAC (signed COG reads)
        "snap",          # ESA SNAP gpt runner (subprocess)
        "stac",          # STAC inventory (pystac-client)
        "stac_import",   # STAC COG import
        "openeo_adapter",  # optional openEO back-end (CDSE): process graphs, batch jobs, contract conversion
        "public_data",   # runtime download of public reference datasets (EuroCrops, WorldCereal calendars)
        "datacube",      # external xarray/Zarr/NetCDF cubes and optional odc-stac loading <-> file contracts
    }),
    # Workflows that compose core and adapters: configuration, orchestration, provenance, studies.
    APPLICATION: frozenset({
        "acceptance", "audit", "benchmark", "cache", "cleanup", "config", "demo", "experiments", "pipeline", "preflight", "provenance",
        "public_benchmark",  # frozen-protocol public benchmark (EuroCrops + Sentinel-2 STAC)
        "evidence",      # per-parcel evidence packages (JSON + HTML + PNG) and batch index
        "registry",      # local file-based model and benchmark registry
    }),
    # Entry points. Nothing below this layer may import it.
    INTERFACE: frozenset({"cli", "web", "web_jobs", "web_map", "web_review"}),
}

#: Web modules may only be imported by the CLI (``geocrop serve``) and by each other.
WEB_MODULES = frozenset({"web", "web_jobs", "web_map", "web_review"})
WEB_IMPORTERS = frozenset({"cli"}) | WEB_MODULES

#: Third-party libraries core modules may use: the scientific/geospatial stack and optional ML backends.
CORE_THIRD_PARTY = frozenset({
    "numpy", "scipy", "pandas", "sklearn", "joblib", "rasterio", "geopandas", "shapely", "pyproj", "affine",
    "skimage", "torch", "lightgbm", "catboost", "omnicloudmask", "segmentation_models_pytorch",
    "matplotlib",   # optional, imported lazily for figures (active_learning.plot_curves)
    "pyarrow",      # Parquet schema reads without loading rows (review.discover)
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
