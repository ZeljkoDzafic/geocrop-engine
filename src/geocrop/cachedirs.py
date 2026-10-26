"""One download cache for everything the engine fetches at run time.

The root is ``$GEOCROP_CACHE``, else ``$XDG_CACHE_HOME/geocrop``, else ``~/.cache/geocrop``. Every kind of
download has a fixed subfolder:

==================  =====================================================================
``ftw/``            Fields of The World checkpoints (``<version>/<file>``)
``presto/``         Presto checkpoint (and the Presto feature provider's weights)
``galileo/``        Galileo weights (feature provider)
``anysat/``         AnySat weights (feature provider)
``<provider>/``     weights of any other (plugin) feature provider
``alphaearth/``     AlphaEarth tile index (``aef_index.parquet``)
``omnicloudmask/``  OmniCloudMask weights (optional ``[cloud]`` extra)
``public/``         public reference datasets: EuroCrops (``eurocrops/``), WorldCereal calendars
==================  =====================================================================

The former per-component variables ``GEOCROP_FTW_CACHE`` (FTW) and ``GEOCROP_DATA_CACHE`` (public datasets)
are deprecated aliases: when set they still win for their subfolder, with a `FutureWarning` (shown to
users by default).

Pure path logic (environment variables only), so core modules can resolve cache paths without network code.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

ENV = "GEOCROP_CACHE"
#: subfolder -> deprecated environment variable that used to point directly at that subfolder
DEPRECATED_ALIASES = {"ftw": "GEOCROP_FTW_CACHE", "public": "GEOCROP_DATA_CACHE"}
SUBFOLDERS = ("ftw", "presto", "galileo", "anysat", "alphaearth", "omnicloudmask", "public")


def cache_root() -> Path:
    """``$GEOCROP_CACHE``, else ``$XDG_CACHE_HOME/geocrop``, else ``~/.cache/geocrop``."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "geocrop"


def cache_dir(sub: str) -> Path:
    """Cache subfolder `sub` (not created). A set deprecated alias for that subfolder still wins, with a warning."""
    if not sub or "/" in sub or "\\" in sub or sub in (".", ".."):
        raise ValueError(f"Invalid cache subfolder {sub!r}")
    legacy = DEPRECATED_ALIASES.get(sub)
    if legacy and os.environ.get(legacy):
        warnings.warn(f"${legacy} is deprecated; set ${ENV} (the cache root) instead, this folder is then "
                      f"${ENV}/{sub}", FutureWarning, stacklevel=2)
        return Path(os.environ[legacy]).expanduser()
    return cache_root() / sub
