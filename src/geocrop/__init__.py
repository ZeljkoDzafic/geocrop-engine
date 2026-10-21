"""GeoCrop Engine: open Sentinel-1/Sentinel-2 crop-type classification and parcel-level verification."""
from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("geocrop-engine")
except _metadata.PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.7.0"

__all__ = ["__version__"]
