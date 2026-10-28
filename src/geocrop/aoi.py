"""AOI-aware processing: the footprint of an irregular AOI on the reference grid and the blocks it touches.

The reference grid is the bounding box of the AOI, so for an irregular AOI most of its pixels lie outside it
(Lijevce polje: 3.7 x the AOI area). `geocrop grid` therefore also writes an AOI mask raster next to the grid
(``<grid>_aoi.tif``: uint8, 1 inside the AOI, 0 outside) and a block index (``<grid>_aoi.json``). Stages that
read or compute per block use :class:`AoiMask` to skip every block that cannot influence a pixel inside the
AOI and write nothing (``SPARSE_OK``: unwritten tiles read as no-data) for the rest of the grid.

Two granularities are used, and the difference matters for correctness (docs/aoi-aware.md):

* *active blocks* (intermediate products: scenes, composites, feature stacks): every block whose window,
  grown by the halo, contains an AOI pixel. They are computed completely, including the pixels of such a
  block that lie outside the AOI, because neighbourhood operations (probability smoothing) at the AOI border
  read them.
* *AOI pixels* (final products: classes, probabilities, uncertainty, AOA map): the mask itself; everything
  outside is no-data.

Without a mask nothing in this module is used and every stage behaves exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .hashing import sha256
from .raster import RasterInputError, blocks, expand

SUFFIX = "_aoi"
BLOCK = 512                       # block size of the block index (stages with other block sizes compute their own)
SCHEMA = "geocrop-aoi/1"


def default_mask_path(reference: str | Path) -> Path:
    """Where `geocrop grid` writes the mask and where it is auto-discovered: ``<grid stem>_aoi.tif``."""
    ref = Path(reference)
    return ref.with_name(ref.stem + SUFFIX + ".tif")


def index_path(mask: str | Path) -> Path:
    return Path(mask).with_suffix(".json")


def creation_options(aoi: AoiMask | None) -> dict:
    """GDAL creation options for rasters written block by block under a mask: blocks that are never written
    (or are entirely no-data) take no disk space and read back as no-data."""
    return {"SPARSE_OK": "TRUE"} if aoi is not None else {}


class AoiMask:
    """Read access to an AOI mask raster (non-zero = inside) with block-level questions.

    `halo` is the number of pixels around the AOI that intermediate products must also hold (smoothing radius,
    window size); `active` / `windows` use it unless told otherwise.
    """

    def __init__(self, path: str | Path, halo: int = 0):
        path = Path(path)
        if not path.is_file():
            raise RasterInputError(f"AOI mask {path} does not exist (create it with geocrop grid --aoi ...)")
        if int(halo) < 0:
            raise ValueError("AOI halo must be >= 0 pixels")
        try:
            with rasterio.open(path) as ds:
                self.crs, self.transform, self.width, self.height = ds.crs, ds.transform, ds.width, ds.height
                self.count = ds.count
        except rasterio.errors.RasterioIOError as exc:
            raise RasterInputError(f"AOI mask {path} cannot be opened as a raster: {exc}") from exc
        if self.count != 1:
            raise RasterInputError(f"AOI mask {path} must have one band, has {self.count}")
        self.path, self.halo = path, int(halo)
        self._dense: np.ndarray | None = None
        self._counts: dict | None = None
        self._ds = None
        self._identity: str | None = None
        self._windows: dict = {}

    def __repr__(self) -> str:
        return f"AoiMask({str(self.path)!r}, halo={self.halo})"

    def identity(self) -> str:
        """Content digest of the mask and its halo: what a cached scene or a stage fingerprint depends on."""
        if self._identity is None:
            self._identity = sha256(self.path)[:20]
        return f"{self._identity}:h{self.halo}"

    def with_halo(self, halo: int) -> AoiMask:
        other = AoiMask(self.path, halo)
        other._dense, other._counts, other._identity = self._dense, self._counts, self._identity
        return other

    # ------------------------------------------------------------------ grid check
    def check_grid(self, ds, what: str = "raster") -> None:
        """Require the mask to lie on the grid of the open dataset `ds` (same CRS, shape and transform)."""
        problems = []
        if ds.crs != self.crs:
            problems.append(f"CRS {self.crs} != {ds.crs}")
        if (ds.height, ds.width) != (self.height, self.width):
            problems.append(f"shape {(self.height, self.width)} != {(ds.height, ds.width)}")
        if not ds.transform.almost_equals(self.transform):
            problems.append("transform differs")
        if problems:
            raise RasterInputError(f"AOI mask {self.path} is not on the grid of {what} {ds.name}: "
                                   f"{'; '.join(problems)}. Recreate it with geocrop grid for this study grid.")

    # ------------------------------------------------------------------ pixels
    def _open(self):
        if self._ds is None:
            self._ds = rasterio.open(self.path)
        return self._ds

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def read(self, window: Window) -> np.ndarray:
        """Boolean AOI membership of `window` (any size; pixels outside the raster are False)."""
        r0, c0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        out = np.zeros((h, w), dtype=bool)
        r1, c1 = min(self.height, r0 + h), min(self.width, c0 + w)
        top, left = max(0, r0), max(0, c0)
        if r1 <= top or c1 <= left:
            return out
        data = self._open().read(1, window=Window(left, top, c1 - left, r1 - top))
        out[top - r0:r1 - r0, left - c0:c1 - c0] = data > 0
        return out

    def dense(self) -> np.ndarray:
        """The whole mask as a boolean array (cached). Use only where a full-grid array exists anyway."""
        if self._dense is None:
            out = np.zeros((self.height, self.width), dtype=bool)
            for w in blocks(self.width, self.height, 2048):
                out[int(w.row_off):int(w.row_off + w.height), int(w.col_off):int(w.col_off + w.width)] = self.read(w)
            self._dense = out
        return self._dense

    # ------------------------------------------------------------------ blocks
    def active(self, window: Window, halo: int | None = None) -> bool:
        """True when `window` grown by `halo` pixels (default: this mask's halo) contains an AOI pixel."""
        h = self.halo if halo is None else int(halo)
        return bool(self.read(expand(window, h) if h else window).any())

    def windows(self, size: int = BLOCK, halo: int | None = None) -> list[Window]:
        """The `size` x `size` blocks of the grid that are active (cached per size and halo)."""
        key = (size, self.halo if halo is None else int(halo))
        if key not in self._windows:
            self._windows[key] = [w for w in blocks(self.width, self.height, size) if self.active(w, key[1])]
        return list(self._windows[key])

    def block_index(self, size: int = BLOCK, halo: int | None = None) -> list[list[int]]:
        """``[[block_row, block_col], ...]`` of the active blocks (row/column counted in blocks of `size`)."""
        return [[int(w.row_off) // size, int(w.col_off) // size] for w in self.windows(size, halo)]

    def counts(self) -> dict:
        """AOI pixels and grid pixels."""
        if self._counts is None:
            n = sum(int(self.read(w).sum()) for w in blocks(self.width, self.height, 2048))
            self._counts = {"aoi_pixels": n, "grid_pixels": self.width * self.height}
        return self._counts

    def work_pixels(self, size: int = BLOCK, halo: int | None = None) -> int:
        """Pixels inside the active blocks: what the stages actually read and compute."""
        return int(sum(int(w.width) * int(w.height) for w in self.windows(size, halo)))

    def summary(self, size: int = BLOCK, halo: int | None = None) -> dict:
        c = self.counts()
        active = self.windows(size, halo)
        total = -(-self.width // size) * -(-self.height // size)
        work = int(sum(int(w.width) * int(w.height) for w in active))
        res = abs(self.transform.a * self.transform.e) ** 0.5
        px_km2 = res * res / 1e6
        return {"aoi_pixels": c["aoi_pixels"], "grid_pixels": c["grid_pixels"],
                "aoi_share_of_grid": round(c["aoi_pixels"] / c["grid_pixels"], 4) if c["grid_pixels"] else None,
                "aoi_km2": round(c["aoi_pixels"] * px_km2, 2), "grid_km2": round(c["grid_pixels"] * px_km2, 2),
                "block_px": size, "halo_px": self.halo if halo is None else int(halo),
                "blocks_total": total, "blocks_active": len(active), "work_pixels": work,
                "work_share_of_grid": round(work / c["grid_pixels"], 4) if c["grid_pixels"] else None}

    def footprint(self, halo: int | None = None, size: int = BLOCK):
        """Union of the active blocks as a shapely polygon in the grid CRS (for catalogue pre-filtering)."""
        from shapely.geometry import box
        from shapely.ops import unary_union
        parts = []
        for w in self.windows(size, halo):
            x0, y0 = self.transform * (w.col_off, w.row_off)
            x1, y1 = self.transform * (w.col_off + w.width, w.row_off + w.height)
            parts.append(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        return unary_union(parts)


# --------------------------------------------------------------------------- writing

def write_index(mask: AoiMask, size: int = BLOCK, halo: int | None = None) -> Path:
    """Write ``<mask stem>.json``: share of the grid inside the AOI and the active 512 px blocks."""
    body = {"schema": SCHEMA, "mask": mask.path.name, **mask.summary(size, halo),
            "blocks": mask.block_index(size, halo),
            "note": "blocks = [row, col] of the blocks (block_px) that contain an AOI pixel within halo_px; "
                    "pipeline stages derive their own active blocks from the mask with the configured "
                    "grid.aoi_halo_px (and the smoothing radius)."}
    path = index_path(mask.path)
    path.write_text(json.dumps(body, indent=2), encoding="utf8")
    return path


def rasterize(geometry, reference: str | Path, output: str | Path) -> Path:
    """Write the AOI mask of a shapely geometry (in the reference CRS) on the grid of `reference`."""
    from rasterio.features import geometry_mask
    with rasterio.open(reference) as ref:
        profile = {"driver": "GTiff", "width": ref.width, "height": ref.height, "count": 1, "dtype": "uint8",
                   "crs": ref.crs, "transform": ref.transform, "nodata": 0, "compress": "deflate", "tiled": True,
                   "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            for w in blocks(ref.width, ref.height, 2048):
                m = geometry_mask([geometry], out_shape=(int(w.height), int(w.width)),
                                  transform=ref.window_transform(w), invert=True)
                dst.write(m.astype("uint8"), 1, window=w)
    return out


# --------------------------------------------------------------------------- configuration

def required_halo(cfg: dict) -> int:
    """Pixels around the AOI that the configured pipeline needs: the probability-smoothing radius."""
    smoothing = (cfg.get("predict") or {}).get("smoothing")
    if not smoothing:
        return 0
    settings = {"enabled": True} if smoothing is True else dict(smoothing)
    if not settings.get("enabled", False):
        return 0
    return int(settings.get("window", 7)) // 2


def resolve_path(cfg: dict, output_dir: str | Path | None = None) -> Path | None:
    """The AOI mask file of a configuration, or None when none is configured or found.

    ``grid.aoi_mask``: a path; ``false`` disables the mask (also the auto-discovery); ``null`` (default) looks
    for ``<grid.reference stem>_aoi.tif`` next to the reference grid and otherwise, when the optional vector
    ``grid.aoi`` is given, rasterises it onto the reference grid into ``<output_dir>/aoi/``.
    """
    g = cfg["grid"]
    if g.get("aoi_mask") is False:
        return None
    if g.get("aoi_mask"):
        path = Path(g["aoi_mask"])
        if not path.is_file():
            raise FileNotFoundError(f"grid.aoi_mask {path} does not exist")
        return path
    ref = g.get("reference")
    if not ref:
        return None
    found = default_mask_path(ref)
    if found.is_file():
        return found
    if g.get("aoi") and Path(ref).is_file():
        import geopandas as gpd
        vector = Path(g["aoi"])
        if not vector.is_file():
            raise FileNotFoundError(f"grid.aoi {vector} does not exist")
        with rasterio.open(ref) as ds:
            crs = ds.crs
        frame = gpd.read_file(vector).to_crs(crs)
        geom = frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.geometry.unary_union
        target = Path(output_dir or Path(ref).parent) / "aoi" / "aoi_mask.tif"
        if not target.is_file() or target.stat().st_mtime < vector.stat().st_mtime:
            rasterize(geom, ref, target)
        return target
    return None


def from_config(cfg: dict, output_dir: str | Path | None = None) -> AoiMask | None:
    """The configured AOI mask with the halo the pipeline needs (`grid.aoi_halo_px` or the smoothing radius,
    whichever is larger), or None: then every stage runs on the whole grid exactly as before."""
    path = resolve_path(cfg, output_dir)
    if path is None:
        return None
    return AoiMask(path, max(int(cfg["grid"].get("aoi_halo_px") or 0), required_halo(cfg)))
