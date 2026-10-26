"""Plugin registries: extend GeoCrop with classifiers, indices, segmenters, exporters, validators and
foundation-model feature providers.

Third-party packages declare entry points in their ``pyproject.toml``; GeoCrop finds them with
``importlib.metadata`` the first time a name is looked up that is not built in (lazy discovery). Built-ins
are registered in code through the same registries, so a plugin is indistinguishable from a built-in once
loaded. See docs/adr/0002-plugin-entry-points.md and ``examples/plugin_example``.

============  =====================  ==============================================================
Registry      Entry-point group      Contract of the registered object
============  =====================  ==============================================================
estimators    ``geocrop.estimators`` ``factory(params: dict, seed: int) -> estimator``; the estimator
                                     has ``fit(X, y)`` returning itself, ``predict_proba(X)`` returning
                                     ``(n, n_classes)`` probabilities and, after fit, ``classes_``.
                                     ``params`` is the YAML ``model:`` section (including
                                     ``algorithm``); ignore unknown keys. Selected with
                                     ``model: {algorithm: <name>}``.
indices       ``geocrop.indices``    a ``geocrop.indices.Index`` (or a zero-argument callable returning
                                     one) whose ``name`` equals the entry-point name. Usable wherever an
                                     index name is accepted (``features.s2_indices``, CBM markers, ...).
segmenters    ``geocrop.segmenters`` an object (or a zero-argument class) satisfying
                                     ``geocrop.segmenters.Segmenter``: ``name`` and
                                     ``segment(raster, output, **params) -> dict`` writing a polygon
                                     layer to ``output`` and returning at least ``{"path", "segments"}``.
exporters     ``geocrop.exporters``  ``export(source, destination, **options) -> Path | dict``.
validators    ``geocrop.validators`` ``validate(path, **options) -> dict`` with a ``problems`` list
                                     (empty when the input passes).
feature_      ``geocrop.feature_     an object (or a zero-argument class) satisfying
providers     providers``            ``geocrop.feature_providers.FeatureProvider``: ``name`` (equal to
                                     the entry-point name), ``dims``, ``licence``, ``weights`` (pinned
                                     URL + sha256), ``contract`` (input contract), ``load_model``,
                                     ``embed_pixels`` and ``embed_parcels``. Selected with
                                     ``features.embeddings.source`` / ``geocrop embeddings --source``.
============  =====================  ==============================================================

A plugin that fails to import or violates its contract is skipped with a ``PluginWarning``; it never
prevents GeoCrop from starting. Entry points cannot replace a built-in name.

Programmatic registration (tests, notebooks)::

    from geocrop import plugins
    plugins.estimators.register("my_rf", lambda params, seed: MyClassifier(seed=seed))
"""
from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from importlib import metadata
from typing import Any

BUILTIN = "builtin"


class PluginWarning(UserWarning):
    """A plugin could not be loaded or does not satisfy its contract."""


def _entry_points(group: str):
    try:
        return list(metadata.entry_points(group=group))
    except Exception as exc:  # corrupt distribution metadata must not break the engine
        warnings.warn(f"geocrop: cannot read entry points of group {group!r}: {exc}", PluginWarning, stacklevel=3)
        return []


def _load_spec(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    obj = importlib.import_module(module)
    for part in filter(None, attr.split(".")):
        obj = getattr(obj, part)
    return obj


class Registry:
    """Name -> object mapping for one extension point.

    `provider` is the module whose import registers the built-ins (imported on first use, so that
    ``plugins.estimators.get("rf")`` works without importing ``geocrop.estimators`` first).
    """

    def __init__(self, kind: str, group: str, check: Callable[[str, Any], Any] | None = None,
                 provider: str | None = None):
        self.kind, self.group, self.provider = kind, group, provider
        self._check = check
        self._items: dict[str, Any] = {}
        self._lazy: dict[str, str] = {}
        self._sources: dict[str, str] = {}
        self._provider_loaded = provider is None
        self._discovered = False

    # ------------------------------------------------------------------ registration
    def register(self, name: str, obj: Any = None, *, replace: bool = False, source: str = "code"):
        """Register `obj` under `name`; usable as a decorator when `obj` is omitted."""
        if obj is None:
            def decorator(target):
                self.register(name, target, replace=replace, source=source)
                return target
            return decorator
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind} name must be a non-empty string")
        if not replace and (name in self._items or name in self._lazy):
            raise ValueError(f"{self.kind} {name!r} is already registered ({self._sources[name]}); "
                             "pass replace=True to override it deliberately")
        obj = self._check(name, obj) if self._check else obj
        self._lazy.pop(name, None)
        self._items[name] = obj
        self._sources[name] = source
        return obj

    def register_lazy(self, name: str, spec: str, *, source: str = BUILTIN) -> None:
        """Register ``"module:attribute"``, imported only when `name` is first requested."""
        if name in self._items or name in self._lazy:
            raise ValueError(f"{self.kind} {name!r} is already registered ({self._sources[name]})")
        self._lazy[name] = spec
        self._sources[name] = source

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)
        self._lazy.pop(name, None)
        self._sources.pop(name, None)

    # ------------------------------------------------------------------ discovery
    def _ensure_provider(self) -> None:
        if not self._provider_loaded:
            self._provider_loaded = True
            importlib.import_module(self.provider)

    def discover(self, force: bool = False) -> None:
        """Load entry points of `group` once (again with `force`). Broken plugins only warn."""
        self._ensure_provider()
        if self._discovered and not force:
            return
        self._discovered = True
        for ep in _entry_points(self.group):
            if ep.name in self._items or ep.name in self._lazy:
                if not self._sources.get(ep.name, "").startswith("entry point"):
                    warnings.warn(f"geocrop: ignoring {self.kind} plugin {ep.name!r} ({ep.value}): the name is "
                                  f"already registered ({self._sources[ep.name]})", PluginWarning, stacklevel=3)
                continue
            try:
                self.register(ep.name, ep.load(), source=f"entry point {ep.value}")
            except Exception as exc:  # a broken plugin is reported, never fatal
                warnings.warn(f"geocrop: ignoring {self.kind} plugin {ep.name!r} ({ep.value}) from entry-point "
                              f"group {self.group!r}: {type(exc).__name__}: {exc}", PluginWarning, stacklevel=3)

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> Any:
        self._ensure_provider()
        if name not in self._items and name not in self._lazy:
            self.discover()
        if name in self._lazy:
            spec = self._lazy[name]
            self.register(name, _load_spec(spec), replace=True, source=self._sources[name])
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {', '.join(self.names()) or 'none'}. "
                           f"Plugins register under the entry-point group {self.group!r}.") from None

    def names(self) -> list[str]:
        self.discover()
        return list(dict.fromkeys([*self._items, *self._lazy]))

    def declared_names(self) -> list[str]:
        """Names registered so far plus entry-point names, without importing any plugin or the built-in provider
        (cheap enough for command-line choices; `names` loads everything)."""
        return list(dict.fromkeys([*self._items, *self._lazy, *(ep.name for ep in _entry_points(self.group))]))

    def source(self, name: str) -> str:
        self.get(name)
        return self._sources[name]

    def __contains__(self, name: object) -> bool:
        self._ensure_provider()
        if name not in self._items and name not in self._lazy:
            self.discover()
        return name in self._items or name in self._lazy

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, group={self.group!r})"


# ---------------------------------------------------------------------- contracts

def _callable(kind: str):
    def check(name: str, obj: Any) -> Any:
        if not callable(obj):
            raise TypeError(f"{kind} {name!r} must be callable, got {type(obj).__name__}")
        return obj
    return check


def _index(name: str, obj: Any) -> Any:
    if not hasattr(obj, "compute") and callable(obj):
        obj = obj()
    missing = [a for a in ("name", "sensor", "inputs", "formula", "reference", "compute") if not hasattr(obj, a)]
    if missing:
        raise TypeError(f"index {name!r} must be a geocrop.indices.Index; missing {missing}")
    if obj.name != name:
        raise ValueError(f"index registered as {name!r} is named {obj.name!r}; the names must match")
    return obj


def _segmenter(name: str, obj: Any) -> Any:
    if isinstance(obj, type):
        obj = obj()
    if not callable(getattr(obj, "segment", None)):
        raise TypeError(f"segmenter {name!r} needs a segment(raster, output, **params) -> dict method")
    return obj


def _feature_provider(name: str, obj: Any) -> Any:
    if isinstance(obj, type):
        obj = obj()
    required = ("name", "dims", "licence", "weights", "contract", "load_model", "embed_pixels", "embed_parcels")
    missing = [a for a in required if not hasattr(obj, a)]
    if missing:
        raise TypeError(f"feature provider {name!r} must satisfy geocrop.feature_providers.FeatureProvider; "
                        f"missing {missing}")
    if obj.name != name:
        raise ValueError(f"feature provider registered as {name!r} is named {obj.name!r}; the names must match")
    return obj


estimators = Registry("estimator", "geocrop.estimators", _callable("estimator factory"), provider="geocrop.estimators")
indices = Registry("index", "geocrop.indices", _index, provider="geocrop.indices")
segmenters = Registry("segmenter", "geocrop.segmenters", _segmenter, provider="geocrop.segmenters")
exporters = Registry("exporter", "geocrop.exporters", _callable("exporter"))
validators = Registry("validator", "geocrop.validators", _callable("validator"))
feature_providers = Registry("feature provider", "geocrop.feature_providers", _feature_provider,
                             provider="geocrop.feature_providers")

# Built-ins implemented outside the core are named, not imported: they load on first use.
exporters.register_lazy("cog", "geocrop.export:to_cog")
exporters.register_lazy("geoparquet", "geocrop.export:to_geoparquet")
validators.register_lazy("raster", "geocrop.provenance:check_raster")
validators.register_lazy("vector", "geocrop.provenance:check_vector")

REGISTRIES: dict[str, Registry] = {"estimators": estimators, "indices": indices, "segmenters": segmenters,
                                   "exporters": exporters, "validators": validators,
                                   "feature_providers": feature_providers}


def describe() -> dict[str, dict[str, str]]:
    """Every registered name with its source (``builtin``, ``code`` or the entry point)."""
    return {kind: {name: reg._sources[name] for name in reg.names()} for kind, reg in REGISTRIES.items()}


if __name__ == "__main__":  # python -m geocrop.plugins
    import json

    from geocrop.plugins import describe as _describe  # the imported module holds the registrations

    print(json.dumps(_describe(), indent=2))
