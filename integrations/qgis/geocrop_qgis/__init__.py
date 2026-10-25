"""Standalone QGIS plugin; the engine runs separately behind its authenticated HTTP API."""


def classFactory(iface):
    from .plugin import GeoCropPlugin
    return GeoCropPlugin(iface)
