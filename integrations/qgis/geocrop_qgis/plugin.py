from qgis.core import QgsApplication
from qgis.PyQt.QtWidgets import QAction

from .provider import GeoCropProvider


class GeoCropPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.action = None

    def initGui(self):
        self.provider = GeoCropProvider()
        if not QgsApplication.processingRegistry().addProvider(self.provider):
            raise RuntimeError("Could not register the GeoCrop Processing provider")
        self.action = QAction(self.provider.icon(), "Run GeoCrop command…", self.iface.mainWindow())
        self.action.triggered.connect(self.open_dialog)
        self.iface.addPluginToMenu("&GeoCrop", self.action)

    def open_dialog(self):
        import processing
        processing.execAlgorithmDialog("geocrop:runcommand", {})

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu("&GeoCrop", self.action)
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
