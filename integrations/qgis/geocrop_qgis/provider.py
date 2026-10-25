from pathlib import Path

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms import DemoAlgorithm, LoadJobAlgorithm, RunCommandAlgorithm


class GeoCropProvider(QgsProcessingProvider):
    def id(self):
        return "geocrop"

    def name(self):
        return "GeoCrop Engine"

    def longName(self):
        return "GeoCrop Engine · HTTP jobs and result layers"

    def icon(self):
        return QIcon(str(Path(__file__).with_name("icon.svg")))

    def loadAlgorithms(self):
        for algorithm in (RunCommandAlgorithm, DemoAlgorithm, LoadJobAlgorithm):
            self.addAlgorithm(algorithm())
