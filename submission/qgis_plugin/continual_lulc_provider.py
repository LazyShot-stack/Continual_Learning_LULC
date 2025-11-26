from qgis.core import QgsProcessingProvider
from .continual_lulc_algorithm import ContinualLULCAlgorithm

class ContinualLULCProvider(QgsProcessingProvider):
    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(ContinualLULCAlgorithm())

    def id(self):
        return 'continuallulc'

    def name(self):
        return 'Continual LULC'

    def icon(self):
        return QgsProcessingProvider.icon(self)
