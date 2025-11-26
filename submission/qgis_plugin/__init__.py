from qgis.core import QgsApplication
from .continual_lulc_provider import ContinualLULCProvider


class ContinualLULCPlugin:
    """Plugin wrapper that registers the processing provider on load
    and removes it on unload. QGIS expects the object returned by
    classFactory to implement initGui() and unload()."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        # Register the processing provider
        try:
            self.provider = ContinualLULCProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)
        except Exception as e:
            # Avoid crashing QGIS; errors will appear in the QGIS log
            print(f"ContinualLULC: failed to register provider: {e}")

    def unload(self):
        # Remove provider if it was registered
        try:
            if self.provider is not None:
                QgsApplication.processingRegistry().removeProvider(self.provider)
                self.provider = None
        except Exception as e:
            print(f"ContinualLULC: failed to remove provider: {e}")


def classFactory(iface):
    return ContinualLULCPlugin(iface)
