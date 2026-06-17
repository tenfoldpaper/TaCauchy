"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Register UI extensions (optional for headless mode).
try:
    from .ui_extension_example import UsdrtExamplePythonExtension
    __all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData", "UsdrtExamplePythonExtension"]
except ImportError:
    # UI module not available (e.g., in headless mode)
    __all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData"]
