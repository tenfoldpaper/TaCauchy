from dataclasses import MISSING

from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .taxim_sim import TaximSimulator

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class TaximSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = TaximSimulator

    calib_folder_path: str = ""

    device: str = "cuda"

    with_shadow: bool = False

    tactile_img_res: tuple = (320, 240)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """

    gelpad_height: float = MISSING
    """Used for computing indentation depth from height map"""

    # Asset Data
    gelpad_to_camera_min_distance: float = MISSING
    """Min distance of camera to the gelpad.
    Used for computing the indentation depth out of the
    camera height map.
    """

    convert_to_grayscale: bool = False
    """Convert RGB output to grayscale (for monochrome sensors like 9DTact).
    
    When True, the output will still be in RGB format (3 channels) but all
    channels will have the same grayscale value computed using standard weights:
    gray = 0.299*R + 0.587*G + 0.114*B
    """

    apply_color_correction: bool = False
    """Apply color correction to match calibration image tone.
    
    When True, applies mean-std matching to adjust the output image's color
    distribution to match the background calibration image. This is useful
    when the simulated images appear too dark or have different tone compared
    to the calibration data.
    
    Algorithm: For each channel, normalizes the image to zero mean and unit
    variance, then scales and shifts to match the background image statistics.
    """
