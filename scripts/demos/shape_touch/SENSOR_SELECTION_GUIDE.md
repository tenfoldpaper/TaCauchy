# Sensor Selection Guide

## Overview

The modular contact force visualization demo now supports both **GelSight Mini** and **DIGIT** tactile sensors. You can select which sensor to use via command-line argument.

## Usage

### Run with GelSight Mini (default)

```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py
```

or explicitly:

```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor gelsight
```

### Run with DIGIT sensor

```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor digit
```

## Sensor Configurations

### GelSight Mini
- **Robot USD**: `Robots/Franka/GelSight_Mini/Single_Adapter/uipc_gelpad.usd`
- **Case prim**: `gelsight_mini_case`
- **Gelpad prim**: `gelsight_mini_gelpad`
- **IK offset**: `[0.0, 0.0, 0.131]` m
- **Resolution**: 64x64
- **Calibration**: Default GelSight Mini calibration

### DIGIT
- **Robot USD**: `Robots/Franka/digit/Single_Adapter/digit_uipc_gelpad.usd`
- **Case prim**: `digit_case`
- **Gelpad prim**: `digit_gelpad`
- **IK offset**: `[0.0, 0.0, 0.132]` m (slightly different geometry)
- **Resolution**: 64x64
- **Calibration**: `Sensors/digit/calibs/640x480`

## Implementation Details

### New Files Created

1. **`source/tacex_assets/tacex_assets/robots/franka/franka_digit_single_uipc.py`**
   - DIGIT robot configuration for UIPC simulation
   - Exports:
     - `FRANKA_PANDA_ARM_SINGLE_DIGIT_UIPC_CFG`
     - `FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG`

2. **Updated `source/tacex_assets/tacex_assets/robots/franka/__init__.py`**
   - Added exports for DIGIT configurations

### Configuration Functions

The modular demo includes helper functions to create sensor-specific configurations:

```python
def create_robot_cfg(sensor_type: str) -> ArticulationCfg:
    """Create robot configuration based on sensor type."""
    
def create_sensor_cfg(sensor_type: str) -> GelSightMiniCfg:
    """Create sensor configuration based on sensor type."""
    
def create_gelpad_cfg(sensor_type: str) -> UipcObjectCfg:
    """Create gelpad configuration based on sensor type."""
    
def create_gelpad_attachment_cfg(sensor_type: str) -> UipcIsaacAttachmentsCfg:
    """Create gelpad attachment configuration based on sensor type."""
```

### Dynamic Configuration

The `BallRollingEnvCfg` class now includes:

```python
@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    # Sensor type selection
    sensor_type: str = "gelsight"  # "gelsight" or "digit"
    
    def __post_init__(self):
        """Initialize sensor-specific configurations after dataclass initialization."""
        self.robot = create_robot_cfg(self.sensor_type).replace(prim_path="/World/envs/env_.*/Robot")
        self.gelpad_cfg = create_gelpad_cfg(self.sensor_type)
        self.gelpad_attachment_cfg = create_gelpad_attachment_cfg(self.sensor_type)
        self.gsmini = create_sensor_cfg(self.sensor_type)
```

## Differences Between Sensors

| Feature | GelSight Mini | DIGIT |
|---------|--------------|-------|
| Gelpad size | ~15mm x 15mm | ~18mm x 24mm |
| IK offset | 0.131m | 0.132m |
| Case prim name | `gelsight_mini_case` | `digit_case` |
| Gelpad prim name | `gelsight_mini_gelpad` | `digit_gelpad` |
| Calibration data | Default | `digit/calibs/640x480` |

## Visualization Features (Same for Both Sensors)

1. **3D Scene Visualization**
   - Cyan wireframe: Tet mesh structure
   - Red arrows: Contact forces
   - Yellow wireframe: Ball mesh

2. **2D Force Visualization**
   - Tactile Image + Force Arrows
   - Interpolated Normal Forces (20×20 grid)
   - Interpolated Tangential Forces (20×20 grid)

3. **Deformation Visualization**
   - Blue: No deformation
   - Green: Medium deformation
   - Yellow/Red: Maximum deformation

## Command-Line Arguments

```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py \
    --sensor gelsight \              # Sensor type: gelsight or digit
    --num_envs 1 \                   # Number of environments
    --force_scale 0.1 \              # Force arrow scale factor
    --debug_vis \                    # Enable debug visualization
    --debug_force \                  # Enable force debug output
    --debug_deformation \            # Enable deformation debug output
    --debug_position \               # Enable position monitoring
    --debug_attachment \             # Enable attachment debug output
    --debug_all                      # Enable all debug output
```

## Troubleshooting

### Issue: "No module named 'tacex_assets.robots.franka.franka_digit_single_uipc'"

**Solution**: Make sure you've installed the tacex_assets package:
```bash
cd source/tacex_assets
pip install -e .
```

### Issue: "DIGIT USD file not found"

**Solution**: Verify the DIGIT USD file exists:
```bash
ls source/tacex_assets/tacex_assets/data/Robots/Franka/digit/Single_Adapter/digit_uipc_gelpad.usd
```

### Issue: Gelpad not attaching correctly

**Solution**: Check that the case prim name matches the sensor type:
- GelSight Mini: `gelsight_mini_case`
- DIGIT: `digit_case`

The attachment configuration automatically uses the correct case name based on `sensor_type`.

## Future Enhancements

Potential improvements:
1. Support for dual-sensor configurations (gripper with two sensors)
2. Automatic sensor detection from USD file
3. Sensor-specific force scaling based on gelpad size
4. Per-sensor visualization presets
