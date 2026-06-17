# GelSight 相机参数调整指南

## 问题描述
`run_shape_touch.py` 中的 GelSight 相机视角过于广角，需要调整相机参数以获得更自然的视野。

## 相机参数位置

### USD 资产文件
相机参数存储在 USD 文件中：
- **GelSight Mini**: `source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Sensor.usd`
- **DIGIT**: `source/tacex_assets/tacex_assets/data/Sensors/digit/Sensor.usd`

### 原始参数值

**GelSight Mini**:
```
focalLength: 20.0
horizontalAperture: 14.3
verticalAperture: 18.6
```

**DIGIT**:
```
focalLength: 26.0
horizontalAperture: 20.0
verticalAperture: 18.0
```

## 调整方法

### 方法 1：运行时修改（推荐）

已在 `scripts/demos/shape_touch/run_shape_touch.py` 中添加 `_adjust_camera_parameters()` 方法。

**默认行为**：使用原始 USD 文件中的参数值（不做任何调整）

**如何调整**：
在 `_adjust_camera_parameters()` 方法中取消注释并修改这些行：

```python
# 方式 1: 直接设置具体数值
camera.GetHorizontalApertureAttr().Set(10.0)  # 原始 GelSight Mini: 14.3
camera.GetVerticalApertureAttr().Set(13.0)    # 原始 GelSight Mini: 18.6

# 方式 2: 按比例调整（推荐）
camera.GetHorizontalApertureAttr().Set(original_h_aperture * 0.7)  # 减少 30%
camera.GetVerticalApertureAttr().Set(original_v_aperture * 0.7)    # 减少 30%

# 方式 3: 同时调整焦距
camera.GetFocalLengthAttr().Set(original_focal * 1.25)  # 增加 25%
```

**参数效果**：
- **horizontalAperture/verticalAperture**（光圈）：
  - 更小的值 = 更窄的视野（减少广角效果）
  - 更大的值 = 更宽的视野（增加广角效果）
  
- **focalLength**（焦距）：
  - 更大的值 = 更窄的视野（类似长焦镜头）
  - 更小的值 = 更宽的视野（类似广角镜头）wo

**推荐调整范围**：
```python
# 轻微调整（减少 20-30% 广角）
horizontalAperture: 10.0 - 12.0
verticalAperture: 13.0 - 15.0

# 中等调整（减少 40-50% 广角）
horizontalAperture: 8.0 - 10.0
verticalAperture: 11.0 - 13.0

# 强烈调整（类似长焦效果）
horizontalAperture: 6.0 - 8.0
verticalAperture: 9.0 - 11.0
focalLength: 25.0 - 30.0
```

### 方法 2：直接修改 USD 文件（永久修改）

使用 Python 脚本修改 USD 文件：

```python
from pxr import Usd, UsdGeom

# 打开 USD 文件
stage = Usd.Stage.Open('source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Sensor.usd')

# 获取相机 prim
camera_prim = stage.GetPrimAtPath('/GelsightMiniCase/Camera')
camera = UsdGeom.Camera(camera_prim)

# 修改参数
camera.GetHorizontalApertureAttr().Set(10.0)
camera.GetVerticalApertureAttr().Set(13.0)
camera.GetFocalLengthAttr().Set(25.0)  # 可选

# 保存
stage.Save()
```

## 测试建议

1. 先使用方法 1（运行时修改）测试不同参数
2. 找到满意的参数后，可以选择使用方法 2 永久修改 USD 文件
3. 建议从轻微调整开始，逐步增加调整幅度

## 相关文件

- 运行脚本: `scripts/demos/shape_touch/run_shape_touch.py`
- GelSight Mini USD: `source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Sensor.usd`
- DIGIT USD: `source/tacex_assets/tacex_assets/data/Sensors/digit/Sensor.usd`
- 相机配置: `source/tacex/tacex/gelsight_sensor_cfg.py`

## 运行测试

```bash
# 测试 GelSight Mini
python scripts/demos/shape_touch/run_shape_touch.py --sensor_type gsmini

# 测试 DIGIT
python scripts/demos/shape_touch/run_shape_touch.py --sensor_type digit
```
