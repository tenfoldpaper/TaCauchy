# Simple Tactile Demo - 使用说明

这是一个简洁的触觉传感器仿真演示脚本，完全按照 `contact_force_visualization_demo_modular copy.py` 的实现方式。

## 功能特性

1. **点基础力可视化**
   - 法向压力：绿色圆圈（大小表示压力）
   - 切向牵引力：红色圆圈（大小表示牵引力）

2. **插值力可视化**（20×20 规则网格）
   - 法向力：颜色编码的压力分布
   - 切向力：方向箭头显示
   - 噪声过滤（10% 百分位阈值）
   - 实时窗口显示（与参考文件相同的实现）

3. **触觉 RGB 图像**
   - 点击窗口保存图像

## 使用方法

```bash
# GelSight Mini 传感器
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight

# DIGIT 传感器
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor digit

# 9DTact 传感器
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor 9dtact
```

## 参数说明

- `--sensor`: 传感器类型（gelsight, digit, 9dtact）
- `--num_envs`: 环境数量（默认：1）
- `--debug_vis`: 启用调试可视化

## 交互控制

- **拖动红色立方体**：控制机器人末端执行器位置
- **点击触觉图像窗口**：保存当前触觉图像到 `tactile_images/`
- **Ctrl+C**：停止仿真

## 可视化窗口

脚本会自动创建以下窗口：

1. **Normal Pressure** - 点基础法向压力（绿色圆圈）
2. **Tangential Traction** - 点基础切向牵引力（红色圆圈）
3. **Normal Force (Interpolated)** - 插值法向力（20×20 网格，颜色编码）
4. **Tangential Force (Interpolated)** - 插值切向力（20×20 网格，方向箭头）
5. **Tactile RGB** - 触觉图像（点击保存）

## 技术实现

### 与参考文件的一致性

本脚本完全按照 `contact_force_visualization_demo_modular copy.py` 的实现方式：

1. **插值可视化器初始化**
   ```python
   InterpolatedForceVisualizer(
       grid_size=(20, 20),
       image_size=(512, 512),
       enable_window=True,  # 始终启用窗口
       noise_threshold_percentile=10.0
   )
   ```

2. **2D 投影方法**
   - 使用手动归一化投影（与参考文件相同）
   - 不使用传感器标定参数
   - 基于三角形中心的 X-Y 坐标范围

3. **数据流程**
   ```
   应力提取 → 三角形中心 → 2D 投影 → 插值 → 可视化
   ```

### 插值算法
- 使用 scipy.interpolate.griddata 进行三次插值
- 自动降级到线性插值（如果三次插值失败）
- 噪声过滤：移除低于 10% 百分位的数据点

### 数据来源
- 应力数据：从 UIPC 仿真中提取柯西应力
- 三角形中心：用于插值的采样点
- 2D 投影：手动归一化到图像坐标

## 输出文件

- 触觉图像：`tactile_images/tactile_rgb_*.png`

## 故障排除

### 问题：插值图像全黑
**原因**：没有接触或力太小
**解决方案**：拖动红色立方体使传感器接触物体

### 问题：插值噪声太多
**解决方案**：增加噪声过滤阈值（需修改代码中的 `noise_threshold_percentile`）

## 模块化设计

本脚本使用 `tacex_uipc.utils.visualization` 中的模块化可视化工具：

- `TactileForceVisualizer`: 点基础力可视化
- `TactileImageViewer`: 触觉图像显示和保存
- `InterpolatedForceVisualizer`: 插值力可视化（20×20 网格）
- `StressExtractor`: 应力数据提取

所有可视化器都支持独立的窗口管理和资源清理。

## 与参考文件的对比

| 特性 | simple_tactile_demo.py | contact_force_visualization_demo_modular copy.py |
|------|------------------------|--------------------------------------------------|
| 插值网格 | 20×20 | 30×30 |
| 插值窗口 | ✅ 始终启用 | ✅ 始终启用 |
| 2D 投影方法 | ✅ 手动归一化 | ✅ 手动归一化 |
| 噪声过滤 | ✅ 10% 百分位 | ✅ 可配置百分位 |
| 实现方式 | ✅ 完全一致 | ✅ 参考实现 |

