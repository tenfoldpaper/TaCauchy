# 插值可视化实现说明

## 概述

`simple_tactile_demo.py` 现在完全按照 `contact_force_visualization_demo_modular copy.py` 的方式实现了插值可视化功能。

## 关键实现细节

### 1. 插值可视化器初始化

```python
interp_viz = InterpolatedForceVisualizer(
    grid_size=(20, 20),  # 20x20 规则网格
    image_size=(512, 512),
    enable_window=True,  # 始终启用窗口（与参考文件一致）
    noise_threshold_percentile=10.0  # 噪声过滤阈值
)
```

**与参考文件的区别**：
- 参考文件使用 30×30 网格
- simple_tactile_demo 使用 20×20 网格（更快）

### 2. 2D 投影方法

完全采用参考文件的手动归一化方法：

```python
# 计算三角形中心的 X-Y 范围
x_min, x_max = tri_centers[:, 0].min(), tri_centers[:, 0].max()
y_min, y_max = tri_centers[:, 1].min(), tri_centers[:, 1].max()

# 手动归一化到图像坐标
img_h, img_w = 512, 512
margin = 50
tri_positions_2d = []
for pos in tri_centers:
    nx = (pos[0] - x_min) / max(x_max - x_min, 0.001)
    ny = (pos[1] - y_min) / max(y_max - y_min, 0.001)
    px = margin + nx * (img_w - 2 * margin)
    py = margin + (1 - ny) * (img_h - 2 * margin)
    tri_positions_2d.append([px, py])
tri_positions_2d = np.array(tri_positions_2d)
```

**优点**：
- 不依赖传感器标定参数
- 自动适应不同的接触区域大小
- 与参考文件完全一致

### 3. 插值调用

```python
# 获取压力幅值
pressure_magnitudes = np.abs(stress_data.normal_pressure)

# 调用插值可视化（与参考文件相同的参数顺序）
interp_viz.visualize_from_values(
    pressure_magnitudes,           # 法向压力幅值
    stress_data.tangential_traction,  # 切向牵引力向量
    tri_positions_2d,              # 2D 投影位置
)
```

## 数据流程

```
UIPC 仿真
    ↓
应力提取 (StressExtractor)
    ↓
三角形中心计算
    ↓
手动 2D 投影（归一化）
    ↓
插值到 20×20 网格 (scipy.griddata)
    ↓
可视化显示（OpenCV 窗口）
```

## 插值算法

### 噪声过滤
```python
# 计算阈值（10% 百分位）
threshold = np.percentile(normal_magnitudes, 10.0)

# 过滤低于阈值的点
valid_mask = normal_magnitudes >= threshold
```

### 插值方法
1. **首选**：三次插值 (cubic)
2. **降级**：线性插值 (linear) - 如果三次插值失败
3. **填充值**：0.0 - 对于网格外的点

### 可视化

**法向力窗口**：
- 颜色映射：JET colormap
- 显示：圆圈（大小 = 压力）
- 统计信息：最大值、平均值、活跃单元数

**切向力窗口**：
- 颜色映射：HSV colormap
- 显示：箭头（方向 + 大小）
- 统计信息：最大值、平均值、活跃单元数

## 性能考虑

### 计算开销
- 三角形中心计算：O(N)
- 2D 投影：O(N)
- 插值：O(N × M)，其中 M = 20×20 = 400
- 可视化：O(M)

### 优化策略
1. 使用较小的网格（20×20 vs 30×30）
2. 噪声过滤减少插值点数
3. 每帧更新（无跳帧）

## 与参考文件的对比

| 特性 | simple_tactile_demo.py | reference (modular copy) |
|------|------------------------|--------------------------|
| 插值网格大小 | 20×20 | 30×30 |
| 窗口启用 | ✅ 始终 | ✅ 始终 |
| 2D 投影 | ✅ 手动归一化 | ✅ 手动归一化 |
| 噪声过滤 | ✅ 10% 百分位 | ✅ 可配置百分位 |
| 插值方法 | ✅ cubic → linear | ✅ cubic → linear |
| 参数顺序 | ✅ 完全一致 | ✅ 参考实现 |
| 实现方式 | ✅ 完全一致 | ✅ 原始实现 |

## 使用示例

```bash
# 运行脚本（插值自动启用）
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight
```

## 预期输出

启动后会看到 5 个窗口：
1. Normal Pressure（点基础）
2. Tangential Traction（点基础）
3. Normal Force (Interpolated)（20×20 网格）
4. Tangential Force (Interpolated)（20×20 网格）
5. Tactile RGB（触觉图像）

## 故障排除

### 问题：插值窗口全黑
**原因**：传感器未接触物体
**解决**：拖动红色立方体使传感器接触

### 问题：插值结果有噪声
**原因**：噪声过滤阈值太低
**解决**：增加 `noise_threshold_percentile` 值

### 问题：插值失败
**原因**：接触点太少（< 4 个）
**解决**：增加接触力或调整网格密度

## 总结

现在 `simple_tactile_demo.py` 的插值实现与参考文件完全一致，可以实时显示插值力场，无需额外的命令行参数。
