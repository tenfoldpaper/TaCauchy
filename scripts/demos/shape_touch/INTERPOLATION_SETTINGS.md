# 插值可视化设置说明

## 如何调整插值点数

插值点数在 `simple_tactile_demo.py` 中设置，位于可视化器初始化部分：

```python
interp_viz = InterpolatedForceVisualizer(
    grid_size=(30, 30),  # 👈 这里设置插值网格大小
    image_size=(512, 512),
    enable_window=True,
    noise_threshold_percentile=10.0
)
```

### 常用网格大小

| 网格大小 | 总点数 | 性能 | 细节 | 推荐场景 |
|---------|--------|------|------|---------|
| (10, 10) | 100 | ⚡⚡⚡ 最快 | ⭐ 粗糙 | 快速预览 |
| (20, 20) | 400 | ⚡⚡ 快 | ⭐⭐ 中等 | 实时仿真 |
| (30, 30) | 900 | ⚡ 中等 | ⭐⭐⭐ 好 | 平衡性能和细节（默认）|
| (40, 40) | 1600 | 🐌 慢 | ⭐⭐⭐⭐ 很好 | 高质量可视化 |
| (50, 50) | 2500 | 🐌🐌 很慢 | ⭐⭐⭐⭐⭐ 极好 | 离线分析 |

### 修改示例

```python
# 快速模式（实时性能优先）
grid_size=(15, 15)

# 标准模式（平衡）
grid_size=(30, 30)

# 高质量模式（细节优先）
grid_size=(50, 50)
```

## 图像旋转设置

### 当前设置：不旋转

插值图像现在**不会随相机旋转**，始终保持固定方向。

修改位置：`source/tacex_uipc/tacex_uipc/utils/visualization/force_visualizer.py`

```python
# 当前代码（不旋转）
cv2.imshow("Normal Force (Interpolated)", normal_img)
cv2.imshow("Tangential Force (Interpolated)", tangent_img)
```

### 如果需要旋转

如果你想让图像旋转 90 度（例如匹配相机视角），可以修改为：

```python
# 旋转 90 度（顺时针）
rotated_normal = cv2.rotate(normal_img, cv2.ROTATE_90_CLOCKWISE)
rotated_tangent = cv2.rotate(tangent_img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Normal Force (Interpolated)", rotated_normal)
cv2.imshow("Tangential Force (Interpolated)", rotated_tangent)
```

可用的旋转选项：
- `cv2.ROTATE_90_CLOCKWISE` - 顺时针 90°
- `cv2.ROTATE_90_COUNTERCLOCKWISE` - 逆时针 90°
- `cv2.ROTATE_180` - 180°

## 其他可调参数

### 1. 图像大小

```python
image_size=(512, 512)  # 默认 512x512
# 可选：(256, 256), (640, 640), (1024, 1024)
```

### 2. 噪声过滤阈值

```python
noise_threshold_percentile=10.0  # 过滤低于 10% 百分位的数据
# 增加值（如 20.0）= 更激进的过滤，更少噪声
# 减少值（如 5.0）= 保留更多数据，可能有噪声
```

### 3. 背景颜色

在 `force_visualizer.py` 中修改：

```python
# 当前：白色背景
img = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255

# 黑色背景
img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

# 灰色背景
img = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 128
```

### 4. 箭头粗细

在 `_visualize_tangent_forces` 方法中：

```python
cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 3, tipLength=0.3)
#                                                  ^ 这里调整粗细
# 1 = 细线
# 2 = 中等
# 3 = 粗线（当前）
# 4+ = 很粗
```

## 性能优化建议

### 实时仿真（60+ FPS）
```python
grid_size=(15, 15)
noise_threshold_percentile=15.0
```

### 平衡模式（30-60 FPS）
```python
grid_size=(30, 30)  # 当前默认
noise_threshold_percentile=10.0
```

### 高质量模式（< 30 FPS）
```python
grid_size=(50, 50)
noise_threshold_percentile=5.0
```

## 快速参考

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| 网格大小 | `simple_tactile_demo.py` | (30, 30) | 插值点数 |
| 图像大小 | `simple_tactile_demo.py` | (512, 512) | 显示分辨率 |
| 噪声过滤 | `simple_tactile_demo.py` | 10.0 | 百分位阈值 |
| 旋转 | `force_visualizer.py` | 无 | 图像旋转 |
| 背景色 | `force_visualizer.py` | 白色 | 背景颜色 |
| 箭头粗细 | `force_visualizer.py` | 3 | 线条宽度 |

## 示例配置

### 配置 1：快速预览
```python
grid_size=(15, 15)
image_size=(256, 256)
noise_threshold_percentile=20.0
```

### 配置 2：标准使用（当前）
```python
grid_size=(30, 30)
image_size=(512, 512)
noise_threshold_percentile=10.0
```

### 配置 3：高质量分析
```python
grid_size=(50, 50)
image_size=(1024, 1024)
noise_threshold_percentile=5.0
```
