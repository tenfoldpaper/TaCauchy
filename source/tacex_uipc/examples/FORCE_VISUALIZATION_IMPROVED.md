# 力可视化优化说明

## 更新内容

优化了 `contact_force_visualization_demo.py` 中的 3D 力可视化方式，使其更加直观和清晰。

## 新的可视化方式

### 之前（旧版本）
- ❌ 黄色箭头：合力（总力）
- ❌ 绿色箭头：法向力
- ❌ 红色箭头：切向力
- ❌ 白色点：接触点
- **问题**：箭头太多，互相遮挡，难以区分

### 现在（新版本）
- ✅ **绿色圆点**：法向力（点的大小 = 力的大小）
- ✅ **红色箭头**：切向力（箭头方向和长度表示力的方向和大小）
- ✅ **青色线框**：Tet mesh 结构
- **优势**：清晰直观，法向力和切向力分别用不同的视觉元素表示

## 可视化原理

### 法向力（Normal Force）
- **表示方式**：绿色圆点
- **大小计算**：`point_size = force_magnitude × normal_point_scale`
- **范围限制**：5 到 50 像素
- **物理意义**：垂直于表面的压力，只有大小没有方向（总是垂直）

### 切向力（Tangential Force）
- **表示方式**：红色箭头
- **长度计算**：`arrow_length = force_magnitude × force_scale`
- **线宽**：3 像素
- **物理意义**：平行于表面的摩擦力，有方向和大小

## 使用方法

### 基本运行
```bash
python source/tacex_uipc/examples/contact_force_visualization_demo.py
```

### 调整切向力箭头长度
```bash
# 增大箭头（更容易看到）
python source/tacex_uipc/examples/contact_force_visualization_demo.py --force_scale 0.2

# 减小箭头（避免太长）
python source/tacex_uipc/examples/contact_force_visualization_demo.py --force_scale 0.05
```

### 调整法向力圆点大小
```bash
# 增大圆点（更明显）
python source/tacex_uipc/examples/contact_force_visualization_demo.py --normal_point_scale 200.0

# 减小圆点（更精细）
python source/tacex_uipc/examples/contact_force_visualization_demo.py --normal_point_scale 50.0
```

### 同时调整两者
```bash
python source/tacex_uipc/examples/contact_force_visualization_demo.py \
    --force_scale 0.15 \
    --normal_point_scale 150.0
```

## 参数说明

### `--force_scale` (默认: 0.1)
- 控制切向力箭头的长度
- 公式：`arrow_length = tangential_force_magnitude × force_scale`
- 建议范围：0.05 ~ 0.3
- 太小：箭头看不清
- 太大：箭头太长，超出视野

### `--normal_point_scale` (默认: 100.0)
- 控制法向力圆点的大小
- 公式：`point_size = normal_force_magnitude × normal_point_scale`
- 建议范围：50.0 ~ 300.0
- 太小：圆点太小，看不清
- 太大：圆点太大，遮挡其他元素
- 注意：点的大小会被限制在 5-50 像素之间

## 可视化效果

运行脚本后，在 3D 场景中你会看到：

1. **青色线框** - Tet mesh 的四面体结构
2. **绿色圆点** - 法向力
   - 小圆点 = 小的法向力
   - 大圆点 = 大的法向力
3. **红色箭头** - 切向力
   - 箭头方向 = 力的方向
   - 箭头长度 = 力的大小

## 调试输出

脚本会打印详细的可视化信息：

```
[FORCE_VIS] Drawing 3D forces: 15 contact points
[FORCE_VIS] Drew 15 normal force points (green circles)
[FORCE_VIS] Normal force point sizes: min=8, max=35, mean=18.5
[FORCE_VIS] Drew 12 tangential force arrows (red)
```

这些信息帮助你了解：
- 有多少个接触点
- 法向力点的大小范围
- 有多少个切向力箭头

## 技术细节

### 法向力点大小计算
```python
point_size = max(5, min(50, int(normal_mag * normal_point_scale)))
```
- `normal_mag`：法向力大小（牛顿）
- `normal_point_scale`：缩放因子
- 最小 5 像素，最大 50 像素

### 切向力箭头绘制
```python
if tangent_mag > 0.0001:  # 只绘制显著的力
    tangent_end = start_pos + tangent * force_scale
    dd.draw_lines([start_pos], [tangent_end], [red], [3])
```

### 过滤小力
- 法向力 < 0.0001 N：不显示
- 切向力 < 0.0001 N：不显示
- 避免噪声和微小力的干扰

## 与 2D 可视化的关系

脚本同时提供两种可视化：

1. **3D 场景可视化**（本次优化）
   - 绿色圆点：法向力
   - 红色箭头：切向力
   - 青色线框：Tet mesh

2. **2D 触觉图像可视化**（单独窗口）
   - 在触觉传感器图像上叠加力箭头
   - 使用 OpenCV 绘制
   - 包含统计信息和图例

## 物理意义

### 为什么法向力用圆点？
- 法向力总是垂直于表面
- 方向是固定的（向内或向外）
- 只需要表示大小，不需要表示方向
- 圆点大小直观表示力的强度

### 为什么切向力用箭头？
- 切向力平行于表面
- 方向可以是任意的（摩擦力方向）
- 需要同时表示方向和大小
- 箭头是最直观的表示方式

## 示例场景

### 球体压在 gelpad 上
- **法向力**：接触区域出现绿色圆点，中心最大（压力最大）
- **切向力**：如果球在滚动，会看到红色箭头指向滚动方向

### 物体滑动
- **法向力**：接触区域的绿色圆点
- **切向力**：红色箭头指向滑动方向（摩擦力）

## 性能说明

- 每帧重绘所有力可视化
- 使用批量绘制（`draw_points` 一次绘制所有圆点）
- 性能影响很小（< 1ms）
- 适合实时可视化

## 故障排除

### 看不到绿色圆点
1. 检查是否有法向力：查看控制台输出
2. 增大 `--normal_point_scale`
3. 确保物体与 gelpad 有接触

### 看不到红色箭头
1. 检查是否有切向力：查看控制台输出
2. 增大 `--force_scale`
3. 尝试让物体滑动或滚动

### 圆点太大/太小
- 调整 `--normal_point_scale` 参数
- 默认 100.0，可以尝试 50.0 ~ 300.0

### 箭头太长/太短
- 调整 `--force_scale` 参数
- 默认 0.1，可以尝试 0.05 ~ 0.3
