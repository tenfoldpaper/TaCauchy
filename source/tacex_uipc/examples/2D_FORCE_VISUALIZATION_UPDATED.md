# 2D 力可视化窗口更新说明

## 更新内容

修改了 `ForceVisualizer` 类（2D OpenCV 窗口），优化了力的显示方式。

## 新的 2D 可视化方式

### Contact Forces 窗口（OpenCV）

**之前：**
- ❌ 黄色箭头：合力
- ❌ 绿色箭头：法向力
- ❌ 红色箭头：切向力
- **问题**：三种箭头重叠，难以区分

**现在：**
- ✅ **红色箭头**：切向力（方向和长度表示力）
- ✅ **绿色圆圈**：法向力（圆圈大小表示力的大小）
- **优势**：清晰直观，不同类型的力用不同视觉元素

## 3D 场景可视化（未改变）

3D 场景中仍然显示所有力的箭头：
- 黄色箭头：合力
- 绿色箭头：法向力
- 红色箭头：切向力
- 白色点：接触点
- 青色线框：Tet mesh

## 可视化原理

### 2D 窗口 - 法向力（绿色圆圈）

```python
def _draw_normal_circle(self, image, x, y, normal_mag):
    radius = int(normal_mag * radius_scale)  # radius_scale = 50.0
    radius = max(3, min(30, radius))  # 限制在 3-30 像素
    cv2.circle(image, (x, y), radius, green, -1)  # 填充圆
    cv2.circle(image, (x, y), radius, white, 1)   # 白色边框
```

- **小圆圈**：法向力小（轻微接触）
- **大圆圈**：法向力大（强烈压力）
- **白色边框**：提高可见度

### 2D 窗口 - 切向力（红色箭头）

```python
def _draw_tangent_arrow(self, image, x, y, tangent):
    # 计算箭头方向和长度
    dx = tangent[0] * scale
    dy = -tangent[1] * scale
    # 绘制箭头
    cv2.arrowedLine(image, (x, y), (end_x, end_y), red, 2)
```

- **箭头方向**：切向力的方向（摩擦力方向）
- **箭头长度**：切向力的大小
- **线宽 2 像素**：清晰可见

## 使用方法

```bash
# 基本运行
python source/tacex_uipc/examples/contact_force_visualization_demo.py

# 调整力箭头缩放
python source/tacex_uipc/examples/contact_force_visualization_demo.py --force_scale 0.15
```

## 可视化窗口

运行脚本后会看到三个窗口：

### 1. Isaac Sim 主窗口（3D 场景）
- 青色线框：Tet mesh
- 黄/绿/红箭头：所有力
- 白色点：接触点

### 2. Contact Forces 窗口（2D，本次更新）
- **红色箭头**：切向力
- **绿色圆圈**：法向力（大小 = 力的强度）
- 统计信息和图例

### 3. Tet Mesh Deformation 窗口（2D）
- 颜色编码的变形热图
- 蓝色（无变形）→ 红色（最大变形）

## 图例说明

Contact Forces 窗口左下角的图例：

```
→ Tangent Force      (红色箭头)
● Normal Force       (绿色圆圈)
  (circle size = magnitude)
Contacts: 15
```

顶部统计信息：
```
Max Force: 0.234 N
Total: [0.12, -0.05, 0.18] N
Max Normal: 0.20N, Tangent: 0.08N
```

## 参数调整

### 法向力圆圈大小

在代码中调整 `radius_scale`（默认 50.0）：

```python
# 在 _draw_normal_circle 方法中
radius_scale = 50.0  # 增大 = 圆圈更大，减小 = 圆圈更小
```

建议范围：30.0 ~ 100.0

### 切向力箭头长度

使用命令行参数：

```bash
# 增大箭头
python ... --force_scale 0.2

# 减小箭头
python ... --force_scale 0.05
```

## 技术细节

### 圆圈大小计算

```python
radius = int(normal_mag * 50.0)  # 力 × 缩放因子
radius = max(3, min(30, radius))  # 限制范围
```

- 最小半径：3 像素（确保可见）
- 最大半径：30 像素（避免太大）
- 力 0.1N → 半径 5 像素
- 力 0.5N → 半径 25 像素

### 箭头长度计算

```python
scale = force_scale * image_size * 1000
dx = tangent[0] * scale
dy = -tangent[1] * scale
length = sqrt(dx² + dy²)
length = max(5, min(80, length))  # 限制范围
```

- 最小长度：5 像素
- 最大长度：80 像素

### 过滤小力

```python
if force_mag < 0.0001:  # < 0.1 mN
    return  # 不绘制
```

避免噪声和微小力的干扰。

## 物理意义

### 为什么法向力用圆圈？

1. **方向固定**：法向力总是垂直于表面
2. **只需表示大小**：不需要显示方向
3. **直观**：圆圈大小直接对应力的强度
4. **不遮挡**：圆圈比箭头占用空间小

### 为什么切向力用箭头？

1. **方向重要**：切向力可以是任意方向（摩擦力）
2. **需要表示方向和大小**：箭头同时显示两者
3. **物理直观**：箭头是力的标准表示方式

## 示例场景

### 球体静止压在 gelpad 上

**2D 窗口显示：**
- 接触区域出现绿色圆圈
- 中心圆圈最大（压力最大）
- 边缘圆圈较小（压力较小）
- 没有红色箭头（无切向力）

### 球体滚动

**2D 窗口显示：**
- 绿色圆圈：接触区域的法向力
- 红色箭头：指向滚动方向（摩擦力）
- 箭头长度表示摩擦力大小

### 物体滑动

**2D 窗口显示：**
- 绿色圆圈：压力分布
- 红色箭头：指向滑动方向（动摩擦力）
- 箭头较长（滑动摩擦力通常较大）

## 与 3D 可视化的对比

| 特性 | 3D 场景 | 2D 窗口 (Contact Forces) |
|------|---------|--------------------------|
| 合力 | ✅ 黄色箭头 | ❌ 不显示 |
| 法向力 | ✅ 绿色箭头 | ✅ 绿色圆圈（大小=力） |
| 切向力 | ✅ 红色箭头 | ✅ 红色箭头 |
| 接触点 | ✅ 白色点 | ✅ 圆圈/箭头起点 |
| Tet mesh | ✅ 青色线框 | ❌ 不显示 |
| 优势 | 完整的力信息 | 清晰的力分解 |

## 代码结构

### 新增方法

```python
class ForceVisualizer:
    def _draw_tangent_arrow(self, image, x, y, tangent):
        """绘制切向力箭头（红色）"""
        
    def _draw_normal_circle(self, image, x, y, normal_mag):
        """绘制法向力圆圈（绿色，大小=力）"""
```

### 修改方法

```python
def draw_force_arrows(self, base_image, force_data, vertex_positions_2d):
    """主绘制方法 - 调用上述两个方法"""
    
def _add_legend(self, image, force_data):
    """更新图例 - 显示箭头和圆圈"""
```

## 调试输出

脚本不会为 2D 可视化打印额外的调试信息，但你可以在统计框中看到：
- 接触点数量
- 最大力
- 总力向量
- 最大法向力和切向力

## 性能说明

- 2D 可视化使用 OpenCV，非常高效
- 每帧重绘整个图像
- 性能影响可忽略（< 1ms）
- 适合实时可视化

## 故障排除

### 看不到绿色圆圈
1. 检查是否有法向力：查看统计框
2. 增大 `radius_scale`（在代码中）
3. 确保物体与 gelpad 有接触

### 圆圈太小
- 在 `_draw_normal_circle` 中增大 `radius_scale`
- 默认 50.0，可以改为 80.0 或 100.0

### 圆圈太大
- 减小 `radius_scale`
- 可以改为 30.0 或 40.0

### 看不到红色箭头
1. 检查是否有切向力：查看统计框
2. 增大 `--force_scale` 参数
3. 尝试让物体滑动或滚动

## 总结

这次更新使 2D Contact Forces 窗口的可视化更加清晰：
- ✅ 移除了合力和法向力箭头
- ✅ 法向力改为圆圈（大小表示力）
- ✅ 切向力保持箭头（方向和长度）
- ✅ 图例更新，说明更清晰
- ✅ 3D 场景保持不变（显示所有力）

这种方式更符合物理直觉，也更容易理解力的分布！
