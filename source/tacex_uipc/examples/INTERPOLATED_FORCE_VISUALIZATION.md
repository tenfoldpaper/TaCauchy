# 插值力可视化说明

## 新增功能

添加了 `InterpolatedForceVisualizer` 类，将稀疏的接触点力数据插值到规则的 100×100 网格上，并在两个新的 OpenCV 窗口中显示。

## 可视化窗口

运行脚本后会看到**5个窗口**：

### 1. Isaac Sim 主窗口（3D 场景）
- 青色线框：Tet mesh
- 黄/绿/红箭头：所有力
- 白色点：接触点

### 2. Contact Forces 窗口（2D，稀疏）
- 红色箭头：切向力
- 绿色圆圈：法向力
- 显示原始接触点的力

### 3. Normal Force (Interpolated) 窗口（2D，新增）
- **100×100 规则网格**
- **绿色圆圈**：法向力
- 圆圈大小 = 插值后的力大小
- 平滑的力分布

### 4. Tangential Force (Interpolated) 窗口（2D，新增）
- **100×100 规则网格**
- **红色箭头**：切向力
- 箭头方向和长度 = 插值后的力
- 平滑的力场

### 5. Tet Mesh Deformation 窗口（2D）
- 变形热图

## 插值原理

### 问题
原始接触点是稀疏且不规则的：
- 接触点数量：10-30 个（取决于接触情况）
- 位置：随机分布，不规则
- 难以看出整体的力分布模式

### 解决方案
使用**三次插值**（cubic interpolation）将稀疏数据插值到规则网格：

```python
from scipy.interpolate import griddata

# 稀疏数据点
points = [(x1, y1), (x2, y2), ..., (xN, yN)]  # N 个接触点
values = [f1, f2, ..., fN]  # 对应的力值

# 规则网格
grid_x, grid_y = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100))

# 插值
force_grid = griddata(points, values, (grid_x, grid_y), method='cubic')
```

### 插值方法
- **cubic**：三次插值，平滑且连续
- **fill_value=0.0**：网格外的区域填充为 0
- **负值截断**：法向力不能为负，截断到 0

## 可视化细节

### 法向力网格（绿色圆圈）

```python
# 对于每个网格单元 (i, j)
force_mag = normal_grid[i, j]

# 计算圆圈半径
normalized_force = force_mag / max_force
radius = normalized_force * max_radius

# 颜色强度
intensity = int(255 * normalized_force)
color = (0, intensity, 0)  # 绿色

# 绘制圆圈
cv2.circle(image, center, radius, color, -1)
cv2.circle(image, center, radius, white, 1)  # 白色边框
```

**特点：**
- 小圆圈 = 小法向力
- 大圆圈 = 大法向力
- 颜色深浅也表示力的大小
- 白色边框提高可见度

### 切向力网格（红色箭头）

```python
# 对于每个网格单元 (i, j)
fx = tangent_x_grid[i, j]
fy = tangent_y_grid[i, j]
force_mag = sqrt(fx^2 + fy^2)

# 计算箭头方向和长度
normalized_force = force_mag / max_force
dx = fx / force_mag * normalized_force * arrow_scale
dy = -fy / force_mag * normalized_force * arrow_scale

# 颜色强度
intensity = int(255 * normalized_force)
color = (0, 0, intensity)  # 红色

# 绘制箭头
cv2.arrowedLine(image, start, end, color, 1)
```

**特点：**
- 箭头方向 = 切向力方向
- 箭头长度 = 切向力大小
- 颜色深浅也表示力的大小
- 规则排列，易于观察模式

## 统计信息

每个插值窗口显示：
- **Grid**: 网格大小（100×100）
- **Max**: 最大力值（牛顿）
- **Active cells**: 有显著力的网格单元数量

## 使用方法

```bash
# 基本运行（自动显示插值窗口）
python source/tacex_uipc/examples/contact_force_visualization_demo.py

# 调整力箭头缩放
python source/tacex_uipc/examples/contact_force_visualization_demo.py --force_scale 0.15
```

## 参数调整

### 网格大小

在代码中修改：

```python
interpolated_force_visualizer = InterpolatedForceVisualizer(
    grid_size=(100, 100),  # 改为 (50, 50) 或 (200, 200)
    image_size=(512, 512)
)
```

- **50×50**：更快，但不够平滑
- **100×100**：平衡（推荐）
- **200×200**：更平滑，但计算慢

### 圆圈/箭头大小

在 `_visualize_normal_forces` 中：

```python
max_radius = min(cell_w, cell_h) * 0.4  # 改为 0.3 或 0.5
```

在 `_visualize_tangent_forces` 中：

```python
arrow_scale = min(cell_w, cell_h) * 0.8  # 改为 0.6 或 1.0
```

## 应用场景

### 1. 球体压在 gelpad 上

**法向力插值窗口：**
- 中心区域：大绿色圆圈（压力集中）
- 周围区域：逐渐变小的圆圈（压力扩散）
- 平滑的压力分布

**切向力插值窗口：**
- 如果球静止：几乎没有箭头
- 如果球滚动：箭头指向滚动方向

### 2. 物体滑动

**法向力插值窗口：**
- 接触区域的压力分布

**切向力插值窗口：**
- 箭头指向滑动方向
- 箭头长度表示摩擦力大小
- 可以看到摩擦力场的分布

### 3. 复杂接触

**优势：**
- 稀疏的接触点 → 平滑的力场
- 容易识别力的分布模式
- 规则网格便于分析

## 技术细节

### 插值算法

使用 `scipy.interpolate.griddata` 的 cubic 方法：

**优点：**
- 平滑且连续
- 二阶导数连续
- 适合力场可视化

**缺点：**
- 可能产生振荡（Runge 现象）
- 边界外推不准确（使用 fill_value=0）

### 性能

- **插值计算**：~5-10ms（100×100 网格）
- **绘制**：~2-3ms
- **总计**：~10-15ms per frame
- 对实时可视化影响很小

### 内存

- 100×100 网格：
  - normal_grid: 10000 × 8 bytes = 80 KB
  - tangent_x_grid: 80 KB
  - tangent_y_grid: 80 KB
  - 总计：~240 KB
- 可忽略不计

## 与原始可视化的对比

| 特性 | Contact Forces (原始) | Interpolated (插值) |
|------|----------------------|---------------------|
| 数据点 | 稀疏（10-30个） | 密集（100×100=10000个） |
| 分布 | 不规则 | 规则网格 |
| 平滑度 | 离散 | 平滑连续 |
| 模式识别 | 困难 | 容易 |
| 计算量 | 小 | 中等 |
| 适用场景 | 精确位置 | 整体分布 |

## 代码结构

```python
class InterpolatedForceVisualizer:
    def __init__(self, grid_size, image_size):
        """初始化，创建规则网格"""
        
    def interpolate_forces(self, force_data, vertex_positions_2d):
        """插值到规则网格"""
        # 使用 scipy.interpolate.griddata
        
    def visualize(self, force_data, vertex_positions_2d):
        """创建并显示可视化"""
        
    def _visualize_normal_forces(self, normal_grid):
        """绘制法向力网格（绿色圆圈）"""
        
    def _visualize_tangent_forces(self, tangent_x_grid, tangent_y_grid):
        """绘制切向力网格（红色箭头）"""
```

## 调试输出

脚本会打印：

```
[InterpolatedForceVis] OpenCV windows created
```

如果窗口创建失败，会保存到文件：
- `/tmp/force_normal_interpolated.png`
- `/tmp/force_tangent_interpolated.png`

## 故障排除

### 看不到插值窗口
1. 检查是否有接触：查看 Contact Forces 窗口
2. 检查控制台是否有错误信息
3. 检查 `/tmp/` 目录是否有保存的图像

### 插值结果不平滑
1. 增加网格大小：(100, 100) → (200, 200)
2. 检查接触点数量：至少需要 4-5 个点
3. 接触点太少时插值效果不好

### 插值结果有振荡
1. 这是 cubic 插值的特性（Runge 现象）
2. 可以改用 'linear' 方法（但不够平滑）
3. 增加接触点数量可以改善

### 性能问题
1. 减小网格大小：(100, 100) → (50, 50)
2. 降低更新频率：每 N 帧更新一次
3. 使用 'linear' 代替 'cubic'

## 扩展功能

### 保存插值数据

```python
# 在 interpolate_forces 后
np.save('normal_grid.npy', normal_grid)
np.save('tangent_x_grid.npy', tangent_x_grid)
np.save('tangent_y_grid.npy', tangent_y_grid)
```

### 导出为热图

```python
import matplotlib.pyplot as plt

plt.imshow(normal_grid, cmap='hot', interpolation='bilinear')
plt.colorbar(label='Normal Force (N)')
plt.savefig('normal_force_heatmap.png')
```

### 3D 表面图

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid_x, grid_y, normal_grid, cmap='viridis')
plt.show()
```

## 总结

插值力可视化提供了：
- ✅ 规则的 100×100 网格
- ✅ 平滑的力分布
- ✅ 易于识别模式
- ✅ 两个独立窗口（法向力和切向力）
- ✅ 实时更新
- ✅ 统计信息显示

这使得力的分布更加直观和易于分析！
