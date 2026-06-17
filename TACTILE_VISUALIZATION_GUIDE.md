# TacEx 触觉传感器可视化系统开发文档

## 目录

1. [概述](#概述)
2. [系统架构](#系统架构)
3. [核心可视化工具](#核心可视化工具)
4. [使用指南](#使用指南)
5. [实现细节](#实现细节)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)

---

## 概述

### 设计目标

TacEx 的触觉可视化系统旨在解决以下问题：

1. **复杂接触数据的可理解性** - 将高维接触数据可视化为直观的 2D 图像
2. **实时可视化反馈** - 在仿真运行时即时显示接触力分布
3. **多传感器支持** - 统一的可视化框架支持 GelSight、Digit、9D Touch 等
4. **研究和调试** - 便于理解网格细化的实际效果

### 核心特性

✅ **多层次可视化**：
- 点力可视化（direct force visualization）
- 插值力场可视化（regular grid visualization）
- RGB 图像显示（tactile image viewer）

✅ **灵活的数据处理**：
- 支持稀疏接触数据自动插值
- 噪声自动过滤
- 自适应颜色映射

✅ **交互式操作**：
- 点击保存可视化结果
- 实时窗口调整
- 双向数据流支持

---

## 系统架构

### 整体框架

```
┌─────────────────────────────────────────────────────┐
│     触觉可视化系统（Tactile Visualization）        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │ 1. 数据获取层（Data Acquisition）         │    │
│  │ ├─ StressExtractor (应力提取)             │    │
│  │ ├─ ContactForceExtractor (接触力提取)    │    │
│  │ └─ SensorDataCollector (传感器数据收集)  │    │
│  └────────────────────────────────────────────┘    │
│                    ↓ 数据流                        │
│  ┌────────────────────────────────────────────┐    │
│  │ 2. 数据处理层（Data Processing）         │    │
│  │ ├─ ForceInterpolator (力插值器)          │    │
│  │ ├─ NoiseFilter (噪声过滤器)              │    │
│  │ ├─ CoordinateTransformer (坐标变换)      │    │
│  │ └─ AspectRatioHandler (宽高比处理)      │    │
│  └────────────────────────────────────────────┘    │
│                    ↓ 处理结果                      │
│  ┌────────────────────────────────────────────┐    │
│  │ 3. 可视化层（Visualization）              │    │
│  │ ├─ TactileForceVisualizer (点力可视化)   │    │
│  │ ├─ InterpolatedForceVisualizer (栅格)   │    │
│  │ ├─ TactileImageViewer (图像显示器)      │    │
│  │ └─ MeshVisualizer (网格可视化)           │    │
│  └────────────────────────────────────────────┘    │
│                    ↓ 可视结果                      │
│  ┌────────────────────────────────────────────┐    │
│  │ 4. 输出层（Output）                       │    │
│  │ ├─ OpenCV Windows (窗口显示)             │    │
│  │ ├─ PNG Export (图像导出)                  │    │
│  │ ├─ Data Logging (数据日志)                │    │
│  │ └─ Real-time Streaming (实时流)          │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 组件关系图

```
仿真环境 (IsaacSim/Isaac Gym)
    ↓ 物理数据
    ├─→ UipcObject (可形变对象)
    │    ↓
    │    ├─→ 网格顶点位置 (nodal_pos)
    │    └─→ 应力数据 (stress_data)
    │
    ├─→ GelSight Mini (传感器)
    │    ↓
    │    └─→ RGB 图像
    │
    └─→ ContactForce (接触力)
         ↓
         └─→ 法向压力 + 切向剪力

         |
         v  数据处理和可视化
         
    ┌─────────────────────────┐
    │  可视化处理器链         │
    ├─────────────────────────┤
    │ 1. 数据验证             │
    │ 2. 坐标转换             │
    │    - 世界坐标           │
    │    - 本地坐标           │
    │    - 2D 投影坐标        │
    │ 3. 插值和平滑           │
    │ 4. 噪声过滤             │
    │ 5. 颜色映射             │
    │ 6. 图像生成             │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │  可视化输出             │
    ├─────────────────────────┤
    │ • OpenCV 窗口          │
    │ • PNG 文件             │
    │ • 实时显示             │
    │ • 性能指标             │
    └─────────────────────────┘
```

---

## 核心可视化工具

### 1. TactileForceVisualizer（点力可视化器）

#### 概述

直接可视化接触点上的法向和切向力。每个点用一个形状表示（圆形表示法向力，箭头表示切向力）。

#### 类定义

```python
class TactileForceVisualizer:
    """
    简单触觉力可视化器，用于演示应用。
    
    在两个 OpenCV 窗口中显示法向压力和切向牵引力分布。
    """
    
    def __init__(self, 
                 image_size: tuple = (512, 512),
                 show_legend: bool = True,
                 save_dir: str = "/home/ly/data/hengfei/getdata/force_field"):
        """初始化可视化器"""
        self.image_size = image_size          # 输出图像大小
        self.show_legend = show_legend        # 是否显示图例
        self.save_dir = Path(save_dir)        # 保存目录
        self.max_normal = 0.0                 # 法向力最大值（用于归一化）
        self.max_tangent = 0.0                # 切向力最大值（用于归一化）
```

#### 核心方法

```python
def visualize(self, stress_data, gelpad_positions: np.ndarray):
    """
    从应力数据进行可视化。
    
    Args:
        stress_data: 应力数据对象，包含：
            - normal_pressure: (N,) 法向压力数组
            - tangential_traction: (N, 3) 切向剪力数组
            - triangle_indices: (N, 3) 三角形索引
        gelpad_positions: (V, 3) 网格顶点位置
    """
    # 1. 提取力数据
    normal_pressure = np.abs(stress_data.normal_pressure)
    tangential_traction = stress_data.tangential_traction
    
    # 2. 计算三角形中心
    tri_indices = stress_data.triangle_indices
    tri_centers = (gelpad_positions[tri_indices[:, 0]] +
                   gelpad_positions[tri_indices[:, 1]] +
                   gelpad_positions[tri_indices[:, 2]]) / 3.0
    
    # 3. 投影到 2D
    positions_2d = self._project_to_2d(tri_centers)
    
    # 4. 绘制可视化
    normal_img = self._draw_scalar_field(positions_2d, normal_pressure, 
                                        "Normal Pressure", "Pa", "green")
    tangent_img = self._draw_tangent_arrows(positions_2d, tangential_traction,
                                           "Tangential Traction", "Pa")
    
    # 5. 显示
    cv2.imshow("Normal Pressure", normal_img)
    cv2.imshow("Tangential Traction", tangent_img)
```

#### 渲染详解

##### 法向力渲染（绿色圆形）

```python
def _draw_scalar_field(self, positions_2d, values, title, unit, color_map="green"):
    """
    绘制标量场（法向力）。
    
    使用圆形大小和颜色深度表示力的大小。
    """
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255  # 白色背景
    
    # 按力的大小排序（小的先画，大的后画，这样大的会在小的上面）
    sorted_indices = np.argsort(values)
    
    for idx in sorted_indices:
        pos = positions_2d[idx]
        val = values[idx]
        
        # 归一化力的大小
        norm_val = val / max(self.max_normal, 0.001)
        
        # 计算圆的半径
        radius = int(4 + norm_val * 18)  # 4-22 像素
        
        # 计算颜色（从浅绿到深绿）
        # 注意：OpenCV 使用 BGR 顺序
        intensity = int(255 * (1 - norm_val ** 0.5))  # 平方根映射
        color = (intensity, 255, intensity)  # BGR: (B, G, R)
        
        # 画圆
        cv2.circle(img, (x, y), radius, color, -1)  # -1 表示填充
    
    return img
```

**颜色映射**：
- 弱力（norm_val ≈ 0）：浅绿 (255, 255, 255) → 白色
- 中等力（norm_val ≈ 0.5）：中绿 (128, 255, 128)
- 强力（norm_val ≈ 1.0）：深绿 (0, 255, 0)

##### 切向力渲染（红色箭头）

```python
def _draw_tangent_arrows(self, positions_2d, tangential_traction, title, unit):
    """
    绘制切向力为红色箭头。
    
    箭头方向表示力的方向，长度和颜色表示力的大小。
    """
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    
    # 提取切向分量 (X, Y)
    tangent_xy = tangential_traction[:, :2]  # (N, 2)
    tangent_mag = np.linalg.norm(tangent_xy, axis=1)  # (N,)
    tangent_dirs = tangent_xy / (tangent_mag[:, np.newaxis] + 1e-6)  # 单位化
    
    sorted_indices = np.argsort(tangent_mag)
    
    for idx in sorted_indices:
        pos = positions_2d[idx]
        mag = tangent_mag[idx]
        direction = tangent_dirs[idx]
        
        x, y = int(pos[0]), int(pos[1])
        
        # 归一化大小
        norm_mag = mag / max(self.max_tangent, 0.001)
        
        # 箭头长度（20-80 像素）
        arrow_len = int(20 + norm_mag * 60)
        
        # 箭头厚度（3-13 像素）
        thickness = max(3, int(3 + norm_mag * 10))
        
        # 颜色（从浅红到深红）
        color_intensity = norm_mag ** 0.5
        if color_intensity < 0.3:
            # 很弱：浅红
            intensity = int(255 - color_intensity * 600)
            color = (intensity, intensity, 255)
        else:
            # 中等到强：深红
            intensity = int(255 * (1 - color_intensity * 0.9))
            color = (0, 0, 255 - intensity)
        
        # 计算箭头端点
        # 注意：图像坐标系 Y 向下，需要翻转
        end_x = int(x + direction[0] * arrow_len)
        end_y = int(y - direction[1] * arrow_len)  # 翻转 Y
        
        # 画箭头
        cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness, tipLength=0.25)
        
        # 在起点画圆（覆盖箭头的起点）
        circle_radius = int(4 + norm_mag * 18)
        cv2.circle(img, (x, y), circle_radius, color, -1)
    
    return img
```

**箭头设计**：
- **方向**：从接触点指向切向力方向
- **长度**：与力大小成正比（20-80 像素）
- **颜色**：从浅红到深红
- **粗细**：从细（3px）到粗（13px）

#### 交互功能

```python
def _mouse_callback_normal(self, event, x, y, flags, param):
    """点击保存法向力图像"""
    if event == cv2.EVENT_LBUTTONDOWN:
        filename = f"normal_pressure_{self.save_counter_normal:04d}.png"
        filepath = self.save_dir / filename
        cv2.imwrite(str(filepath), self.current_normal_img)
        self.save_counter_normal += 1
        print(f"✓ 已保存: {filepath}")
```

---

### 2. InterpolatedForceVisualizer（插值力场可视化器）

#### 概述

使用规则网格对稀疏的接触点进行插值，生成连续的力场可视化。比点力可视化提供更好的连续性。

#### 关键特性

```python
class InterpolatedForceVisualizer:
    """
    在规则网格上可视化插值的力。
    
    特性：
    - 自动插值稀疏接触数据
    - 噪声自动过滤
    - 自适应宽高比处理
    """
    
    def __init__(self,
                 grid_size: tuple = (100, 100),
                 image_size: tuple = (512, 512),
                 enable_window: bool = False,
                 noise_threshold_percentile: float = 10.0,
                 save_dir: str = "/tmp/force_field"):
        """
        初始化插值可视化器。
        
        Args:
            grid_size: 插值网格大小 (height, width)
            image_size: 输出图像大小 (width, height)
            enable_window: 是否创建 OpenCV 窗口
            noise_threshold_percentile: 噪声阈值百分位数 (0-100)
                低于此百分位数的值被视为噪声并过滤
            save_dir: 点击保存时的目录
        """
        self.grid_size = grid_size
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size[1]),
            np.linspace(0, 1, grid_size[0])
        )
        self.noise_threshold_percentile = noise_threshold_percentile
```

#### 核心工作流

```
输入：稀疏接触数据
  │
  ├─→ [1] 坐标系转换
  │       从世界坐标 → 本地坐标 → 2D 投影坐标
  │
  ├─→ [2] 噪声过滤
  │       使用百分位数阈值过滤低值点
  │
  ├─→ [3] 插值处理
  │       使用 scipy.interpolate.griddata
  │       方法：cubic (首选) 或 linear (备选)
  │
  ├─→ [4] 颜色映射
  │       将插值值映射到 RGB 颜色
  │
  └─→ 输出：连续力场图像
```

#### 插值算法

```python
def _interpolate_values(self, 
                        normal_magnitudes: np.ndarray,
                        tangent_x: np.ndarray,
                        tangent_y: np.ndarray,
                        vertex_positions_2d: np.ndarray,
                        noise_threshold_percentile: float = 10.0):
    """
    使用给定值进行插值到规则网格。
    
    原理：
    1. 接收稀疏点和对应的值
    2. 使用三次样条插值到规则网格
    3. 自动噪声过滤
    """
    
    # ─── 噪声过滤 ───
    if noise_threshold_percentile > 0:
        threshold = np.percentile(normal_magnitudes, noise_threshold_percentile)
        valid_mask = normal_magnitudes >= threshold
        
        # 过滤太激进时保留所有点
        if np.sum(valid_mask) < 4:
            valid_mask = normal_magnitudes > 0.0001
        
        # 应用过滤
        vertex_positions_2d = vertex_positions_2d[valid_mask]
        normal_magnitudes = normal_magnitudes[valid_mask]
        tangent_x = tangent_x[valid_mask]
        tangent_y = tangent_y[valid_mask]
    
    # ─── 插值 ───
    try:
        # 优先使用三次样条（更平滑）
        normal_grid = griddata(
            vertex_positions_2d[:, :2],  # 2D 坐标
            normal_magnitudes,           # 值
            (self.grid_x, self.grid_y),  # 目标网格
            method='cubic',              # 三次样条
            fill_value=0.0               # 无数据处理
        )
    except Exception:
        # 降级到线性插值
        normal_grid = griddata(
            vertex_positions_2d[:, :2],
            normal_magnitudes,
            (self.grid_x, self.grid_y),
            method='linear',
            fill_value=0.0
        )
    
    # 确保无负值
    normal_grid = np.maximum(normal_grid, 0.0)
    
    return normal_grid, tangent_x_grid, tangent_y_grid
```

#### 可视化方法

```python
def visualize_from_values(self,
                          normal_magnitudes: np.ndarray,
                          tangential_vectors: np.ndarray,
                          vertex_positions_2d: np.ndarray = None):
    """
    从原始值直接可视化插值力场。
    
    Args:
        normal_magnitudes: (N,) 法向力大小
        tangential_vectors: (N, 2) 切向力向量 (X, Y)
        vertex_positions_2d: (N, 2) 顶点 2D 位置
    """
    
    # 1. 插值
    normal_grid, tangent_x_grid, tangent_y_grid = self._interpolate_values(
        normal_magnitudes, tangential_vectors[:, 0], tangential_vectors[:, 1],
        vertex_positions_2d,
        noise_threshold_percentile=self.noise_threshold_percentile
    )
    
    # 2. 生成图像
    normal_img = self._grid_to_image(normal_grid, "Normal Force")
    tangent_img = self._grid_to_image_vector(tangent_x_grid, tangent_y_grid, "Tangential Force")
    
    # 3. 存储用于保存
    self.current_normal_interp_img = normal_img
    self.current_tangent_interp_img = tangent_img
    
    # 4. 显示
    if self.use_cv_window:
        cv2.imshow("Normal Force (Interpolated)", normal_img)
        cv2.imshow("Tangential Force (Interpolated)", tangent_img)
        cv2.waitKey(1)
```

---

### 3. TactileImageViewer（触觉RGB图像查看器）

#### 概述

显示传感器的 RGB 图像输出（例如 GelSight 的凝胶表面形变图像）。

#### 实现

```python
class TactileImageViewer:
    """
    触觉 RGB 图像查看器。
    
    显示传感器的原始 RGB 输出，通常用于视觉调试。
    """
    
    def __init__(self, 
                 window_name: str = "Tactile RGB",
                 save_dir: str = "./tactile_images"):
        self.window_name = window_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_counter = 0
        self.current_img = None
        
        # 创建窗口
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            cv2.setMouseCallback(window_name, self._mouse_callback)
            print(f"[TactileImageViewer] ✓ 窗口创建: '{window_name}'")
            print(f"[TactileImageViewer] 点击保存到: {save_dir}")
        except cv2.error as e:
            print(f"[TactileImageViewer] ✗ 无法创建窗口: {e}")
    
    def show(self, img: np.ndarray):
        """
        显示 RGB 图像。
        
        Args:
            img: (H, W, 3) RGB 或 BGR 图像
        """
        if img is None:
            return
        
        # 确保是 uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        self.current_img = img
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """点击保存图像"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_img is not None:
            filename = f"{self.window_name.replace(' ', '_')}_{self.save_counter:04d}.png"
            filepath = self.save_dir / filename
            cv2.imwrite(str(filepath), self.current_img)
            self.save_counter += 1
            print(f"[TactileImageViewer] ✓ 已保存: {filepath}")
```

---

## 使用指南

### 1. 基础使用流程

```python
from tacex_uipc.utils.visualization import (
    TactileForceVisualizer,
    InterpolatedForceVisualizer,
    TactileImageViewer
)
from tacex_uipc.sim import StressExtractor

# ─── 初始化可视化器 ───
force_viz = TactileForceVisualizer(
    image_size=(512, 512),
    save_dir="/path/to/save"
)

interpolated_viz = InterpolatedForceVisualizer(
    grid_size=(50, 50),
    image_size=(512, 512),
    enable_window=True,
    noise_threshold_percentile=15.0,
    save_dir="/path/to/save"
)

image_viewer = TactileImageViewer(
    window_name="GelSight RGB",
    save_dir="/path/to/save"
)

stress_extractor = StressExtractor()

# ─── 在仿真循环中使用 ───
while simulation_app.is_running():
    # 仿真步进
    env.sim.step()
    
    # 获取数据
    gelpad.update(dt)
    stress_data = stress_extractor.get_object_surface_stress(gelpad)
    gelpad_pos = gelpad.data.nodal_pos_w.cpu().numpy()
    
    # 可视化
    # 1. 点力可视化
    force_viz.visualize(stress_data, gelpad_pos)
    
    # 2. 插值力场（需要坐标变换）
    sensor_pos = robot.data.body_link_pos_w[0, sensor_idx]
    sensor_quat = robot.data.body_link_quat_w[0, sensor_idx]
    
    # 转换到本地坐标系
    gelpad_local = transform_to_sensor_frame(gelpad_pos, sensor_pos, sensor_quat)
    positions_2d = project_to_2d(gelpad_local)
    
    interpolated_viz.visualize_from_values(
        normal_magnitudes=stress_data.normal_pressure,
        tangential_vectors=stress_data.tangential_traction[:, :2],
        vertex_positions_2d=positions_2d
    )
    
    # 3. RGB 图像
    tactile_rgb = sensor.data.output.get("tactile_rgb")[0]
    image_viewer.show(tactile_rgb)
```

### 2. 坐标系变换详解

```python
def transform_to_sensor_frame(world_points, sensor_pos, sensor_quat_wxyz):
    """
    将世界坐标转换为传感器本地坐标系。
    
    Args:
        world_points: (N, 3) 世界坐标点
        sensor_pos: (3,) 传感器位置
        sensor_quat_wxyz: (4,) 传感器方向四元数 (w, x, y, z)
    
    Returns:
        local_points: (N, 3) 本地坐标点
    """
    # 提取四元数分量
    w, x, y, z = sensor_quat_wxyz
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    # 转换：p_local = R^T * (p_world - t)
    local_points = []
    for p in world_points:
        p_local = R.T @ (p - sensor_pos)
        local_points.append(p_local)
    
    return np.array(local_points)


def project_to_2d(local_points_3d, preserve_aspect_ratio=True):
    """
    将 3D 本地坐标投影到 2D 图像坐标。
    
    假设：
    - Z 轴为法线方向（垂直于接触面）
    - X-Y 平面为接触面
    
    Args:
        local_points_3d: (N, 3) 本地坐标
        preserve_aspect_ratio: 是否保持宽高比
    
    Returns:
        points_2d: (N, 2) 2D 投影坐标 (pixel)
    """
    x = local_points_3d[:, 0]
    y = local_points_3d[:, 1]
    
    # 计算投影范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_range = max(x_max - x_min, 0.001)
    y_range = max(y_max - y_min, 0.001)
    
    # 计算宽高比
    aspect_ratio = x_range / y_range
    
    # 确定输出图像尺寸
    base_size = 512
    margin = 50
    
    if preserve_aspect_ratio:
        if aspect_ratio > 1.0:
            img_w = base_size
            img_h = int(base_size / aspect_ratio)
        else:
            img_h = base_size
            img_w = int(base_size * aspect_ratio)
    else:
        img_w = img_h = base_size
    
    # 归一化并映射到图像坐标
    available_w = img_w - 2 * margin
    available_h = img_h - 2 * margin
    
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    
    x_img = margin + x_norm * available_w
    y_img = margin + (1 - y_norm) * available_h  # Y 翻转（图像坐标系）
    
    return np.column_stack([x_img, y_img]).astype(int)
```

### 3. 配置最佳实践

```python
# ───────────────────────────────────────────
# 场景1：实时演示（速度优先）
# ───────────────────────────────────────────
demo_config = {
    "force_viz": {
        "image_size": (512, 512),
        "show_legend": False
    },
    "interp_viz": {
        "grid_size": (40, 40),        # 较小网格，更快
        "enable_window": True,
        "noise_threshold_percentile": 20.0,  # 激进过滤
    }
}

# ───────────────────────────────────────────
# 场景2：离线分析（精度优先）
# ───────────────────────────────────────────
analysis_config = {
    "force_viz": {
        "image_size": (1024, 1024),   # 大尺寸
        "show_legend": True
    },
    "interp_viz": {
        "grid_size": (200, 200),      # 高分辨率
        "enable_window": False,       # 无窗口显示
        "noise_threshold_percentile": 5.0,   # 保守过滤
    }
}

# ───────────────────────────────────────────
# 场景3：调试开发
# ───────────────────────────────────────────
debug_config = {
    "force_viz": {
        "image_size": (512, 512),
        "show_legend": True
    },
    "interp_viz": {
        "grid_size": (100, 100),
        "enable_window": True,
        "noise_threshold_percentile": 10.0,  # 平衡
    }
}
```

---

## 实现细节

### 1. 坐标系统

#### 三层坐标系

```
┌─────────────────────────────────────────┐
│ 世界坐标系 (World Frame)                 │
│ - 绝对位置                              │
│ - 所有对象的参考系                      │
│ - Z 通常是竖直向上                      │
└─────────────────────────────────────────┘
           ↓ 相对转换
┌─────────────────────────────────────────┐
│ 传感器本地坐标系 (Sensor Local Frame)    │
│ - 原点在传感器中心                      │
│ - Z 轴指向凝胶表面外侧（法线）          │
│ - X-Y 平面是接触面                      │
└─────────────────────────────────────────┘
           ↓ 投影变换
┌─────────────────────────────────────────┐
│ 2D 图像坐标系 (Image Frame)              │
│ - 像素坐标 (u, v)                       │
│ - 原点在左上角                          │
│ - Y 向下                                 │
└─────────────────────────────────────────┘
```

#### 变换实现

```python
# 世界坐标 → 本地坐标
p_local = R^T @ (p_world - t_sensor)

# 本地坐标 → 2D 投影
x_pixel = (x_local - x_min) / (x_max - x_min) * img_width
y_pixel = img_height - (y_local - y_min) / (y_max - y_min) * img_height  # Y翻转
```

### 2. 噪声过滤策略

```python
def apply_noise_filter(values, threshold_percentile):
    """
    基于百分位数的噪声过滤。
    
    原理：假设低值是噪声，高值是真实接触。
    """
    
    # 计算阈值
    threshold = np.percentile(values, threshold_percentile)
    
    # 创建掩码
    valid_mask = values >= threshold
    
    # 过滤太激进时的处理
    if np.sum(valid_mask) < 4:
        # 至少保留 4 个点用于插值
        sorted_indices = np.argsort(values)[::-1]  # 降序
        valid_mask[sorted_indices[:4]] = True
    
    return valid_mask

# 使用示例
# threshold_percentile = 10.0
# 去掉最小的 10% 的值（认为是噪声）
# 保留最大的 90% 的值（认为是真实信号）
```

### 3. 插值方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|--------|
| **Cubic** | 平滑，连续二阶导数 | 计算慢，可能振荡 | 离线分析、展示 |
| **Linear** | 快速，保守 | 表面不平滑，可能有棱角 | 实时应用、快速预览 |
| **Nearest** | 最快 | 完全不平滑 | 性能关键应用 |

```python
# 动态选择插值方法
def select_interpolation_method(num_points, is_realtime):
    if is_realtime or num_points < 100:
        return 'linear'  # 快速
    else:
        return 'cubic'   # 光滑
```

### 4. 颜色映射策略

#### 法向力颜色映射

```python
def normal_force_colormap(normalized_value):
    """
    将归一化的法向力映射到绿色色调。
    
    norm_val: [0, 1] 范围
    """
    # 方案 1: 线性映射
    intensity = int(255 * (1 - norm_val))
    return (intensity, 255, intensity)  # BGR
    
    # 方案 2: 非线性映射（对数）
    log_val = np.log1p(norm_val)
    intensity = int(255 * (1 - log_val / np.log(2)))
    return (intensity, 255, intensity)
    
    # 方案 3: 非线性映射（平方根）
    sqrt_val = np.sqrt(norm_val)
    intensity = int(255 * (1 - sqrt_val))
    return (intensity, 255, intensity)
```

**建议**：使用平方根映射，因为它能更好地展示中等力的细节。

#### 切向力颜色映射

```python
def tangent_force_colormap(normalized_magnitude):
    """
    将归一化的切向力映射到红色色调。
    
    使用幂函数强调中等值的差异。
    """
    color_intensity = normalized_magnitude ** 0.5
    
    if color_intensity < 0.3:
        # 很弱：浅红 (light pink)
        intensity = int(255 - color_intensity * 600)
        return (intensity, intensity, 255)
    else:
        # 中等到强：深红
        intensity = int(255 * (1 - color_intensity * 0.9))
        return (0, 0, 255 - intensity)
```

---

## 最佳实践

### 1. 窗口管理

```python
# ✅ 正确：使用上下文管理
class VisualizationManager:
    def __init__(self):
        self.force_viz = TactileForceVisualizer()
        self.interp_viz = InterpolatedForceVisualizer()
        self.image_viewer = TactileImageViewer()
    
    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()


# 使用
with VisualizationManager() as vis_mgr:
    # 仿真循环
    pass
# 自动清理

# ❌ 错误：遗留窗口
def bad_visualization():
    for i in range(100):
        viz = TactileForceVisualizer()  # 每次创建新窗口！
        # 窗口资源泄漏
```

### 2. 性能优化

```python
# ✅ 优化策略
class OptimizedVisualizer:
    def __init__(self):
        self.skip_count = 0
        self.skip_interval = 3  # 每 3 帧显示一次
    
    def visualize(self, data):
        """每 N 帧更新一次显示"""
        self.skip_count += 1
        if self.skip_count % self.skip_interval != 0:
            return
        
        # 实际可视化
        self._do_visualization(data)

# 在实时应用中使用
# FPS 从 30 → ~100 fps（通过跳帧）
```

### 3. 内存管理

```python
# ✅ 及时释放大数组
def efficient_visualization(stress_data, positions):
    """有效地处理大型数据"""
    
    # 1. 立即转换到 numpy 并释放原始数据
    stress = stress_data.normal_pressure.cpu().numpy()
    del stress_data  # 及时删除
    
    # 2. 使用原地操作
    stress_normalized = stress / stress.max()  # 避免复制
    
    # 3. 处理后立即显示，不保存中间结果
    viz.visualize(stress_normalized, positions)
    
    # 4. 显式清理
    del stress, stress_normalized
    gc.collect()
```

### 4. 错误处理

```python
# ✅ 健壮的可视化
def robust_visualization(stress_data, positions):
    """处理各种边界情况"""
    
    try:
        # 1. 输入验证
        if stress_data is None:
            print("[WARNING] 空应力数据")
            return
        
        if len(positions) == 0:
            print("[WARNING] 空位置数据")
            return
        
        # 2. 数据清理
        normal_pressure = np.abs(stress_data.normal_pressure)
        normal_pressure = np.nan_to_num(normal_pressure, nan=0.0, posinf=0.0)
        
        # 3. 检查有效数据量
        valid_count = np.sum(normal_pressure > 0.0001)
        if valid_count < 4:
            print(f"[WARNING] 有效接触点过少: {valid_count}")
            return
        
        # 4. 执行可视化
        viz.visualize(stress_data, positions)
        
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")
        traceback.print_exc()
```

---

## 常见问题

### Q1: 为什么可视化显示为空（全白）？

**答**：可能的原因和解决方案：

```python
# 原因 1: 没有接触数据
if len(stress_data.triangle_indices) == 0:
    print("无接触数据，无法显示")
    # 解决：确保物体确实与传感器接触

# 原因 2: 坐标变换错误
# 检查四元数格式
print(f"四元数格式: {quat}")  # 应该是 (w, x, y, z)

# 原因 3: 数据为 NaN 或 Inf
if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
    print("坐标包含 NaN 或 Inf")
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0)

# 原因 4: 数值太小
if np.max(normal_pressure) < 0.0001:
    print(f"法向压力太小: max={np.max(normal_pressure)}")
    # 检查单位（Pa vs MPa）
```

### Q2: 可视化性能差（低 FPS）

**答**：优化策略：

```python
# 1. 跳帧显示
if frame_count % 5 == 0:  # 每 5 帧显示一次
    visualize()

# 2. 降低分辨率
viz = InterpolatedForceVisualizer(
    grid_size=(50, 50),       # 从 (100, 100) 降低
    image_size=(256, 256)     # 从 (512, 512) 降低
)

# 3. 使用线性插值（而非三次样条）
# 在 griddata 调用中使用 method='linear'

# 4. 异步显示
import threading
viz_thread = threading.Thread(target=visualize, args=(data,), daemon=True)
viz_thread.start()
```

### Q3: 箭头方向不对

**答**：检查坐标系转换：

```python
# 问题：图像坐标系 Y 向下，世界坐标系 Y 可能向上
# 解决：显式翻转 Y 分量

# 错误：
end_y = int(y + direction[1] * arrow_length)

# 正确：
end_y = int(y - direction[1] * arrow_length)  # 减号！

# 或在数据转换时处理
tangent_directions[:, 1] *= -1  # 翻转 Y
```

### Q4: 插值结果有棱角（不光滑）

**答**：改进插值方法：

```python
# 1. 增加网格分辨率
grid_size = (200, 200)  # 从 (100, 100) 增加

# 2. 使用更高阶插值
method = 'cubic'  # 而非 'linear'

# 3. 减少噪声阈值（保留更多点）
noise_threshold_percentile = 5.0  # 从 10.0 降低

# 4. 降低过滤强度
valid_mask = normal_magnitudes > 0.00001  # 更宽松的阈值
```

### Q5: 颜色映射看起来很暗/很亮

**答**：调整颜色映射函数：

```python
# 当前（太暗）：
intensity = int(255 * (1 - norm_val ** 0.5))

# 选项 1: 减少指数（更亮）
intensity = int(255 * (1 - norm_val ** 0.3))

# 选项 2: 添加偏移（更亮）
intensity = int(255 * (0.5 + 0.5 * (1 - norm_val ** 0.5)))

# 选项 3: 使用对数缩放（更平衡）
log_val = np.log1p(norm_val * 10) / np.log(11)
intensity = int(255 * (1 - log_val))
```

### Q6: 点击保存不工作

**答**：调试保存功能：

```python
# 1. 检查目录权限
import os
save_dir = Path("/path/to/save")
if not os.access(save_dir.parent, os.W_OK):
    print(f"无写入权限: {save_dir.parent}")

# 2. 检查文件名冲突
def _mouse_callback(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        filename = f"force_{self.counter:04d}.png"
        filepath = self.save_dir / filename
        
        # 调试打印
        print(f"尝试保存: {filepath}")
        print(f"目录存在: {filepath.parent.exists()}")
        print(f"图像有效: {self.current_img is not None}")
        
        if cv2.imwrite(str(filepath), self.current_img):
            print("✓ 保存成功")
            self.counter += 1
        else:
            print("✗ 保存失败")

# 3. 检查图像格式
if self.current_img.dtype != np.uint8:
    self.current_img = self.current_img.astype(np.uint8)
```

---

## 总结

### 核心设计原则

1. **模块化**：每个可视化器独立工作，互不干扰
2. **灵活性**：支持多种数据源和可视化方式
3. **鲁棒性**：优雅处理异常，不中断仿真
4. **可扩展性**：易于添加新的可视化器或数据源

### 快速参考

| 任务 | 工具 | 方法 |
|------|------|------|
| 显示接触点力 | TactileForceVisualizer | `visualize()` |
| 显示连续力场 | InterpolatedForceVisualizer | `visualize_from_values()` |
| 显示RGB图像 | TactileImageViewer | `show()` |
| 坐标变换 | 手动实现 | `transform_to_sensor_frame()` |
| 数据投影 | 手动实现 | `project_to_2d()` |

### 后续改进方向

- [ ] GPU 加速插值（CUDA）
- [ ] 3D 可视化支持（Open3D/Vispy）
- [ ] 多传感器融合显示
- [ ] 实时数据流/录制
- [ ] 交互式参数调整面板

---

**最后更新**：2026-04-01  
**维护者**：TacEx 开发团队  
**相关文件**：
- `source/tacex_uipc/tacex_uipc/utils/visualization/simple_visualizers.py` - 简单可视化器
- `source/tacex_uipc/tacex_uipc/utils/visualization/force_visualizer.py` - 力可视化器
- `scripts/demos/shape_touch/simple_tactile_demo.py` - 完整示例
