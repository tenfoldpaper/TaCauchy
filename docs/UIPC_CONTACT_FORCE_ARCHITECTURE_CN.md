# TacEx UIPC 接触力提取架构文档

## 概述

本文档描述了TacEx框架中基于libuipc的接触力提取系统的架构设计和实现原理。该系统能够从UIPC软体仿真中提取接触力信息，并将其分解为法向力和切向力分量，用于触觉传感器的力学反馈。

---

## 1. 理论基础

### 1.1 libuipc 接触系统原理

libuipc使用基于优化的隐式时间积分方法来处理接触问题。在每个仿真步骤中：

1. **能量函数**：系统通过最小化能量函数来求解下一时刻的状态
2. **梯度计算**：接触约束贡献的梯度（gradient）表示接触力的信息
3. **时间步长缩放**：在libuipc中，能量/梯度/Hessian矩阵已经被时间步长Δt²缩放

### 1.2 力的计算公式

根据牛顿第二定律，从优化梯度恢复接触力：

```
f = -g / (Δt)²
```

其中：
- `f` = 接触力向量 (N)
- `g` = 接触梯度向量 (从libuipc获取)
- `Δt` = 仿真时间步长 (s)

### 1.3 力的分解

对于每个接触点，将总力分解为法向和切向分量：

```
法向力: f_n = (f · n) × n
切向力: f_t = f - f_n
```

其中 `n` 是接触点的表面法向量。

---

## 2. 系统架构

### 2.1 核心模块

```
┌─────────────────────────────────────────────────────────────────┐
│                        TacEx UIPC Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐     ┌──────────────────────────────────┐  │
│  │    UipcSim       │────▶│   ContactForceExtractor          │  │
│  │  (仿真管理器)     │     │   (接触力提取器)                  │  │
│  └──────────────────┘     └──────────────────────────────────┘  │
│          │                              │                        │
│          ▼                              ▼                        │
│  ┌──────────────────┐     ┌──────────────────────────────────┐  │
│  │  libuipc Engine  │     │      ContactForceData            │  │
│  │  (UIPC仿真引擎)   │     │      (力数据容器)                 │  │
│  └──────────────────┘     └──────────────────────────────────┘  │
│          │                              │                        │
│          ▼                              ▼                        │
│  ┌──────────────────┐     ┌──────────────────────────────────┐  │
│  │ContactSystemFeature│   │   Visualization / Output          │  │
│  │  (接触系统特性)    │    │   (可视化/输出)                   │  │
│  └──────────────────┘     └──────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

```
source/tacex_uipc/tacex_uipc/
├── sim/
│   ├── __init__.py                    # 模块导出
│   ├── uipc_sim.py                    # UIPC仿真主类
│   └── contact_force_extractor.py     # 接触力提取核心模块
│
└── examples/
    ├── contact_force_demo.py              # 基础demo
    ├── contact_force_print_demo.py        # 打印输出demo
    └── contact_force_visualization_demo.py # 可视化demo
```

---

## 3. 核心类详解

### 3.1 ContactForceData (数据容器)

```python
@dataclass
class ContactForceData:
    """接触力数据容器"""
    
    # 基础数据
    vertex_indices: np.ndarray      # 接触顶点索引 (N,)
    contact_gradient: np.ndarray    # 原始梯度 (N, 3)
    contact_forces: np.ndarray      # 接触力 (N, 3)
    
    # 分解后的力
    normal_forces: np.ndarray       # 法向力分量 (N, 3)
    tangential_forces: np.ndarray   # 切向力分量 (N, 3)
    
    # 力的幅值
    normal_force_magnitudes: np.ndarray     # 法向力大小 (N,)
    tangential_force_magnitudes: np.ndarray # 切向力大小 (N,)
    
    # 几何信息
    surface_normals: np.ndarray     # 表面法向量 (N, 3)
    dt: float                       # 时间步长
```

### 3.2 ContactForceExtractor (核心提取器)

```python
class ContactForceExtractor:
    """从libuipc仿真中提取接触力"""
    
    def __init__(self, uipc_sim: UipcSim):
        self._uipc_sim = uipc_sim
        self._contact_feature = None  # ContactSystemFeature
        
    def initialize(self) -> bool:
        """初始化，获取ContactSystemFeature"""
        features = self._uipc_sim.engine.features()
        self._contact_feature = features.find(ContactSystemFeature.FeatureName)
        return self._contact_feature is not None
    
    def extract_contact_forces(self) -> ContactForceData:
        """提取当前帧的接触力"""
        # 1. 创建几何体接收梯度数据
        gradient_geo = Geometry()
        
        # 2. 从libuipc获取接触梯度
        self._contact_feature.contact_gradient(gradient_geo)
        
        # 3. 解析顶点索引和梯度值
        instances = gradient_geo.instances()
        i_attr = instances.find('i')      # 顶点索引
        grad_attr = instances.find('grad') # 梯度值
        
        # 4. 计算力: f = -gradient / dt²
        forces = -gradients / (dt * dt)
        
        # 5. 分解为法向和切向分量
        self._compute_force_decomposition()
        
        return self._force_data
    
    def get_object_forces(self, uipc_object) -> ContactForceData:
        """获取特定物体的接触力"""
        all_forces = self.extract_contact_forces()
        # 过滤属于指定物体的力
        return filtered_forces
```

### 3.3 UipcSim 集成

```python
class UipcSim:
    """UIPC仿真管理器"""
    
    def __init__(self, cfg: UipcSimCfg):
        self._contact_force_extractor = None
        
    def _setup_engine(self):
        # ... 设置引擎 ...
        
        # 初始化接触力提取器
        self._contact_force_extractor = ContactForceExtractor(self)
        if self._contact_force_extractor.initialize():
            print("[UipcSim] Contact force extractor initialized successfully.")
    
    @property
    def contact_force_extractor(self) -> ContactForceExtractor:
        """获取接触力提取器"""
        return self._contact_force_extractor
    
    def get_contact_forces(self) -> ContactForceData:
        """便捷方法：获取当前帧的接触力"""
        if self._contact_force_extractor:
            return self._contact_force_extractor.extract_contact_forces()
        return ContactForceData()
```

---

## 4. 数据流程

### 4.1 单帧处理流程

```
┌─────────────┐
│  仿真步进    │
│  sim.step() │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  ContactSystemFeature           │
│  contact_gradient(geometry)     │
│  填充几何体的接触梯度信息         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  解析 Geometry.instances()      │
│  - 'i' 属性: 顶点索引            │
│  - 'grad' 属性: 梯度向量         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  力计算                         │
│  f = -gradient / (dt * dt)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  力分解                         │
│  f_n = (f · n) × n             │
│  f_t = f - f_n                  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  输出 ContactForceData          │
│  包含所有力学信息                │
└─────────────────────────────────┘
```

### 4.2 libuipc API 调用

```python
# 1. 获取ContactSystemFeature
features = engine.features()
contact_feature = features.find(ContactSystemFeature.FeatureName)

# 2. 创建几何体接收数据
gradient_geo = Geometry()

# 3. 调用contact_gradient填充数据
contact_feature.contact_gradient(gradient_geo)

# 4. 访问实例数据
instances = gradient_geo.instances()
instance_size = instances.size()  # 接触点数量

# 5. 读取属性
import pyuipc as uipc
i_attr = instances.find('i')
i_view = uipc.view(i_attr)
vertex_indices = np.array(i_view).flatten()

grad_attr = instances.find('grad')
grad_view = uipc.view(grad_attr)
gradients = np.array(grad_view).reshape(-1, 3)
```

---

## 5. 使用示例

### 5.1 基础用法

```python
from tacex_uipc import UipcSim, UipcSimCfg

# 创建仿真
cfg = UipcSimCfg(dt=1/60)
uipc_sim = UipcSim(cfg)

# 运行仿真循环
while running:
    uipc_sim.step()
    
    # 获取接触力
    force_data = uipc_sim.get_contact_forces()
    
    if force_data.vertex_indices is not None and len(force_data.vertex_indices) > 0:
        print(f"接触点数量: {len(force_data.vertex_indices)}")
        print(f"总力: {np.sum(force_data.contact_forces, axis=0)} N")
        print(f"最大法向力: {np.max(force_data.normal_force_magnitudes)} N")
        print(f"最大切向力: {np.max(force_data.tangential_force_magnitudes)} N")
```

### 5.2 获取特定物体的力

```python
# 获取gelpad的接触力
extractor = uipc_sim.contact_force_extractor
gelpad_forces = extractor.get_object_forces(gelpad_object)

# 获取特定marker点的力
marker_indices = np.array([10, 20, 30, 40])  # marker顶点索引
marker_forces = extractor.get_marker_forces(gelpad_object, marker_indices)

print(f"Marker力: {marker_forces['forces']}")
print(f"有接触的Marker: {marker_forces['has_contact']}")
```

### 5.3 可视化

```python
# 3D可视化 (Isaac Sim debug_draw)
from isaacsim.util.debug_draw import _debug_draw
debug_draw = _debug_draw.acquire_debug_draw_interface()

for idx, force in zip(force_data.vertex_indices, force_data.contact_forces):
    start_pos = vertex_positions[idx]
    end_pos = start_pos + force * scale
    debug_draw.draw_lines([start_pos], [end_pos], [(1, 1, 0, 1)], [2])

# 2D可视化 (OpenCV)
import cv2
for i, (idx, force) in enumerate(zip(force_data.vertex_indices, force_data.contact_forces)):
    x, y = project_to_2d(vertex_positions[idx])
    dx, dy = force[0] * scale, -force[2] * scale
    cv2.arrowedLine(image, (x, y), (x + dx, y + dy), (0, 255, 255), 2)
```

---

## 6. 接触力统计输出示例

运行demo时的输出：

```
[ContactForceExtractor] Instance size: 3

[Step 30] === Contact Force Info ===
  Number of contact points: 3
  Total force: [0.0000, 0.0000, 0.0049] N
  Max force magnitude: 0.0022 N
  Max normal force: 0.0022 N
  Max tangential force: 0.0017 N

[Step 30] Contact Force Summary:
  Vertices in contact: 3
  Total Force: [0.0000, 0.0000, 0.0049] N
  Max Force Magnitude: 0.0022 N
  Max Normal Force: 0.0022 N
  Max Tangential Force: 0.0017 N
```

---

## 7. 限制与注意事项

### 7.1 当前限制

1. **接触点数量**：取决于网格分辨率，gelpad约424个顶点
2. **全局索引**：libuipc返回的是全局DoF索引，需要映射到物体局部索引
3. **法向量计算**：当前使用顶点法向量近似，可能不够精确

### 7.2 性能考虑

- 每帧调用 `contact_gradient()` 有一定开销
- 建议在需要时才提取力信息（如每N帧一次）
- 3D可视化使用 `debug_draw` 可能影响渲染性能

### 7.3 未来改进

1. 支持按物体过滤接触力（需要维护顶点索引映射表）
2. 添加接触力历史记录和平滑处理
3. 支持更精确的表面法向量计算
4. 与触觉传感器的marker系统深度集成

---

## 8. 参考资料

- [libuipc官方文档](https://github.com/spiriMirror/libuipc)
- [libuipc-samples: contact_system_feature示例](https://github.com/spiriMirror/libuipc-samples/blob/main/python/20_contact_system_feature/main.py)
- TacEx文档: `docs/FORCE_INFORMATION_IN_TACEX.md`
- UIPC力信息文档: `docs/UIPC_FORCE_INFORMATION.md`

---

## 附录：关键代码位置

| 文件 | 功能 |
|------|------|
| `tacex_uipc/sim/contact_force_extractor.py` | 接触力提取核心实现 |
| `tacex_uipc/sim/uipc_sim.py` | UIPC仿真集成 |
| `tacex_uipc/sim/__init__.py` | 模块导出 |
| `examples/contact_force_visualization_demo.py` | 可视化demo |
| `examples/contact_force_print_demo.py` | 打印输出demo |

---

*文档版本: 1.0*  
*更新日期: 2026年1月19日*
