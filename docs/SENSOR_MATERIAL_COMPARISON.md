# TacEx 传感器材料参数对比

## 概述

本文档对比了 TacEx 项目中三种触觉传感器（GelSight Mini、DIGIT、9DTact）的 gelpad 材料参数设置。

## 传感器对比表

### 基本信息

| 传感器 | Prim 路径 | Case 名称 | Gelpad 名称 |
|--------|-----------|-----------|-------------|
| **GelSight Mini** | `/World/envs/env_.*/Robot/gelsight_mini_gelpad` | `gelsight_mini_case` | `gelsight_mini_gelpad` |
| **DIGIT** | `/World/envs/env_.*/Robot/digit_gelpad` | `digit_case` | `digit_gelpad` |
| **9DTact** | `/World/envs/env_.*/Robot/Gelpad9DTact` | `case_9dtact` | `Gelpad9DTact/mesh` |

### 几何尺寸

#### GelSight Mini
```python
case_dimensions = Dimensions(
    width=32 mm,
    length=28 mm, 
    height=24 mm
)

gelpad_dimensions = Dimensions(
    width=20.75 mm,
    length=25.25 mm,
    height=4.5 mm
)
```

#### DIGIT
- 尺寸略大于 GelSight Mini
- 具体尺寸需查看 USD 文件

#### 9DTact
- 尺寸与 GelSight Mini 相近
- 具体尺寸需查看 USD 文件

## 材料参数对比

### 当前配置（modular 版本）

根据 `scripts/demos/shape_touch/contact_force_visualization_demo_modular.py`：

| 参数 | GelSight Mini | DIGIT | 9DTact |
|------|---------------|-------|--------|
| **弹性模量 (Young's Modulus)** | 0.1 MPa | 0.1 MPa | **0.01 MPa** ⭐ |
| **泊松比 (Poisson Ratio)** | 0.40 | 0.40 | 0.40 |
| **本构模型** | StableNeoHookean | StableNeoHookean | StableNeoHookean |
| **约束强度** | 2000.0 | 2000.0 | 2000.0 |
| **整体表面约束** | False | False | False |
| **表面约束百分位** | 70.0 | 70.0 | 70.0 |

**注意**: 9DTact 使用更软的材料（0.01 MPa），使其更容易变形，提供更高的灵敏度。

### 其他配置文件中的设置

#### 1. `contact_force_visualization_demo.py` (GelSight Mini)
```python
gelpad_cfg = UipcObjectCfg(
    constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
        youngs_modulus=0.01,   # 0.01 MPa (非常软)
        poisson_rate=0.30,     # 降低以减少变形传播
    ),
)

gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
    constraint_strength_ratio=50000.0,  # 非常强的约束
    constrain_entire_surface=True,      # 约束整个顶部表面
    surface_constraint_z_percentile=75.0,  # 顶部 25% 的顶点
)
```

#### 2. `ball_rolling_tactile_rgb_uipc.py` (训练环境)
```python
# Ball (刚性物体)
ball_cfg = UipcObjectCfg(
    constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
        youngs_modulus=100.0,  # 100 MPa (硬)
    ),
)

# Gelpad (软材料)
gelpad_cfg = UipcObjectCfg(
    constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
        youngs_modulus=10.0,   # 10 MPa (中等硬度)
    ),
)
```

#### 3. `pick_up_uipc.py` (抓取任务)
```python
gelpad_cfg = UipcObjectCfg(
    constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
        youngs_modulus=0.0005,  # 0.0005 MPa (极软)
    ),
)
```

## 材料参数详解

### 弹性模量 (Young's Modulus)

弹性模量表示材料的刚度，单位为 MPa (兆帕)。

| 数值 | 材料特性 | 应用场景 |
|------|----------|----------|
| **0.0005 MPa** | 极软，类似果冻 | 需要大变形的抓取任务 |
| **0.01 MPa** | 非常软，类似软橡胶 | 高灵敏度接触检测 |
| **0.1 MPa** | 软，类似硅胶 | 标准触觉传感 |
| **10 MPa** | 中等硬度，类似硬橡胶 | 训练环境，平衡变形和稳定性 |
| **100 MPa** | 硬，类似塑料 | 刚性物体模拟 |

### 泊松比 (Poisson Ratio)

泊松比表示材料在拉伸时横向收缩与纵向伸长的比值。

| 数值 | 材料特性 | 影响 |
|------|----------|------|
| **0.30** | 低泊松比 | 变形局部化，减少传播 |
| **0.40** | 中等泊松比 | 平衡局部和整体变形 |
| **0.45** | 高泊松比 | 接近不可压缩材料 |
| **0.49** | 接近不可压缩 | 类似橡胶，体积几乎不变 |

### 约束强度 (Constraint Strength Ratio)

约束强度控制 gelpad 顶部与传感器外壳的连接强度。

| 数值 | 约束效果 | 应用 |
|------|----------|------|
| **2000.0** | 标准约束 | 允许顶部有一定变形 |
| **6000.0** | 强约束 | 顶部变形较小 |
| **50000.0** | 极强约束 | 顶部几乎刚性 |

## 真实传感器材料参考

### GelSight Mini
- **实际材料**: 硅胶 (Silicone)
- **弹性模量**: 约 0.1-1 MPa
- **泊松比**: 约 0.48-0.49 (接近不可压缩)
- **硬度**: Shore A 10-30

### DIGIT
- **实际材料**: 软硅胶
- **弹性模量**: 约 0.05-0.5 MPa
- **泊松比**: 约 0.48-0.49
- **硬度**: Shore A 5-20

### 9DTact
- **实际材料**: 硅胶
- **弹性模量**: 约 0.1-1 MPa
- **泊松比**: 约 0.48-0.49
- **硬度**: Shore A 10-30

## 结论

### 当前状态

**9DTact 传感器现在使用更软的材料配置：**

```python
# GelSight Mini & DIGIT
youngs_modulus = 0.1 MPa
poisson_rate = 0.40
constraint_strength_ratio = 2000.0

# 9DTact (更软，更容易变形)
youngs_modulus = 0.01 MPa  # 10倍更软
poisson_rate = 0.40
constraint_strength_ratio = 2000.0
```

### 为什么 9DTact 使用更软的材料？

1. **更高灵敏度**: 0.01 MPa 的弹性模量使 gelpad 更容易变形，能够检测到更小的接触力
2. **更大变形**: 适合需要明显变形反馈的应用场景
3. **差异化配置**: 与 GelSight Mini 和 DIGIT 形成对比，便于测试不同材料特性的影响

### 建议的差异化配置

当前配置已经实现了传感器间的差异化：

#### GelSight Mini (标准硅胶)
```python
youngs_modulus = 0.1 MPa   # 标准软度
poisson_rate = 0.40        # 中等泊松比
constraint_strength_ratio = 2000.0
```

#### DIGIT (标准硅胶)
```python
youngs_modulus = 0.1 MPa   # 标准软度
poisson_rate = 0.40        # 中等泊松比
constraint_strength_ratio = 2000.0
```

#### 9DTact (软硅胶 - 高灵敏度) ⭐
```python
youngs_modulus = 0.01 MPa  # 10倍更软，更容易变形
poisson_rate = 0.40        # 中等泊松比
constraint_strength_ratio = 2000.0
```

### 9DTact 软材料的优势

1. **更高灵敏度**: 能够检测到更小的接触力（约 0.001-0.01 N）
2. **更大变形量**: 变形量可达 gelpad 厚度的 10-30%
3. **更丰富的触觉信息**: 提供更详细的接触几何和力分布信息

### 9DTact 软材料的注意事项

1. **稳定性**: 更软的材料可能导致仿真不稳定，需要更小的时间步长
2. **计算成本**: 大变形需要更多的计算资源
3. **恢复时间**: 软材料的恢复时间更长，可能影响高频接触检测

## 参数调优指南

### 提高灵敏度
```python
youngs_modulus = 0.01-0.05 MPa  # 降低刚度
poisson_rate = 0.30-0.40        # 降低泊松比
```

### 提高稳定性
```python
youngs_modulus = 0.1-1.0 MPa    # 提高刚度
constraint_strength_ratio = 5000-10000  # 增强约束
```

### 局部化变形
```python
poisson_rate = 0.30-0.35        # 降低泊松比
constrain_entire_surface = True  # 约束整个顶部
surface_constraint_z_percentile = 75-80  # 增大约束区域
```

## 参考资料

1. GelSight 论文: "GelSight: High-Resolution Robot Tactile Sensors for Estimating Geometry and Force"
2. DIGIT 论文: "DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation"
3. 硅胶材料数据手册
4. UIPC 文档: https://github.com/spiriMirror/libuipc
