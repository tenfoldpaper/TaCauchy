# Shape Touch Demo 脚本总结

本目录包含触觉传感器与多物体交互的演示脚本。

## 脚本对比

### 1. 简洁版本（推荐用于快速演示）

**文件**: `simple_tactile_demo.py` + `simple_tactile_env.py`

**特点**:
- ✅ 代码简洁（~400 行总计）
- ✅ 核心功能完整
- ✅ 易于理解和修改
- ✅ 模块化设计

**功能**:
- 多物体 UIPC 交互（cylinder, cone, flat_slab, sphere2）
- Goal 控制机器人
- 法向/切向力可视化
- 触觉 RGB 图像显示和保存

**使用**:
```bash
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight
```

**适用场景**: 快速演示、教学、原型开发

---

### 2. 完整版本（用于深度分析）

**文件**: `contact_force_visualization_demo_modular copy.py`

**特点**:
- ✅ 功能全面（~3000+ 行）
- ✅ 详细的性能监控
- ✅ 多种可视化选项
- ✅ 丰富的调试工具

**功能**:
- 所有简洁版功能
- UIPC 性能基准测试
- 变形可视化（接触面和约束面）
- 网格结构可视化
- 应力张量可视化
- Z 方向合力曲线
- 表面三角形受力分析
- 详细的调试输出

**使用**:
```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular\ copy.py --sensor gelsight
```

**适用场景**: 论文实验、性能分析、深度调试

---

## 模块化架构

两个脚本都使用相同的模块化可视化工具：

```
source/tacex_uipc/tacex_uipc/utils/visualization/
├── __init__.py                    # 导出所有可视化工具
├── force_visualizer.py            # 力可视化（2D 窗口）
├── deformation_visualizer.py      # 变形可视化
├── mesh_visualizer.py             # 3D 网格可视化
├── surface_filter.py              # 表面过滤工具
├── simple_visualizers.py          # 简化版可视化器（新增）
└── README.md                      # 可视化工具文档
```

### 导入示例

```python
# 简洁版使用
from tacex_uipc.utils.visualization import (
    TactileForceVisualizer,
    TactileRGBVisualizer,
)

# 完整版使用
from tacex_uipc.utils.visualization import (
    ForceVisualizer,
    InterpolatedForceVisualizer,
    DeformationVisualizer,
    draw_3d_forces,
    draw_tet_mesh,
    filter_contact_surface_vertices,
)
```

---

## 多物体 UIPC 交互配置

两个脚本都支持多个物体与传感器同时交互。关键配置：

### 1. 定义 UIPC 物体

```python
uipc_shapes = {
    "cylinder": UipcObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_uipc",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.03]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/cylinder.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        mesh_cfg=None,
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=100.0,
            kinematic=True,
        ),
        mass_density=1e3,
    ),
    # ... 其他物体
}
```

### 2. 注册到场景

```python
for shape_name, shape_cfg in cfg.uipc_shapes.items():
    self.uipc_shapes[shape_name] = UipcObject(shape_cfg, self.uipc_sim)
    self.scene.uipc_objects[f"{shape_name}_uipc"] = self.uipc_shapes[shape_name]
```

### 重要提示

⚠️ **UIPC 只能与 UIPC 物体交互**，不能与 PhysX 刚体交互！

- ✅ 正确: `UipcObjectCfg` → 可以与 gelpad 交互
- ❌ 错误: `RigidObjectCfg` → 无法与 gelpad 交互

---

## 传感器支持

三种传感器都支持：

| 传感器 | 命令行参数 | 分辨率 | 特点 |
|--------|-----------|--------|------|
| GelSight Mini | `--sensor gelsight` | 640×480 | 彩色，标准尺寸 |
| DIGIT | `--sensor digit` | 640×480 | 彩色，高精度 |
| 9DTact | `--sensor 9dtact` | 640×480 | 单色，薄型设计 |

---

## 快速开始

### 1. 简洁演示（推荐新手）

```bash
# 运行简洁版
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight

# 查看文档
cat scripts/demos/shape_touch/SIMPLE_DEMO_README.md
```

### 2. 完整分析（推荐研究）

```bash
# 运行完整版
python scripts/demos/shape_touch/contact_force_visualization_demo_modular\ copy.py --sensor gelsight

# 启用所有调试输出
python scripts/demos/shape_touch/contact_force_visualization_demo_modular\ copy.py \
    --sensor gelsight \
    --debug_all \
    --show_legends
```

---

## 文件清单

```
scripts/demos/shape_touch/
├── simple_tactile_demo.py                              # 简洁版主脚本
├── simple_tactile_env.py                               # 简洁版环境配置
├── contact_force_visualization_demo_modular copy.py    # 完整版脚本
├── SIMPLE_DEMO_README.md                               # 简洁版文档
├── SUMMARY.md                                          # 本文档
└── visualization_usage_example.py                      # 可视化工具使用示例
```

---

## 下一步

1. **学习基础**: 从 `simple_tactile_demo.py` 开始
2. **理解模块**: 阅读 `source/tacex_uipc/tacex_uipc/utils/visualization/README.md`
3. **深入研究**: 使用完整版脚本进行性能分析
4. **自定义开发**: 基于简洁版修改，添加自己的功能

---

## 技术支持

- 项目文档: `docs/`
- 可视化工具: `source/tacex_uipc/tacex_uipc/utils/visualization/`
- 传感器配置: `source/tacex_assets/tacex_assets/robots/franka/`
- 问题反馈: 查看项目 README
