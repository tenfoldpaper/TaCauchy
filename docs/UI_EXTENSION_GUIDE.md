# UIPC UI Extension 使用指南

## 概述

UIPC UI Extension提供了一个图形界面，用于为Isaac Sim中的网格生成四面体网格。现在支持gelpad中心区域网格加密功能。

## 打开UI Extension

1. 启动Isaac Sim
2. 在菜单栏选择 `Window` -> `Extensions`
3. 搜索 `tacex_uipc`
4. 启用该扩展
5. 窗口 "Generate Tet Meshes for the IPC simulation" 会自动出现

## UI界面说明

### 基础参数 (Basic Parameters)

#### Edge Length Ratio (边长比例)
- **范围**: 0.01 - 1.0
- **默认值**: 0.5
- **说明**: 控制整体网格的相对边长。值越小，网格越密集
- **建议值**: 
  - 粗网格: 0.6-0.8
  - 中等网格: 0.4-0.6
  - 细网格: 0.2-0.4

#### Epsilon Ratio (包络比例)
- **范围**: 0.001 - 0.1
- **默认值**: 0.01
- **说明**: 控制包络大小，影响特征保留
- **建议值**: 保持默认 0.01

#### Stop Quality (停止质量)
- **范围**: 1 - 50
- **默认值**: 10
- **说明**: AMIPS能量优化的停止阈值。值越大，优化越少
- **建议值**: 8-15

#### Max Iterations (最大迭代次数)
- **范围**: 10 - 200
- **默认值**: 80
- **说明**: 网格优化的最大迭代次数
- **建议值**: 60-100

### 中心区域加密参数 (Center Refinement)

#### Enable Center Refinement (启用中心加密)
- **类型**: 复选框
- **默认值**: 未选中
- **说明**: 启用后，会在gelpad中心区域生成更密集的网格

#### Center Region Ratio (中心区域比例)
- **范围**: 0.1 - 0.8
- **默认值**: 0.3
- **说明**: 中心区域相对于整体的比例
- **建议值**:
  - 小接触面: 0.2-0.25
  - 中等接触面: 0.3-0.35
  - 大接触面: 0.4-0.5

#### Center Edge Length (中心边长)
- **范围**: 0.05 - 0.5
- **默认值**: 0.25 (1/4)
- **说明**: 中心区域的相对边长。值越小，网格越密
- **建议值**:
  - 低密度: 0.3-0.4
  - 中等密度: 0.2-0.3
  - 高密度: 0.15-0.2
  - 极高密度: 0.1-0.15

#### Transition Ratio (过渡比例)
- **范围**: 0.05 - 0.3
- **默认值**: 0.1
- **说明**: 中心和外围区域之间的过渡区域大小
- **建议值**: 0.08-0.15

### 快速预设 (Quick Presets)

提供四个预设按钮，可以快速应用常用配置：

#### Standard (标准)
- 不启用中心加密
- Edge Length: 0.5
- 适用于：一般用途，不需要特别关注中心区域

#### Fast (快速)
- 启用中心加密
- Edge Length: 0.667 (1/1.5)
- Center Ratio: 0.2
- Center Edge: 0.4 (1/2.5)
- 适用于：快速原型开发，性能优先

#### Balanced (平衡) - 推荐
- 启用中心加密
- Edge Length: 0.5
- Center Ratio: 0.3
- Center Edge: 0.25 (1/4)
- 适用于：大多数应用场景，平衡精度和性能

#### High (高精度)
- 启用中心加密
- Edge Length: 0.5
- Center Ratio: 0.4
- Center Edge: 0.167 (1/6)
- 适用于：需要高精度的研究和验证

## 使用步骤

### 1. 生成四面体网格

1. **选择网格对象**
   - 在场景中选择一个Mesh prim（必须是UsdGeom.Mesh类型）
   - 确保选择的是Mesh本身，而不是其父Xform

2. **配置参数**
   - 方式1：使用快速预设按钮（推荐新手）
   - 方式2：手动调整各个参数

3. **生成网格**
   - 点击 "Compute Tet Mesh" 按钮
   - 等待生成完成（控制台会显示进度）
   - 结果会显示在窗口顶部

4. **查看结果**
   - 生成的四面体网格会以红色线框显示
   - 表面网格会以蓝色显示
   - 信息面板会显示：
     - 四面体顶点数量
     - 四面体数量
     - 表面三角形数量
     - 使用的配置参数

### 2. 更新表面网格

如果需要更新渲染网格的拓扑结构：

1. 选择已经生成了四面体网格的prim
2. 点击 "Update Surface Mesh" 按钮
3. 表面网格会根据UIPC的表面提取算法更新

### 3. 创建附件 (Attachment)

用于将刚体与UIPC软体连接：

1. 先选择刚体对象
2. 按住Ctrl，再选择四面体网格对象
3. 点击 "Create Attachment" 按钮
4. 附件点会以红色点显示，连接线为黄色

## 使用示例

### 示例1: 为Gelpad生成标准网格

```
1. 在场景中选择 /World/GelPad/Mesh
2. 点击 "Standard" 预设按钮
3. 点击 "Compute Tet Mesh"
4. 等待生成完成
```

### 示例2: 为Gelpad生成高精度中心加密网格

```
1. 在场景中选择 /World/GelPad/Mesh
2. 点击 "High" 预设按钮
3. 点击 "Compute Tet Mesh"
4. 查看结果，中心区域应该有更密集的网格
```

### 示例3: 自定义参数

```
1. 在场景中选择 /World/GelPad/Mesh
2. 设置参数：
   - Edge Length Ratio: 0.5
   - Enable Center Refinement: ✓
   - Center Region Ratio: 0.4
   - Center Edge Length: 0.167
   - Transition Ratio: 0.15
3. 点击 "Compute Tet Mesh"
```

### 示例4: 为不同接触场景优化

**小接触面积（如手指按压）**:
```
- Edge Length: 0.5
- Enable Center Refinement: ✓
- Center Ratio: 0.25
- Center Edge: 0.2
- Transition: 0.1
```

**大接触面积（如手掌按压）**:
```
- Edge Length: 0.5
- Enable Center Refinement: ✓
- Center Ratio: 0.4
- Center Edge: 0.286 (1/3.5)
- Transition: 0.15
```

**滚动接触（如球滚动）**:
```
- Edge Length: 0.5
- Enable Center Refinement: ✓
- Center Ratio: 0.35
- Center Edge: 0.25
- Transition: 0.12
```

## 可视化说明

### 网格可视化

生成网格后，会在场景中显示：

- **红色线框**: 四面体网格的边
- **蓝色线框**: 表面三角形网格
- **白色点**: 表面顶点

### 中心加密可视化

启用中心加密后：
- 中心区域的网格线会更密集
- 可以清楚地看到从中心到外围的密度渐变

## 生成的数据

生成的四面体网格数据会作为自定义属性保存在USD prim上：

- `tet_points`: 四面体顶点坐标
- `tet_indices`: 四面体索引
- `tet_surf_points`: 表面顶点坐标
- `tet_surf_indices`: 表面三角形索引

这些数据可以在 "Raw USD Properties" 面板中查看。

## 性能提示

### 网格生成时间

- **标准网格**: 几秒到十几秒
- **中心加密（低）**: +5-10%时间
- **中心加密（中）**: +10-20%时间
- **中心加密（高）**: +30-50%时间

### 内存使用

网格数量增加会导致内存使用增加：
- 标准: 基准
- 中心加密（低）: +2%
- 中心加密（中）: +8%
- 中心加密（高）: +34%

### 优化建议

1. **从粗网格开始**: 先用较大的edge length测试
2. **逐步细化**: 根据需要逐步减小参数
3. **使用预设**: 预设配置已经过优化
4. **监控日志**: 查看控制台输出了解生成进度

## 故障排除

### 问题1: 点击按钮没有反应

**可能原因**:
- 没有选择正确的prim
- 选择的不是Mesh类型

**解决方案**:
- 确保选择的是Mesh prim，不是Xform
- 在Hierarchy面板中展开对象，选择Mesh子节点

### 问题2: 生成失败或报错

**可能原因**:
- 输入网格有问题（自相交、退化三角形等）
- 参数设置不合理

**解决方案**:
- 检查输入网格的质量
- 尝试增大epsilon_r参数
- 使用预设配置

### 问题3: 网格太密或太稀

**解决方案**:
- 太密: 增大edge_length_r和center_edge_length_r
- 太稀: 减小这些参数
- 使用预设作为起点

### 问题4: 中心加密效果不明显

**解决方案**:
- 确保勾选了 "Enable Center Refinement"
- 减小center_edge_length_r值
- 增大center_region_ratio值

## 高级技巧

### 1. 批量处理

虽然UI一次只能处理一个对象，但可以：
1. 为第一个对象配置参数
2. 生成网格
3. 选择下一个对象
4. 直接点击 "Compute Tet Mesh"（参数会保持）

### 2. 保存配置

当前UI不支持保存配置，但可以：
- 记录常用的参数组合
- 在代码中使用预设配置（见 `examples/gelpad_mesh_configs.py`）

### 3. 调试模式

生成网格时，控制台会输出详细信息：
- 输入网格统计
- 细分过程信息
- 最终网格统计

设置 `log_level=0` 可以看到更详细的WildMeshing日志。

## 与代码集成

UI生成的网格可以在代码中使用：

```python
from tacex_uipc.objects import UipcObjectCfg

# 使用UI生成的网格数据
gelpad_cfg = UipcObjectCfg(
    prim_path="/World/GelPad/Mesh",
    # 不需要指定mesh_cfg，会自动使用已生成的数据
)
```

## 参考资料

- [Gelpad中心加密详细指南](GELPAD_CENTER_REFINEMENT_GUIDE.md)
- [快速开始指南](../GELPAD_CENTER_REFINEMENT_README.md)
- [预设配置](../examples/gelpad_mesh_configs.py)

## 更新日志

### v1.1 (2026-01-26)
- ✅ 添加中心区域加密参数控制
- ✅ 添加四个快速预设按钮
- ✅ 改进UI布局和参数说明
- ✅ 增加配置信息显示
- ✅ 扩大窗口尺寸以容纳新控件

### v1.0
- 初始版本
- 基础四面体网格生成
- 表面网格更新
- 附件创建
