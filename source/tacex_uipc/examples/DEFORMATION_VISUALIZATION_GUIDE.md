# Tet Mesh变形可视化指南

## 概述

`contact_force_visualization_demo.py` 现在包含了两个可视化窗口：

1. **Contact Forces** - 接触力可视化
2. **Tet Mesh Deformation** - Tet mesh变形可视化（新增！）

## 运行方法

```bash
python source/tacex_uipc/examples/contact_force_visualization_demo.py
```

## 可视化窗口说明

### 窗口1: Contact Forces（接触力）
- **黄色箭头** - 总力
- **绿色箭头** - 法向力
- **红色箭头** - 切向力
- **箭头长度** - 力的大小

### 窗口2: Tet Mesh Deformation（变形）
- **蓝色点** - 无变形或极小变形
- **绿色点** - 中等变形
- **黄色/红色点** - 最大变形
- **白色圆圈** - Attachment约束点（应该几乎不变形）
- **右侧色条** - 变形量标尺（单位：mm）

## 显示的统计信息

### 变形统计
- **Vertices**: 顶点总数
- **Mean Def**: 平均变形量（mm）
- **Max Def**: 最大变形量（mm）
- **Attachment Points**: 约束点数量

### 力统计
- **Contacts**: 接触点数量
- **Max Force**: 最大力（N）
- **Total Force**: 总力向量（N）
- **Max Normal/Tangent**: 最大法向/切向力（N）

## 控制台输出

每30步打印一次详细信息：

### 位置监控
```
[Position Monitor - Step 30]
  Gelpad center: [x, y, z]
  Gelpad Z range: [min, max]
  Case position: [x, y, z]
  Ball center: [x, y, z]
```

### 变形统计
```
[Deformation Stats - Step 30]
  Mean deformation: X.XXX mm
  Max deformation: X.XXX mm
  Attachment points mean def: X.XXX mm
  ⚠️  WARNING: Large attachment point deformation!  # 如果>1mm
```

### 接触力信息
```
[Step 30] === Contact Force Info ===
  Number of contact points: N
  Total force: [fx, fy, fz] N
  Max force magnitude: X.XXXX N
  Max normal force: X.XXXX N
  Max tangential force: X.XXXX N
```

## 验证UIPC是否正常工作

### 1. 检查变形可视化
- ✅ 当ball接触gelpad时，应该看到红色/黄色区域（变形）
- ✅ Attachment点（白色圆圈）应该保持蓝色或浅绿色（几乎不变形）
- ❌ 如果所有点都是蓝色，说明没有变形 → UIPC可能没工作

### 2. 检查控制台输出
- ✅ `Mean deformation` 应该 > 0 当有接触时
- ✅ `Attachment points mean def` 应该 < 1mm
- ❌ 如果attachment点变形 > 1mm，说明约束太弱

### 3. 检查力可视化
- ✅ 应该看到黄色/绿色/红色箭头
- ✅ 箭头应该指向gelpad内部（压入方向）
- ❌ 如果没有箭头，说明没有检测到接触

## 常见问题

### Q: 窗口没有出现
**A**: 检查控制台输出：
- 如果看到 "OpenCV window not available"，图像会保存到 `/tmp/deformation_visualization.png`
- 使用图像查看器打开该文件

### Q: 所有点都是蓝色（无变形）
**A**: 可能的原因：
1. Ball还没接触到gelpad - 等待几秒
2. UIPC仿真没有运行 - 检查控制台错误
3. Gelpad位置不对 - 查看位置监控输出

### Q: Attachment点变形很大（>1mm）
**A**: 这表示约束强度不够：
- 增加 `constraint_strength_ratio` 在配置中
- 检查attachment数据是否正确加载

### Q: 变形看起来不合理
**A**: 检查：
1. 初始位置是否正确加载
2. 当前位置是否正确获取
3. 坐标系是否一致

## 技术细节

### 变形计算
```python
deformation = ||current_position - initial_position||
```

### 颜色映射
```python
if deformation < 50%:
    color = Blue → Green
else:
    color = Green → Red
```

### 数据来源
- **初始位置**: `env._uipc_gelpad.init_vertex_pos`
- **当前位置**: `env._uipc_gelpad.data.nodal_pos_w`
- **Attachment索引**: `env.attachment.attachment_points_idx`

## 下一步

如果变形可视化工作正常，你可以：

1. **集成到传感器** - 使用真实变形数据替代相机深度图
2. **对比两种方法** - 相机深度图 vs UIPC变形
3. **优化性能** - 如果变形计算太慢

参考spec文档：`.kiro/specs/uipc-gelpad-deformation-visualization/requirements.md`
