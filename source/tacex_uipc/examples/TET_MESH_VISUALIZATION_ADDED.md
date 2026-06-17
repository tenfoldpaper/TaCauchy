# Tet Mesh Visualization Added to Contact Force Demo

## 更新内容

在 `contact_force_visualization_demo.py` 中添加了 tet mesh 可视化功能。

## 新增功能

### 1. Tet Mesh 可视化

```python
def draw_tet_mesh(env: BallRollingEnv):
    """绘制四面体网格结构"""
```

- **颜色**: 青色 (Cyan) 线框
- **线宽**: 3 像素
- **显示内容**: 所有四面体的边（每个四面体有6条边）

### 2. Tet 拓扑获取

```python
def _get_tet_indices_from_uipc(env: BallRollingEnv) -> np.ndarray:
    """从UIPC simulation获取tet mesh拓扑"""
```

- 直接从 UIPC simulation 获取四面体拓扑
- 使用 `geometry.tetrahedra().topo()` 方法
- 自动处理不同的数据格式 (shape 转换)

## 可视化效果

运行脚本后，你会在 3D 场景中看到：

1. **青色线框** - Tet mesh 结构（四面体网格）
2. **黄色箭头** - 总接触力
3. **绿色箭头** - 法向力
4. **红色箭头** - 切向力
5. **白色点** - 接触点
6. **变形热图** - 在单独的窗口中显示

## 使用方法

```bash
# 运行脚本（默认显示 tet mesh）
python source/tacex_uipc/examples/contact_force_visualization_demo.py

# 调整力箭头缩放
python source/tacex_uipc/examples/contact_force_visualization_demo.py --force_scale 0.2
```

## 可选功能

如果需要显示附着点调试信息，取消注释这一行：

```python
# 在主循环中
# draw_attachment_debug(env)  # 取消注释以显示附着点
```

这会额外显示：
- 红色球：附着点位置
- 蓝色球：目标位置
- 青色线：约束连接

## 技术细节

### Tet Mesh 绘制逻辑

每个四面体有 4 个顶点，6 条边：
- (v0, v1), (v0, v2), (v0, v3)
- (v1, v2), (v1, v3), (v2, v3)

代码遍历所有四面体，绘制每条边作为线段。

### 数据获取流程

1. 从 `env.uipc_sim.scene.geometries()` 获取几何体
2. 使用 `obj_id` 查找对应的 gelpad 几何体
3. 获取 `tetrahedra().topo()` 拓扑数据
4. 转换为 numpy 数组并处理 shape
5. 展平为 1D 数组 (每 4 个索引表示一个四面体)

### 与 gelpad_position_monitor.py 的区别

- `gelpad_position_monitor.py`: 完整的诊断工具，包含多种可视化选项和详细报告
- `contact_force_visualization_demo.py`: 专注于力和变形可视化，tet mesh 作为辅助参考

## 调试输出

脚本会打印：
```
[TET_VIS] Got 428 indices (107 tetrahedra)
[TET_VIS] Drawing 107 tetrahedra
[TET_VIS] Drew 642 edges
```

如果看不到 tet mesh，检查：
1. 控制台是否有 `[TET_VIS]` 输出
2. 是否成功获取了拓扑数据
3. debug_draw 接口是否可用

## 性能说明

- Tet mesh 每帧重绘（跟随变形）
- 107 个四面体 = 642 条边
- 对性能影响很小（< 1ms）
