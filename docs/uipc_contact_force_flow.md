# UIPC 力与位移计算流程（contact_force_visualization_demo_modular.py）

本文档说明脚本 `contact_force_visualization_demo_modular.py` 中 **UIPC 力计算** 与 **位移/变形计算** 的完整流程，并给出对应的力学原理与关键实现位置。

## 1. 总体流程概览

运行主循环中每一帧大致按以下顺序执行：

1. PhysX 步进 + UIPC 更新  
   - 机器人控制、仿真步进、UIPC 网格刷新。
2. 读取 UIPC 的接触梯度并计算接触力  
   - `ContactForceExtractor` 从 libuipc 的 ContactSystemFeature 取出 contact gradient，并通过 `f = -g / dt^2` 计算力。
3. 计算位移/变形  
   - 底部接触面：将当前顶点与初始顶点进行刚体对齐（Kabsch），得到“局部变形”。
   - 顶部约束面：同样用 Kabsch 去除整体刚体运动，得到局部位移。
4. 可视化  
   - 3D：力箭头、tet mesh、glass mesh。  
   - 2D：触觉图 + 力箭头、插值力图、变形热力图。

主循环中关键调用位置：

```
1082:1163:scripts/demos/shape_touch/contact_force_visualization_demo_modular.py
            # Extract contact forces
            gelpad_forces = force_extractor.get_object_forces(env._uipc_gelpad)
...
            # 3D Visualization
            draw_3d_forces(...)
            draw_tet_mesh(...)
            draw_ball_mesh(...)
            draw_glass_mesh(...)
...
            # Deformation Visualization (contact surface)
            deformation_visualizer.visualize_deformation(...)
...
            # Top surface deformation visualization
            top_surface_visualizer.visualize_displacement(...)
```

## 2. 力的计算流程

### 2.1 数据来源：ContactSystemFeature 的 contact gradient

`ContactForceExtractor` 使用 libuipc 的 ContactSystemFeature 读取接触系统的 **梯度**。  
该梯度是对接触势能的梯度（在 libuipc 里按 `dt^2` 缩放）。

```
166:212:source/tacex_uipc/tacex_uipc/sim/contact_force_extractor.py
        self._contact_feature.contact_gradient(gradient_geo)
        ...
        gradients = np.array(grad_view).reshape(-1, 3)
        self._force_data.contact_gradient = gradients
        # Compute forces: f = -g / dt^2
        self._force_data.contact_forces = -gradients / (dt * dt)
```

### 2.2 接触力计算

核心公式：

- **力 = - 梯度 / dt²**  
  由于 libuipc 的接触能量梯度被乘上 `dt^2`，需要除以 `dt^2` 才能得到真实力。

### 2.3 法向/切向分解

如果需要法向与切向分解，`ContactForceExtractor` 会用接触顶点的表面法线进行分解：

```
247:258:source/tacex_uipc/tacex_uipc/sim/contact_force_extractor.py
        dot_products = np.sum(forces * normals, axis=1, keepdims=True)
        normal_forces = dot_products * normals
        tangential_forces = forces - normal_forces
```

表面法线由当前 UIPC surface 的三角面片计算得到（每个顶点累积临近三角形法线并归一化）。

## 3. 位移 / 变形计算流程

### 3.1 底部接触面变形（Contact Surface）

1. 从 gelpad 的 **当前顶点位置** 和 **初始顶点位置** 得到位移。  
2. 如果 `remove_rigid_motion=True`，先用 Kabsch 对齐初始与当前点云，去除整体刚体平移/旋转，得到 **局部变形**。

```
130:176:source/tacex_uipc/tacex_uipc/utils/visualization/deformation_visualizer.py
        deformation = np.linalg.norm(current_positions - aligned_initial, axis=1)
...
    def _align_initial_to_current(...):
        # Kabsch alignment (SVD)
```

### 3.2 顶部约束面位移（Top Surface）

脚本里 `TopSurfaceDeformationVisualizer` 做了同样的 Kabsch 对齐逻辑，用于估计顶部“局部位移”。

```
192:214:scripts/demos/shape_touch/contact_force_visualization_demo_modular.py
            h = init_centered.T @ curr_centered
            u, _, vt = np.linalg.svd(h)
            r = vt.T @ u.T
            ...
            aligned_initial = (init_centered @ r) + curr_centroid
            displacement = np.linalg.norm(top_current - aligned_initial, axis=1)
```

## 4. 力学原理简述

### 4.1 接触力

UIPC 的接触系统基于 **势能（potential energy）** 的形式。  
接触系统输出能量对位移的梯度 `g`（并按照 `dt^2` 缩放）。  
通过牛顿第二定律可还原力：

```
f = - g / dt^2
```

这里的 `g` 是对 **接触势能** 的梯度，负号表示力的方向与能量梯度方向相反（能量下降方向）。

### 4.2 变形（位移）

物体的变形量由 **当前顶点位置** 与 **初始顶点位置** 的差得到。  
若要剔除整体刚体运动（平移/旋转），使用 **Kabsch 刚体对齐**，只保留局部形变：

- 对齐后再做 L2 距离得到每个顶点的局部变形量。

### 4.3 约束与刚性层

顶面玻璃层作为高刚度 UIPC 物体并与 case 绑定，提供“近似刚体”的边界条件，从而抑制 gelpad 顶面的局部变形。

## 5. 关键实现位置索引

- 主循环与力/位移可视化  
  `scripts/demos/shape_touch/contact_force_visualization_demo_modular.py`
- 接触力提取与分解  
  `source/tacex_uipc/tacex_uipc/sim/contact_force_extractor.py`
- 变形计算（Kabsch 对齐）  
  `source/tacex_uipc/tacex_uipc/utils/visualization/deformation_visualizer.py`
- 顶面局部位移可视化  
  `scripts/demos/shape_touch/contact_force_visualization_demo_modular.py`

