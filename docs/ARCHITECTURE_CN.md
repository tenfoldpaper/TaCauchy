# TacEx 代码结构与关键函数实现说明

本文档面向 `IROS_zhf/root_cn.tex` 所对应的中文技术报告，补充说明当前仓库中与视觉触觉仿真、UIPC 软体接触、力场提取和任务环境相关的代码实现。重点不是逐文件罗列，而是从系统层面解释代码的整体结构、关键模块之间的调用关系，以及重要函数是如何完成核心功能的。

## 1. 代码整体结构

当前工程可以分为四个主要层次。

### 1.1 触觉传感器抽象层

路径主要位于：

- `source/tacex/tacex/`
- `source/tacex/tacex/simulation_approaches/`

这一层负责把“视觉触觉传感器”抽象为 Isaac Lab 可管理的传感器对象。核心类是 `GelSightSensor`，它统一管理：

- 传感器相机数据采集
- 深度图到高度图的转换
- 压入深度估计
- 光学触觉图像生成
- 标记点运动模拟

在设计上，`GelSightSensor` 本身不直接实现所有仿真细节，而是通过 `GelSightSimulator` 抽象基类挂接不同的模拟后端，例如：

- `TaximSimulator`：基于 Taxim 的 GPU 光学仿真
- `FOTSMarkerSimulator`：基于 FOTS 的 marker motion 仿真
- `ManiSkillSimulator`：基于 FEM/ManiSkill 风格的触觉建模

因此，这一层的核心思想是“统一传感器接口 + 可替换仿真后端”。

### 1.2 UIPC 软体仿真与力学层

路径主要位于：

- `source/tacex_uipc/tacex_uipc/sim/`
- `source/tacex_uipc/tacex_uipc/objects/`
- `source/tacex_uipc/tacex_uipc/utils/`

这一层负责有限元软体、接触、应力和力场的物理计算，是整个项目与论文中 TaCauchy/UIPC 思路最直接对应的实现区域。主要模块包括：

- `UipcSim`：UIPC 世界、场景、求解器和渲染同步的总控类
- `UipcObject`：把 USD 中的 gelpad 或交互物体转换为 UIPC 中的四面体对象
- `UipcIsaacAttachments`：把 UIPC 软体点与 Isaac 刚体/机械臂建立约束
- `ContactForceExtractor`：从 UIPC 接触系统中提取顶点接触梯度并恢复力
- `StressExtractor`：读取四面体 Cauchy 应力并投影到表面三角片
- `MeshGenerator`：利用 WildMeshing 生成或细化四面体网格

这一层构成了“真实表面形变”和“力学真值”的来源。

### 1.3 任务环境与强化学习层

路径主要位于：

- `source/tacex_tasks/tacex_tasks/`
- `scripts/demos/shape_touch/`

这一层负责把机器人、触觉传感器、UIPC 物体、奖励函数和重置逻辑组织成 Isaac Lab 环境。主要形式有两类：

- 正式任务：如 `ball_rolling_tactile/ball_rolling_tactile_rgb_uipc.py`
- 示例环境：如 `scripts/demos/shape_touch/simple_tactile_env.py`

这部分代码的作用是把前两层的物理与触觉能力，封装成可运行、可训练、可可视化的任务接口。

### 1.4 可视化与调试层

路径主要位于：

- `source/tacex_uipc/tacex_uipc/utils/visualization/`
- `scripts/demos/shape_touch/contact_force_visualization_demo.py`
- `scripts/demos/shape_touch/contact_force_visualization_demo_modular.py`

这一层用于绘制：

- 接触力矢量
- 表面三角片法向压力
- 切向牵引
- 形变网格
- 低分辨率力图和插值力图

从论文写作角度看，这些代码并不是物理主链路的一部分，但它们对验证应力提取、接触点分布和力场重建结果非常关键。

## 2. 系统运行主链路

从代码实现上看，整个系统的数据流大致如下：

1. 使用 `UipcObject` 和 `MeshGenerator` 为 gelpad/物体建立四面体网格。
2. 使用 `UipcSim` 初始化 UIPC 场景、接触参数、求解器和物理时间步。
3. 使用 `UipcIsaacAttachments` 将 gelpad 顶部或边界点约束到传感器外壳/机械臂刚体上。
4. 在 Isaac Lab 环境 `UipcRLEnv` 的 `step()` 中推进 PhysX 和 UIPC，并同步渲染网格。
5. `GelSightSensor` 从相机深度图恢复 height map，并调用 `TaximSimulator` 等模块生成触觉 RGB 图像。
6. 若需要力学量，则由 `ContactForceExtractor` 提取顶点接触力，由 `StressExtractor` 提取表面牵引力、法向压力和切向牵引。
7. 任务环境根据触觉图像、压入深度、接触力或目标状态计算 observation、reward 和 done。

因此，代码整体并不是“先渲染、再估计力”，而是“先有物理形变与接触，再生成视觉触觉结果”，这与论文中强调的物理一致性是统一的。

## 3. 关键模块与重要函数实现

## 3.1 `GelSightSensor`：统一管理相机、height map 和触觉输出

文件路径：

- `source/tacex/tacex/gelsight_sensor.py`

### 3.1.1 构造函数 `__init__`

该函数完成三件事：

1. 初始化 Isaac Lab 传感器基类 `SensorBase`
2. 根据配置构建光学模拟器和 marker 模拟器
3. 创建统一的数据缓存 `GelSightSensorData`

这里最关键的实现点是“配置驱动实例化”。例如：

- `self.cfg.optical_sim_cfg.simulation_approach_class(...)`

意味着具体使用 Taxim、FOTS 还是其他方法，不是在 `GelSightSensor` 内写死，而是由配置类指定。这保证了传感器主类保持稳定，而后端可以替换。

### 3.1.2 初始化函数 `_initialize_impl`

该函数负责完成运行前的真实初始化，包括：

- 创建 `XFormPrim` 视图定位所有传感器 prim
- 创建 `TiledCamera`
- 初始化图像缓存
- 初始化 optical simulator / marker simulator
- 绑定 indentation depth 的计算函数

这一步的关键思想是：`GelSightSensor` 只持有统一接口，不直接知道后端如何工作，但它会在初始化阶段为后端准备好相机、缓存、device 和 batch 维度。

### 3.1.3 更新主函数 `_update_buffers_impl`

这是传感器每一帧更新时最核心的函数。它的执行顺序可以概括为：

1. 更新 frame 计数
2. 更新相机数据
3. 调用 `_get_height_map()` 恢复 gel 表面高度
4. 调用 `compute_indentation_depth_func()` 估计压入深度
5. 生成 `camera_depth` / `camera_rgb`
6. 调用 `optical_simulator.optical_simulation()` 生成 `tactile_rgb`
7. 调用 `marker_motion_simulator.marker_motion_simulation()` 生成 marker 运动

这条链路表明，触觉图像并不是单独产生的，而是以相机深度和 height map 为中间层。也就是说，这段代码实现的是：

`camera depth -> height map -> indentation depth -> tactile rendering`

这种分层实现方式便于以后加入新的 optical simulator，而不需要重写传感器主逻辑。

## 3.2 `TaximSimulator`：把 height map 转成触觉 RGB 图像

文件路径：

- `source/tacex/tacex/simulation_approaches/gpu_taxim/taxim_sim.py`

### 3.2.1 初始化函数 `_initialize_impl`

这个函数负责：

- 读取 calibration folder
- 初始化 Taxim 后端对象
- 创建 background 图像
- 根据目标分辨率对背景图做 resize
- 为单色传感器记录背景颜色统计量

这里有一个实现细节很重要：代码不仅支持普通 GelSight，也显式支持 DIGIT/9DTact 一类非标准颜色风格的传感器。因此初始化阶段会可选地打开：

- `convert_to_grayscale`
- `apply_color_correction`

也就是说，TacEx 不是简单把所有传感器都渲染成同一种 RGB 风格，而是通过背景图统计量对输出做风格校正。

### 3.2.2 核心函数 `optical_simulation`

该函数的核心流程是：

1. 从 `GelSightSensor` 读取当前 `height_map`
2. 若分辨率不匹配则 resize
3. 调用 `self._taxim.render_direct(...)`
4. 得到 RGB 触觉图像
5. 若需要灰度化，则按亮度权重变换
6. 若需要颜色校正，则执行逐通道均值-方差匹配

其本质可以概括为：使用形变高度图驱动光照纹理渲染，再根据目标传感器的视觉风格进行后处理。

### 3.2.3 函数 `compute_indentation_depth`

该函数先取 `height_map` 最小值，再结合 `gelpad_to_camera_min_distance` 和 `gelpad_height` 计算物体压入 gel 的深度。实现上采用：

- 先把高度图从毫米转成米
- 找到最小距离点
- 判断物体是否进入 gelpad 厚度范围
- 输出以毫米计的压入深度

这段实现虽然不复杂，但它把“图像几何变化”转换成“可用于奖励/接触判断的标量深度”，因此在任务环境中很常用。

## 3.3 `UipcSim`：统一管理 UIPC 世界、接触和渲染同步

文件路径：

- `source/tacex_uipc/tacex_uipc/sim/uipc_sim.py`

### 3.3.1 配置类 `UipcSimCfg`

该配置类封装了求解器所需的主要参数，包括：

- 时间步 `dt`
- 重力
- 接触参数 `d_hat`、摩擦、接触阻力
- Newton 迭代参数
- 线性系统求解器参数

这部分代码体现的是“把物理求解参数暴露为结构化配置”，方便在任务层按传感器类型或实验条件调参。

### 3.3.2 构造函数 `__init__`

构造函数会完成以下工作：

1. 创建 UIPC `Engine` 和 `World`
2. 从 `Scene.default_config()` 出发覆写求解参数
3. 创建地面对象
4. 设置默认接触模型
5. 初始化顶点偏移、表面偏移和渲染同步成员

这里最关键的实现点是：它不是简单包一层 Python 接口，而是把 UIPC 场景配置、接触模型和渲染同步机制都纳入一个统一对象里。因此环境层只需要持有 `UipcSim`，不必分别管理 solver、scene、surface IO 和 force extractor。

### 3.3.3 `setup_sim`

该函数在场景构建完成后调用，主要做：

- `world.init(scene)` 初始化物理世界
- `world.retrieve()` 拉回当前状态
- 为每个 `UipcObject` 计算全局顶点偏移
- 将 `self.step` 注册为 Isaac Sim 的 physics callback
- 初始化 `ContactForceExtractor`

这一步的作用是把独立创建的 UIPC 对象真正接入全局系统，并建立“每个对象在全局自由度中的索引位置”，后面写回顶点、提取接触梯度都依赖这个映射。

### 3.3.4 `update_render_meshes`

该函数从 `SceneIO` 中读取最新表面三角网格顶点，并写回到 Fabric/USD 渲染 prim。它解决的是一个常见问题：

- UIPC 内部算出来的形变网格，不会自动体现在 Isaac 的可视化网格上。

因此该函数本质上是在做“物理网格到渲染网格的同步”。没有它，触觉相机看到的表面不会随 FEM 形变更新。

## 3.4 `UipcObject`：把 USD 物体转换为 UIPC 可求解对象

文件路径：

- `source/tacex_uipc/tacex_uipc/objects/uipc_object.py`

### 3.4.1 构造函数 `__init__`

这个函数负责从 USD prim 中提取或生成四面体网格，并创建 UIPC 所需的 mesh。其主要流程是：

1. 找到对应 mesh prim
2. 若 `mesh_cfg is None`，则读取预计算的 `tet_points`、`tet_indices`
3. 否则调用 `MeshGenerator.generate_tet_mesh_for_prim`
4. 将局部顶点变换到世界坐标
5. 修正四面体顶点顺序，保证体积为正
6. 调用 `tetmesh(...)` 创建 UIPC 四面体网格
7. 标注表面、三角形朝向，并提取表面用于渲染
8. 调用 `_create_constitutions(mesh)` 指定材料模型

这段实现把“USD 资产”变成了“UIPC 中可参与 FEM 求解的对象”，是物理链路中的入口。

### 3.4.2 `_ensure_positive_tet_volumes`

这是一个很实用的几何修正函数。它通过四面体体积符号判断网格方向是否反了；若体积为负，则交换顶点顺序。这样做的原因是：

- 若四面体朝向错误，有限元中的法向、体积和应力计算都会出错。

因此这个函数属于典型的“前处理鲁棒性保护”。

### 3.4.3 `_create_constitutions`

该函数根据配置选择材料本构模型：

- `StableNeoHookean`：用于可形变 gelpad
- `AffineBodyConstitution`：用于近似刚体或运动学物体

对于 `StableNeoHookean`，代码通过 `ElasticModuli.youngs_poisson(...)` 把杨氏模量和泊松比转成材料参数，并应用到整个四面体网格。对于 `AffineBodyConstitution`，则使用刚度参数 `m_kappa`，并可通过 `builtin.is_fixed` 把对象设为运动学固定体。

这一函数对应论文中的“材料建模”部分，是从参数配置到 FEM 本构求解的桥梁。

## 3.5 `MeshGenerator`：实现自适应四面体网格生成

文件路径：

- `source/tacex_uipc/tacex_uipc/utils/mesh_gen.py`

### 3.5.1 `generate_tet_mesh_for_prim`

该函数从 USD mesh 读取顶点与三角面后，调用 `compute_tet_mesh` 完成体网格生成，并返回：

- `tet_mesh_points`
- `tet_indices`
- `surf_points`
- `surf_indices`

这一步是从表面网格进入体网格的关键接口。

### 3.5.2 `_compute_center_region` 与 `_build_sizing_field_func`

这是当前代码里和论文“接触区域自适应加密”最贴近的实现。

其核心思想是：

1. 先从包围盒判断最薄方向，也就是 gelpad 的厚度方向
2. 自动确定需要细化的表面是 top、bottom 还是 both
3. 计算平面中心和细化半径
4. 构建一个位置相关的 sizing field
5. 在中心区域返回更小的目标边长，在边缘返回更大的边长
6. 用余弦过渡带平滑连接细网格和粗网格

这意味着代码并不是统一细化整个 gelpad，而是把有限计算预算集中到接触最重要的中间区域，从而兼顾精度和速度。

### 3.5.3 `compute_tet_mesh`

这是网格生成的总控函数。它会：

- 打印输入几何尺寸和参数
- 可选应用中心细化 sizing field
- 调用 WildMeshing 四面体化
- 提取表面网格
- 统计四面体数量、表面点数、边长和三角片面积

从工程角度说，这个函数不仅负责算网格，也负责输出足够多的统计信息，便于比较不同细化参数对网格密度和接触分辨率的影响。

## 3.6 `UipcIsaacAttachments`：建立软体顶点与刚体外壳的约束

文件路径：

- `source/tacex_uipc/tacex_uipc/sim/uipc_attachments.py`

### 3.6.1 `compute_attachment_data`

这是 attachment 的关键函数。它的实现机制不是简单按名字匹配点，而是：

1. 强制 PhysX 预加载碰撞体
2. 找到目标刚体 prim
3. 遍历 tet mesh 顶点
4. 对每个顶点执行 `sweep_sphere_closest`
5. 若球扫掠命中目标刚体碰撞体，则视为 attachment 点
6. 将顶点位置变换到刚体局部坐标系，记录 attachment offset

换句话说，这里 attachment 的生成方式是“基于空间邻近和碰撞查询自动选点”，而不是手工指定顶点编号。这使得不同传感器 mesh 或不同外壳形状都可以复用同一逻辑。

### 3.6.2 `_initialize_impl`

初始化阶段进一步完成：

- 根据 articulation 或 rigid object 获取刚体姿态
- 对整个约束顶点集合重新计算 offset
- 支持 `constrain_entire_surface=True`，把 top surface 作为整体刚性约束面

这部分实现决定了 gelpad 上哪些区域被外壳“固定”，哪些区域允许自由变形，因此会直接影响接触区形变模式。

## 3.7 `ContactForceExtractor`：从接触梯度恢复顶点力和压强

文件路径：

- `source/tacex_uipc/tacex_uipc/sim/contact_force_extractor.py`

### 3.7.1 `extract_contact_forces`

这是接触力提取主函数。其流程是：

1. 通过 `ContactSystemFeature` 获取当前接触系统特征
2. 调用 `contact_gradient(...)` 读取 libuipc 返回的梯度
3. 解析出顶点索引和梯度向量
4. 使用公式 `f = -gradient / dt^2` 恢复真实力
5. 计算接触点的 nodal area
6. 若启用分解，则继续计算法向/切向分量

这里最重要的不是 API 调用本身，而是它遵循了 UIPC 的缩放规则：接触系统内部保存的是乘以 `dt^2` 的梯度，因此必须再除以 `dt^2` 才得到实际力。这是论文里“从物理系统直接恢复力场真值”的核心代码之一。

### 3.7.2 `_compute_force_decomposition`

该函数根据表面法向，将总接触力分解为：

- 法向力 `f_n = (f · n) n`
- 切向力 `f_t = f - f_n`

随后再根据 nodal area 归一化得到 pressure、normal pressure 和 tangential pressure。

这意味着输出不仅有“力”，还有“压强”，从而更接近视觉触觉研究里常用的局部压力分布和三轴力场表示。

### 3.7.3 `_get_surface_normals` 与 `_compute_nodal_areas`

这两个函数分别负责：

- 从表面三角网格计算顶点法向
- 根据每个三角形面积平均分配顶点面积

虽然这部分看起来是辅助函数，但它们实际上决定了法向分解是否稳定，以及 pressure 归一化是否合理，因此是力场重建中不可缺少的几何基础。

## 3.8 `StressExtractor`：从 Cauchy 应力投影到表面牵引

文件路径：

- `source/tacex_uipc/tacex_uipc/sim/stress_extractor.py`

### 3.8.1 `get_object_surface_stress`

该函数是论文中表面应力投影算法的直接代码实现。它会：

1. 从四面体对象中读取 `cauchy_stress`
2. 读取表面三角片拓扑
3. 根据 `builtin.is_surf` 筛选表面三角片
4. 根据 `builtin.parent_id` 找到每个表面三角片所属四面体
5. 计算三角片单位法向
6. 用 `traction = sigma * n` 计算牵引力
7. 用 `normal_pressure = traction · n` 计算法向压力
8. 用 `tangential = traction - p n` 计算切向牵引

也就是说，这个函数正对应论文里的表面 Cauchy 应力提取算法，并且输出的数据结构已经天然适合作为高分辨率表面力场。

## 3.9 `UipcRLEnv`：在 Isaac Lab 中整合 PhysX、UIPC 和触觉传感器

文件路径：

- `source/tacex_uipc/tacex_uipc/envs/direct_uipc_rl_env.py`

### 3.9.1 构造函数 `__init__`

这个类继承自 `DirectRLEnv`，在 Isaac Lab 标准环境框架上新增了 UIPC 支持。构造过程中主要完成：

- 创建 `SimulationContext`
- 如果配置了 `uipc_sim`，则构建 `UipcSim`
- 创建 `UipcInteractiveScene`
- 执行 `_setup_scene()`
- 初始化事件管理器、UI 和空间定义
- 在 `sim.reset()` 后调用 `uipc_sim.setup_sim()`

因此，`UipcRLEnv` 的价值不是替代 Isaac Lab，而是把 UIPC 软体仿真插进 Isaac Lab 的标准 step/reset 生命周期中。

### 3.9.2 `step`

这是环境推进的主循环。它的关键实现顺序是：

1. 预处理 action
2. 在 decimation 循环内调用 `_apply_action()`
3. `scene.write_data_to_sim()`
4. `sim.step(render=False)`
5. 若到达渲染时刻，则先 `uipc_sim.update_render_meshes()`，再 `sim.render()`
6. `scene.update(dt)`
7. 出循环后计算 reward、done、reset 和 observation

这一顺序特别关键，因为它保证了：

- 先有 UIPC 形变
- 再同步到渲染表面
- 最后触觉相机和 observation 看到的是更新后的 gel 表面

因此该函数实际上是“物理、渲染、传感器”三条链路的同步点。

## 3.10 任务环境：以 `BallRollingTactileRGBUipcEnv` 和 `SimpleTactileEnv` 为例

代表文件：

- `source/tacex_tasks/tacex_tasks/ball_rolling_tactile/ball_rolling_tactile_rgb_uipc.py`
- `scripts/demos/shape_touch/simple_tactile_env.py`

### 3.10.1 `_setup_scene`

在任务环境里，`_setup_scene` 通常负责组装以下元素：

- 机械臂 articulation
- Plate 或其他刚体场景
- `GelSightSensor`
- `UipcObject` gelpad
- `UipcObject` 交互物体
- `UipcIsaacAttachments`

这说明任务环境本质上是一个组装层，真正的传感器和物理逻辑都来自前面几个基础模块。

### 3.10.2 `_pre_physics_step` 与 `_apply_action`

这两类函数通常完成：

- 将 RL action 缩放为末端位姿控制量
- 用 `DifferentialIKController` 把末端命令转成关节目标
- 将关节目标写入机械臂

因此动作链路是：

`policy action -> IK command -> joint target -> contact motion`

### 3.10.3 `_reset_idx`

以 `ball_rolling_tactile_rgb_uipc.py` 为例，重置不仅会恢复机械臂关节，还会：

- 重新设置目标位置
- 调用 `self.gsmini.reset(env_ids=...)`
- 把 gelpad 和 object 的 UIPC 顶点写回初始位置

这表明该环境的 reset 不只是重置刚体状态，也包括软体顶点状态。否则 gelpad 的形变会跨 episode 累积，造成错误的初始条件。

### 3.10.4 `_compute_intermediate_values` 和 `_compute_rewards`

这两个函数一般被写成 `torch.jit.script`，目的是提高批量环境下的计算效率。其逻辑通常包括：

- 计算末端到物体距离
- 计算物体到目标距离
- 计算姿态误差
- 根据 indentation depth 判断是否稳定接触
- 根据位置误差、姿态误差和动作变化构造奖励

从实现思路上说，任务奖励并不直接从 RGB 图像端反推接触状态，而是充分利用了物理层已经提供的几何量和压入深度，这也是该代码库相对于纯图像触觉环境的一大特点。

## 4. 代码实现与论文方法的对应关系

如果把当前代码与报告中的核心方法一一对应，可以得到如下映射。

- 网格生成与接触区域加密：`MeshGenerator`
- 超弹性 gelpad 建模：`UipcObject._create_constitutions`
- UIPC 物理求解与场景推进：`UipcSim`
- 软体与传感器外壳的固定关系：`UipcIsaacAttachments`
- 由形变表面生成视觉触觉图像：`GelSightSensor` + `TaximSimulator`
- 从接触系统恢复顶点力：`ContactForceExtractor`
- 从 Cauchy 应力恢复表面牵引/压强：`StressExtractor`
- 用于学习和验证的任务环境：`UipcRLEnv` 及 `tacex_tasks/*`

这说明 TacEx 当前代码并不是单一的渲染工具，而是一个从网格、材料、接触、力场到图像输出都相互连通的完整仿真栈。

## 5. 总结

从工程实现角度看，TacEx 的核心优势有三点。

第一，它通过 `GelSightSensor` 和 `GelSightSimulator` 把不同视觉触觉传感器统一成可配置接口，使光学仿真层具有良好的可扩展性。

第二，它通过 `UipcSim`、`UipcObject`、`ContactForceExtractor` 和 `StressExtractor` 建立了从有限元形变到力学真值输出的完整链路，因此不仅能生成“看起来合理”的触觉图像，也能输出法向力、切向力、压力和表面牵引等高价值监督信号。

第三，它通过 `UipcRLEnv` 和多个任务环境把上述能力嵌入 Isaac Lab 训练框架，使该系统不仅适合离线分析，也适合强化学习、模仿学习和在线力场重建等下游任务。

若从本文报告的扩展目标来看，这套代码已经具备“高分辨率三维力场真值生成器”的基本结构：`StressExtractor` 和 `ContactForceExtractor` 负责生成原始力学场，`visualization` 和 `demo` 代码提供场分布观察接口，而后续只需在此基础上继续增加规则网格投影、低秩压缩和在线回归模型，即可自然衔接到“压缩降维的三维力场在线重建”这一研究方向。
