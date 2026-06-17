# TacEx 触觉传感器实现与自定义指南

本文档用中文总结 TacEx 中触觉传感器（以 GelSight 为例）的实现结构、关键源码文件与它们的作用，并提供详细的步骤与示例，教你如何替换或实现自定义传感器。

---

## 目标读者
- 想理解 TacEx 中触觉仿真如何实现的工程师或研究者
- 想把 GelSight 替换为自定义传感器并在环境中使用的开发者

## 总览（高层）
TacEx 在 Isaac Lab / IsaacSim 框架上实现触觉仿真。流程概括：
1. 在资产（USD）里包含传感器相关 prim（case、gelpad、camera 等）。
2. 在环境配置（EnvCfg）中声明传感器配置（例如 `GelSightMiniCfg`）。
3. 在环境的 `_setup_scene()` 中实例化传感器类（`GelSightSensor`）并注册到 `scene.sensors`。
4. 传感器类初始化对应模拟器（Taxim / FOTS / UIPC），并在每帧计算/更新传感器输出，输出存放在 `sensor.data`（数据类）。
5. GUI 可通过 prim 的 `debug_<datatype>` 属性或相机纹理显示传感器图像。

---

## 关键文件清单（路径 + 作用）
下面列出仓库内与 GelSight/触觉仿真直接相关的重要文件与作用，便于定位与修改：

- `source/tacex/tacex/gelsight_sensor.py`
  - 主传感器类，继承 IsaacLab 的 `SensorBase`。
  - 负责 prim 查找/创建、模拟器初始化、每帧数据更新、暴露 `data` 接口。

- `source/tacex/tacex/gelsight_sensor_cfg.py`
  - 传感器配置类（`GelSightSensorCfg`），定义 prim 路径、尺寸、相机子配置、输出 `data_types`、optical/marker 模拟器配置等。

- `source/tacex/tacex/gelsight_sensor_data.py`
  - 数据容器类，定义传感器输出字段（如 tactile_rgb、marker_motion、height_map、camera_depth 等）。

- `source/tacex/tacex/simulation_approaches/`
  - 包含多个仿真实现子模块：`gpu_taxim`（GPU Taxim 实现）、`fots`（marker motion 模拟）、`uipc`（若支持）等。
  - `calibration.py` 提供校准文件路径常量（`CALIB_GELSIGHT` 等）。

- `source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py`
  - GelSight Mini 的预置配置（`GelSightMiniCfg`），包含尺寸、默认分辨率、marker/optical 模拟参数、calib 路径等。

- 示例/演示脚本与任务：
  - `scripts/demos/pick_up/pick_up_rigid.py`（演示 Franka + 双 GelSight 的交互 demo）
  - `scripts/demos/shape_touch/run_shape_touch.py`（演示如何采集形状触觉并直接展示 sensor debug）
  - `source/tacex_tasks/tacex_tasks/ball_rolling_tactile/*`（任务示例）
  - `source/tacex_uipc/examples/*`（UIPC 示例）

---

## 运行时的数据流（简述）
1. 环境初始化（`Env` 实例化） → 读取 `EnvCfg`（其中包含传感器 cfg）。
2. `_setup_scene()` 中利用 `GelSightSensor(cfg)` 创建传感器对象，并 `scene.sensors[...] = sensor` 注册。
3. 传感器在 `_initialize_impl()` 中找到或创建相关 prim（相机、gelpad），并实例化模拟器（Taxim/FOTS）。
4. 每帧：环境主循环触发 `scene.update()` → 调用传感器的 `_update_buffers_impl(dt)`，由模拟器产出图像/marker 数据并写入 `sensor.data`。
5. 可视化：当 `cfg.debug_vis=True` 且 prim 上 `debug_<datatype>` 为 True 时，Omniverse 会显示对应纹理/Camera 面板。

---

## 为什么 `run_shape_touch.py` 更容易直接显示传感器画面？
- `run_shape_touch.py` 在 `run_simulator()` 里显式设置了 prim 的 `debug_<data_type>` 属性为 True，从而强制打开 prim 的 debug 输出：
```py
for data_type in env.cfg.gsmini.data_types:
    env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
```
- `pick_up_rigid.py` 未显式设置该 prim 属性，它依赖 GUI 布局（debug panel 是否展开）和是否打开 Camera/Texture 面板。因此在默认 UI 下你可能看不到传感器图像。

---

## 如何替换 GelSight 为自定义传感器（逐步指南）
以下步骤从完整性角度给出实现自定义传感器的工作流（包含示例代码片段），假设你想实现一个名为 `CustomSensor` 的触觉传感器。

### 步骤 0 — 规划
- 决定传感器类型：视觉型（图像）、marker 型（位移场）、力阵列、单点力/触发等。
- 决定输出数据：张量形状、数据类型（例如 `custom_map`, `force_vector`）和帧率（update_period）。
- 决定模拟方法：直接读取物理量（如接触力）、基于图像的仿真（需要光学模型）、或简化的经验模型/神经近似。

### 步骤 1 — 新建配置类（cfg）
- 路径建议：`source/tacex/tacex/custom_sensor_cfg.py`
- 继承自 `SensorBaseCfg` 或对现有的 `GelSightSensorCfg` 进行定制。

示例：
```py
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

@configclass
class CustomSensorCfg(SensorBaseCfg):
    prim_path: str = "/World/envs/env_.*/Robot/custom_sensor"
    device: str = "cuda"
    debug_vis: bool = True
    data_types: list[str] = ["custom_map"]
    resolution: tuple[int,int] = (32,32)
    # 其他自定义参数
```

### 步骤 2 — 新建数据类（data）
- 路径建议：`source/tacex/tacex/custom_sensor_data.py`
- 用于封装输出张量并初始化空缓冲。

示例：
```py
from dataclasses import dataclass
import torch

@dataclass
class CustomSensorData:
    custom_map: torch.Tensor | None = None

    def init(self, num_envs, h, w, device):
        self.custom_map = torch.zeros((num_envs, h, w), device=device)
```

### 步骤 3 — 新建传感器类（继承 SensorBase）
- 路径建议：`source/tacex/tacex/custom_sensor.py`
- 必须实现的接口：`_initialize_impl()`, `_update_buffers_impl(self, dt)`, `reset()` 等。

示例模板：
```py
from isaaclab.sensors import SensorBase
from .custom_sensor_cfg import CustomSensorCfg
from .custom_sensor_data import CustomSensorData

class CustomSensor(SensorBase):
    cfg: CustomSensorCfg

    def __init__(self, cfg: CustomSensorCfg):
        super().__init__(cfg)
        self._data = CustomSensorData()

    def _initialize_impl(self):
        # 查找或创建相关 prim（如 case、gelpad、camera），初始化 buffers
        self._data.init(self.num_envs, self.cfg.resolution[1], self.cfg.resolution[0], self.device)

    def _update_buffers_impl(self, dt: float):
        # 每帧逻辑：计算/读取并写入 self._data.custom_map
        # 例如，读取 contact forces 或调用自定义模拟器
        pass

    def reset(self, env_ids=None):
        # 重置数据
        if env_ids is None:
            self._data.init(self.num_envs, self.cfg.resolution[1], self.cfg.resolution[0], self.device)
        else:
            self._data.custom_map[env_ids] = 0
```

### 步骤 4 — （可选）模拟器模块
- 如果你的传感器需要复杂的仿真逻辑（如光学渲染、marker-tracking），把它实现成一个 `simulation_approaches/<your_sim>` 模块，参考 `gpu_taxim`, `fots` 的结构和接口。
- 提供 `compute(...)` 函数将物理输入转换为传感器输出。

### 步骤 5 — 在 tacex_assets 中添加预设配置（便于复用）
- 在 `source/tacex_assets/tacex_assets/sensors/` 下建立 `custom_sensor/` 目录并放置一个 `custom_cfg.py`，导出 `CUSTOM_SENSOR_CFG`，便于在多个环境中复用。

### 步骤 6 — 在环境中挂载
- 在某个 `EnvCfg`（例如 `BallRollingEnvCfg`）内添加一项 `custom_sensor = CUSTOM_SENSOR_CFG.replace(prim_path=..., debug_vis=True)`。
- 在环境 `_setup_scene()` 中实例化并注册：
```py
self.custom_sensor = CustomSensor(self.cfg.custom_sensor)
self.scene.sensors["custom_sensor"] = self.custom_sensor
```

### 步骤 7 — 可视化与调试
- 确保 `cfg.debug_vis=True`。
- 若要程序化强制显示：在 `run_simulator()` 启动前或 `env.reset()` 后设置 prim 的 debug 属性（与 `run_shape_touch.py` 一致）：
```py
for dt in env.cfg.custom_sensor.data_types:
    env.custom_sensor._prim_view.prims[0].GetAttribute(f"debug_{dt}").Set(True)
```
- 或在 GUI 中手动展开 `debug_frame` / 打开 Camera 面板。

### 步骤 8 — 集成到 observation / RL（如果需要）
- 在环境的 `_get_observations()` 中把自定义传感器数据加入 observation dict（例如 `obs['tactile'] = env.scene.sensors['custom_sensor'].data.custom_map`）。
- 若用于训练，更新 reward/done 逻辑以使用触觉信息（例如 `in_contact` 奖励）。

### 步骤 9 — 测试与性能
- 先在 demo 脚本中集成并运行单个 env（`--num_envs 1 --debug_vis`）。
- 观察 FPS，若太慢则降低分辨率或增加 `update_period`（不必每帧计算）。
- 使用 GPU（`device='cuda'`）优先运行性能敏感的模拟器。

---

## 实用提示与常见故障排查
- 无法看到画面：检查是否使用 `--headless`，检查 `debug_vis` 与 prim `debug_<datatype>`，并展开 GUI 的 Debug 面板。
- prim view 访问失败：不要在传感器 prim 未创建前直接访问 `_prim_view`，对设置 prim 属性使用 try/except 或延迟到 env.reset 后。
- 校准文件缺失：Taxim 依赖校准数据（`TACEX_ASSETS_DATA_DIR` 下），确认资产路径正确。
- UIPC/FOTS 并发限制：FOTS/Taxim 在高并发（大量 envs）下可能受限，先以小规模测试。

---

## 最小模板（快速复制）
- `custom_sensor_cfg.py`、`custom_sensor_data.py`、`custom_sensor.py` 的模板见上文“示例模板”段落，拷贝并按需修改。

---

## 下一步（我可以帮你做的）
- 我可以把上面提到的最小模板文件创建到仓库（`source/tacex/tacex/` 下），并把一个 demo 环境改为使用该自定义传感器；或
- 我可以把此文档写入仓库的 `docs/`（已完成），也可以把文档放入 `README.md` 或其他位置。

告诉我你希望我现在：
- [ ] 把模板文件创建进代码仓库并做一个可运行示例；还是
- [ ] 只把这份 Markdown 提交（已完成）；还是
- [ ] 直接在某个 demo 中替换并运行测试（需要你确认运行环境）。

---

文档生成：`docs/TacEx_tactile_sensors_guide.md`。祝你“农”得顺利——如果“农”是指要大量生成数据或做批量采集，我可以继续帮你自动化数据保存/命名与批量运行脚本。
