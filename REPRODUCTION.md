# TaCauchy 安装与复现指南

本文档说明如何在 Ubuntu 22.04 上从零安装 TaCauchy，并复现论文中的核心演示（UIPC-FEM 软体仿真 + Cauchy 应力提取 + 视觉触觉图像 + 力场可视化）。

## 1. 系统要求

| 项目 | 推荐配置 |
|------|----------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA GPU（需 CUDA，文档测试环境：RTX 5090 + Driver 550 + CUDA 12.8） |
| Python | 3.11 |
| Isaac Sim | **5.0.0** |
| Isaac Lab | **v2.2.1** |
| 编译工具 | GCC 11.4、CMake ≥ 3.26、vcpkg |

> 完整硬件与软件依赖见 [docs/source/installation/Local-Installation.md](docs/source/installation/Local-Installation.md)。

## 2. 克隆仓库

```bash
git lfs install
git clone --recurse-submodules https://github.com/figsama/TaCauchy.git
cd TaCauchy
./scripts/setup_assets.sh
./scripts/setup_libuipc_patches.sh
```

若已克隆但未拉取子模块：

```bash
git submodule update --init --recursive
```

## 3. 拉取并检查传感器资产（必需）

本仓库通过 **Git LFS** 管理 USD 模型、校准 `dataPack.npz`、纹理和策略文件（见 [ASSETS.md](ASSETS.md)）。普通 `git clone` 可能只得到 LFS pointer，因此运行演示前请执行：

```bash
./scripts/setup_assets.sh
```

该脚本会拉取 `source/tacex_assets/tacex_assets/data/` 下的 LFS 文件，并检查 GelSight Mini、DIGIT、9DTact、Franka adapter 和测试物体是否齐全。

## 4. 应用 TaCauchy 的 libuipc 本地补丁（必需）

`source/tacex_uipc/libuipc` 固定到可公开克隆的 `DH-Ng/libuipc@837808a361398f18e9526e36a0764b1ec0b741ac`，TaCauchy 额外需要的 Cauchy stress / CUDA 兼容性改动保存在 `patches/`：

```bash
./scripts/setup_libuipc_patches.sh
```

该脚本会初始化子模块并把 `patches/libuipc-tacauchy.patch` 与 `patches/muda-tacauchy.patch` 应用到本地子模块。若已应用，脚本会自动跳过。

## 5. 安装 Isaac Sim 5.0 与 Isaac Lab 2.2.1

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com

# 验证 Isaac Sim
isaacsim
```

安装 Isaac Lab：

```bash
sudo apt install cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
git checkout v2.2.1
conda activate env_isaaclab
./isaaclab.sh --install
cd ..
```

验证 Isaac Lab（可选）：

```bash
conda activate env_isaaclab
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

## 6. 安装 TaCauchy 核心包

```bash
cd TaCauchy
conda activate env_isaaclab

# 安装 tacex、tacex_assets、tacex_tasks
./tacex.sh -i
```

## 7. 编译安装 UIPC 扩展（TaCauchy 核心）

UIPC 需要 vcpkg 与 CUDA。首次编译请参考 [docs/source/installation/Local-Installation.md](docs/source/installation/Local-Installation.md) 中的 UIPC 章节。

简要步骤：

```bash
# 1) 安装 vcpkg（若尚未安装）
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export CMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"

# 2) 确保 TaCauchy 的 libuipc 补丁已应用
./scripts/setup_libuipc_patches.sh

# 3) 安装 libuipc 构建依赖
conda env update -n env_isaaclab --file source/tacex_uipc/libuipc/conda/env.yaml

# 4) 按 GPU 架构设置 CUDA（示例：RTX 4090/5090 为 89）
export CMAKE_CUDA_ARCHITECTURES=89

# 5) 可编辑安装 tacex_uipc（会编译 libuipc C++ 扩展）
pip install -e source/tacex_uipc -v
```

> 编译耗时较长（首次约 10–30 分钟，视机器而定）。若失败，请检查 GCC 版本、vcpkg 路径与 CUDA 架构号。

## 8. 复现核心演示

### 8.1 精简力场 + 触觉 RGB 演示（推荐入口）

```bash
conda activate env_isaaclab
cd TaCauchy

# GelSight Mini
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight

# DIGIT
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor digit

# 9DTact
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor 9dtact
```

**交互说明**：
- 拖动场景中的红色立方体，使传感器与物体接触
- 窗口显示：点状法向/切向力、20×20 插值力场、触觉 RGB 图像
- 点击触觉图像窗口可保存 PNG 到 `tactile_images/`

如需保存触觉图像，点击触觉图像窗口即可写入 `tactile_images/`。

### 8.2 完整模块化力场可视化

```bash
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py
```

### 8.3 性能基准（无 GUI）

```bash
python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor gelsight --headless
```

### 8.4 Taxim 光学仿真验证（不含 UIPC 应力）

```bash
python scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

### 8.5 RL 任务示例

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0 \
  --num_envs 1 --enable_cameras
```

## 9. 论文对应模块

| 论文概念 | 代码位置 |
|----------|----------|
| UIPC-FEM 软体求解 | `source/tacex_uipc/tacex_uipc/sim/uipc_sim.py` |
| Cauchy 应力提取 | `source/tacex_uipc/tacex_uipc/sim/stress_extractor.py` |
| 接触力/牵引力 | `source/tacex_uipc/tacex_uipc/sim/contact_force_extractor.py` |
| 自适应网格细化 | `source/tacex_uipc/tacex_uipc/utils/mesh_gen.py` |
| 多传感器配置 | `source/tacex_assets/tacex_assets/sensors/` |
| 力场可视化 | `source/tacex_uipc/tacex_uipc/utils/visualization/` |
| 主演示脚本 | `scripts/demos/shape_touch/` |

## 10. 常见问题

### Q: 启动时报找不到 USD 资产

运行 `./scripts/setup_assets.sh`，或确认 `source/tacex_assets/tacex_assets/data/` 下存在传感器与机器人模型，且文件不是 Git LFS pointer。

### Q: `pip install -e source/tacex_uipc` 编译失败

- 确认 `CMAKE_TOOLCHAIN_FILE` 指向 vcpkg
- 确认 GCC 11.4 与 `CMAKE_CUDA_ARCHITECTURES` 与 GPU 匹配
- 参考 [docs/source/installation/Troubleshooting.md](docs/source/installation/Troubleshooting.md)

### Q: 插值力场窗口全黑

传感器尚未产生有效接触——拖动交互物体使 gelpad 压入目标表面。

### Q: Isaac Sim 扩展未加载

确认 Isaac Sim / Isaac Lab 版本与本文档一致，并在 Isaac Sim Extension Manager 中启用 TaCauchy 相关扩展。

### Q: `libuipc` 编译后没有 Cauchy stress / force field 输出

确认已经运行 `./scripts/setup_libuipc_patches.sh`，并在应用补丁后重新执行 `pip install -e source/tacex_uipc -v`。

如有问题，请在 GitHub Issues 中反馈，并附上 GPU 型号、Isaac Sim/Lab 版本与完整报错日志。
