# 资产文件说明 (Assets)

TaCauchy 需要 USD 模型、校准 `dataPack.npz`、纹理、策略文件等二进制资产才能完整复现 GelSight Mini、DIGIT 和 9DTact 演示。本仓库现在通过 **Git LFS** 管理这些运行时资产，避免 GitHub 普通 Git blob 的 100 MB 限制。

## 一键拉取与检查

```bash
git lfs install
./scripts/setup_assets.sh
```

该脚本会执行 `git lfs pull --include="source/tacex_assets/tacex_assets/data/**"`，并检查核心演示需要的资产是否真实存在且不是 LFS pointer。

## 已纳入 LFS 的核心资产

| 目录 | 内容 |
|------|------|
| `source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/` | GelSight Mini USD、纹理与 Taxim/FOTS 标定 |
| `source/tacex_assets/tacex_assets/data/Sensors/digit/` | DIGIT USD、纹理与标定 |
| `source/tacex_assets/tacex_assets/data/Sensors/9dtact/` | 9DTact USD、纹理与标定 |
| `source/tacex_assets/tacex_assets/data/Robots/Franka/` | Franka + GelSight Mini / DIGIT / 9DTact adapter USD 与 mesh |
| `source/tacex_assets/tacex_assets/data/Props/` | 演示物体、形状测试集与 plate/ball assets |
| `source/tacex_assets/tacex_assets/data/Policies/` | 复现实验所需策略文件 |

## 不纳入仓库的文件

以下文件保持忽略，不影响复现：

- `*backup*`、`*.backup*`、`*.before_*`、`*.broken`：本地修复过程中的备份/坏版本。
- `*.zip`：已解压资产的归档副本。
- `source/tacex_uipc/build/`：本地编译生成的 `.so`、CMake 和 vcpkg 产物。
- `logs/`、`outputs/`、`tactile_data/`、`tactile_images/`：运行时输出。

## 演示所需最小资产

`simple_tactile_demo.py` 至少需要：

- `data/Sensors/GelSight_Mini/`、`data/Sensors/digit/` 或 `data/Sensors/9dtact/` 中对应传感器目录。
- `data/Robots/Franka/<sensor>/Single_Adapter/` 下的 UIPC gelpad USD。
- `data/Props/tactile_test_shapes/` 和 `data/Props/plate.usd` 测试几何体。

若缺少资产，Isaac Sim 会在加载场景时报 `Failed to open USD` 或 Taxim 标定文件读取错误。

## 编译产物

`source/tacex_uipc/build/` 下的动态库不提交到 GitHub，需要在本地通过以下命令生成：

```bash
./scripts/setup_libuipc_patches.sh
pip install -e source/tacex_uipc -v
```
