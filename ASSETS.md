# 资产文件说明 (Assets)

TaCauchy **代码仓库不包含** 大型二进制资产（USD 模型、校准数据、纹理等），以符合 GitHub 文件大小限制并保持仓库轻量。运行演示前需单独安装资产。

## 一键安装

```bash
./scripts/setup_assets.sh
```

该脚本从上游 [TacEx](https://github.com/DH-Ng/TacEx) 通过 Git sparse checkout + LFS 拉取 `source/tacex_assets/tacex_assets/data/` 并复制到本地。

## 手动安装

```bash
git lfs install
git clone --recurse-submodules https://github.com/DH-Ng/TacEx.git /tmp/TacEx
rsync -a /tmp/TacEx/source/tacex_assets/tacex_assets/data/ \
         ./source/tacex_assets/tacex_assets/data/
```

## 被排除的大文件（>100MB，GitHub 禁止）

| 文件 | 大小 |
|------|------|
| `Sensors/digit/Sensor.usd` | ~141 MB |
| `Sensors/digit/Case.usd` | ~139 MB |

## 被排除的大型文件（>20MB，建议 LFS 或本地保留）

| 文件 | 大小 |
|------|------|
| `Sensors/GelSight_Mini/Gelpad_extremly_high_res.usd` | ~52 MB |
| `Sensors/*/calibs/640x480/dataPack.npz` | ~22 MB |
| `Robots/Franka/*/uipc_*_gelpad.usd` | ~40–44 MB |

完整列表见上游 TacEx 的 `LARGE_FILES_EXCLUDED.md`。

## 演示所需最小资产

`simple_tactile_demo.py` 至少需要：

- `data/Sensors/GelSight_Mini/`（或 DIGIT / 9DTact 对应目录）
- `data/Robots/Franka/<sensor>/Single_Adapter/` 下的 gelpad USD
- `data/Props/tactile_test_shapes/` 测试几何体

若缺少资产，Isaac Sim 会在加载场景时报 `Failed to open USD` 类错误。

## 编译产物（不在仓库中）

`source/tacex_uipc/build/` 下的 `.so` 文件需在本地通过 `pip install -e source/tacex_uipc` 编译生成。
