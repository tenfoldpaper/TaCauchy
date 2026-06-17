# Isaac Sim 5.1 扩展快速启动指南

## 你的环境

- Isaac Sim 安装方式: 通过 Isaac Lab (conda 环境)
- Isaac Lab 路径: `/home/ly/IsaacLab/`
- Conda 环境: `tacex`
- 启动命令: `./isaaclab.sh -s`

## 步骤 1: 启动 Isaac Sim GUI

```bash
cd /home/ly/IsaacLab
./isaaclab.sh -s
```

等待 Isaac Sim GUI 完全启动。

## 步骤 2: 添加扩展搜索路径

1. 在 Isaac Sim 中，点击菜单 `Window` -> `Extensions`
2. 点击右上角的 **汉堡菜单图标** (☰)
3. 选择 `Settings`
4. 找到 `Extension Search Paths` 部分
5. 点击 `+` 号添加新路径
6. 输入你的 TacEx source 目录的**绝对路径**：
   ```
   /home/ly/data/hengfei/Chemistry_with_sensor/TacEx_original/source
   ```
7. 点击汉堡菜单，选择 `Refresh`

## 步骤 3: 在 GUI 中验证环境（可选但推荐）

1. 在 Isaac Sim 中，点击菜单 `Window` -> `Script Editor`
2. 在 Script Editor 中，点击 `Open` 按钮
3. 导航到并打开：
   ```
   /home/ly/data/hengfei/Chemistry_with_sensor/TacEx_original/source/tacex_uipc/check_extension_gui.py
   ```
4. 点击 `Run` 按钮（或按 Ctrl+Enter）
5. 查看输出结果

**或者直接复制粘贴运行：**

在 Script Editor 中复制粘贴以下代码并运行：

```python
import sys
sys.path.insert(0, '/home/ly/data/hengfei/Chemistry_with_sensor/TacEx_original/source/tacex_uipc')
from check_extension_gui import check_extension_in_gui
check_extension_in_gui()
```

## 步骤 4: 启用 tacex_uipc 扩展

1. 在扩展管理器的搜索框中输入 `tacex_uipc`
2. 在 `Third Party` 类别下找到 `tacex_uipc` 扩展
3. 切换开关以启用扩展
4. 你应该看到一个新窗口出现：`Generate Tet Meshes for the IPC simulation:`

## 步骤 5: 测试扩展功能

### 测试 1: 生成四面体网格

1. 在 Isaac Sim 中创建一个简单的立方体：
   - `Create` -> `Mesh` -> `Cube`
2. 在场景树中选择 `Cube` 下的 `Mesh` (不是 Xform!)
3. 在 tacex_uipc 窗口中点击 `Compute Tet Mesh` 按钮
4. 你应该看到四面体网格以线条形式显示

### 测试 2: 创建附着点

1. 创建一个刚体和一个网格
2. 先选择刚体，然后按住 Shift 选择网格
3. 点击 `Create Attachment` 按钮
4. 你应该看到附着点的可视化

## 故障排除

### 问题 1: 扩展未出现在列表中

**检查：**
```bash
# 确认路径正确
ls /home/ly/data/hengfei/Chemistry_with_sensor/TacEx_original/source/tacex_uipc/config/extension.toml
```

**解决：**
- 确保路径是绝对路径
- 确保指向 `source` 目录，不是项目根目录
- 在扩展管理器中点击 Refresh
- 重启 Isaac Sim

### 问题 2: debug_draw 模块未找到

在 Script Editor 中运行：

```python
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.debug_draw")
```

然后重新启用 tacex_uipc 扩展。

### 问题 3: 可视化不工作

1. 确保 `omni.isaac.debug_draw` 扩展已启用
2. 检查终端输出中的错误信息
3. 尝试重启 Isaac Sim

### 问题 4: 导入错误

确保以下扩展都已启用（在扩展管理器中搜索）：
- `isaaclab`
- `isaaclab_assets`
- `omni.isaac.debug_draw`

## 验证成功的标志

✅ 在扩展管理器中能找到 `tacex_uipc`  
✅ 启用后出现 UI 窗口  
✅ 能够生成四面体网格并看到可视化  
✅ 终端没有错误信息  

## 下一步

成功启用扩展后，参考以下文档了解更多功能：

- [UI 扩展教程](docs/source/tutorials/tacex_uipc_ui_extension.md)
- [扩展设置指南](source/tacex_uipc/EXTENSION_SETUP.md)
- [迁移指南](docs/ISAAC_SIM_5_MIGRATION.md)

## 常用命令

```bash
# 启动 Isaac Sim GUI
cd /home/ly/IsaacLab && ./isaaclab.sh -s

# 运行 standalone 脚本（不需要 GUI）
cd /home/ly/IsaacLab && ./isaaclab.sh -p /path/to/script.py

# 激活 conda 环境
conda activate tacex
```

## 需要帮助？

如果遇到问题：
1. 查看终端输出中的错误信息
2. 在 Script Editor 中运行 `check_extension_gui.py`
3. 查看故障排除部分
4. 检查相关文档
