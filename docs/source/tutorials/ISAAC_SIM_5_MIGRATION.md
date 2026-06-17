# Isaac Sim 5.1 迁移指南

## 概述

TacEx 项目最初是为 Isaac Sim 4.5 开发的。本文档说明如何将 `tacex_uipc` UI 扩展迁移到 Isaac Sim 5.1。

## 主要变化

### 1. Debug Draw 导入路径

**Isaac Sim 4.x:**
```python
import omni.isaac.debug_draw._debug_draw as _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
```

**Isaac Sim 5.x:**
```python
from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()
```

### 2. 扩展配置更新

在 `extension.toml` 中添加了 `omni.isaac.debug_draw` 依赖：

```toml
[dependencies]
"isaaclab" = {}
"isaaclab_assets" = {}
"isaaclab_rl" = {}
"isaaclab_tasks" = {}
"omni.isaac.debug_draw" = {}  # 新增
```

### 3. 向后兼容性

更新后的代码包含了向后兼容逻辑，可以同时支持 Isaac Sim 4.x 和 5.x：

```python
try:
    # 尝试 Isaac Sim 5.x 路径
    from isaacsim.util.debug_draw import _debug_draw
    _draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    # 回退到 Isaac Sim 4.x 路径
    import omni.isaac.debug_draw._debug_draw as _debug_draw_module
    _draw = _debug_draw_module.acquire_debug_draw_interface()
```

## 安装步骤

### 1. 确保 Isaac Sim 5.1 正确安装

确认你的 Isaac Sim 5.1 安装路径，例如：
```bash
~/.local/share/ov/pkg/isaac-sim-5.1.0/
```

### 2. 在 Isaac Sim 中添加扩展搜索路径

1. 启动 Isaac Sim 5.1
2. 打开扩展管理器：`Window` -> `Extensions`
3. 点击 **汉堡菜单图标** (☰)，选择 `Settings`
4. 在 `Extension Search Paths` 中：
   - 点击 `+` 号
   - 添加 TacEx 的 `source` 目录的绝对路径，例如：
     ```
     /home/your_username/Projects/TacEx/source
     ```
   - 如果还没有，也添加 Isaac Lab 的扩展目录：
     ```
     /path/to/IsaacLab/source
     ```
5. 点击汉堡菜单，选择 `Refresh`

### 3. 启用扩展

1. 在扩展管理器中搜索 `tacex_uipc`
2. 在 `Third Party` 类别下找到它
3. 切换开关以启用扩展
4. 你应该能看到 `tacex_uipc` UI 窗口出现

## 故障排除

### 问题：扩展未出现在列表中

**解决方案：**
1. 检查路径是否正确（必须是绝对路径）
2. 确保路径指向 `source` 目录，而不是项目根目录
3. 在扩展管理器中点击 `Refresh`
4. 检查终端输出是否有错误信息

### 问题：导入错误

**解决方案：**
1. 确保所有依赖的扩展都已启用：
   - `isaaclab`
   - `isaaclab_assets`
   - `isaaclab_rl`
   - `isaaclab_tasks`
   - `omni.isaac.debug_draw`

2. 如果 `omni.isaac.debug_draw` 未找到，尝试手动启用：
   ```python
   from omni.isaac.core.utils.extensions import enable_extension
   enable_extension("omni.isaac.debug_draw")
   ```

### 问题：可视化功能不工作

**解决方案：**
更新后的代码包含了一个 dummy 实现，即使 debug_draw 不可用，扩展也能加载。检查终端输出中的警告信息。

## 验证安装

成功安装后，你应该能够：

1. 在 Isaac Sim GUI 中看到 `tacex_uipc` UI 窗口
2. 选择一个网格 prim 并点击 "Compute Tet Mesh" 按钮
3. 看到四面体网格以线条形式绘制出来

## 其他参考

- [Isaac Sim 5.1 文档](https://docs.isaacsim.omniverse.nvidia.com/latest/)
- [Debug Draw API 文档](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/debugging/ext_isaacsim_util_debug_draw.html)
- [Isaac Lab 迁移指南](https://isaac-sim.github.io/IsaacLab/main/source/refs/migration.html)

## 已更新的文件

- `source/tacex_uipc/tacex_uipc/ui_extension.py` - 更新了导入路径和向后兼容性
- `source/tacex_uipc/config/extension.toml` - 添加了必要的依赖项
