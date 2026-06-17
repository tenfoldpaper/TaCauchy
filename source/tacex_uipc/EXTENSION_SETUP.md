# tacex_uipc 扩展设置指南

## 快速开始

### 1. 检查兼容性（可选但推荐）

在 Isaac Sim 中运行兼容性检查脚本：

```bash
# 进入 Isaac Sim 安装目录
cd ~/.local/share/ov/pkg/isaac-sim-5.1.0/

# 运行检查脚本
./python.sh /path/to/TacEx/source/tacex_uipc/check_extension.py
```

这将验证所有必需的依赖是否可用。

### 2. 在 Isaac Sim GUI 中启用扩展

1. 启动 Isaac Sim
2. 打开扩展管理器：`Window` -> `Extensions`
3. 点击汉堡菜单 (☰) -> `Settings`
4. 在 `Extension Search Paths` 中添加：
   ```
   /path/to/TacEx/source
   ```
5. 点击汉堡菜单 (☰) -> `Refresh`
6. 搜索 `tacex_uipc`
7. 在 `Third Party` 类别下找到并启用它

### 3. 验证安装

成功启用后，你应该看到一个名为 "Generate Tet Meshes for the IPC simulation:" 的窗口。

## 版本兼容性

- **Isaac Sim 5.1+**: 完全支持（推荐）
- **Isaac Sim 4.5+**: 通过向后兼容支持

## 功能

扩展提供以下功能：

1. **Compute Tet Mesh**: 为选中的网格生成四面体网格
2. **Update Surface Mesh**: 基于四面体网格更新表面网格（仅用于可视化）
3. **Create Attachment**: 创建刚体和四面体网格之间的附着点
4. **Extract/Set primvars:st**: UV 贴图操作（实验性功能）

## 使用示例

### 生成四面体网格

1. 在场景中选择一个网格 prim（必须选择 Mesh，不是 Xform）
2. 点击 "Compute Tet Mesh" 按钮
3. 四面体网格将以线条形式可视化
4. 保存 USD 文件以保留预计算的数据

### 创建附着点

1. 选择刚体（可以选择 Xform）
2. 按住 `Shift` 并选择四面体网格
3. 点击 "Create Attachment" 按钮
4. 附着点将被可视化
5. 保存 USD 文件以保留数据

## 故障排除

### 扩展未出现

**检查项：**
- 路径必须是绝对路径
- 路径应指向 `source` 目录
- 点击 Refresh 后等待几秒
- 检查终端是否有错误信息

### 导入错误

**解决方案：**
1. 运行 `check_extension.py` 脚本查看具体缺失的依赖
2. 确保以下扩展已启用：
   - `isaaclab`
   - `isaaclab_assets`
   - `omni.isaac.debug_draw`

### 可视化不工作

如果四面体网格或附着点没有显示：
- 检查终端输出中的警告
- 确保 `omni.isaac.debug_draw` 扩展已启用
- 尝试手动启用：
  ```python
  from omni.isaac.core.utils.extensions import enable_extension
  enable_extension("omni.isaac.debug_draw")
  ```

### 性能问题

如果四面体网格生成很慢：
- 减小网格分辨率
- 调整 `TetMeshCfg` 参数（需要修改源代码）
- 使用预计算的四面体数据

## 配置

默认的四面体网格配置在 `ui_extension.py` 中：

```python
tet_cfg = TetMeshCfg(edge_length_r=0.25)
```

可用参数：
- `edge_length_r`: 边长比例（越小越精细，但计算更慢）
- `max_its`: 最大迭代次数
- `epsilon_r`: 收敛阈值

要修改配置，编辑 `source/tacex_uipc/tacex_uipc/ui_extension.py` 中的 `_generate_tet_mesh` 函数。

## 更多信息

- [完整教程](../../docs/source/tutorials/tacex_uipc_ui_extension.md)
- [Isaac Sim 5.1 迁移指南](../../docs/ISAAC_SIM_5_MIGRATION.md)
- [UIPC 文档](../../docs/install_uipc.md)

## 支持

如果遇到问题：
1. 运行 `check_extension.py` 检查依赖
2. 查看终端输出中的错误信息
3. 参考故障排除部分
4. 查看相关文档
