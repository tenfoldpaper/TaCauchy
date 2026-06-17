#!/usr/bin/env python3
"""
在 Isaac Sim GUI 中运行的扩展检查脚本

使用方法：
1. 启动 Isaac Sim: ./isaaclab.sh -s
2. 在 Isaac Sim 的 Script Editor 中运行此脚本
   或者通过菜单: Window -> Script Editor
3. 复制此脚本内容并执行
"""

def check_extension_in_gui():
    """在 Isaac Sim GUI 中检查扩展依赖"""
    import sys
    
    print("=" * 70)
    print("tacex_uipc 扩展兼容性检查 (GUI 模式)")
    print("=" * 70)
    
    results = {}
    
    # 检查基础 Omniverse 模块
    print("\n检查 Omniverse 基础模块...")
    try:
        import omni.ext
        results['omni.ext'] = True
        print("✓ omni.ext")
    except ImportError as e:
        results['omni.ext'] = False
        print(f"✗ omni.ext: {e}")
    
    try:
        import omni.ui
        results['omni.ui'] = True
        print("✓ omni.ui")
    except ImportError as e:
        results['omni.ui'] = False
        print(f"✗ omni.ui: {e}")
    
    try:
        import omni.usd
        results['omni.usd'] = True
        print("✓ omni.usd")
    except ImportError as e:
        results['omni.usd'] = False
        print(f"✗ omni.usd: {e}")
    
    try:
        from omni.physx import get_physx_interface
        results['omni.physx'] = True
        print("✓ omni.physx")
    except ImportError as e:
        results['omni.physx'] = False
        print(f"✗ omni.physx: {e}")
    
    # 检查 debug_draw
    print("\n检查 debug_draw 模块...")
    debug_draw_version = None
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        results['debug_draw'] = True
        debug_draw_version = "Isaac Sim 5.x"
        print(f"✓ debug_draw (Isaac Sim 5.x 路径)")
    except ImportError:
        try:
            import omni.isaac.debug_draw._debug_draw as _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
            results['debug_draw'] = True
            debug_draw_version = "Isaac Sim 4.x"
            print(f"✓ debug_draw (Isaac Sim 4.x 路径)")
        except ImportError as e:
            results['debug_draw'] = False
            print(f"✗ debug_draw: 两种路径都无法导入")
            print(f"  提示: 尝试启用 omni.isaac.debug_draw 扩展")
    
    # 检查 USD
    print("\n检查 USD 模块...")
    try:
        import pxr
        from pxr import Sdf, UsdGeom
        results['pxr'] = True
        print("✓ pxr (USD)")
    except ImportError as e:
        results['pxr'] = False
        print(f"✗ pxr: {e}")
    
    # 检查 tacex_uipc 模块
    print("\n检查 tacex_uipc 模块...")
    try:
        from tacex_uipc.sim import UipcIsaacAttachments
        results['tacex_uipc.sim'] = True
        print("✓ tacex_uipc.sim")
    except ImportError as e:
        results['tacex_uipc.sim'] = False
        print(f"✗ tacex_uipc.sim: {e}")
    
    try:
        from tacex_uipc.utils import MeshGenerator, TetMeshCfg
        results['tacex_uipc.utils'] = True
        print("✓ tacex_uipc.utils")
    except ImportError as e:
        results['tacex_uipc.utils'] = False
        print(f"✗ tacex_uipc.utils: {e}")
    
    # 检查版本信息
    print("\n" + "=" * 70)
    print("环境信息")
    print("=" * 70)
    
    try:
        import carb
        version = carb.get_framework().get_version_string()
        print(f"Omniverse 框架版本: {version}")
    except:
        print("无法检测 Omniverse 框架版本")
    
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    
    if debug_draw_version:
        print(f"Debug Draw 版本: {debug_draw_version}")
    
    # 总结
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    # 详细建议
    print("\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    
    if success_count == total_count:
        print("✓ 所有依赖都可用！")
        print("\n下一步:")
        print("1. 在扩展管理器中搜索 'tacex_uipc'")
        print("2. 启用该扩展")
        print("3. 你应该能看到 'Generate Tet Meshes for the IPC simulation' 窗口")
    else:
        print("需要解决以下问题:\n")
        
        if not results.get('omni.ext') or not results.get('omni.ui'):
            print("⚠ 基础 Omniverse 模块缺失")
            print("  - 确保在 Isaac Sim GUI 中运行此脚本")
            print("  - 不要在 standalone 模式下运行\n")
        
        if not results.get('debug_draw'):
            print("⚠ debug_draw 模块未找到")
            print("  解决方案:")
            print("  1. 在扩展管理器中搜索 'omni.isaac.debug_draw'")
            print("  2. 启用该扩展")
            print("  3. 或者在 Script Editor 中运行:")
            print("     from omni.isaac.core.utils.extensions import enable_extension")
            print("     enable_extension('omni.isaac.debug_draw')\n")
        
        if not results.get('tacex_uipc.sim') or not results.get('tacex_uipc.utils'):
            print("⚠ tacex_uipc 模块未找到")
            print("  解决方案:")
            print("  1. 确保已添加扩展搜索路径:")
            print("     Window -> Extensions -> 汉堡菜单 -> Settings")
            print("     添加: /path/to/TacEx/source")
            print("  2. 点击 Refresh")
            print("  3. 确保 tacex_uipc 扩展已启用\n")
    
    print("=" * 70)
    
    return results

# 如果直接运行此脚本
if __name__ == "__main__":
    check_extension_in_gui()
