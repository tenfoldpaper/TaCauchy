#!/usr/bin/env python3
"""
检查 tacex_uipc 扩展的依赖和兼容性

这个脚本可以在 Isaac Sim 环境中运行，用于验证扩展是否可以正确加载。
"""

import sys

def check_imports():
    """检查所有必需的导入"""
    print("=" * 60)
    print("检查 tacex_uipc 扩展依赖...")
    print("=" * 60)
    
    results = {}
    
    # 检查基础 Omniverse 模块
    try:
        import omni.ext
        results['omni.ext'] = '✓'
        print("✓ omni.ext")
    except ImportError as e:
        results['omni.ext'] = f'✗ {e}'
        print(f"✗ omni.ext: {e}")
    
    try:
        import omni.ui
        results['omni.ui'] = '✓'
        print("✓ omni.ui")
    except ImportError as e:
        results['omni.ui'] = f'✗ {e}'
        print(f"✗ omni.ui: {e}")
    
    try:
        import omni.usd
        results['omni.usd'] = '✓'
        print("✓ omni.usd")
    except ImportError as e:
        results['omni.usd'] = f'✗ {e}'
        print(f"✗ omni.usd: {e}")
    
    try:
        from omni.physx import get_physx_interface
        results['omni.physx'] = '✓'
        print("✓ omni.physx")
    except ImportError as e:
        results['omni.physx'] = f'✗ {e}'
        print(f"✗ omni.physx: {e}")
    
    # 检查 debug_draw (尝试两种路径)
    print("\n检查 debug_draw 模块...")
    debug_draw_found = False
    
    try:
        from isaacsim.util.debug_draw import _debug_draw
        results['debug_draw'] = '✓ (Isaac Sim 5.x 路径)'
        print("✓ debug_draw (Isaac Sim 5.x 路径: isaacsim.util.debug_draw)")
        debug_draw_found = True
    except ImportError:
        try:
            import omni.isaac.debug_draw._debug_draw
            results['debug_draw'] = '✓ (Isaac Sim 4.x 路径)'
            print("✓ debug_draw (Isaac Sim 4.x 路径: omni.isaac.debug_draw)")
            debug_draw_found = True
        except ImportError as e:
            results['debug_draw'] = f'✗ {e}'
            print(f"✗ debug_draw: 两种路径都无法导入")
            print(f"  - Isaac Sim 5.x 路径失败")
            print(f"  - Isaac Sim 4.x 路径失败")
    
    # 检查 USD 相关模块
    print("\n检查 USD 模块...")
    try:
        import pxr
        from pxr import Sdf, UsdGeom
        results['pxr'] = '✓'
        print("✓ pxr (USD)")
    except ImportError as e:
        results['pxr'] = f'✗ {e}'
        print(f"✗ pxr: {e}")
    
    # 检查 tacex_uipc 模块
    print("\n检查 tacex_uipc 模块...")
    try:
        from tacex_uipc.sim import UipcIsaacAttachments
        results['tacex_uipc.sim'] = '✓'
        print("✓ tacex_uipc.sim")
    except ImportError as e:
        results['tacex_uipc.sim'] = f'✗ {e}'
        print(f"✗ tacex_uipc.sim: {e}")
    
    try:
        from tacex_uipc.utils import MeshGenerator, TetMeshCfg
        results['tacex_uipc.utils'] = '✓'
        print("✓ tacex_uipc.utils")
    except ImportError as e:
        results['tacex_uipc.utils'] = f'✗ {e}'
        print(f"✗ tacex_uipc.utils: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    success_count = sum(1 for v in results.values() if '✓' in str(v))
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓ 所有依赖都可用！扩展应该可以正常工作。")
        return True
    else:
        print("\n✗ 某些依赖缺失。请检查上面的错误信息。")
        if not debug_draw_found:
            print("\n注意: debug_draw 模块未找到，但扩展仍可加载（可视化功能将受限）")
        return False

def check_isaac_sim_version():
    """尝试检测 Isaac Sim 版本"""
    print("\n" + "=" * 60)
    print("检测 Isaac Sim 版本...")
    print("=" * 60)
    
    try:
        import carb
        version = carb.get_framework().get_version_string()
        print(f"Omniverse 框架版本: {version}")
    except:
        print("无法检测 Omniverse 框架版本")
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")

if __name__ == "__main__":
    print("\ntacex_uipc 扩展兼容性检查工具")
    print("=" * 60)
    
    check_isaac_sim_version()
    print()
    success = check_imports()
    
    print("\n" + "=" * 60)
    if success:
        print("建议: 尝试在 Isaac Sim 扩展管理器中启用 tacex_uipc 扩展")
    else:
        print("建议: 解决上述依赖问题后再尝试启用扩展")
    print("=" * 60)
