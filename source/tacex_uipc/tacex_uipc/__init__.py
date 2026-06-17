"""
TacEx UIPC Package

IMPORTANT: This package requires Isaac Sim's SimulationApp to be initialized before importing.
All imports from this package must happen AFTER SimulationApp is created.

Example:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    # Now you can import from tacex_uipc
    from tacex_uipc import UipcRLEnv, UipcSim
"""

# Delay imports until SimulationApp is initialized
# This prevents ModuleNotFoundError when importing tacex_uipc before SimulationApp starts
def _check_simulation_app():
    """Check if SimulationApp is initialized."""
    try:
        import omni.kit.app
        return True
    except ModuleNotFoundError:
        return False

# Only import if SimulationApp is already initialized
# Otherwise, users must import after creating SimulationApp
if _check_simulation_app():
    from .envs import UipcInteractiveScene, UipcRLEnv
    from .objects import UipcObject, UipcObjectCfg, UipcObjectDeformableData, UipcObjectRigidData
    from .sim import UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcSim, UipcSimCfg
    from .sim import ContactForceExtractor, ContactForceData, compute_force_from_gradient
    
    # Register UI extensions (optional for headless mode).
    try:
        from .ui_extension import *  # noqa: F403
    except ImportError:
        # UI module not available (e.g., in headless mode)
        pass
else:
    # Provide helpful error message
    import sys
    
    class _LazyImport:
        """Lazy import wrapper that provides helpful error messages."""
        def __init__(self, module_name):
            self.module_name = module_name
        
        def __getattr__(self, name):
            raise ImportError(
                f"Cannot import '{self.module_name}' because Isaac Sim's SimulationApp is not initialized.\n"
                "Please initialize SimulationApp before importing from tacex_uipc:\n\n"
                "    from isaaclab.app import AppLauncher\n"
                "    app_launcher = AppLauncher(headless=True)\n"
                "    simulation_app = app_launcher.app\n\n"
                "    # Now you can import from tacex_uipc\n"
                "    from tacex_uipc import UipcRLEnv, UipcSim\n"
            )
    
    # Create lazy import stubs
    sys.modules[__name__ + '.envs'] = _LazyImport('tacex_uipc.envs')
    sys.modules[__name__ + '.objects'] = _LazyImport('tacex_uipc.objects')
    sys.modules[__name__ + '.sim'] = _LazyImport('tacex_uipc.sim')
