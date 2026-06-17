"""
Gelpad Mesh Configuration Examples

This file contains various mesh configuration presets for different use cases.
You can import and use these configurations directly in your code.

Usage:
    from examples.gelpad_mesh_configs import GELPAD_MESH_PRESETS
    
    cfg = GELPAD_MESH_PRESETS["high_precision"]
    mesh_gen = MeshGenerator(config=cfg)
"""

from source.tacex_uipc.tacex_uipc.utils.mesh_gen import TetMeshCfg


# ============================================================================
# Standard Presets (without center refinement)
# ============================================================================

STANDARD_COARSE = TetMeshCfg(
    edge_length_r=1/1.5,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=False,
)

STANDARD_MEDIUM = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=False,
)

STANDARD_FINE = TetMeshCfg(
    edge_length_r=1/3,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=False,
)


# ============================================================================
# Center Refinement Presets
# ============================================================================

# Fast prototyping - minimal performance impact
FAST_PROTOTYPE = TetMeshCfg(
    edge_length_r=1/1.5,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.2,
    center_edge_length_r=1/2.5,
    refinement_transition_ratio=0.1,
)

# Balanced - recommended for most applications
BALANCED = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.3,
    center_edge_length_r=1/4,
    refinement_transition_ratio=0.1,
)

# High precision - for detailed deformation capture
HIGH_PRECISION = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.4,
    center_edge_length_r=1/5,
    refinement_transition_ratio=0.15,
)

# Ultra high precision - for research and validation
ULTRA_PRECISION = TetMeshCfg(
    edge_length_r=1/3,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=100,
    enable_center_refinement=True,
    center_region_ratio=0.5,
    center_edge_length_r=1/6,
    refinement_transition_ratio=0.15,
)


# ============================================================================
# Application-Specific Presets
# ============================================================================

# For tactile sensing with small contact areas
SMALL_CONTACT = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.25,  # Smaller center region
    center_edge_length_r=1/5,  # Very dense in center
    refinement_transition_ratio=0.1,
)

# For large contact areas (e.g., palm pressing)
LARGE_CONTACT = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.4,  # Larger center region
    center_edge_length_r=1/3.5,  # Moderate density
    refinement_transition_ratio=0.15,
)

# For rolling/sliding contact (e.g., ball rolling)
ROLLING_CONTACT = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.35,
    center_edge_length_r=1/4,
    refinement_transition_ratio=0.12,
)

# For multi-point contact (e.g., grasping)
MULTI_POINT_CONTACT = TetMeshCfg(
    edge_length_r=1/2.5,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.3,
    center_edge_length_r=1/4.5,
    refinement_transition_ratio=0.1,
)


# ============================================================================
# Performance-Optimized Presets
# ============================================================================

# Real-time simulation - prioritize performance
REALTIME = TetMeshCfg(
    edge_length_r=1/1.5,
    epsilon_r=1.5e-2,
    stop_quality=15,
    max_its=60,
    enable_center_refinement=True,
    center_region_ratio=0.2,
    center_edge_length_r=1/2.5,
    refinement_transition_ratio=0.08,
)

# Offline training - balance between speed and accuracy
OFFLINE_TRAINING = TetMeshCfg(
    edge_length_r=1/2,
    epsilon_r=1e-2,
    stop_quality=10,
    max_its=80,
    enable_center_refinement=True,
    center_region_ratio=0.3,
    center_edge_length_r=1/4,
    refinement_transition_ratio=0.1,
)

# Validation/Testing - prioritize accuracy
VALIDATION = TetMeshCfg(
    edge_length_r=1/3,
    epsilon_r=8e-3,
    stop_quality=8,
    max_its=100,
    enable_center_refinement=True,
    center_region_ratio=0.4,
    center_edge_length_r=1/6,
    refinement_transition_ratio=0.15,
)


# ============================================================================
# Preset Dictionary
# ============================================================================

GELPAD_MESH_PRESETS = {
    # Standard presets
    "standard_coarse": STANDARD_COARSE,
    "standard_medium": STANDARD_MEDIUM,
    "standard_fine": STANDARD_FINE,
    
    # Center refinement presets
    "fast_prototype": FAST_PROTOTYPE,
    "balanced": BALANCED,
    "high_precision": HIGH_PRECISION,
    "ultra_precision": ULTRA_PRECISION,
    
    # Application-specific presets
    "small_contact": SMALL_CONTACT,
    "large_contact": LARGE_CONTACT,
    "rolling_contact": ROLLING_CONTACT,
    "multi_point_contact": MULTI_POINT_CONTACT,
    
    # Performance-optimized presets
    "realtime": REALTIME,
    "offline_training": OFFLINE_TRAINING,
    "validation": VALIDATION,
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_preset(name: str) -> TetMeshCfg:
    """
    Get a mesh configuration preset by name.
    
    Args:
        name: Name of the preset (see GELPAD_MESH_PRESETS keys)
        
    Returns:
        TetMeshCfg configuration object
        
    Raises:
        KeyError: If preset name is not found
    """
    if name not in GELPAD_MESH_PRESETS:
        available = ", ".join(GELPAD_MESH_PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available presets: {available}")
    
    return GELPAD_MESH_PRESETS[name]


def list_presets():
    """Print all available presets with descriptions."""
    print("\nAvailable Gelpad Mesh Presets:")
    print("=" * 80)
    
    categories = {
        "Standard Presets": ["standard_coarse", "standard_medium", "standard_fine"],
        "Center Refinement Presets": ["fast_prototype", "balanced", "high_precision", "ultra_precision"],
        "Application-Specific": ["small_contact", "large_contact", "rolling_contact", "multi_point_contact"],
        "Performance-Optimized": ["realtime", "offline_training", "validation"],
    }
    
    for category, presets in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for preset_name in presets:
            cfg = GELPAD_MESH_PRESETS[preset_name]
            refinement = "✓" if cfg.enable_center_refinement else "✗"
            print(f"  {preset_name:25s} - Refinement: {refinement}, Edge: {cfg.edge_length_r:.3f}")
            if cfg.enable_center_refinement:
                print(f"                              Center ratio: {cfg.center_region_ratio:.2f}, "
                      f"Center edge: {cfg.center_edge_length_r:.3f}")
    
    print("\n" + "=" * 80)


def compare_presets(*preset_names):
    """
    Compare multiple presets side by side.
    
    Args:
        *preset_names: Names of presets to compare
    """
    print("\nPreset Comparison:")
    print("=" * 80)
    
    params = [
        "edge_length_r",
        "enable_center_refinement",
        "center_region_ratio",
        "center_edge_length_r",
        "epsilon_r",
        "stop_quality",
        "max_its",
    ]
    
    # Header
    print(f"{'Parameter':<25s}", end="")
    for name in preset_names:
        print(f"{name:<20s}", end="")
    print()
    print("-" * 80)
    
    # Parameters
    for param in params:
        print(f"{param:<25s}", end="")
        for name in preset_names:
            cfg = get_preset(name)
            value = getattr(cfg, param)
            if isinstance(value, bool):
                value_str = "✓" if value else "✗"
            elif isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            print(f"{value_str:<20s}", end="")
        print()
    
    print("=" * 80)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Gelpad Mesh Configuration Examples")
    print("=" * 80)
    
    # List all presets
    list_presets()
    
    # Compare some presets
    print("\n")
    compare_presets("fast_prototype", "balanced", "high_precision")
    
    # Example: Get a specific preset
    print("\n\nExample: Using a preset in code")
    print("-" * 80)
    print("""
from examples.gelpad_mesh_configs import get_preset
from tacex_uipc.utils.mesh_gen import MeshGenerator

# Get the balanced preset
cfg = get_preset("balanced")

# Create mesh generator
mesh_gen = MeshGenerator(config=cfg)

# Generate mesh
tet_points, tet_indices, surf_points, surf_indices = mesh_gen.generate_tet_mesh_for_prim(usd_prim)
    """)
    print("-" * 80)
