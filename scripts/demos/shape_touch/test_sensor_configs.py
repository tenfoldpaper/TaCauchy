#!/usr/bin/env python3
"""
Test script to verify sensor configurations are properly set up.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")

    try:
        from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
            FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
        )
        print("✓ GelSight Mini UIPC config imported successfully")
    except Exception as e:
        print(f"✗ Failed to import GelSight Mini config: {e}")
        return False

    try:
        from tacex_assets.robots.franka.franka_digit_single_uipc import (
            FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG,
        )
        print("✓ DIGIT UIPC config imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DIGIT config: {e}")
        return False

    try:
        from tacex_assets import TACEX_ASSETS_DATA_DIR
        print(f"✓ TACEX_ASSETS_DATA_DIR: {TACEX_ASSETS_DATA_DIR}")
    except Exception as e:
        print(f"✗ Failed to import TACEX_ASSETS_DATA_DIR: {e}")
        return False

    return True


def test_usd_files():
    """Test that USD files exist."""
    print("\nTesting USD files...")

    from tacex_assets import TACEX_ASSETS_DATA_DIR

    gsmini_path = f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/uipc_gelpad.usd"
    digit_path = f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/digit/Single_Adapter/digit_uipc_gelpad.usd"

    if os.path.exists(gsmini_path):
        print(f"✓ GelSight Mini USD exists: {gsmini_path}")
    else:
        print(f"✗ GelSight Mini USD not found: {gsmini_path}")
        return False

    if os.path.exists(digit_path):
        print(f"✓ DIGIT USD exists: {digit_path}")
    else:
        print(f"✗ DIGIT USD not found: {digit_path}")
        return False

    return True


def test_configs():
    """Test that configurations are properly set up."""
    print("\nTesting configurations...")

    from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
        FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
    )
    from tacex_assets.robots.franka.franka_digit_single_uipc import (
        FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG,
    )
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    # Check GelSight Mini config
    gsmini_cfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG
    expected_gsmini_path = f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Single_Adapter/uipc_gelpad.usd"

    if gsmini_cfg.spawn.usd_path == expected_gsmini_path:
        print(f"✓ GelSight Mini config USD path correct")
    else:
        print(f"✗ GelSight Mini config USD path mismatch:")
        print(f"  Expected: {expected_gsmini_path}")
        print(f"  Got: {gsmini_cfg.spawn.usd_path}")
        return False

    # Check DIGIT config
    digit_cfg = FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG
    expected_digit_path = f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/digit/Single_Adapter/digit_uipc_gelpad.usd"

    if digit_cfg.spawn.usd_path == expected_digit_path:
        print(f"✓ DIGIT config USD path correct")
    else:
        print(f"✗ DIGIT config USD path mismatch:")
        print(f"  Expected: {expected_digit_path}")
        print(f"  Got: {digit_cfg.spawn.usd_path}")
        return False

    # Check actuator configs
    if gsmini_cfg.actuators["panda_shoulder"].stiffness == 400.0:
        print(f"✓ GelSight Mini actuator stiffness correct (400.0)")
    else:
        print(f"✗ GelSight Mini actuator stiffness incorrect: {gsmini_cfg.actuators['panda_shoulder'].stiffness}")
        return False

    if digit_cfg.actuators["panda_shoulder"].stiffness == 400.0:
        print(f"✓ DIGIT actuator stiffness correct (400.0)")
    else:
        print(f"✗ DIGIT actuator stiffness incorrect: {digit_cfg.actuators['panda_shoulder'].stiffness}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Sensor Configuration Test")
    print("=" * 70)

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_usd_files():
        all_passed = False

    if not test_configs():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
