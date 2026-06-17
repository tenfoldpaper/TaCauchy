#!/bin/bash
# Example commands for running the modular demo with different sensors

echo "=========================================="
echo "Contact Force Visualization Demo"
echo "Sensor Selection Examples"
echo "=========================================="
echo ""

# Function to print command and wait for user
run_example() {
    echo "Command: $1"
    echo "Description: $2"
    echo ""
    read -p "Press Enter to run this command (or Ctrl+C to skip)..."
    eval $1
    echo ""
    echo "=========================================="
    echo ""
}

# Example 1: Run with GelSight Mini (default)
run_example \
    "python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py" \
    "Run with GelSight Mini sensor (default)"

# Example 2: Run with DIGIT sensor
run_example \
    "python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor digit" \
    "Run with DIGIT sensor"

# Example 3: Run with GelSight Mini and debug output
run_example \
    "python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor gelsight --debug_force --debug_deformation" \
    "Run with GelSight Mini and debug output"

# Example 4: Run with DIGIT and all debug output
run_example \
    "python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor digit --debug_all" \
    "Run with DIGIT sensor and all debug output"

# Example 5: Run with custom force scale
run_example \
    "python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py --sensor digit --force_scale 0.2" \
    "Run with DIGIT sensor and larger force arrows"

echo "All examples completed!"
