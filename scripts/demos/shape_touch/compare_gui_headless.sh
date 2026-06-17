#!/bin/bash
# Compare GUI vs Headless performance for tactile simulation

SENSOR=${1:-gelsight}
NUM_ENVS=${2:-1}
STEPS=${3:-500}

echo "=========================================="
echo "Tactile Simulation Performance Comparison"
echo "=========================================="
echo "Sensor: $SENSOR"
echo "Environments: $NUM_ENVS"
echo "Steps: $STEPS"
echo ""

# Run GUI benchmark
echo "Running GUI benchmark..."
python scripts/demos/shape_touch/benchmark_tactile_performance.py \
    --sensor $SENSOR \
    --num_envs $NUM_ENVS \
    --steps $STEPS \
    --save_results

echo ""
echo "Waiting 5 seconds before headless test..."
sleep 5

# Run Headless benchmark
echo "Running Headless benchmark..."
python scripts/demos/shape_touch/benchmark_tactile_performance.py \
    --sensor $SENSOR \
    --num_envs $NUM_ENVS \
    --steps $STEPS \
    --headless \
    --save_results

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "Results saved to outputs/benchmarks/"
echo "=========================================="
