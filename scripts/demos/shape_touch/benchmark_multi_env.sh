#!/bin/bash
# Benchmark tactile simulation with multiple environments
# Compare GUI vs Headless performance

SENSOR=${1:-digit}
NUM_ENVS=${2:-10}
STEPS=${3:-500}

echo "=========================================="
echo "Multi-Environment Performance Benchmark"
echo "=========================================="
echo "Sensor: $SENSOR"
echo "Environments: $NUM_ENVS"
echo "Steps: $STEPS"
echo ""

# Run GUI benchmark
echo "=========================================="
echo "1/2: Running GUI benchmark with $NUM_ENVS environments..."
echo "=========================================="
python scripts/demos/shape_touch/benchmark_tactile_performance.py \
    --sensor $SENSOR \
    --num_envs $NUM_ENVS \
    --steps $STEPS \
    --save_results

echo ""
echo "Waiting 10 seconds before headless test..."
sleep 10

# Run Headless benchmark
echo ""
echo "=========================================="
echo "2/2: Running Headless benchmark with $NUM_ENVS environments..."
echo "=========================================="
python scripts/demos/shape_touch/benchmark_tactile_performance.py \
    --sensor $SENSOR \
    --num_envs $NUM_ENVS \
    --steps $STEPS \
    --headless \
    --save_results

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to outputs/benchmarks/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/demos/shape_touch/analyze_benchmark_results.py"
