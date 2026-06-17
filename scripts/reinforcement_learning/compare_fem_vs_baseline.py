#!/usr/bin/env python3
"""
对比脚本：比较FEM (UIPC) vs 基线方法 (Taxim/FOTS) 的RL训练性能

这个脚本会：
1. 训练使用Taxim/FOTS的基线环境
2. 训练使用UIPC FEM的环境
3. 对比两者的训练曲线和最终性能
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 定义要对比的环境配置
ENVIRONMENTS = {
    "baseline_taxim": {
        "task": "TacEx-Ball-Rolling-Taxim-Fots-v0",
        "num_envs": 64,  # Taxim较慢，使用较少环境
        "description": "基线方法 (Taxim/FOTS 光学仿真)",
    },
    "fem_uipc": {
        "task": "TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0",
        "num_envs": 16,  # UIPC FEM，先用少量环境测试
        "description": "FEM方法 (UIPC 有限元仿真 + 柯西应力)",
    },
}


def run_training(env_name: str, config: dict, max_iterations: int = 500, seed: int = 42):
    """运行单个环境的训练"""
    print(f"\n{'='*80}")
    print(f"开始训练: {config['description']}")
    print(f"环境: {config['task']}")
    print(f"并行环境数: {config['num_envs']}")
    print(f"最大迭代数: {max_iterations}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "scripts/reinforcement_learning/skrl/train.py",
        "--task", config["task"],
        "--num_envs", str(config["num_envs"]),
        "--max_iterations", str(max_iterations),
        "--seed", str(seed),
        "--enable_cameras",
        "--headless",  # 无头模式，加快训练
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {env_name} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {env_name} 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {env_name} 训练被用户中断")
        return False


def main():
    parser = argparse.ArgumentParser(description="对比FEM vs 基线方法的RL性能")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="每个环境的最大训练迭代数 (默认: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["baseline_taxim", "fem_uipc"],
        help="只运行指定的环境",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过基线训练，只训练FEM",
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TacEx RL 对比实验: FEM (UIPC) vs 基线 (Taxim/FOTS)")
    print("="*80)

    results = {}

    # 运行训练
    if args.only:
        # 只运行指定的环境
        env_name = args.only
        config = ENVIRONMENTS[env_name]
        success = run_training(env_name, config, args.max_iterations, args.seed)
        results[env_name] = success
    else:
        # 运行所有环境
        for env_name, config in ENVIRONMENTS.items():
            if args.skip_baseline and env_name == "baseline_taxim":
                print(f"\n⊘ 跳过 {env_name}")
                continue

            success = run_training(env_name, config, args.max_iterations, args.seed)
            results[env_name] = success

    # 打印总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    for env_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{env_name:20s}: {status}")

    print("\n提示:")
    print("- 查看训练日志: logs/skrl/ball_rolling/")
    print("- 使用 tensorboard 查看训练曲线:")
    print("  tensorboard --logdir logs/skrl/ball_rolling/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
