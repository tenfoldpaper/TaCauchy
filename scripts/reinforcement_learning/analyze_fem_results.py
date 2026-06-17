#!/usr/bin/env python3
"""
分析FEM vs 基线方法的RL训练结果

功能:
1. 从tensorboard日志中提取训练曲线
2. 对比不同方法的收敛速度和最终性能
3. 生成对比图表
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def find_training_logs(log_dir: Path) -> Dict[str, List[Path]]:
    """查找所有训练日志"""
    logs = {
        "baseline_taxim": [],
        "fem_uipc": [],
    }

    for run_dir in log_dir.glob("*"):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name.lower()
        if "taxim" in run_name or "fots" in run_name:
            logs["baseline_taxim"].append(run_dir)
        elif "uipc" in run_name:
            logs["fem_uipc"].append(run_dir)

    return logs


def extract_tensorboard_data(log_dir: Path):
    """从tensorboard日志中提取数据"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("错误: 需要安装 tensorboard")
        print("运行: pip install tensorboard")
        return None

    # 查找事件文件
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"警告: 在 {log_dir} 中未找到事件文件")
        return None

    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }

    return data


def plot_comparison(baseline_data: dict, fem_data: dict, output_dir: Path):
    """生成对比图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 关键指标
    metrics = [
        ("Reward/total", "总奖励"),
        ("Reward/success_reward", "成功奖励"),
        ("Metrics/obj_goal_distance", "物体-目标距离"),
        ("Loss/policy_loss", "策略损失"),
    ]

    for metric_key, metric_name in metrics:
        plt.figure(figsize=(10, 6))

        # 绘制基线
        if baseline_data and metric_key in baseline_data:
            steps = baseline_data[metric_key]["steps"]
            values = baseline_data[metric_key]["values"]
            plt.plot(steps, values, label="基线 (Taxim/FOTS)", alpha=0.7)

        # 绘制FEM
        if fem_data and metric_key in fem_data:
            steps = fem_data[metric_key]["steps"]
            values = fem_data[metric_key]["values"]
            plt.plot(steps, values, label="FEM (UIPC)", alpha=0.7)

        plt.xlabel("训练步数")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}对比")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_file = output_dir / f"{metric_key.replace('/', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ 保存图表: {output_file}")


def generate_summary_report(baseline_data: dict, fem_data: dict, output_dir: Path):
    """生成总结报告"""
    report = {
        "baseline_taxim": {},
        "fem_uipc": {},
    }

    # 提取关键统计数据
    for name, data in [("baseline_taxim", baseline_data), ("fem_uipc", fem_data)]:
        if not data:
            continue

        if "Reward/total" in data:
            rewards = data["Reward/total"]["values"]
            report[name]["final_reward_mean"] = float(np.mean(rewards[-100:]))
            report[name]["final_reward_std"] = float(np.std(rewards[-100:]))
            report[name]["max_reward"] = float(np.max(rewards))

        if "Metrics/obj_goal_distance" in data:
            distances = data["Metrics/obj_goal_distance"]["values"]
            report[name]["final_distance_mean"] = float(np.mean(distances[-100:]))
            report[name]["final_distance_std"] = float(np.std(distances[-100:]))
            report[name]["min_distance"] = float(np.min(distances))

    # 保存JSON报告
    report_file = output_dir / "summary_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 保存报告: {report_file}")

    # 打印到控制台
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)

    for name in ["baseline_taxim", "fem_uipc"]:
        if not report[name]:
            continue

        print(f"\n{name}:")
        for key, value in report[name].items():
            print(f"  {key:30s}: {value:.4f}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="分析FEM vs 基线的RL结果")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/skrl/ball_rolling",
        help="训练日志目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/skrl/ball_rolling/analysis",
        help="输出目录",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if not log_dir.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        return

    print(f"分析日志目录: {log_dir}")

    # 查找日志
    logs = find_training_logs(log_dir)

    print(f"\n找到的训练运行:")
    for method, runs in logs.items():
        print(f"  {method}: {len(runs)} 个运行")

    # 提取最新的运行数据
    baseline_data = None
    fem_data = None

    if logs["baseline_taxim"]:
        latest_baseline = sorted(logs["baseline_taxim"])[-1]
        print(f"\n提取基线数据: {latest_baseline.name}")
        baseline_data = extract_tensorboard_data(latest_baseline)

    if logs["fem_uipc"]:
        latest_fem = sorted(logs["fem_uipc"])[-1]
        print(f"提取FEM数据: {latest_fem.name}")
        fem_data = extract_tensorboard_data(latest_fem)

    if not baseline_data and not fem_data:
        print("\n错误: 未找到有效的训练数据")
        return

    # 生成对比图表
    print(f"\n生成对比图表...")
    plot_comparison(baseline_data, fem_data, output_dir)

    # 生成总结报告
    generate_summary_report(baseline_data, fem_data, output_dir)

    print(f"\n✓ 分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
