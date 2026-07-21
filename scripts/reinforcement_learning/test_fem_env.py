#!/usr/bin/env python3
"""
快速测试FEM环境是否正常工作

这个脚本会：
1. 加载UIPC FEM环境
2. 运行几个episode
3. 检查观察、奖励、done信号是否正常
4. 打印性能统计
"""

import argparse
import time

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试FEM环境")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_steps", type=int, default=100, help="测试步数")
parser.add_argument("--headless", action="store_true", help="无头模式")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import tacex_tasks  # noqa: F401


def test_environment(task_name: str, num_envs: int, num_steps: int):
    """测试环境"""
    print(f"\n{'='*80}")
    print(f"测试环境: {task_name}")
    print(f"并行环境数: {num_envs}")
    print(f"测试步数: {num_steps}")
    print(f"{'='*80}\n")

    # 创建环境
    print("创建环境...")
    env = gym.make(task_name, num_envs=num_envs, enable_cameras=True)

    print(f"✓ 环境创建成功")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    # 重置环境
    print("\n重置环境...")
    obs, info = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  观察键: {obs.keys() if isinstance(obs, dict) else 'single tensor'}")

    # 运行测试步骤
    print(f"\n运行 {num_steps} 步...")
    episode_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(num_envs, device=env.unwrapped.device)
    num_resets = 0

    start_time = time.time()

    for step in range(num_steps):
        # 随机动作
        actions = torch.rand(num_envs, env.action_space.shape[0], device=env.unwrapped.device) * 2 - 1

        # 执行步骤
        obs, rewards, terminated, truncated, info = env.step(actions)

        # 累积统计
        episode_rewards += rewards
        episode_lengths += 1

        # 检查重置
        dones = terminated | truncated
        if dones.any():
            num_resets += dones.sum().item()
            episode_rewards[dones] = 0
            episode_lengths[dones] = 0

        # 打印进度
        if (step + 1) % 20 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) * num_envs / elapsed
            print(f"  步骤 {step+1}/{num_steps} | FPS: {fps:.1f} | 重置次数: {num_resets}")

    elapsed_time = time.time() - start_time
    total_steps = num_steps * num_envs
    fps = total_steps / elapsed_time

    # 打印统计
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")
    print(f"总步数: {total_steps}")
    print(f"总时间: {elapsed_time:.2f} 秒")
    print(f"FPS: {fps:.1f}")
    print(f"重置次数: {num_resets}")
    print(f"平均episode长度: {episode_lengths.mean().item():.1f}")
    print(f"平均奖励: {episode_rewards.mean().item():.3f}")
    print(f"{'='*80}\n")

    # 关闭环境
    env.close()

    return {
        "fps": fps,
        "num_resets": num_resets,
        "avg_episode_length": episode_lengths.mean().item(),
        "avg_reward": episode_rewards.mean().item(),
    }


def main():
    """主函数"""
    # 测试UIPC FEM环境
    try:
        results = test_environment(
            task_name="TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0",
            num_envs=args_cli.num_envs,
            num_steps=args_cli.num_steps,
        )

        print("\n✓ 环境测试通过！")
        print("\n建议:")
        if results["fps"] < 50:
            print("  - FPS较低，考虑减少并行环境数或降低传感器分辨率")
        if results["num_resets"] == 0:
            print("  - 没有episode完成，考虑增加测试步数")

        print("\n可以开始训练了！运行:")
        print(f"  python scripts/reinforcement_learning/skrl/train.py \\")
        print(f"      --task TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0 \\")
        print(f"      --num_envs {args_cli.num_envs} \\")
        print(f"      --enable_cameras \\")
        print(f"      --headless")

    except Exception as e:
        print(f"\n✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
