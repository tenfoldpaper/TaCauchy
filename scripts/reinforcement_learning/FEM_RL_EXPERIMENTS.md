# FEM vs 基线方法的强化学习对比实验

## 概述

本目录包含用于对比FEM (UIPC) 和基线方法 (Taxim/FOTS) 在强化学习任务中性能的脚本。

## 实验设置

### 对比的方法

1. **基线方法 (Taxim/FOTS)**
   - 环境: `TacEx-Ball-Rolling-Taxim-Fots-v0`
   - 光学仿真: 使用Taxim进行形变近似 + FOTS进行标记点运动
   - 特点: 快速但物理保真度较低

2. **FEM方法 (UIPC)**
   - 环境: `TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0`
   - 物理仿真: 使用UIPC有限元方法 + 柯西应力计算
   - 特点: 高物理保真度，包含真实的应力-应变关系

### 任务描述

**Ball Rolling任务**: 机器人需要使用触觉传感器将球推到目标位置。

- 观察空间: 
  - 本体感觉: 末端执行器位置、姿态、目标位置等 (14维)
  - 视觉: 触觉传感器RGB图像 (32×32×3)
- 动作空间: 末端执行器相对位置和姿态 (6维)
- 奖励函数: 包含目标跟踪、成功奖励、行为惩罚等多个组件

## 使用方法

### 1. 运行对比实验

```bash
# 运行完整对比实验（基线 + FEM）
python scripts/reinforcement_learning/compare_fem_vs_baseline.py \
    --max_iterations 1000 \
    --seed 42

# 只运行FEM方法
python scripts/reinforcement_learning/compare_fem_vs_baseline.py \
    --only fem_uipc \
    --max_iterations 1000

# 跳过基线，只训练FEM
python scripts/reinforcement_learning/compare_fem_vs_baseline.py \
    --skip-baseline \
    --max_iterations 1000
```

### 2. 分析结果

训练完成后，使用分析脚本生成对比图表：

```bash
python scripts/reinforcement_learning/analyze_fem_results.py \
    --log_dir logs/skrl/ball_rolling \
    --output_dir logs/skrl/ball_rolling/analysis
```

这将生成：
- 训练曲线对比图（总奖励、成功率、距离误差等）
- JSON格式的性能总结报告
- 统计数据（均值、标准差、最大/最小值）

### 3. 使用TensorBoard查看实时训练

```bash
tensorboard --logdir logs/skrl/ball_rolling/
```

然后在浏览器中打开 `http://localhost:6006`

## 预期结果

### FEM方法的优势

1. **物理保真度**: 
   - 真实的应力-应变关系
   - 准确的接触力计算
   - 体积守恒的形变

2. **泛化能力**:
   - 更好的sim-to-real迁移潜力
   - 对物体属性变化更鲁棒

3. **信息丰富度**:
   - 提供柯西应力张量
   - 法向和切向力分解
   - 接触压力分布

### 可能的权衡

1. **计算成本**: FEM方法可能比基线方法慢
2. **样本效率**: 需要更多训练样本才能收敛
3. **并行度**: 可能需要调整并行环境数量

## 实验配置调整

### 修改环境数量

编辑 `compare_fem_vs_baseline.py` 中的 `ENVIRONMENTS` 字典：

```python
ENVIRONMENTS = {
    "baseline_taxim": {
        "num_envs": 64,  # 调整这里
    },
    "fem_uipc": {
        "num_envs": 16,  # 调整这里
    },
}
```

### 修改训练超参数

编辑对应的agent配置文件：
- `source/tacex_tasks/tacex_tasks/ball_rolling_tactile/agents/skrl_ppo_tactile_rgb_cfg.yaml`

关键参数：
- `learning_rate`: 学习率
- `rollouts`: 每次迭代的rollout步数
- `mini_batches`: mini-batch数量
- `discount_factor`: 折扣因子

## 论文实验建议

### 实验1: 收敛速度对比
- 训练步数: 5000-10000 iterations
- 指标: 达到目标性能所需的样本数

### 实验2: 最终性能对比
- 训练步数: 充分训练直到收敛
- 指标: 最终成功率、平均奖励、目标距离误差

### 实验3: 泛化能力测试
- 在训练后测试不同物体属性（质量、摩擦系数等）
- 指标: 性能下降程度

### 实验4: 消融实验
- 对比使用/不使用柯西应力信息的性能
- 对比不同网格细化策略的影响

## 故障排除

### 内存不足
- 减少 `num_envs`
- 减少传感器分辨率
- 使用更粗的网格

### 训练不稳定
- 降低学习率
- 增加 `mini_batches`
- 调整奖励权重

### UIPC仿真错误
- 检查网格质量参数
- 调整接触参数 `d_hat`
- 确保附着点配置正确

## 相关文件

- 环境定义: `source/tacex_tasks/tacex_tasks/ball_rolling_tactile/`
- UIPC配置: `source/tacex_uipc/tacex_uipc/`
- 传感器配置: `source/tacex_assets/tacex_assets/sensors/`
- 训练日志: `logs/skrl/ball_rolling/`

## 引用

如果使用这些实验结果，请引用：
- TacEx框架
- UIPC物理引擎
- Isaac Lab平台
