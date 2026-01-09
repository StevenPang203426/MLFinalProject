# 基线实验使用指南

## 概述

本目录包含使用 TRL 框架对 qwen2-1.5b 模型进行 RLHF 微调的完整实现。

## 文件说明

- `train_baseline.py`: RLHF 训练脚本（使用 PPO）
- `evaluate_baseline.py`: 模型评估脚本（支持 ROUGE、BLEU 等指标）
- `README.md`: 实验说明文档

## 快速开始

### 1. 数据准备

确保已经将数据转换为训练格式：

```bash
# 数据应该已经转换完成，位于：
# - data/train_converted.jsonl
# - data/valid_converted.jsonl
# - data/test_converted.jsonl
```

### 2. 训练模型

#### 使用默认配置

```bash
cd FinalProject/experiments/baseline
python train_baseline.py --config baseline_config.yaml
```

#### 指定模型路径

```bash
python train_baseline.py \
    --config baseline_config.yaml \
    --model_name "C:/Users/lenovo/Desktop/PDML/models/qwen2-1.5b"
```

#### 指定训练数据

```bash
python train_baseline.py \
    --config baseline_config.yaml \
    --train_data "../data/train_converted.jsonl"
```

#### 启用 wandb 日志

```bash
python train_baseline.py \
    --config baseline_config.yaml \
    --use_wandb
```

### 3. 评估模型

#### 评估训练后的模型

```bash
python evaluate_baseline.py \
    --model_path "../results/baseline/final_model" \
    --test_data "../data/test_converted.jsonl" \
    --output_file "evaluation_results.json"
```

#### 评估原始模型（基线对比）

```bash
python evaluate_baseline.py \
    --model_path "C:/Users/lenovo/Desktop/PDML/models/qwen2-1.5b" \
    --test_data "../data/test_converted.jsonl" \
    --output_file "baseline_results.json"
```

#### 快速测试（只评估部分样本）

```bash
python evaluate_baseline.py \
    --model_path "../results/baseline/final_model" \
    --test_data "../data/test_converted.jsonl" \
    --max_samples 100
```

## 配置说明

配置文件位于 `../../configs/baseline_config.yaml`，主要配置项：

### 模型配置
```yaml
model:
  name: "../models/qwen2-1.5b"  # 模型路径
```

### 数据配置
```yaml
data:
  train_path: "./data/train_converted.jsonl"  # 训练数据
  task_type: "summarization"  # 任务类型
```

### 训练配置
```yaml
training:
  output_dir: "./results/baseline"  # 输出目录
  num_epochs: 3  # 训练轮数
  batch_size: 128  # 批次大小
  learning_rate: 1.41e-5  # 学习率
```

### PPO 配置
```yaml
ppo:
  ppo_epochs: 4  # PPO 内部迭代次数
  target_kl: 0.1  # 目标 KL 散度
  init_kl_coef: 0.2  # 初始 KL 系数
```

## 训练流程

1. **加载模型和分词器**
   - 从指定路径加载 qwen2-1.5b 模型
   - 设置 pad_token 等必要配置

2. **加载和准备数据**
   - 加载转换后的训练数据
   - 使用 `prepare_ppo_dataset` 准备 PPO 训练格式

3. **初始化奖励模型**
   - 使用简单的基于参考摘要的奖励函数
   - 结合长度匹配和关键词重叠计算奖励

4. **PPO 训练**
   - 创建参考模型（用于 KL 散度计算）
   - 初始化 PPOTrainer
   - 执行多轮训练循环

5. **保存模型**
   - 每个 epoch 保存检查点
   - 训练完成后保存最终模型

## 评估指标

评估脚本会计算以下指标：

- **ROUGE-1, ROUGE-2, ROUGE-L**: 摘要质量指标
- **BLEU**: 生成质量指标
- **平均长度**: 预测和参考的平均长度

## 输出文件

### 训练输出
- `results/baseline/checkpoint-epoch-{N}/`: 每个 epoch 的检查点
- `results/baseline/final_model/`: 最终训练模型

### 评估输出
- `evaluation_results.json`: 评估结果 JSON 文件

## 注意事项

1. **内存要求**: qwen2-1.5b 模型需要足够的 GPU 内存（建议至少 8GB）
2. **训练时间**: 完整训练可能需要数小时，取决于数据量和硬件
3. **奖励函数**: 当前使用简单的奖励函数，可以根据需要改进
4. **数据格式**: 确保使用转换后的数据（`*_converted.jsonl`）

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确保模型文件完整

### 内存不足
- 减小 `batch_size`
- 使用 `gradient_accumulation_steps` 增加有效批次大小

### 训练不稳定
- 调整学习率
- 调整 `target_kl` 和 `init_kl_coef`
- 检查奖励函数是否合理

## 下一步

- 尝试改进奖励函数（见 `improved/` 目录）
- 调整超参数以获得更好效果
- 使用更复杂的评估指标
