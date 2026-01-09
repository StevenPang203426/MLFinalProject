# 基线实验

## 实验说明

本目录包含基线实验的代码和结果。

## 文件结构

```
baseline/
├── README.md              # 本文件
├── train_baseline.py      # 基线训练脚本
├── evaluate_baseline.py   # 基线评估脚本
├── config.yaml            # 实验配置文件
└── results/               # 实验结果（自动生成）
```

## 实验设置

### 模型
- 模型名称：
- 参数量：
- 预训练权重：

### 任务
- 任务类型：summarization（摘要生成）
- 数据集：JSONL 格式，详见 `../../data/DATA_FORMAT.md`

### 训练参数
- 学习率：
- 批次大小：
- 训练轮数：
- 其他超参数：

## 配置文件说明

配置文件位于 `../../configs/baseline_config.yaml`，包含以下主要配置：

### 模型配置
- `model.name`: 模型路径（支持本地路径或 HuggingFace 模型名称）
  - 本地模型示例: `"../models/qwen2-1.5b"` 或 `"C:/Users/lenovo/Desktop/PDML/models/qwen2-1.5b"`
  - HuggingFace 模型示例: `"Qwen/Qwen-1_8B"`

### 训练配置
- `training.output_dir`: 输出目录
- `training.learning_rate`: 学习率
- `training.batch_size`: 批次大小
- `training.num_epochs`: 训练轮数
- 其他训练超参数

### PPO 配置
- `ppo.ppo_epochs`: PPO 训练轮数
- `ppo.target_kl`: 目标 KL 散度
- `ppo.init_kl_coef`: 初始 KL 系数
- 其他 PPO 超参数

## 运行方式

### 使用默认配置文件
```bash
# 从 experiments/baseline 目录运行
python train_baseline.py --config baseline_config.yaml
```

### 使用自定义配置文件
```bash
# 指定配置文件路径（相对于 configs/ 目录）
python train_baseline.py --config my_custom_config.yaml
```

### 覆盖配置文件中的参数
```bash
# 使用命令行参数覆盖配置文件中的设置
python train_baseline.py \
    --config baseline_config.yaml \
    --model_name "../models/qwen2-1.5b" \
    --output_dir "./my_results" \
    --use_wandb
```

### 数据准备

#### 数据格式要求

数据文件应为 JSONL 格式（每行一个 JSON 对象），每个样本必须包含：
- `prompt` (必需): 输入提示词
- `reference` (可选): 参考输出，用于评估

详细的数据格式说明请参考：`../../data/DATA_FORMAT.md`

#### 示例数据

项目提供了示例数据文件：
- `../../data/train.jsonl.example` - 训练集示例
- `../../data/val.jsonl.example` - 验证集示例
- `../../data/test.jsonl.example` - 测试集示例

#### 验证数据格式

在训练前，建议先验证数据格式：

```bash
# 验证训练数据
python ../../experiments/utils/validate_data.py --data_path ../../data/train.jsonl --task_type summarization

# 验证验证集数据
python ../../experiments/utils/validate_data.py --data_path ../../data/val.jsonl --task_type summarization
```

### 评估
```bash
python evaluate_baseline.py --model_path <model_path>
```

## 实验结果

### 训练指标
- Loss 曲线
- Reward 曲线
- 其他指标

### 评估指标
- 任务特定指标
- 生成质量指标

## 实验记录

（在此记录实验过程中的观察和问题）
