# 奖励模型实现详解

本文档详细说明基于 TRL (Transformers Reinforcement Learning) 框架的奖励模型实现原理、配置和使用方法。

## 📋 目录

1. [奖励模型概述](#1-奖励模型概述)
2. [模型架构与初始化](#2-模型架构与初始化)
3. [奖励计算方式](#3-奖励计算方式)
4. [数据预处理流程](#4-数据预处理流程)
5. [训练配置与优化](#5-训练配置与优化)
6. [奖励模型在 RLHF 中的作用](#6-奖励模型在-rlhf-中的作用)
7. [实际项目实现](#7-实际项目实现)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 奖励模型概述

### 1.1 什么是奖励模型？

奖励模型（Reward Model）是 RLHF 流程中的核心组件，用于评估生成文本的质量。它学习一个函数 `r(x)`，能够对响应 `x` 给出标量奖励值，表示该响应的质量分数。

### 1.2 奖励模型在 RLHF 中的位置

```
第一阶段：SFT（监督微调）
    ↓
第二阶段：奖励模型训练 ← 当前阶段
    ↓
第三阶段：DPO/PPO（策略优化）
```

**关键点**：
- 奖励模型基于 SFT 后的模型进行训练
- 在 DPO 中，奖励模型**不直接参与**策略更新（DPO 使用隐式奖励）
- 在 PPO 中，奖励模型**直接参与**策略更新（用于评估生成响应）

---

## 2. 模型架构与初始化

### 2.1 架构转换

奖励模型通过将因果语言模型（CausalLM）转换为序列分类模型（SequenceClassification）来实现：

```python
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. 首先需要合并 SFT 阶段的 LoRA 权重
sft_lora_path = "qwen2-1.5b-sft-lora-output"
original_model_path = "../../../Qwen2-1.5B"
merged_sft_model_path = "qwen2-1.5b-sft-merged"

# 合并 SFT 权重
if not os.path.exists(merged_sft_model_path):
    temp_base = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    temp_model = PeftModel.from_pretrained(temp_base, sft_lora_path)
    merged_model = temp_model.merge_and_unload()
    merged_model.save_pretrained(merged_sft_model_path)

# 2. 转换为序列分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    merged_sft_model_path,
    num_labels=1,              # 输出单个奖励值（标量）
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)
```

**关键点**：
- `num_labels=1`：输出单个标量奖励值
- 使用 `AutoModelForSequenceClassification` 而不是 `AutoModelForCausalLM`
- 必须基于合并后的 SFT 模型，不能直接使用 LoRA 权重

### 2.2 LoRA 配置

奖励模型使用独立的 LoRA 配置，专门针对序列分类任务：

```python
rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.SEQ_CLS,      # 序列分类任务类型
    modules_to_save=["score"],        # 重要：保存新初始化的分类头
)

model = get_peft_model(model, rm_lora_config)
```

**关键配置说明**：
- `task_type=TaskType.SEQ_CLS`：指定为序列分类任务
- `modules_to_save=["score"]`：**必须包含**，因为分类头是新初始化的，需要完整保存

---

## 3. 奖励计算方式

### 3.1 奖励值特性

**重要**：TRL 的奖励模型**不进行归一化**到 [0, 1] 范围：
- 奖励值可以是任意实数（正数、负数、零）
- 模型学习的是**相对顺序**（chosen > rejected），而不是绝对分数
- 这允许奖励值有更大的动态范围

### 3.2 损失函数：Bradley-Terry 模型

奖励模型使用 Bradley-Terry 损失函数，最大化 `P(chosen > rejected)`：

**数学公式**：
```
L = -log(σ(r_chosen - r_rejected))
```

其中：
- `σ(x) = 1 / (1 + exp(-x))` 是 sigmoid 函数
- `r_chosen` 是 chosen 响应的奖励值
- `r_rejected` 是 rejected 响应的奖励值

**代码实现**（TRL 内部）：
```python
# 无 margin 的情况
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

# 有 margin 的情况（可选）
if "margin" in inputs:
    loss = -nn.functional.logsigmoid(
        rewards_chosen - rewards_rejected - inputs["margin"]
    ).mean()
```

**含义**：
- 使用 log-sigmoid 确保数值稳定性
- margin 参数允许设置奖励差异的最小阈值
- 损失越小，说明模型越能区分 chosen 和 rejected

### 3.3 奖励中心化正则化（可选）

TRL 提供了可选的奖励中心化策略，防止奖励值漂移：

```python
# 在 RewardConfig 中设置
center_rewards_coefficient=0.01  # 可选，默认 None

# 损失函数变为
L_total = L_Bradley-Terry + λ * E[(r_chosen + r_rejected)²]
```

其中 `λ = center_rewards_coefficient`。

**作用**：
- **防止奖励值漂移**：鼓励奖励值接近 0，避免奖励值无限增长
- **正则化效果**：通过惩罚大的奖励值，使模型更稳定
- **推荐值**：`0.01` - `0.1`（如果启用）

---

## 4. 数据预处理流程

### 4.1 数据格式要求

奖励模型需要偏好对数据，格式为：
- `prompt`: 输入提示
- `chosen`: 更好的响应
- `rejected`: 较差的响应

### 4.2 预处理函数

```python
from datasets import load_dataset

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(
        examples["prompt"], 
        examples["chosen"], 
        examples["rejected"]
    ):
        # 拼接 prompt 和响应
        c_tokens = tokenizer(
            f"{prompt} {chosen}", 
            max_length=1024, 
            truncation=True
        )
        r_tokens = tokenizer(
            f"{prompt} {rejected}", 
            max_length=1024, 
            truncation=True
        )
        new_examples["input_ids_chosen"].append(c_tokens["input_ids"])
        new_examples["attention_mask_chosen"].append(c_tokens["attention_mask"])
        new_examples["input_ids_rejected"].append(r_tokens["input_ids"])
        new_examples["attention_mask_rejected"].append(r_tokens["attention_mask"])
    return new_examples

# 加载和预处理数据
dataset = load_dataset("parquet", data_files=preference_data_path, split="train")
train_dataset = dataset.map(preprocess_function, batched=True)
```

### 4.3 DataCollatorForPreference

TRL 的 `RewardTrainer` 内部使用 `DataCollatorForPreference`：

**数据组织方式**：
- 将 chosen 和 rejected 响应拼接成一个批次
- 批次的前半部分是 `chosen_input_ids`
- 批次的后半部分是 `rejected_input_ids`
- 使用动态 padding，每个批次只 padding 到该批次的最大长度

**关键点**：
- 模型对每个响应输出一个标量奖励值
- 前向传播后，通过 `torch.chunk(outputs.logits, chunks=2)` 分离 chosen 和 rejected 的奖励值

---

## 5. 训练配置与优化

### 5.1 RewardConfig 关键参数

```python
from trl import RewardConfig

reward_config = RewardConfig(
    # 基础训练参数
    output_dir="qwen2-1.5b-reward-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # 有效 batch size = 4 × 4 = 16
    learning_rate=5e-5,
    
    # 奖励模型特定参数
    center_rewards_coefficient=None,   # 奖励中心化系数（可选）
    max_length=1024,                   # 最大序列长度
    remove_unused_columns=False,       # 保留原始列以便调试
    
    # 显存优化
    bf16=True,                         # 使用 bf16 避免精度溢出
    gradient_checkpointing=True,       # 梯度检查点
    optim="paged_adamw_32bit",        # 分页优化器
    
    # 日志和监控
    logging_steps=1,
    report_to="wandb",
)
```

### 5.2 显存优化策略

**本项目使用的优化策略**：

1. **混合精度训练**：`bf16=True`
   - 使用 bfloat16 精度，避免 FP16 的精度溢出问题
   - 相比 FP32，显存占用减少约 50%

2. **梯度检查点**：`gradient_checkpointing=True`
   - 以计算时间换取显存空间
   - 显存占用减少约 30-40%

3. **分页优化器**：`optim="paged_adamw_32bit"`
   - 使用 32-bit 分页 AdamW 优化器
   - 显存占用进一步减少

4. **LoRA 参数高效微调**：
   - 只训练少量参数（约 1-2%），大幅减少显存占用

### 5.3 评估指标

训练过程中会自动记录以下指标：

- **loss**: Bradley-Terry 损失值
- **min_reward**: 最小奖励值
- **mean_reward**: 平均奖励值
- **max_reward**: 最大奖励值
- **accuracy**: `P(r_chosen > r_rejected)`，即正确预测偏好的比例
- **margin**: `E[r_chosen - r_rejected]`，平均奖励差异

这些指标用于监控训练过程，帮助判断模型是否收敛。

---

## 6. 奖励模型在 RLHF 中的作用

### 6.1 在 DPO 中的使用

**重要**：在 DPO（直接偏好优化）阶段，奖励模型**不直接使用**：

- **DPO 直接优化策略**：不需要显式的奖励模型
- **使用偏好数据**：DPO 使用相同的 `(prompt, chosen, rejected)` 数据
- **隐式奖励**：DPO 通过最大化 chosen 的似然、最小化 rejected 的似然来学习偏好

**DPO 损失函数**：
```
L_DPO = -log(σ(β * (log π_θ(chosen|prompt) - log π_ref(chosen|prompt) 
                    - log π_θ(rejected|prompt) + log π_ref(rejected|prompt))))
```

其中：
- `β`: 温度参数（正则化强度）
- `π_θ`: 当前策略模型
- `π_ref`: 参考模型（通常是 SFT 模型）

**为什么 DPO 不需要奖励模型？**
- DPO 直接从偏好数据中学习，不需要显式的奖励信号
- 通过对比 chosen 和 rejected 的似然差异来学习偏好
- 这简化了训练流程，避免了奖励模型的训练和校准

### 6.2 在 PPO 中的使用

如果使用 PPO（而不是 DPO），奖励模型会**直接参与策略更新**：

1. **生成阶段**：策略模型生成响应
2. **评估阶段**：奖励模型对生成的响应打分
3. **策略更新**：使用 PPO 算法，基于奖励值更新策略

**PPO 更新公式**：
```
L_PPO = E[min(
    r(θ) * A, 
    clip(r(θ), 1-ε, 1+ε) * A
)]
```

其中：
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` 是重要性采样比率
- `A` 是优势函数，通常基于奖励值计算：`A = R - V(s)`
- `R` 是奖励模型给出的奖励值

---

## 7. 实际项目实现

### 7.1 完整训练流程

```python
"""
RLHF 第二阶段：奖励模型训练
"""
import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 显存优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ========== 路径配置 ==========
sft_lora_path = "qwen2-1.5b-sft-lora-output" 
original_model_path = "../../../Qwen2-1.5B"
merged_sft_model_path = "qwen2-1.5b-sft-merged"
output_dir = "qwen2-1.5b-reward-output"
preference_data_path = "../../data/train-00000-of-00001-3cbd295cedeecf91.parquet"

# ========== 1. 权重合并 ==========
if not os.path.exists(merged_sft_model_path):
    print("正在合并 SFT 权重...")
    temp_base = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    temp_model = PeftModel.from_pretrained(temp_base, sft_lora_path)
    merged_model = temp_model.merge_and_unload()
    merged_model.save_pretrained(merged_sft_model_path)
    del temp_base, temp_model, merged_model
    torch.cuda.empty_cache()

# ========== 2. 奖励模型初始化 ==========
wandb.init(project="RLHF-Reward", name="qwen2-1.5b-rm-lora-final")

tokenizer = AutoTokenizer.from_pretrained(original_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    merged_sft_model_path,
    num_labels=1,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id

# ========== 3. LoRA 配置 ==========
rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"],  # 重要：保存分类头
)
model = get_peft_model(model, rm_lora_config)

# ========== 4. 数据预处理 ==========
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        c_tokens = tokenizer(f"{prompt} {chosen}", max_length=1024, truncation=True)
        r_tokens = tokenizer(f"{prompt} {rejected}", max_length=1024, truncation=True)
        new_examples["input_ids_chosen"].append(c_tokens["input_ids"])
        new_examples["attention_mask_chosen"].append(c_tokens["attention_mask"])
        new_examples["input_ids_rejected"].append(r_tokens["input_ids"])
        new_examples["attention_mask_rejected"].append(r_tokens["attention_mask"])
    return new_examples

dataset = load_dataset("parquet", data_files=preference_data_path, split="train")
train_dataset = dataset.map(preprocess_function, batched=True)

# ========== 5. 训练配置 ==========
reward_config = RewardConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    remove_unused_columns=False,
    max_length=1024,
    logging_steps=1,
)

# ========== 6. 初始化训练器 ==========
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

# ========== 7. 开始训练 ==========
print("开始训练奖励模型...")
trainer.train()
trainer.save_model(output_dir)
wandb.finish()
```

### 7.2 项目配置总结

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank | 16 | LoRA 的秩 |
| LoRA alpha | 32 | LoRA 的缩放因子 |
| Batch size | 4 × 4 = 16 | per_device × gradient_accumulation |
| 学习率 | 5e-5 | 奖励模型训练的学习率 |
| 最大长度 | 1024 | 序列最大长度 |
| 精度 | bf16 | 混合精度训练 |
| 优化器 | paged_adamw_32bit | 分页优化器 |

---

## 8. 常见问题与解决方案

### 8.1 为什么需要合并 SFT 权重？

**问题**：为什么不能直接使用 SFT 的 LoRA 权重？

**解答**：
- `AutoModelForSequenceClassification` 需要完整的模型权重
- LoRA 权重只是适配器，需要与基础模型合并才能使用
- 合并后的模型可以独立加载，不需要原始模型

### 8.2 为什么需要 `modules_to_save=["score"]`？

**问题**：为什么 LoRA 配置中必须包含 `modules_to_save=["score"]`？

**解答**：
- `score` 是分类头，是新初始化的模块
- LoRA 只训练适配器参数，不训练分类头
- `modules_to_save` 确保分类头被完整保存，而不是只保存适配器

### 8.3 奖励值范围是多少？

**问题**：奖励值应该在什么范围内？

**解答**：
- **没有固定范围**：奖励值可以是任意实数
- 模型学习的是相对顺序，不是绝对分数
- 训练过程中奖励值可能会漂移，这是正常的
- 如果担心漂移，可以使用 `center_rewards_coefficient` 进行正则化

### 8.4 DPO 是否需要奖励模型？

**问题**：DPO 训练是否需要先训练奖励模型？

**解答**：
- **不需要**：DPO 直接从偏好数据中学习，不需要显式的奖励模型
- 奖励模型主要用于 PPO 或其他需要显式奖励信号的方法
- 本项目中的奖励模型训练是为了完整展示 RLHF 流程

### 8.5 如何判断奖励模型训练是否成功？

**指标**：
- **accuracy**：应该逐渐接近 1.0（chosen 奖励 > rejected 奖励的比例）
- **margin**：应该逐渐增大（chosen 和 rejected 的平均差异）
- **loss**：应该逐渐减小并收敛

**典型值**：
- accuracy > 0.9：模型能够较好地区分 chosen 和 rejected
- margin > 0.5：chosen 和 rejected 有足够的奖励差异

---

## 9. 总结

### 奖励计算方式
- ✅ 使用回归模型输出标量奖励值
- ✅ Bradley-Terry 损失函数（log-sigmoid）
- ✅ 支持可选的 margin 参数
- ❌ **不进行归一化**到 [0, 1]

### 奖励归一化策略
- ✅ 可选的奖励中心化正则化（`center_rewards_coefficient`）
- ✅ 记录奖励统计信息用于监控
- ❌ 不强制归一化到固定范围

### 奖励与策略更新的交互
- **DPO**：不使用奖励模型，直接优化策略
- **PPO**：使用奖励模型评估生成响应，指导策略更新
- 奖励模型主要用于评估和监控，而不是直接参与策略梯度计算（在 DPO 中）

---

## 📚 参考资料

- **TRL 源代码**：`/usr/local/lib/python3.10/dist-packages/trl/trainer/reward_trainer.py`
- **Bradley-Terry 模型**：用于成对比较的概率模型
- **DPO 论文**：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **PPO 论文**：Proximal Policy Optimization Algorithms
- **RLHF 综述**：Training Language Models to Follow Instructions with Human Feedback

---

## 🔗 相关文档

- [RLHF 三阶段训练 README](../README.md)：完整的训练流程说明
- [DPO 训练脚本](../train_dpo_regularize.py)：正则化版本的 DPO 训练
- [DPO 多奖励平衡版本](../train_dpo_multireward.py)：改进版的 DPO 训练
