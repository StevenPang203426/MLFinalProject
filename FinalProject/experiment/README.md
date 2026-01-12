# RLHF 三阶段训练实验

本项目实现了完整的 RLHF（Reinforcement Learning from Human Feedback）三阶段训练流程，基于 Qwen2-1.5B 模型进行文本摘要任务的强化学习优化。

## 📋 目录结构

```
rlhf/
├── train_sft.py              # 第一阶段：监督微调 (SFT)
├── train_reward.py           # 第二阶段：奖励模型训练
├── train_dpo_baseline.py     # 第三阶段：DPO 训练（基线版，用于对比）
├── train_dpo_regularize.py   # 第三阶段：DPO 训练（改进版，正则化）
├── train_dpo_multireward.py  # 第三阶段：DPO 训练（改进版，正则化与多奖励平衡）
├── README.md                 # 本文件
└── REWARD_MODEL_IMPLEMENTATION.md  # 奖励模型实现详解
```

## 🔧 环境配置

### 路径配置

- **基础模型路径**: `../Qwen2-1.5B`（相对于当前目录）
- **数据路径**: `../data`（相对于当前目录）
  - SFT 数据: `train_converted.jsonl`
  - 偏好数据: `train-00000-of-00001-3cbd295cedeecf91.parquet`

数据来自 `https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons`

### 依赖库

- `transformers`: Hugging Face Transformers 库
- `trl`: Transformers Reinforcement Learning 库
- `peft`: Parameter-Efficient Fine-Tuning 库（LoRA）
- `datasets`: Hugging Face Datasets 库
- `wandb`: Weights & Biases（用于实验跟踪）

## 🚀 三阶段训练流程

### 第一阶段：监督微调 (SFT)

**脚本**: `train_sft.py`

**功能**: 使用监督学习对基础模型进行微调，学习文本摘要任务的基本能力。

**特点**:
- 使用 LoRA（Low-Rank Adaptation）进行参数高效微调
- LoRA rank=16, alpha=32
- 支持梯度检查点和混合精度训练（FP16）
- 使用 `paged_adamw_32bit` 优化器以节省显存

**使用方法**:
```bash
python3 train_sft.py
```

**输出目录**: `qwen2-1.5b-sft-lora-output`

**配置参数**:
- 学习率: `1e-4`
- Batch size: `4` (per device) × `4` (gradient accumulation) = 16
- 最大长度: `1024`
- 训练轮数: `1`

---

### 第二阶段：奖励模型训练

**脚本**: `train_reward.py`

**功能**: 训练一个奖励模型，能够评估生成文本的质量，为后续的强化学习提供奖励信号。

**特点**:
- 自动合并 SFT 阶段的 LoRA 权重
- 将因果语言模型转换为序列分类模型（`AutoModelForSequenceClassification`）
- 使用 Bradley-Terry 损失函数学习偏好对
- 支持奖励中心化正则化（可选）

**使用方法**:
```bash
python3 train_reward.py
```

**输出目录**: `qwen2-1.5b-reward-output`

**中间文件**: `qwen2-1.5b-sft-merged`（合并后的 SFT 模型）

**配置参数**:
- 学习率: `5e-5`
- Batch size: `4` × `4` = 16
- 最大长度: `1024`
- 精度: `bf16`（避免精度溢出）
- 训练轮数: `1`

**奖励模型原理**:
- 模型输出单个标量奖励值（`num_labels=1`）
- 使用回归任务类型
- 损失函数: `L = -log(σ(r_chosen - r_rejected))`
- 奖励值可以是任意实数，不限制在 [0, 1] 范围内

---

### 第三阶段：直接偏好优化 (DPO)

本阶段提供了三个版本的 DPO 训练脚本，用于不同的实验目的：


#### 3.1 DPO 基线版本

**脚本**: `train_dpo_baseline.py`

**功能**: 用于对比实验的基线版本，不包含正则化优化。

**特点**:
- 使用极小的 beta (`0.01`)，模型几乎不受参考模型约束
- 较高的学习率 (`1e-5`)
- 放宽长度限制 (`max_length=2048`)

**使用方法**:
```bash
python3 train_dpo_baseline.py
```

**输出目录**: `qwen2-1.5b-dpo-baseline`

**用途**: 用于 Task 5 的对比实验，展示未优化参数时的训练效果。

#### 3.2 标准 DPO 训练

**脚本**: `train_dpo_regularize.py`

**功能**: 使用 DPO 算法直接优化策略模型，无需显式的奖励模型。

**特点**:
- 使用 SFT 合并后的模型作为基础
- 通过最大化 chosen 响应似然、最小化 rejected 响应似然来学习偏好
- 包含正则化参数 `beta=0.1` 防止奖励黑客问题

**使用方法**:
```bash
python3 train_dpo_regularize.py
```

**输出目录**: `qwen2-1.5b-dpo-output`

**配置参数**:
- Beta (正则化强度): `0.1`
- 学习率: `5e-7`
- Batch size: `2` × `8` = 16
- 最大长度: `1024`
- 最大 prompt 长度: `512`


#### 3.3 DPO 改进版本（多奖励平衡）

**脚本**: `train_dpo_multireward.py`

**功能**: 包含多项改进的 DPO 训练版本。

**改进点**:

1. **奖励塑形 (Reward Shaping)**
   - 相关性奖励：基于 prompt 和响应之间的词汇重叠
   - 简洁性奖励：理想长度区间（20-60 词）给予奖励，过长或过短给予惩罚

2. **多奖励平衡 (Multi-Reward Balancing)**
   - 相关性权重: `0.7`
   - 简洁性权重: `0.3`
   - 加权组合多个奖励指标

3. **Verifier（验证器）过滤**
   - 检测词汇多样性（重复率）
   - 如果 chosen 响应未通过验证而 rejected 通过，自动反转偏好
   - 基于平衡后的得分重新确定偏好

**使用方法**:
```bash
python3 train_dpo_multireward.py
```

**输出目录**: `qwen2-1.5b-dpo-refined-v2`

**配置参数**:
- Beta: `0.1`（适中的正则化强度）
- 学习率: `5e-7`（较低的学习率保持稳定）
- 最大长度: `1024`（硬性长度限制）

---

## 📊 训练流程依赖关系

```
基础模型 (Qwen2-1.5B)
    ↓
[第一阶段] SFT 训练
    ↓
qwen2-1.5b-sft-lora-output
    ↓ (合并权重)
qwen2-1.5b-sft-merged
    ↓
[第二阶段] 奖励模型训练
    ↓
qwen2-1.5b-reward-output
    ↓
[第三阶段] DPO 训练
    ↓
qwen2-1.5b-dpo-baseline (基线版)
qwen2-1.5b-dpo-output (改进版1)
qwen2-1.5b-dpo-refined-v2 (改进版2)
```

## 🔍 关键技术细节

### LoRA 配置

所有阶段均使用 LoRA 进行参数高效微调：

```python
LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,          # LoRA alpha
    target_modules=[         # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"   # 或 "SEQ_CLS" (奖励模型)
)
```

### 显存优化策略

- **梯度检查点**: `gradient_checkpointing=True`
- **混合精度训练**: `fp16=True` 或 `bf16=True`
- **分页优化器**: `optim="paged_adamw_32bit"`
- **动态显存分配**: `PYTORCH_ALLOC_CONF=expandable_segments:True`

### DPO 损失函数

DPO 使用以下损失函数：

```
L_DPO = -log(σ(β * (log π_θ(chosen|prompt) - log π_ref(chosen|prompt) 
                    - log π_θ(rejected|prompt) + log π_ref(rejected|prompt))))
```

其中：
- `β`: 温度参数（正则化强度）
- `π_θ`: 当前策略模型
- `π_ref`: 参考模型（通常是 SFT 模型）

## 📈 实验跟踪

所有训练脚本均集成了 Weights & Biases (wandb) 进行实验跟踪：

- **SFT 项目**: `RLHF-SFT`
- **奖励模型项目**: `RLHF-Reward`
- **DPO 项目**: `RLHF-DPO`

训练过程中的关键指标（损失、学习率、奖励统计等）会自动记录到 wandb。

## 📝 注意事项

1. **数据格式要求**:
   - SFT 数据: JSONL 格式，包含 `prompt` 和 `reference` 字段
   - 偏好数据: Parquet 格式，包含 `prompt`、`chosen` 和 `rejected` 字段

2. **模型路径**:
   - 确保基础模型路径正确
   - SFT 训练后会自动生成 LoRA 权重，需要合并后才能用于后续阶段

3. **显存要求**:
   - 建议至少 24GB VRAM
   - 如果显存不足，可以减小 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`

4. **训练顺序**:
   - 必须按照 SFT → 奖励模型 → DPO 的顺序进行训练
   - 奖励模型训练会自动合并 SFT 权重

## 🔗 相关文档

- [奖励模型实现详解](REWARD_MODEL_IMPLEMENTATION.md): 详细的奖励模型实现原理和配置说明

## 📚 参考资料

- TRL (Transformers Reinforcement Learning) 库文档
- DPO 论文: Direct Preference Optimization
- PPO 论文: Proximal Policy Optimization
- Bradley-Terry 模型: 用于成对比较的概率模型
