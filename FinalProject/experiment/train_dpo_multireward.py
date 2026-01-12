"""
RLHF Stage 3: Refined Direct Preference Optimization (DPO)
Improvements: Multi-Reward Balancing, Reward Shaping, and Verifier Filtering.
"""
import os
import torch
import wandb
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# ========== 1. 基础配置与环境 ==========
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
sft_model_path = ""
original_model_dir = ""
output_dir = ""
preference_data_path = ""

wandb.init(project="RLHF-DPO", name="qwen2-1.5b-dpo-refined-final")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 2. 改进逻辑：奖励塑形、权重平衡与 Verifier ==========

def apply_refinement_logic(example):
    """
    此函数在数据层面实现方案 2, 3, 4：
    - 奖励塑形：通过相关性和长度惩罚重新评估摘要
    - 多奖励平衡：加权组合多个指标
    - Verifier：剔除或修正不符合逻辑的样本
    """
    prompt = example['prompt']
    chosen = example['chosen']
    rejected = example['rejected']

    # --- [方案 2 & 3: 辅助奖励计算与权重平衡] ---
    def get_scores(text):
        # 奖励项 A: 相关性 (Relevance) - 模拟摘要内容覆盖率
        rel_score = len(set(prompt.split()) & set(text.split())) / (len(set(text.split())) + 1)
        
        # 奖励项 B: 简洁性塑形 (Conciseness Shaping) - 理想长度设为 30-60 词
        length = len(text.split())
        if length > 60:
            conc_score = -0.1 * (length - 60) # 长度惩罚
        elif length < 20:
            conc_score = -0.2 * (20 - length) # 太短也惩罚
        else:
            conc_score = 0.5 # 理想区间奖励
            
        # 方案 3: 多奖励权重平衡 (相关性 0.7 + 简洁性 0.3)
        return 0.7 * rel_score + 0.3 * conc_score

    score_c = get_scores(chosen)
    score_r = get_scores(rejected)

    # --- [方案 4: Verifier (验证器) 逻辑] ---
    # 简单的硬约束验证器：检测重复率和幻觉风险（词汇多样性低于阈值视为无效）
    def is_valid(text):
        words = text.split()
        if len(words) == 0: return False
        diversity = len(set(words)) / len(words)
        return diversity > 0.4  # 词汇太重复则验证不通过

    v_c = is_valid(chosen)
    v_r = is_valid(rejected)

    # 决策逻辑：
    # 1. 如果 Chosen 没过验证而 Rejected 过了，强行反转偏好
    if not v_c and v_r:
        return {"chosen": rejected, "rejected": chosen}
    
    # 2. 如果都通过或都没过，基于平衡后的得分重新确定偏好
    if score_r > score_c:
        return {"chosen": rejected, "rejected": chosen}
    
    return {"chosen": chosen, "rejected": rejected}

# ========== 3. 数据预处理 ==========
print("Loading and Refining dataset...")
raw_dataset = load_dataset("parquet", data_files=preference_data_path, split="train")
# 应用改进逻辑
refined_dataset = raw_dataset.map(apply_refinement_logic)

# ========== 4. 加载模型 ==========
print("Loading SFT model for DPO...")
model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置
dpo_lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# ========== 5. DPO 训练配置 (包含正则化改进) ==========
dpo_config = DPOConfig(
    output_dir=output_dir,
    beta=0.1,                  # 核心改进：适中的正则化强度，防止 Reward 坠入负值
    learning_rate=5e-7,        # 较低的学习率保持训练稳定
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=1024,           # 改进：硬性长度限制
    max_prompt_length=512,
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="wandb",
    logging_steps=1,
    remove_unused_columns=False # 保持原始列以便调试
)

# ========== 6. 启动训练 ==========
trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    args=dpo_config,
    train_dataset=refined_dataset,
    processing_class=tokenizer,
    peft_config=dpo_lora_config,
)

print("Starting Refined DPO training...")
trainer.train()

# 保存最终模型
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
wandb.finish()