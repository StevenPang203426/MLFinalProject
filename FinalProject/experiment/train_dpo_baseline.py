"""
RLHF Stage 3: DPO Baseline (Using SFT model but Unoptimized params)
用于 Task 5 的对比实验：虽然使用 SFT 模型，但不包含正则化超参数优化
"""
import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ========== Paths ==========
# 同样使用 SFT 后的模型
sft_model_path = ""
# 修复 Tokenizer 报错：直接指向原始模型目录获取词表
original_model_dir = ""
output_dir = ""
preference_data_path = ""

wandb.init(project="RLHF-DPO", name="qwen2-1.5b-dpo-baseline")
 
# 1. 修复加载报错：从原始目录加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载 SFT 合并后的模型
print("Loading SFT-Merged Model for Baseline...")
model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. LoRA 配置 (基线组可以使用相同的 LoRA 结构)
dpo_lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 4. 加载偏好数据
dataset = load_dataset("parquet", data_files=preference_data_path, split="train")

# ========== 5. 基线组配置 (关键：取消正则化和长度约束) ==========
dpo_config = DPOConfig(
    output_dir=output_dir,
    beta=0.01,                # 优化前：极小的 beta。模型几乎不受参考模型约束，极易发生训练崩溃或作弊
    learning_rate=1e-5,       # 优化前：较高的学习率
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=2048,          # 优化前：放宽长度限制。模型会发现“写得越长分越高”
    max_prompt_length=512,
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="wandb",
    logging_steps=1
)

# 6. 初始化训练器
trainer = DPOTrainer(
    model=model,
    ref_model=None,           # 自动处理参考模型
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=dpo_lora_config,
)

# 7. 运行
print("Starting Baseline DPO training (No Regularization)...")
trainer.train()
trainer.save_model(output_dir)
wandb.finish()