"""
RLHF 第二阶段：奖励模型 (Reward Model) 训练 - 彻底修复加载冲突版
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
sft_lora_path = ""
original_model_path = ""
merged_sft_model_path = ""
output_dir = ""
preference_data_path = ""

# ========== 1. 权重合并 (解决 AttributeError) ==========
if not os.path.exists(merged_sft_model_path):
    print("正在合并 SFT 权重以准备奖励模型训练...")
    # 以 CausalLM 加载以匹配 SFT 的适配器结构
    temp_base = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    # 挂载 SFT LoRA
    temp_model = PeftModel.from_pretrained(temp_base, sft_lora_path)
    # 合并并保存
    merged_model = temp_model.merge_and_unload()
    merged_model.save_pretrained(merged_sft_model_path)
    # 释放显存
    del temp_base, temp_model, merged_model
    torch.cuda.empty_cache()
    print(f"SFT 权重已成功合并至: {merged_sft_model_path}")

# ========== 2. 奖励模型初始化 ==========
wandb.login(key='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')
wandb.init(project="RLHF-Reward", name="qwen2-1.5b-rm-lora-final")

tokenizer = AutoTokenizer.from_pretrained(original_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载合并后的 SFT 模型作为奖励模型底座...")
# 现在作为分类模型加载，不会再报 prepare_inputs_for_generation 错误
model = AutoModelForSequenceClassification.from_pretrained(
    merged_sft_model_path,
    num_labels=1,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id

# 3. RM 专属 LoRA 配置 (Task 4: 改进奖励实现)
rm_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.SEQ_CLS, 
    modules_to_save=["score"], # 重要：保存新初始化的分类头
)
model = get_peft_model(model, rm_lora_config)

# 4. 数据预处理
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # 奖励模型学习人类偏好： chosen 应该比 rejected 得分更高
        c_tokens = tokenizer(f"{prompt} {chosen}", max_length=1024, truncation=True)
        r_tokens = tokenizer(f"{prompt} {rejected}", max_length=1024, truncation=True)
        new_examples["input_ids_chosen"].append(c_tokens["input_ids"])
        new_examples["attention_mask_chosen"].append(c_tokens["attention_mask"])
        new_examples["input_ids_rejected"].append(r_tokens["input_ids"])
        new_examples["attention_mask_rejected"].append(r_tokens["attention_mask"])
    return new_examples

dataset = load_dataset("parquet", data_files=preference_data_path, split="train")
train_dataset = dataset.map(preprocess_function, batched=True)

# 5. Reward 配置 (Task 3: 奖励机制分析 - 使用混合精度提升性能)
reward_config = RewardConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True, # 使用 bf16 避免精度溢出
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    remove_unused_columns=False,
    max_length=1024,
    logging_steps=1,
)

# 6. 初始化训练器
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    processing_class=tokenizer, 
    train_dataset=train_dataset,
)

# 7. 开始训练
print("开始训练奖励模型...")
try:
    trainer.train()
    trainer.save_model(output_dir)
    print(f"奖励模型训练完成，保存至: {output_dir}")
except Exception as e:
    print(f"训练失败: {e}")
    raise e
finally:
    wandb.finish()