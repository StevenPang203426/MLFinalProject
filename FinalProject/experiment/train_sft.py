"""
RLHF 第一阶段：SFT（监督微调）- 源码匹配版
"""

import os
# 显存优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import wandb
import torch

# ========== wandb 配置 ==========
wandb.login(key='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')
wandb.init(project="RLHF-SFT", name="qwen2-1.5b-sft-lora")

# ========== 路径配置 ==========
model_path = ""
output_dir = ""
data_path = ""

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

# 3. LoRA 配置 (显存优化的核心)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. 数据预处理
print("正在处理数据...")
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        prompt = item.get("prompt", "")
        reference = item.get("reference", "")
        if prompt and reference:
            # 拼接文本
            full_text = f"{prompt}{reference}{tokenizer.eos_token}"
            data.append({"content": full_text}) # 使用 content 键名
dataset = Dataset.from_list(data)

# 5. 格式化函数 (匹配源码中的 formatting_func)
def formatting_prompts_func(example):
    # 根据文档，该函数作用于单个 sample 字典并返回字符串
    return example["content"]

# 6. SFT 配置
# 注意：如果 SFTConfig 依然报错，可以将 SFTConfig 换成 transformers 的 TrainingArguments
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,           
    per_device_train_batch_size=4, # 从 1 增加到 4，速度提升 4 倍
    gradient_accumulation_steps=4, # 保持总 Batch 为 16 (4*4)
    learning_rate=1e-4,
    save_strategy="steps",        
    save_steps=500,
    max_length=1024,      
    gradient_checkpointing=True,   # 建议先保持 True，防止 Batch=4 时 OOM
    fp16=True,
    optim="paged_adamw_32bit",     # 必须保留这个，它是防止 OOM 的最后防线
    report_to="wandb",
    logging_steps=1,
)

# 7. 初始化训练器 (严格匹配你提供的源码参数)
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer, # 源码中使用的是 processing_class 而不是 tokenizer
    peft_config=lora_config,    # 直接在初始化时传入 PEFT 配置，更简洁
    formatting_func=formatting_prompts_func,
)

# 8. 开始训练
print("开始训练...")
try:
    sft_trainer.train()
    sft_trainer.save_model(output_dir)
    print(f"模型已保存到: {output_dir}")
except Exception as e:
    print(f"训练失败: {e}")
    raise e
finally:
    wandb.finish()