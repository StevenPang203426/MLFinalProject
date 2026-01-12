"""
RLHF Stage 3: Direct Preference Optimization (DPO)
Refined for 24GB VRAM with Tokenizer Fix
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
sft_model_path = ""
original_model_path = ""
output_dir = ""
preference_data_path = ""

wandb.init(project="RLHF-DPO", name="qwen2-1.5b-dpo-final")

# 1. Load Tokenizer from ORIGINAL path to avoid NoneType/Vocab errors
tokenizer = AutoTokenizer.from_pretrained(original_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Merged SFT Model
print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. LoRA Configuration (Saves VRAM)
dpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 4. Load Data
dataset = load_dataset("parquet", data_files=preference_data_path, split="train")

# 5. DPO Config (Includes Task 4 improvements: Regularization via Beta)
dpo_config = DPOConfig(
    output_dir=output_dir,
    beta=0.1,                  # Task 4: Regularization - prevents reward hacking
    learning_rate=5e-7,        # Stable LR for DPO
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=512,
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="wandb",
    logging_steps=1
)

# 6. Initialize Trainer
# ref_model=None + LoRA automatically handles the reference model logic efficiently
trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=dpo_lora_config,
)

# 7. Execute
print("Starting DPO training...")
trainer.train()
trainer.save_model(output_dir)
wandb.finish()