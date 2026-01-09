"""
基线实验训练脚本
使用 TRL 框架进行 RLHF 微调（PPO）
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
import wandb

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.data_utils import load_dataset, prepare_ppo_dataset


class SummaryDataset(Dataset):
    """摘要数据集"""
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt", "")
        reference = item.get("reference", "")
        
        # 编码 prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": prompt_tokens["input_ids"].squeeze(),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(),
            "prompt": prompt,
            "reference": reference
        }


class RewardModel:
    """简单的奖励模型（基于参考摘要的相似度）"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, prompts: List[str], summaries: List[str], references: List[str] = None) -> torch.Tensor:
        """
        计算奖励
        
        Args:
            prompts: 提示词列表
            summaries: 生成的摘要列表
            references: 参考摘要列表（可选）
        
        Returns:
            奖励张量
        """
        rewards = []
        for summary, reference in zip(summaries, references):
            if reference:
                # 简单的基于长度的奖励（可以替换为更复杂的奖励函数）
                # 这里使用简单的长度匹配和内容相似度
                summary_tokens = len(self.tokenizer.encode(summary))
                reference_tokens = len(self.tokenizer.encode(reference))
                
                # 长度奖励（鼓励生成接近参考长度的摘要）
                length_reward = 1.0 - abs(summary_tokens - reference_tokens) / max(reference_tokens, 1)
                length_reward = max(0.0, length_reward)
                
                # 简单的关键词匹配奖励（可以替换为更复杂的相似度计算）
                summary_words = set(summary.lower().split())
                reference_words = set(reference.lower().split())
                if reference_words:
                    overlap = len(summary_words & reference_words) / len(reference_words)
                else:
                    overlap = 0.0
                
                # 综合奖励
                reward = 0.5 * length_reward + 0.5 * overlap
                rewards.append(reward)
            else:
                # 如果没有参考，使用默认奖励
                rewards.append(0.5)
        
        return torch.tensor(rewards, dtype=torch.float32)


def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径（可以是相对路径或绝对路径）
    
    Returns:
        配置字典
    """
    # 如果路径是相对路径，尝试从项目根目录或脚本所在目录查找
    if not os.path.isabs(config_path):
        # 获取项目根目录（FinalProject）
        script_dir = Path(__file__).parent  # experiments/baseline
        project_root = script_dir.parent.parent  # FinalProject
        
        # 尝试从项目根目录查找配置文件
        config_file = project_root / "configs" / config_path
        if not config_file.exists():
            # 如果不存在，尝试直接使用提供的路径
            config_file = Path(config_path)
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="基线实验训练")
    parser.add_argument(
        "--config", 
        type=str, 
        default="baseline_config.yaml", 
        help="配置文件路径（相对于 configs/ 目录）"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None, 
        help="模型名称或路径（如果提供，将覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="输出目录（如果提供，将覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true", 
        help="是否使用 wandb（如果提供，将覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="训练数据路径（如果提供，将覆盖配置文件中的设置）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置文件
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 从配置文件获取参数（命令行参数优先）
    model_name = args.model_name or config['model']['name']
    output_dir = args.output_dir or config['training']['output_dir']
    use_wandb = args.use_wandb or config.get('wandb', {}).get('enabled', False)
    train_data_path = args.train_data or config['data'].get('train_path', 'data/train_converted.jsonl')
    task_type = config['data'].get('task_type', 'summarization')
    
    # 处理模型路径：如果是相对路径，转换为绝对路径
    if not os.path.isabs(model_name) and not model_name.startswith(('http://', 'https://')):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        model_path = project_root / model_name
        if model_path.exists():
            model_name = str(model_path.resolve())
            print(f"使用本地模型路径: {model_name}")
    
    # 处理数据路径
    if not os.path.isabs(train_data_path):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        train_data_path = project_root / train_data_path
        train_data_path = str(train_data_path.resolve())
    
    # 处理输出目录
    if not os.path.isabs(output_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / output_dir
        output_dir = str(output_dir.resolve())
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 wandb
    if use_wandb:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'rl-reward-design'),
            name=wandb_config.get('name', 'baseline'),
            tags=wandb_config.get('tags', [])
        )
    
    # 加载模型和分词器
    print(f"\n加载模型: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        sys.exit(1)
    
    # 加载训练数据
    print(f"\n加载训练数据: {train_data_path}")
    train_data = load_dataset(train_data_path)
    print(f"✓ 成功加载 {len(train_data)} 个样本")
    
    # 准备 PPO 数据集
    print("准备 PPO 数据集...")
    ppo_data = prepare_ppo_dataset(train_data, task_type=task_type)
    print(f"✓ 数据集准备完成，共 {len(ppo_data)} 个样本")
    
    # 创建数据集
    dataset = SummaryDataset(ppo_data, tokenizer, max_length=512)
    
    # 初始化奖励模型
    reward_model = RewardModel(tokenizer)
    
    # 从配置文件获取 PPO 配置
    ppo_cfg = config.get('ppo', {})
    training_cfg = config.get('training', {})
    
    # 配置 PPO
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=training_cfg.get('learning_rate', 1.41e-5),
        batch_size=training_cfg.get('batch_size', 128),
        mini_batch_size=training_cfg.get('mini_batch_size', 128),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=ppo_cfg.get('target_kl', 0.1),
        ppo_epochs=ppo_cfg.get('ppo_epochs', 4),
        seed=training_cfg.get('seed', 42),
        init_kl_coef=ppo_cfg.get('init_kl_coef', 0.2),
        adap_kl_ctrl=ppo_cfg.get('adap_kl_ctrl', True),
        cliprange=ppo_cfg.get('cliprange', 0.2),
        cliprange_value=ppo_cfg.get('cliprange_value', 0.2),
        gamma=ppo_cfg.get('gamma', 1.0),
        lam=ppo_cfg.get('lam', 0.95),
    )
    
    print(f"\nPPO 配置:")
    print(f"  - 学习率: {ppo_config.learning_rate}")
    print(f"  - 批次大小: {ppo_config.batch_size}")
    print(f"  - PPO epochs: {ppo_config.ppo_epochs}")
    print(f"  - 目标 KL: {ppo_config.target_kl}")
    print(f"  - 输出目录: {output_dir}")
    
    # 创建参考模型（用于 KL 散度计算）
    print("\n创建参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # 初始化 PPO Trainer
    print("\n初始化 PPO Trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    
    num_epochs = training_cfg.get('num_epochs', 3)
    batch_size = ppo_config.batch_size
    
    # 准备生成参数
    generation_kwargs = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 分批处理数据
        for batch_idx, batch_start in enumerate(range(0, len(dataset), batch_size)):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = [dataset[i] for i in range(batch_start, batch_end)]
            
            # 提取 prompts 和 references
            queries = [item["prompt"] for item in batch]
            references = [item.get("reference", "") for item in batch]
            
            # 编码 queries
            query_tensors = []
            for query in queries:
                query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
                query_tensors.append(query_tokens["input_ids"].squeeze())
            
            # 生成摘要
            model.eval()
            with torch.no_grad():
                responses = []
                response_tensors = []
                
                for query_tensor in query_tensors:
                    query_tensor = query_tensor.unsqueeze(0).to(device)
                    response = ppo_trainer.generate(
                        query_tensor,
                        return_prompt=False,
                        **generation_kwargs
                    )
                    response_tensors.append(response.squeeze())
                    # 解码响应
                    response_text = tokenizer.decode(response.squeeze(), skip_special_tokens=True)
                    responses.append(response_text)
            
            # 计算奖励
            rewards = reward_model(queries, responses, references)
            
            # PPO 训练步骤
            try:
                stats = ppo_trainer.step(
                    queries=query_tensors,
                    responses=response_tensors,
                    scores=rewards.tolist()
                )
                
                if use_wandb and stats:
                    wandb.log({**stats, "epoch": epoch + 1, "batch": batch_idx + 1})
                
                if (batch_idx + 1) % training_cfg.get('logging_steps', 10) == 0:
                    print(f"  Batch {batch_idx + 1}: Reward={rewards.mean().item():.4f}, {stats}")
            except Exception as e:
                print(f"  PPO 训练步骤失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存检查点
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ 保存检查点到: {checkpoint_dir}")
    
    # 保存最终模型
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\n✓ 训练完成！最终模型已保存到: {final_model_dir}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
