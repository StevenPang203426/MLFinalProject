"""
基线实验训练脚本
使用 TRL 或类似框架进行 RLHF 微调
"""

import argparse
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

def parse_args():
    parser = argparse.ArgumentParser(description="基线实验训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-1_8B", help="模型名称")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(project="rl-reward-design", name="baseline")
    
    # 加载模型和分词器
    print(f"加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 配置 PPO
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=1.41e-5,
        batch_size=128,
        mini_batch_size=128,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=0,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
    )
    
    # TODO: 实现训练逻辑
    # 1. 准备数据集
    # 2. 初始化 PPO Trainer
    # 3. 训练循环
    # 4. 记录指标
    
    print("训练完成！")

if __name__ == "__main__":
    main()
