"""
奖励塑形（Reward Shaping）
设计中间步骤的辅助奖励，提供更细粒度的反馈信号
"""

import torch
import torch.nn as nn
from typing import List, Dict

class ShapedRewardModel(nn.Module):
    """
    奖励塑形模型
    为中间步骤提供辅助奖励，解决稀疏奖励问题
    """
    
    def __init__(self, base_reward_model):
        super().__init__()
        self.base_reward_model = base_reward_model
    
    def compute_relevance_reward(self, sequences: List[str], inputs: List[str]) -> torch.Tensor:
        """
        计算相关性奖励（中间奖励）
        评估生成内容与输入的相关性
        """
        # TODO: 实现相关性评估
        # 例如：使用语义相似度、关键词匹配等
        rewards = []
        for seq, inp in zip(sequences, inputs):
            # 简单的关键词匹配示例
            input_words = set(inp.lower().split())
            seq_words = set(seq.lower().split())
            overlap = len(input_words & seq_words) / max(len(input_words), 1)
            rewards.append(overlap)
        return torch.tensor(rewards)
    
    def compute_conciseness_reward(self, sequences: List[str]) -> torch.Tensor:
        """
        计算简洁性奖励（中间奖励）
        评估生成内容的简洁程度
        """
        # TODO: 实现简洁性评估
        # 例如：信息密度、冗余度等
        rewards = []
        for seq in sequences:
            words = seq.split()
            unique_words = len(set(words))
            # 简洁性 = 唯一词数 / 总词数
            conciseness = unique_words / max(len(words), 1)
            rewards.append(conciseness)
        return torch.tensor(rewards)
    
    def forward(self, sequences: List[str], inputs: List[str] = None) -> torch.Tensor:
        """
        计算塑形后的奖励
        """
        # 最终奖励（稀疏奖励）
        final_rewards = self.base_reward_model(sequences)
        
        # 中间奖励（辅助奖励）
        if inputs is not None:
            relevance_rewards = self.compute_relevance_reward(sequences, inputs)
            conciseness_rewards = self.compute_conciseness_reward(sequences)
            
            # 组合奖励（可以设置权重）
            shaped_rewards = (
                final_rewards * 0.7 +  # 最终奖励权重
                relevance_rewards * 0.2 +  # 相关性奖励权重
                conciseness_rewards * 0.1  # 简洁性奖励权重
            )
        else:
            shaped_rewards = final_rewards
        
        return shaped_rewards
