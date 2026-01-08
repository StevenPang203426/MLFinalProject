"""
正则化奖励模型
添加约束防止 reward hacking（如长度惩罚、重复惩罚）
"""

import torch
import torch.nn as nn
from typing import List, Dict

class RegularizedRewardModel(nn.Module):
    """
    带正则化的奖励模型
    添加长度惩罚和重复惩罚来防止 reward hacking
    """
    
    def __init__(self, base_reward_model, length_penalty_weight=0.1, repetition_penalty_weight=0.1):
        super().__init__()
        self.base_reward_model = base_reward_model
        self.length_penalty_weight = length_penalty_weight
        self.repetition_penalty_weight = repetition_penalty_weight
    
    def compute_length_penalty(self, sequences: List[str]) -> torch.Tensor:
        """
        计算长度惩罚
        惩罚过长或过短的序列
        """
        # TODO: 实现长度惩罚逻辑
        # 例如：如果长度超出合理范围，给予负奖励
        penalties = []
        for seq in sequences:
            length = len(seq.split())
            # 假设合理长度范围是 10-100
            if length < 10:
                penalty = -0.1 * (10 - length)
            elif length > 100:
                penalty = -0.1 * (length - 100)
            else:
                penalty = 0.0
            penalties.append(penalty)
        return torch.tensor(penalties)
    
    def compute_repetition_penalty(self, sequences: List[str]) -> torch.Tensor:
        """
        计算重复惩罚
        惩罚包含重复内容的序列
        """
        # TODO: 实现重复惩罚逻辑
        # 例如：检测 n-gram 重复
        penalties = []
        for seq in sequences:
            words = seq.split()
            # 简单的重复检测：检查连续重复的单词
            penalty = 0.0
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    penalty -= 0.05
            penalties.append(penalty)
        return torch.tensor(penalties)
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        计算正则化后的奖励
        """
        # 基础奖励
        base_rewards = self.base_reward_model(sequences)
        
        # 正则化项
        length_penalties = self.compute_length_penalty(sequences)
        repetition_penalties = self.compute_repetition_penalty(sequences)
        
        # 组合奖励
        regularized_rewards = (
            base_rewards 
            + self.length_penalty_weight * length_penalties
            + self.repetition_penalty_weight * repetition_penalties
        )
        
        return regularized_rewards
