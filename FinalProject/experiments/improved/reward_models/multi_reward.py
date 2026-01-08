"""
多奖励权重平衡
平衡多个奖励目标（如简洁性 vs 准确性）
"""

import torch
import torch.nn as nn
from typing import List, Dict

class MultiRewardModel(nn.Module):
    """
    多奖励模型
    平衡多个奖励目标
    """
    
    def __init__(self, reward_models: Dict[str, nn.Module], weights: Dict[str, float] = None):
        super().__init__()
        self.reward_models = nn.ModuleDict(reward_models)
        
        # 默认权重（如果未指定）
        if weights is None:
            weights = {name: 1.0 / len(reward_models) for name in reward_models.keys()}
        self.weights = weights
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化奖励
        将不同尺度的奖励归一化到相同范围
        """
        # TODO: 实现归一化策略
        # 例如：min-max 归一化、z-score 归一化等
        if rewards.std() > 0:
            normalized = (rewards - rewards.mean()) / rewards.std()
        else:
            normalized = rewards
        return normalized
    
    def forward(self, sequences: List[str], **kwargs) -> torch.Tensor:
        """
        计算多奖励加权和
        """
        all_rewards = {}
        
        # 计算每个奖励模型的奖励
        for name, model in self.reward_models.items():
            rewards = model(sequences, **kwargs)
            # 归一化奖励
            normalized_rewards = self.normalize_rewards(rewards)
            all_rewards[name] = normalized_rewards
        
        # 加权组合
        combined_rewards = torch.zeros_like(list(all_rewards.values())[0])
        for name, rewards in all_rewards.items():
            weight = self.weights.get(name, 0.0)
            combined_rewards += weight * rewards
        
        return combined_rewards
    
    def set_weights(self, weights: Dict[str, float]):
        """
        动态调整权重
        """
        self.weights = weights
    
    def get_individual_rewards(self, sequences: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """
        获取各个奖励模型的单独奖励（用于分析）
        """
        individual_rewards = {}
        for name, model in self.reward_models.items():
            rewards = model(sequences, **kwargs)
            individual_rewards[name] = rewards
        return individual_rewards
