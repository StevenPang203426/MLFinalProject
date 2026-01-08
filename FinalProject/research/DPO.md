# DPO (Direct Preference Optimization) 调研报告

## 1. 方法概述

DPO（直接偏好优化）是一种无需显式奖励模型的强化学习方法，直接从偏好数据中优化策略。

## 2. 原理

### 2.1 核心思想
- 绕过奖励模型训练阶段，直接从人类偏好数据中学习
- 使用分类损失函数，将偏好学习转化为分类问题
- 简化 RLHF 的训练流程

### 2.2 数学原理
- 基于 Bradley-Terry 模型建模人类偏好
- 通过最大似然估计直接优化策略
- 避免了 PPO 等强化学习算法的复杂性

## 3. 优势

- 训练流程更简单，只需一个阶段
- 计算效率更高，无需训练奖励模型
- 减少了 reward hacking 的风险
- 训练更稳定，收敛更快

## 4. 局限

- 需要高质量的偏好数据
- 对于复杂任务，可能不如 RLHF 灵活
- 难以处理多轮交互的场景
- 偏好数据的质量直接影响效果

## 5. 适用场景

- 有高质量偏好数据的任务
- 需要快速迭代的场景
- 计算资源有限的情况
- 单轮生成任务（如摘要、翻译）

## 6. 相关论文

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [DPO 官方实现](https://github.com/eric-mitchell/direct-preference-optimization)

## 7. 实现框架

- TRL (支持 DPO)
- DPO 官方实现
- 其他支持 DPO 的框架

## 8. 实验记录

（在此记录相关实验和观察）

---

**调研日期**：  
**调研人**：
