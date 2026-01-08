# RLHF (Reinforcement Learning from Human Feedback) 调研报告

## 1. 方法概述

RLHF（从人类反馈的强化学习）是一种通过人类反馈来训练奖励模型，进而指导强化学习训练的方法。

## 2. 原理

### 2.1 核心思想
- 使用人类反馈来训练奖励模型（Reward Model）
- 通过强化学习优化策略模型，使其输出更符合人类偏好

### 2.2 训练流程
1. **监督微调（SFT）**：在人类标注的数据上微调预训练模型
2. **奖励模型训练**：使用人类偏好数据训练奖励模型
3. **强化学习优化**：使用 PPO 等算法，基于奖励模型优化策略

## 3. 优势

- 能够学习复杂的人类偏好，无需显式定义奖励函数
- 适用于难以形式化的任务（如创意写作、对话生成）
- 已被 ChatGPT、Claude 等大模型成功应用

## 4. 局限

- 需要大量人类标注数据，成本高
- 人类标注可能存在不一致性
- 奖励模型可能存在偏见，导致 reward hacking
- 训练过程复杂，需要多个阶段

## 5. 适用场景

- 对话生成任务
- 文本摘要任务
- 创意写作任务
- 需要符合人类偏好的生成任务

## 6. 相关论文

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [InstructGPT](https://openai.com/research/instruction-following)

## 7. 实现框架

- TRL (Transformers Reinforcement Learning)
- OpenRLHF
- DeepSpeed-Chat

## 8. 实验记录

（在此记录相关实验和观察）

---

**调研日期**：  
**调研人**：
