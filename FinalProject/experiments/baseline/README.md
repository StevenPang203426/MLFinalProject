# 基线实验

## 实验说明

本目录包含基线实验的代码和结果。

## 文件结构

```
baseline/
├── README.md              # 本文件
├── train_baseline.py      # 基线训练脚本
├── evaluate_baseline.py   # 基线评估脚本
├── config.yaml            # 实验配置文件
└── results/               # 实验结果（自动生成）
```

## 实验设置

### 模型
- 模型名称：
- 参数量：
- 预训练权重：

### 任务
- 任务类型：
- 数据集：

### 训练参数
- 学习率：
- 批次大小：
- 训练轮数：
- 其他超参数：

## 运行方式

```bash
# 训练
python train_baseline.py --config config.yaml

# 评估
python evaluate_baseline.py --model_path <model_path>
```

## 实验结果

### 训练指标
- Loss 曲线
- Reward 曲线
- 其他指标

### 评估指标
- 任务特定指标
- 生成质量指标

## 实验记录

（在此记录实验过程中的观察和问题）
