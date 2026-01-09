# Reddit 数据格式转换工具

## 功能说明

`convert_reddit_data.py` 脚本用于将 Reddit 数据格式转换为 RLHF 训练所需的格式。

### 输入格式
```json
{
  "id": "t3_xxx",
  "subreddit": "AskReddit",
  "title": "...",
  "post": "原始文本内容...",
  "summary": "摘要内容..."
}
```

### 输出格式
```json
{
  "prompt": "请为以下文本生成摘要：\n原始文本内容...\n\n摘要：",
  "reference": "摘要内容...",
  "id": "t3_xxx",
  "subreddit": "AskReddit",
  "title": "..."
}
```

## 使用方法

### 基本用法

```bash
# 转换测试集
python experiments/utils/convert_reddit_data.py data/test.jsonl data/test_converted.jsonl

# 转换训练集
python experiments/utils/convert_reddit_data.py data/train.jsonl data/train_converted.jsonl

# 转换验证集
python experiments/utils/convert_reddit_data.py data/valid.jsonl data/valid_converted.jsonl
```

### 高级选项

```bash
# 不保留元数据字段（只保留 prompt 和 reference）
python experiments/utils/convert_reddit_data.py data/train.jsonl data/train_converted.jsonl --no-metadata

# 指定任务类型（默认是 summarization）
python experiments/utils/convert_reddit_data.py data/test.jsonl data/test_converted.jsonl --task-type summarization
```

## 参数说明

- `input_file`: 输入文件路径（包含 `post` 和 `summary` 字段的 JSONL 文件）
- `output_file`: 输出文件路径（将包含 `prompt` 和 `reference` 字段）
- `--task-type`: 任务类型，可选值：`summarization`（默认）、`math`、`dialogue`
- `--no-metadata`: 不保留元数据字段（id, subreddit, title）

## 示例

转换后的数据可以直接用于训练：

```python
from experiments.utils.data_utils import load_dataset, prepare_ppo_dataset

# 加载转换后的数据
data = load_dataset("data/test_converted.jsonl")

# 准备用于 PPO 训练的数据集
ppo_data = prepare_ppo_dataset(data, task_type="summarization")
```

## 注意事项

1. 脚本会自动跳过缺少 `post` 或 `summary` 字段的样本
2. 转换后的 `prompt` 字段已经格式化，包含任务指令
3. 默认会保留元数据字段，便于后续分析和调试
4. 如果输入文件很大，转换可能需要一些时间
