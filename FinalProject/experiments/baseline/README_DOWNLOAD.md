# 模型下载指南

## 问题说明

如果遇到网络连接问题无法从 HuggingFace 下载模型，可以使用以下方法：

## 方法1：使用下载脚本（推荐）

### 步骤1：运行下载脚本

```bash
cd FinalProject/experiments/baseline
python download_model.py
```

### 步骤2：等待下载完成

脚本会自动下载模型权重文件到 `../../models/qwen2-1.5b` 目录。

### 步骤3：运行训练

```bash
python train_baseline.py --config baseline_config.yaml
```

## 方法2：手动下载

### 使用 HuggingFace CLI

```bash
# 安装 huggingface-hub
pip install huggingface-hub

# 下载模型
huggingface-cli download Qwen/Qwen2-1.5B --local-dir C:/Users/lenovo/Desktop/PDML/models/qwen2-1.5b
```

### 使用 Python 脚本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2-1.5B"
local_dir = "C:/Users/lenovo/Desktop/PDML/models/qwen2-1.5b"

# 下载模型
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=local_dir)
model.save_pretrained(local_dir, safe_serialization=True)

# 下载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir)
tokenizer.save_pretrained(local_dir)
```

## 方法3：使用镜像站点

如果无法访问 HuggingFace，可以使用镜像站点：

### 设置环境变量

```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# 然后运行下载脚本
python download_model.py
```

### 或在代码中设置

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 方法4：使用代理

如果有代理，可以设置代理环境变量：

```bash
# Windows PowerShell
$env:HTTP_PROXY="http://your-proxy:port"
$env:HTTPS_PROXY="http://your-proxy:port"

python download_model.py
```

## 验证下载

下载完成后，检查模型目录应包含以下文件：

- `model.safetensors` 或 `pytorch_model.bin`（模型权重）
- `config.json`（模型配置）
- `tokenizer.json`（分词器）
- 其他配置文件

## 常见问题

### Q: 下载很慢怎么办？
A: 使用镜像站点或代理可以加速下载。

### Q: 下载中断了怎么办？
A: 重新运行下载脚本，支持断点续传。

### Q: 如何检查是否下载成功？
A: 检查模型目录中是否有 `model.safetensors` 或 `pytorch_model.bin` 文件。

### Q: 下载后仍然报错？
A: 确保模型目录路径正确，并且包含完整的权重文件。
