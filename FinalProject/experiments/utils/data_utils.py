"""
数据处理工具函数
"""

import json
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    return data

def save_dataset(data: List[Dict[str, Any]], file_path: str):
    """
    保存数据集
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, ensure_ascii=False, indent=2)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

def format_prompt(input_text: str, task_type: str = "summarization") -> str:
    """
    格式化提示词
    """
    if task_type == "summarization":
        return f"请为以下文本生成摘要：\n{input_text}\n\n摘要："
    elif task_type == "math":
        return f"请解答以下数学问题：\n{input_text}\n\n解答："
    else:
        return input_text
