"""
数据处理工具函数
"""

import json
from typing import List, Dict, Any, Optional

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集
    
    Args:
        file_path: 数据文件路径（支持 .json 或 .jsonl 格式）
    
    Returns:
        数据列表，每个元素是一个字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 检查文件格式（支持 .json, .jsonl 以及 .example 等扩展名）
        if '.json' in file_path and not file_path.endswith('.jsonl'):
            # JSON 格式（可能是 .json 或 .json.example）
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，尝试按 JSONL 格式解析
                f.seek(0)
                data = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            # JSONL 格式（.jsonl 或包含 jsonl 的其他扩展名）
            data = [json.loads(line.strip()) for line in f if line.strip()]
    return data

def save_dataset(data: List[Dict[str, Any]], file_path: str):
    """
    保存数据集
    
    Args:
        data: 要保存的数据列表
        file_path: 保存路径（支持 .json 或 .jsonl 格式）
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, ensure_ascii=False, indent=2)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，请使用 .json 或 .jsonl 格式")

def validate_dataset(data: List[Dict[str, Any]], required_fields: Optional[List[str]] = None) -> tuple[bool, List[str]]:
    """
    验证数据集格式
    
    Args:
        data: 数据列表
        required_fields: 必需字段列表，默认为 ["prompt"]
    
    Returns:
        (是否有效, 错误信息列表)
    """
    if required_fields is None:
        required_fields = ["prompt"]
    
    errors = []
    
    if not data:
        errors.append("数据集为空")
        return False, errors
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"第 {i+1} 个样本不是字典类型")
            continue
        
        # 检查必需字段
        for field in required_fields:
            if field not in item:
                errors.append(f"第 {i+1} 个样本缺少必需字段: {field}")
            elif not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"第 {i+1} 个样本的字段 {field} 为空或不是字符串")
    
    return len(errors) == 0, errors

def format_prompt(input_text: str, task_type: str = "summarization") -> str:
    """
    格式化提示词
    
    Args:
        input_text: 输入文本
        task_type: 任务类型（summarization, math, dialogue）
    
    Returns:
        格式化后的提示词
    """
    if task_type == "summarization":
        return f"请为以下文本生成摘要：\n{input_text}\n\n摘要："
    elif task_type == "math":
        return f"请解答以下数学问题：\n{input_text}\n\n解答："
    elif task_type == "dialogue":
        # 对话任务通常 prompt 已经包含完整的对话上下文
        return input_text
    else:
        return input_text

def prepare_ppo_dataset(data: List[Dict[str, Any]], task_type: str = "summarization") -> List[Dict[str, Any]]:
    """
    准备用于 PPO 训练的数据集
    
    Args:
        data: 原始数据列表
        task_type: 任务类型
    
    Returns:
        处理后的数据列表，每个样本包含格式化后的 prompt
    """
    processed_data = []
    
    for item in data:
        # 如果 prompt 字段已存在且已格式化，直接使用
        if "prompt" in item and item["prompt"]:
            prompt = item["prompt"]
        else:
            # 否则尝试从其他字段构建 prompt
            if "input" in item:
                prompt = format_prompt(item["input"], task_type)
            elif "question" in item:
                prompt = format_prompt(item["question"], task_type)
            elif "post" in item:
                # 支持 Reddit 数据格式
                prompt = format_prompt(item["post"], task_type)
            else:
                raise ValueError(f"无法从数据样本中找到有效的输入字段: {item.keys()}")
        
        processed_item = {
            "prompt": prompt,
        }
        
        # 保留参考输出（如果存在）
        if "reference" in item:
            processed_item["reference"] = item["reference"]
        elif "summary" in item:
            # 支持 Reddit 数据格式（summary -> reference）
            processed_item["reference"] = item["summary"]
        
        # 保留其他字段（排除已处理的字段）
        excluded_keys = ["prompt", "reference", "input", "question", "post", "summary"]
        for key in item:
            if key not in excluded_keys:
                processed_item[key] = item[key]
        
        processed_data.append(processed_item)
    
    return processed_data
