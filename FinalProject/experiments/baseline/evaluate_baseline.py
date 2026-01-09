"""
基线实验评估脚本
评估训练后的模型性能，支持 ROUGE、BLEU 等指标
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.data_utils import load_dataset
from experiments.utils.metrics import compute_rouge, compute_bleu, compute_average_length


def parse_args():
    parser = argparse.ArgumentParser(description="基线实验评估")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="模型路径（本地路径或 HuggingFace 模型名称）"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        default="data/test_converted.jsonl",
        help="测试数据路径（默认: data/test_converted.jsonl）"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="evaluation_results.json", 
        help="评估结果输出文件"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试，默认评估全部）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小（默认: 8）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="生成最大长度（默认: 512）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备（auto/cpu/cuda，默认: auto）"
    )
    return parser.parse_args()


def load_test_data(test_data_path: str, max_samples: int = None):
    """
    加载测试数据
    
    Args:
        test_data_path: 测试数据文件路径
        max_samples: 最大样本数（用于快速测试）
    
    Returns:
        数据列表
    """
    # 处理相对路径
    if not os.path.isabs(test_data_path):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        test_data_path = project_root / test_data_path
    
    print(f"加载测试数据: {test_data_path}")
    data = load_dataset(str(test_data_path))
    
    if max_samples and len(data) > max_samples:
        print(f"限制评估样本数为: {max_samples}")
        data = data[:max_samples]
    
    print(f"✓ 成功加载 {len(data)} 个样本")
    return data


def generate_summary(model, tokenizer, prompt: str, max_length: int = 512, device: str = "cuda"):
    """
    生成摘要
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示词
        max_length: 最大生成长度
        device: 设备
    
    Returns:
        生成的摘要文本
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            min_length=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取摘要部分（去掉 prompt）
    if "摘要：" in generated_text:
        summary = generated_text.split("摘要：")[-1].strip()
    else:
        # 如果没有找到"摘要："标记，尝试提取生成的部分
        summary = generated_text[len(prompt):].strip()
    
    return summary


def evaluate_model(model, tokenizer, test_data, batch_size: int = 8, max_length: int = 512, device: str = "cuda"):
    """
    评估模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据
        batch_size: 批次大小
        max_length: 最大生成长度
        device: 设备
    
    Returns:
        评估结果字典
    """
    model.eval()
    predictions = []
    references = []
    
    print("开始生成摘要...")
    for item in tqdm(test_data, desc="评估进度"):
        prompt = item.get("prompt", "")
        reference = item.get("reference", "")
        
        if not prompt:
            print(f"警告: 样本缺少 prompt 字段，已跳过")
            continue
        
        # 生成摘要
        try:
            prediction = generate_summary(model, tokenizer, prompt, max_length, device)
            predictions.append(prediction)
            references.append(reference if reference else "")
        except Exception as e:
            print(f"生成失败: {e}")
            predictions.append("")
            references.append(reference if reference else "")
    
    print(f"\n✓ 完成生成，共 {len(predictions)} 个样本")
    
    # 过滤空预测
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid_pairs:
        print("警告: 没有有效的预测-参考对")
        return {}
    
    valid_predictions, valid_references = zip(*valid_pairs)
    
    print("\n计算评估指标...")
    
    # 计算 ROUGE
    rouge_scores = compute_rouge(list(valid_references), list(valid_predictions))
    
    # 计算 BLEU
    bleu_score = compute_bleu(list(valid_references), list(valid_predictions))
    
    # 计算平均长度
    avg_pred_length = compute_average_length(list(valid_predictions))
    avg_ref_length = compute_average_length(list(valid_references))
    
    results = {
        "num_samples": len(predictions),
        "valid_samples": len(valid_predictions),
        "rouge": rouge_scores,
        "bleu": bleu_score,
        "average_length": {
            "prediction": avg_pred_length,
            "reference": avg_ref_length
        }
    }
    
    return results


def main():
    args = parse_args()
    
    # 确定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 处理模型路径
    model_path = args.model_path
    if not os.path.isabs(model_path) and not model_path.startswith(('http://', 'https://')):
        # 尝试从项目根目录查找
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        potential_path = project_root / model_path
        if potential_path.exists():
            model_path = str(potential_path.resolve())
            print(f"使用本地模型路径: {model_path}")
    
    # 加载模型和分词器
    print(f"\n加载模型: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(device)
        print(f"✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        sys.exit(1)
    
    # 加载测试数据
    test_data = load_test_data(args.test_data, args.max_samples)
    
    # 评估模型
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    results = evaluate_model(
        model, 
        tokenizer, 
        test_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"样本数: {results.get('num_samples', 0)}")
    print(f"有效样本数: {results.get('valid_samples', 0)}")
    print(f"\nROUGE 分数:")
    for key, value in results.get('rouge', {}).items():
        print(f"  {key}: {value:.4f}")
    print(f"\nBLEU 分数: {results.get('bleu', 0):.4f}")
    print(f"\n平均长度:")
    print(f"  预测: {results.get('average_length', {}).get('prediction', 0):.2f} 词")
    print(f"  参考: {results.get('average_length', {}).get('reference', 0):.2f} 词")
    
    # 保存结果
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        script_dir = Path(__file__).parent
        output_path = script_dir / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 评估完成！结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
