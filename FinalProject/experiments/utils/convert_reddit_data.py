"""
Reddit 数据格式转换脚本
将 Reddit 数据格式 {"post": "...", "summary": "..."} 转换为训练格式 {"prompt": "...", "reference": "..."}
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.data_utils import load_dataset, save_dataset, format_prompt


def convert_reddit_to_training_format(
    input_file: str,
    output_file: str,
    task_type: str = "summarization",
    keep_metadata: bool = True
) -> None:
    """
    将 Reddit 数据格式转换为训练格式
    
    Args:
        input_file: 输入文件路径（包含 post 和 summary 字段）
        output_file: 输出文件路径（将包含 prompt 和 reference 字段）
        task_type: 任务类型，默认为 "summarization"
        keep_metadata: 是否保留元数据字段（id, subreddit, title 等）
    """
    print(f"正在加载数据: {input_file}")
    data = load_dataset(input_file)
    print(f"✓ 成功加载 {len(data)} 个样本")
    
    converted_data = []
    skipped_count = 0
    
    for i, item in enumerate(data):
        # 检查必需字段
        if "post" not in item or not item.get("post", "").strip():
            print(f"警告: 第 {i+1} 个样本缺少 'post' 字段，已跳过")
            skipped_count += 1
            continue
        
        if "summary" not in item or not item.get("summary", "").strip():
            print(f"警告: 第 {i+1} 个样本缺少 'summary' 字段，已跳过")
            skipped_count += 1
            continue
        
        # 转换格式
        converted_item = {
            "prompt": format_prompt(item["post"], task_type),
            "reference": item["summary"]
        }
        
        # 保留元数据（如果启用）
        if keep_metadata:
            for key in ["id", "subreddit", "title"]:
                if key in item:
                    converted_item[key] = item[key]
        
        converted_data.append(converted_item)
    
    if skipped_count > 0:
        print(f"⚠ 跳过了 {skipped_count} 个无效样本")
    
    print(f"✓ 成功转换 {len(converted_data)} 个样本")
    print(f"正在保存到: {output_file}")
    save_dataset(converted_data, output_file)
    print(f"✓ 转换完成！")
    
    # 显示转换后的示例
    if converted_data:
        print("\n转换后的数据示例（前 2 个样本）:")
        print("-" * 80)
        for i, item in enumerate(converted_data[:2]):
            print(f"\n样本 {i+1}:")
            print(f"  Prompt (前150字符): {item['prompt'][:150]}...")
            print(f"  Reference: {item['reference']}")
            if keep_metadata and 'subreddit' in item:
                print(f"  Subreddit: {item.get('subreddit', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="将 Reddit 数据格式转换为训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换测试集
  python convert_reddit_data.py data/test.jsonl data/test_converted.jsonl
  
  # 转换训练集，不保留元数据
  python convert_reddit_data.py data/train.jsonl data/train_converted.jsonl --no-metadata
  
  # 转换验证集
  python convert_reddit_data.py data/valid.jsonl data/valid_converted.jsonl
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="输入文件路径（包含 post 和 summary 字段的 JSONL 文件）"
    )
    
    parser.add_argument(
        "output_file",
        type=str,
        help="输出文件路径（将包含 prompt 和 reference 字段的 JSONL 文件）"
    )
    
    parser.add_argument(
        "--task-type",
        type=str,
        default="summarization",
        choices=["summarization", "math", "dialogue"],
        help="任务类型（默认: summarization）"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不保留元数据字段（id, subreddit, title）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 创建输出目录（如果不存在）
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行转换
    try:
        convert_reddit_to_training_format(
            input_file=args.input_file,
            output_file=args.output_file,
            task_type=args.task_type,
            keep_metadata=not args.no_metadata
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
