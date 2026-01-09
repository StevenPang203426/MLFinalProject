"""
数据格式验证脚本
用于检查数据文件是否符合训练要求
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import load_dataset, validate_dataset

def main():
    parser = argparse.ArgumentParser(description="验证数据文件格式")
    parser.add_argument("--data_path", type=str, required=True, help="数据文件路径")
    parser.add_argument("--task_type", type=str, default="summarization", 
                       choices=["summarization", "math", "dialogue"],
                       help="任务类型")
    args = parser.parse_args()
    
    print(f"正在验证数据文件: {args.data_path}")
    print(f"任务类型: {args.task_type}")
    print("-" * 50)
    
    try:
        # 加载数据
        data = load_dataset(args.data_path)
        print(f"✓ 成功加载 {len(data)} 个样本")
        
        # 验证数据格式
        is_valid, errors = validate_dataset(data, required_fields=["prompt"])
        
        if is_valid:
            print("✓ 数据格式验证通过！")
            
            # 显示数据统计信息
            print("\n数据统计:")
            print(f"  - 总样本数: {len(data)}")
            
            has_reference = sum(1 for item in data if "reference" in item and item["reference"])
            print(f"  - 包含参考输出的样本数: {has_reference}")
            
            # 显示前几个样本的示例
            print("\n前 3 个样本预览:")
            for i, item in enumerate(data[:3]):
                print(f"\n样本 {i+1}:")
                print(f"  Prompt (前100字符): {item['prompt'][:100]}...")
                if "reference" in item:
                    print(f"  Reference (前100字符): {item['reference'][:100]}...")
        else:
            print("✗ 数据格式验证失败！")
            print("\n错误信息:")
            for error in errors[:10]:  # 只显示前10个错误
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... 还有 {len(errors) - 10} 个错误")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"✗ 错误: 文件不存在: {args.data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
