"""验证数据转换结果"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.utils.data_utils import load_dataset

def verify_conversion():
    """验证转换后的数据"""
    datasets = {
        "train": "data/train_converted.jsonl",
        "valid": "data/valid_converted.jsonl",
        "test": "data/test_converted.jsonl"
    }
    
    print("=" * 60)
    print("数据转换验证报告")
    print("=" * 60)
    
    total_samples = 0
    for name, path in datasets.items():
        file_path = project_root / path
        if not file_path.exists():
            print(f"❌ {name}: 文件不存在 - {path}")
            continue
        
        try:
            data = load_dataset(str(file_path))
            total_samples += len(data)
            
            # 检查格式
            if len(data) == 0:
                print(f"❌ {name}: 数据为空")
                continue
            
            sample = data[0]
            has_prompt = "prompt" in sample
            has_reference = "reference" in sample
            
            print(f"\n✓ {name.upper()}:")
            print(f"  - 样本数: {len(data):,}")
            print(f"  - 包含 prompt: {has_prompt}")
            print(f"  - 包含 reference: {has_reference}")
            
            if has_prompt and has_reference:
                print(f"  - Prompt 示例 (前100字符): {sample['prompt'][:100]}...")
                print(f"  - Reference 示例: {sample['reference'][:100]}...")
            
        except Exception as e:
            print(f"❌ {name}: 加载失败 - {e}")
    
    print("\n" + "=" * 60)
    print(f"总计: {total_samples:,} 个样本")
    print("=" * 60)

if __name__ == "__main__":
    verify_conversion()
