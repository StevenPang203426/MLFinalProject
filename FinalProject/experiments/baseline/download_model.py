"""
下载模型权重脚本
用于从 HuggingFace 下载模型权重文件到本地目录
"""

import argparse
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_id: str, local_dir: str, resume_download: bool = True):
    """
    下载模型到本地目录
    
    Args:
        model_id: HuggingFace 模型ID（如 'Qwen/Qwen2-1.5B'）
        local_dir: 本地保存目录
        resume_download: 是否支持断点续传
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"保存到: {local_path.resolve()}")
    print("-" * 60)
    
    try:
        # 下载模型
        print("下载模型权重...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(local_path),
            local_files_only=False,
            resume_download=resume_download,
            trust_remote_code=True
        )
        
        # 保存到指定目录
        print(f"\n保存模型到: {local_path}")
        model.save_pretrained(str(local_path), safe_serialization=True)
        
        # 下载并保存分词器
        print("下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(local_path),
            local_files_only=False,
            resume_download=resume_download,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(local_path))
        
        print(f"\n✓ 模型下载完成！")
        print(f"模型已保存到: {local_path.resolve()}")
        
        # 检查文件
        print("\n检查下载的文件:")
        weight_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
        if weight_files:
            print(f"  ✓ 找到 {len(weight_files)} 个权重文件")
            for f in weight_files[:5]:  # 只显示前5个
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.1f} MB)")
            if len(weight_files) > 5:
                print(f"    ... 还有 {len(weight_files) - 5} 个文件")
        else:
            print("  ⚠ 未找到权重文件，可能下载失败")
        
        config_files = list(local_path.glob("*.json"))
        if config_files:
            print(f"  ✓ 找到 {len(config_files)} 个配置文件")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用代理或 VPN")
        print("3. 使用镜像站点（如 hf-mirror.com）")
        print("4. 手动下载模型文件")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载模型权重")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="HuggingFace 模型ID（默认: Qwen/Qwen2-1.5B）"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="../../models/qwen2-1.5b",
        help="本地保存目录（默认: ../../models/qwen2-1.5b）"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="不使用断点续传"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    if not os.path.isabs(args.local_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        local_dir = project_root / args.local_dir
    else:
        local_dir = Path(args.local_dir)
    
    download_model(
        model_id=args.model_id,
        local_dir=str(local_dir),
        resume_download=not args.no_resume
    )


if __name__ == "__main__":
    main()
