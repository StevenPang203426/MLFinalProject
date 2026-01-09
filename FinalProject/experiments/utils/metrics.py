"""
评估指标计算工具
"""

from typing import List, Dict
import numpy as np

def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    计算 ROUGE 分数
    
    Args:
        references: 参考文本列表
        predictions: 预测文本列表
    
    Returns:
        包含 rouge1, rouge2, rougeL 分数的字典
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    except ImportError:
        print("警告: rouge-score 库未安装，无法计算 ROUGE 分数")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def compute_bleu(references: List[str], predictions: List[str]) -> float:
    """
    计算 BLEU 分数
    
    Args:
        references: 参考文本列表
        predictions: 预测文本列表
    
    Returns:
        BLEU 分数
    """
    try:
        from sacrebleu import BLEU
        # sacrebleu 需要将参考文本转换为列表的列表
        refs = [[ref] for ref in references]
        bleu = BLEU()
        score = bleu.corpus_score(predictions, refs)
        return score.score / 100.0  # 转换为 0-1 范围
    except ImportError:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothing = SmoothingFunction().method1
            scores = []
            for ref, pred in zip(references, predictions):
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                scores.append(score)
            return np.mean(scores)
        except ImportError:
            print("警告: sacrebleu 和 nltk 库均未安装，无法计算 BLEU 分数")
            return 0.0

def compute_accuracy(references: List[str], predictions: List[str]) -> float:
    """
    计算准确率（适用于分类任务或数学推理任务）
    """
    correct = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return correct / len(references) if references else 0.0

def compute_average_length(texts: List[str]) -> float:
    """
    计算平均长度
    """
    lengths = [len(text.split()) for text in texts]
    return np.mean(lengths)

def compute_diversity(texts: List[str]) -> float:
    """
    计算多样性（唯一 n-gram 比例）
    """
    # TODO: 实现多样性计算
    pass
