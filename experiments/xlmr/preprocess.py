# -*- coding: utf-8 -*-
"""
XLM-R 数据预处理脚本
"""

import json
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.xlmr.config import INTENTS, TRAIN_RATIO, VAL_RATIO, PROCESSED_DIR, DATA_DIR


def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_samples(data):
    """从对话数据中提取训练样本"""
    samples = []
    
    for dialog in data:
        for turn in dialog["turns"]:
            if turn.get("speaker") == "user":
                utterance = turn["utterance"]
                intent = turn.get("intent")
                
                # 跳过没有意图的样本
                if not intent:
                    continue
                
                # 跳过不在意图列表中的样本
                if intent not in INTENTS:
                    continue
                
                samples.append({
                    "utterance": utterance,
                    "intent": intent
                })
    
    return samples


def split_data(samples, train_ratio, val_ratio):
    """划分训练/验证/测试集"""
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples


def preprocess(file_path, lang="en", save_dir=None):
    """完整预处理流程"""
    print(f"\n{'='*50}")
    print(f"Processing {lang.upper()} data for XLM-R")
    print(f"{'='*50}")
    
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    
    print("Extracting samples...")
    samples = extract_samples(data)
    print(f"  Total samples: {len(samples)}")
    
    # 统计意图分布
    intent_counts = {}
    for s in samples:
        intent_counts[s["intent"]] = intent_counts.get(s["intent"], 0) + 1
    print("  Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"    {intent}: {count}")
    
    print("Splitting data...")
    train_samples, val_samples, test_samples = split_data(
        samples, TRAIN_RATIO, VAL_RATIO
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 构建意图映射
    intent_to_id = {intent: idx for idx, intent in enumerate(INTENTS)}
    
    # 保存处理后的数据
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "train_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "val_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "test_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "intent_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(intent_to_id, f, ensure_ascii=False)
        
        print(f"\nSaved preprocessed data to {save_dir}")
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "intent_to_id": intent_to_id,
        "num_intents": len(INTENTS)
    }


if __name__ == "__main__":
    # 处理英文数据
    preprocess(os.path.join(DATA_DIR, "cafe_dialogue_en.json"), lang="en", save_dir=os.path.join(PROCESSED_DIR, "en"))
    
    # 处理中文数据
    preprocess(os.path.join(DATA_DIR, "cafe_dialogue_zh.json"), lang="zh", save_dir=os.path.join(PROCESSED_DIR, "zh"))
    
    # 处理混合数据
    preprocess(os.path.join(DATA_DIR, "cafe_dialogue_mixed.json"), lang="mixed", save_dir=os.path.join(PROCESSED_DIR, "mixed"))
    
    print("\n" + "="*50)
    print("All preprocessing completed!")
    print("="*50)
