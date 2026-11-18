# -*- coding: utf-8 -*-
"""
数据预处理脚本（修复版）
"""

import json
import os
import sys
from collections import Counter
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.aligner import align_slots, build_tag_to_id
from experiments.bilstm.config import SLOTS, INTENTS, MAX_SEQ_LEN, TRAIN_RATIO, VAL_RATIO, PROCESSED_DIR, DATA_DIR


def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_samples(data, lang="mixed", fuzzy=True):
    """
    从对话数据中提取训练样本
    每个用户消息作为一个样本
    
    Args:
        data: 对话数据
        lang: 语言类型
        fuzzy: 是否启用模糊匹配（默认开启）
    """
    samples = []
    failed_count = 0
    
    for dialog in data:
        for turn in dialog["turns"]:
            if turn.get("speaker") == "user":
                utterance = turn["utterance"]
                intent = turn["intent"]
                slots = turn.get("slots", {})
                
                # 槽位对齐（启用模糊匹配）
                tokens, slot_labels = align_slots(utterance, slots, lang, fuzzy=fuzzy)
                
                # 检查是否有槽位未匹配
                matched_slots = [tag for tag in slot_labels if tag != "O"]
                if len(slots) > 0 and len(matched_slots) == 0:
                    failed_count += 1
                
                samples.append({
                    "tokens": tokens,
                    "intent": intent,
                    "slot_labels": slot_labels
                })
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} samples had unmatched slots")
    
    return samples


def build_vocab(samples):
    """构建词表"""
    word_counts = Counter()
    word_counts["<PAD>"] = float('inf')
    word_counts["<UNK>"] = float('inf')
    
    for sample in samples:
        for token in sample["tokens"]:
            word_counts[token] += 1
    
    word_to_id = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    return word_to_id, id_to_word


def split_data(samples, train_ratio, val_ratio):
    """划分训练/验证/测试集"""
    random.seed(42)  # 固定随机种子，保证可复现
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples


def preprocess(file_path, lang="mixed", save_dir=None):
    """完整预处理流程"""
    print(f"\n{'='*50}")
    print(f"Processing {lang.upper()} data")
    print(f"{'='*50}")
    
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    
    print("Extracting samples...")
    samples = extract_samples(data, lang, fuzzy=True)
    print(f"  Total samples: {len(samples)}")
    
    print("Building vocabulary...")
    word_to_id, id_to_word = build_vocab(samples)
    print(f"  Vocabulary size: {len(word_to_id)}")
    
    # 构建标签映射
    tag_to_id = build_tag_to_id(SLOTS)
    intent_to_id = {intent: idx for idx, intent in enumerate(INTENTS)}
    
    print("Splitting data...")
    train_samples, val_samples, test_samples = split_data(
        samples, TRAIN_RATIO, VAL_RATIO
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 统计槽位标签分布
    tag_counts = Counter()
    for sample in samples:
        for tag in sample["slot_labels"]:
            tag_counts[tag] += 1
    
    non_o_count = sum(v for k, v in tag_counts.items() if k != "O")
    total_count = sum(tag_counts.values())
    print(f"  Non-O tags ratio: {non_o_count/total_count*100:.2f}%")
    print(f"  Unique BIO tags: {len([t for t in tag_counts if t != 'O'])}")
    
    # 保存处理后的数据
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "train_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "val_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "test_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(save_dir, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(word_to_id, f, ensure_ascii=False)
        
        with open(os.path.join(save_dir, "tag_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(tag_to_id, f, ensure_ascii=False)
        
        with open(os.path.join(save_dir, "intent_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(intent_to_id, f, ensure_ascii=False)
        
        print(f"\nSaved preprocessed data to {save_dir}")
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "word_to_id": word_to_id,
        "tag_to_id": tag_to_id,
        "intent_to_id": intent_to_id,
        "vocab_size": len(word_to_id),
        "num_intents": len(INTENTS),
        "num_slots": len(tag_to_id)
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
