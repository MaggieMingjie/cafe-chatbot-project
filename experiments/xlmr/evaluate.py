# -*- coding: utf-8 -*-
"""
XLM-R 评估脚本
"""

import json
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.xlmr.config import MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, PROCESSED_DIR, RESULTS_DIR
from src.xlmr.model import XLMRIntentClassifier
from src.xlmr.dataset import IntentDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "mixed"])
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_dir = os.path.join(PROCESSED_DIR, args.lang)
    with open(os.path.join(data_dir, "test_samples.json"), 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    with open(os.path.join(data_dir, "intent_to_id.json"), 'r', encoding='utf-8') as f:
        intent_to_id = json.load(f)
    
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = XLMRIntentClassifier(args.model_name, len(intent_to_id)).to(device)
    
    # 加载最佳模型
    model_path = os.path.join(RESULTS_DIR, args.lang, "best_model.pt")
    if not os.path.exists(model_path):
        # 尝试从当前目录加载
        model_path = "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 创建数据集
    test_dataset = IntentDataset(test_samples, tokenizer, intent_to_id, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_ids = batch["intent_ids"].to(device)
            
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(intent_ids.cpu().numpy())
    
    # 打印分类报告
    target_names = list(intent_to_id.keys())
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    # 打印混淆矩阵
    print("\n" + "="*50)
    print("Confusion Matrix")
    print("="*50)
    cm = confusion_matrix(all_labels, all_preds)
    print(" " * 15 + " ".join([f"{n[:8]:<8}" for n in target_names]))
    for i, name in enumerate(target_names):
        print(f"{name:12} " + " ".join([f"{cm[i][j]:4d}" for j in range(len(target_names))]))


if __name__ == "__main__":
    main()
