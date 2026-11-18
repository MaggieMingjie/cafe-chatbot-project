# -*- coding: utf-8 -*-
"""
评估脚本：计算详细指标
"""

import json
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.bilstm.config import *
from src.bilstm.model import BiLSTM
from src.bilstm.dataset import DialogueDataset, collate_fn


def load_model(model_path, vocab_size, num_intents, num_slots, device):
    """加载训练好的模型"""
    model = BiLSTM(
        vocab_size=vocab_size,
        num_intents=num_intents,
        num_slots=num_slots,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, test_loader, tag_to_id, intent_to_id, device):
    """评估模型并输出详细结果"""
    model.eval()
    
    all_intent_preds = []
    all_intent_labels = []
    all_slot_preds = []
    all_slot_labels = []
    
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    
    with torch.no_grad():
        for batch in test_loader:
            token_ids = batch["token_ids"].to(device)
            slot_ids = batch["slot_ids"].to(device)
            intent_ids = batch["intent_ids"].to(device)
            lengths = batch["lengths"]
            
            intent_logits, slot_logits = model(token_ids, lengths)
            
            intent_pred = torch.argmax(intent_logits, dim=1)
            slot_pred = torch.argmax(slot_logits, dim=-1)
            
            all_intent_preds.extend(intent_pred.cpu().numpy())
            all_intent_labels.extend(intent_ids.cpu().numpy())
            
            for i, length in enumerate(lengths):
                all_slot_preds.extend(slot_pred[i, :length].cpu().numpy())
                all_slot_labels.extend(slot_ids[i, :length].cpu().numpy())
    
    # 计算各类别准确率
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*50)
    print("Intent Classification Report")
    print("="*50)
    intent_target_names = list(intent_to_id.keys())
    print(classification_report(
        all_intent_labels, all_intent_preds, 
        target_names=intent_target_names, zero_division=0
    ))
    
    print("\n" + "="*50)
    print("Slot Filling Report")
    print("="*50)
    slot_target_names = list(tag_to_id.keys())
    print(classification_report(
        all_slot_labels, all_slot_preds, 
        target_names=slot_target_names, zero_division=0
    ))
    
    return all_intent_preds, all_intent_labels, all_slot_preds, all_slot_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "mixed"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_dir = os.path.join(PROCESSED_DIR, args.lang)
    with open(os.path.join(data_dir, "vocab.json"), 'r') as f:
        word_to_id = json.load(f)
    with open(os.path.join(data_dir, "tag_to_id.json"), 'r') as f:
        tag_to_id = json.load(f)
    with open(os.path.join(data_dir, "intent_to_id.json"), 'r') as f:
        intent_to_id = json.load(f)
    with open(os.path.join(data_dir, "test_samples.json"), 'r') as f:
        test_samples = json.load(f)
    
    # 创建测试集
    test_dataset = DialogueDataset(
        test_samples, word_to_id, tag_to_id, intent_to_id, MAX_SEQ_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 加载模型
    model_path = os.path.join(RESULTS_DIR, args.lang, "best_model.pt")
    model = load_model(model_path, len(word_to_id), len(intent_to_id), len(tag_to_id), device)
    
    # 评估
    evaluate_model(model, test_loader, tag_to_id, intent_to_id, device)


if __name__ == "__main__":
    main()
