# -*- coding: utf-8 -*-
"""
XLM-R 训练脚本
"""

import json
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.xlmr.config import (
    MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE, 
    NUM_EPOCHS, PATIENCE, RESULTS_DIR, PROCESSED_DIR
)
from src.xlmr.model import XLMRIntentClassifier
from src.xlmr.dataset import IntentDataset, collate_fn
from src.xlmr.trainer import Trainer


class Config:
    def __init__(self):
        self.MODEL_NAME = MODEL_NAME
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.PATIENCE = PATIENCE


def load_preprocessed(data_dir):
    """加载预处理后的数据"""
    with open(os.path.join(data_dir, "train_samples.json"), 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    
    with open(os.path.join(data_dir, "val_samples.json"), 'r', encoding='utf-8') as f:
        val_samples = json.load(f)
    
    with open(os.path.join(data_dir, "test_samples.json"), 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    with open(os.path.join(data_dir, "intent_to_id.json"), 'r', encoding='utf-8') as f:
        intent_to_id = json.load(f)
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "intent_to_id": intent_to_id
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "mixed"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()
    
    # 创建配置对象
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.MODEL_NAME = args.model_name
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {config.MODEL_NAME}")
    
    # 加载预处理数据
    data_dir = os.path.join(PROCESSED_DIR, args.lang)
    
    if not os.path.exists(data_dir):
        print(f"Error: Preprocessed data not found at {data_dir}")
        print("Please run preprocess.py first:")
        print("  python preprocess.py")
        return
    
    data = load_preprocessed(data_dir)
    
    print(f"\nLoaded data for {args.lang}:")
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Val: {len(data['val'])} samples")
    print(f"  Test: {len(data['test'])} samples")
    print(f"  Num intents: {len(data['intent_to_id'])}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 创建数据集
    train_dataset = IntentDataset(
        data['train'], tokenizer, data['intent_to_id'], config.MAX_SEQ_LEN
    )
    val_dataset = IntentDataset(
        data['val'], tokenizer, data['intent_to_id'], config.MAX_SEQ_LEN
    )
    test_dataset = IntentDataset(
        data['test'], tokenizer, data['intent_to_id'], config.MAX_SEQ_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = XLMRIntentClassifier(
        model_name=config.MODEL_NAME,
        num_intents=len(data['intent_to_id']),
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # 训练
    history = trainer.train()
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load("best_model.pt"))
    
    print("\n" + "="*50)
    print("Testing on test set...")
    test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # 保存结果
    results_dir = os.path.join(RESULTS_DIR, args.lang)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "num_test_samples": len(data['test']),
        "num_intents": len(data['intent_to_id']),
        "model_name": config.MODEL_NAME,
        "history": history
    }
    
    with open(os.path.join(results_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/results.json")


if __name__ == "__main__":
    main()
