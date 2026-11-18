# -*- coding: utf-8 -*-
"""
BiLSTM训练脚本 - 修复版
"""

import json
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入配置
from experiments.bilstm.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MAX_SEQ_LEN,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PATIENCE, LAMBDA, PROCESSED_DIR, RESULTS_DIR
)
from src.bilstm.model import BiLSTM
from src.bilstm.dataset import DialogueDataset, collate_fn
from src.bilstm.trainer import Trainer


# 创建配置类
class Config:
    def __init__(self):
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.DROPOUT = DROPOUT
        self.PATIENCE = PATIENCE
        self.LAMBDA = LAMBDA


def load_preprocessed(data_dir):
    """加载预处理后的数据"""
    with open(os.path.join(data_dir, "train_samples.json"), 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    
    with open(os.path.join(data_dir, "val_samples.json"), 'r', encoding='utf-8') as f:
        val_samples = json.load(f)
    
    with open(os.path.join(data_dir, "test_samples.json"), 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    with open(os.path.join(data_dir, "vocab.json"), 'r', encoding='utf-8') as f:
        word_to_id = json.load(f)
    
    with open(os.path.join(data_dir, "tag_to_id.json"), 'r', encoding='utf-8') as f:
        tag_to_id = json.load(f)
    
    with open(os.path.join(data_dir, "intent_to_id.json"), 'r', encoding='utf-8') as f:
        intent_to_id = json.load(f)
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "word_to_id": word_to_id,
        "tag_to_id": tag_to_id,
        "intent_to_id": intent_to_id
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "mixed"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 用命令行参数覆盖配置
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预处理数据
    data_dir = os.path.join(PROCESSED_DIR, args.lang)
    
    # 检查数据是否存在
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
    print(f"  Vocab size: {len(data['word_to_id'])}")
    print(f"  Num intents: {len(data['intent_to_id'])}")
    print(f"  Num slots: {len(data['tag_to_id'])}")
    
    # 创建数据集
    train_dataset = DialogueDataset(
        data['train'], data['word_to_id'], data['tag_to_id'], 
        data['intent_to_id'], config.MAX_SEQ_LEN
    )
    val_dataset = DialogueDataset(
        data['val'], data['word_to_id'], data['tag_to_id'], 
        data['intent_to_id'], config.MAX_SEQ_LEN
    )
    test_dataset = DialogueDataset(
        data['test'], data['word_to_id'], data['tag_to_id'], 
        data['intent_to_id'], config.MAX_SEQ_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = BiLSTM(
        vocab_size=len(data['word_to_id']),
        num_intents=len(data['intent_to_id']),
        num_slots=len(data['tag_to_id']),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器（传入num_slots用于计算权重）
    trainer = Trainer(model, train_loader, val_loader, config, device, num_slots=len(data['tag_to_id']))
    
    # 训练
    history = trainer.train()
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load("best_model.pt"))
    
    print("\n" + "="*50)
    print("Testing on test set...")
    test_loss, test_acc, test_intent_f1, test_slot_f1, test_overall_f1 = trainer.evaluate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Intent F1: {test_intent_f1:.4f}")
    print(f"Test Slot F1: {test_slot_f1:.4f}")
    print(f"Test Overall F1: {test_overall_f1:.4f}")
    
    # 保存结果
    results_dir = os.path.join(RESULTS_DIR, args.lang)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_intent_f1": test_intent_f1,
        "test_slot_f1": test_slot_f1,
        "test_overall_f1": test_overall_f1,
        "history": history
    }
    
    with open(os.path.join(results_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
