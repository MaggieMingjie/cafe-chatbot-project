# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\train_en.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import time
import json
from src.paths import MODELS_DIR, RAW_DATA_DIR

from preprocessing.data_loader import CafeDataLoader
from preprocessing.data_processor import CafeDataset, DataCollator
from model.cafe_model import CafeDialogueModel
from training.trainer import CafeTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Coffee Shop Dialogue Model')
    
    # 数据路径
    parser.add_argument('--data_path', type=str, 
                        default=str(RAW_DATA_DIR / "cafe_dialogue_en.json"),
                        help='Dataset path')
    parser.add_argument('--model_save_path', type=str,
                        default=str(MODELS_DIR / "mt5" / "en"),
                        help='Model save path')
    
    # 模型参数
    parser.add_argument('--base_model', type=str, default="google/mt5-small",
                        help='Base model name')
    parser.add_argument('--num_slot_labels', type=int, default=20,
                        help='Number of slot labels')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Whether to use LoRA')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha parameter')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test_run', action='store_true',
                        help='Test run (only run one batch)')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Random seed set to: {seed}")


def main():
    """主训练函数"""
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 记录开始时间
    start_time = time.time()
    
    print("\n" + "="*60)
    print("🍵 Coffee Shop Dialogue Model Training System")
    print("="*60)
    print(f"Training Configuration:")
    print(f"  - Base Model: {args.base_model}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Max Sequence Length: {args.max_length}")
    print(f"  - LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    print("="*60)
    
    # 1. 加载数据
    print("\n" + "="*50)
    print("📂 Step 1: Loading Dataset")
    print("="*50)
    loader = CafeDataLoader(args.data_path)
    
    # 2. 创建分词器
    print("\n" + "="*50)
    print("🔤 Step 2: Initializing Tokenizer")
    print("="*50)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 3. 创建意图标签映射
    intent2id = loader.create_intent_labels()
    print(f"Intent label mapping: {intent2id}")
    print(f"Number of intents: {len(intent2id)}")
    
    # 4. 划分数据集
    train_data, val_data, test_data = loader.split_data(
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=1 - args.train_ratio - args.val_ratio,
        seed=args.seed
    )
    
    # 5. 创建数据集
    print("\n" + "="*50)
    print("📊 Step 3: Creating Datasets")
    print("="*50)
    train_dataset = CafeDataset(
        train_data, 
        tokenizer, 
        intent2id, 
        max_length=args.max_length,
        is_training=True
    )
    val_dataset = CafeDataset(
        val_data, 
        tokenizer, 
        intent2id, 
        max_length=args.max_length,
        is_training=False
    )
    test_dataset = CafeDataset(
        test_data, 
        tokenizer, 
        intent2id, 
        max_length=args.max_length,
        is_training=False
    )
    
    print(f"Training set size: {len(train_dataset)} samples")
    print(f"Validation set size: {len(val_dataset)} samples")
    print(f"Test set size: {len(test_dataset)} samples")
    
    # 6. 创建数据加载器
    print("\n" + "="*50)
    print("🔄 Step 4: Creating DataLoaders")
    print("="*50)
    data_collator = DataCollator(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0
    )
    
    # 7. 如果是测试运行，只取一个batch
    if args.test_run:
        print("\n🧪 Test mode: Only using one batch")
        train_dataloader = [next(iter(train_dataloader))]
        val_dataloader = [next(iter(val_dataloader))]
    
    # 8. 创建模型
    print("\n" + "="*50)
    print("🤖 Step 5: Initializing Model")
    print("="*50)
    model = CafeDialogueModel(
        model_name=args.base_model,
        num_intents=len(intent2id),
        num_slot_labels=args.num_slot_labels,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        confidence_threshold=0.7
    )
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 9. 创建训练器
    print("\n" + "="*50)
    print("🏋️ Step 6: Initializing Trainer")
    print("="*50)
    trainer = CafeTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        intent2id=intent2id,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        model_save_path=args.model_save_path,
        language="en"
    )
    
    # 10. 开始训练
    print("\n" + "="*50)
    print("🚀 Step 7: Start Training")
    print("="*50)
    history = trainer.train()
    
    # 11. 在测试集上评估
    print("\n" + "="*50)
    print("📈 Step 8: Test Set Evaluation")
    print("="*50)
    
    best_model_path = os.path.join(args.model_save_path, f'best_model_{trainer.language}.pt')
    if os.path.exists(best_model_path):
        trainer.load_model(f'best_model_{trainer.language}.pt')
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        trainer.val_dataloader = test_dataloader
        test_metrics = trainer.evaluate()
        
        print(f"\n📊 Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")

        # 保存测试结果到文件
        test_results_path = os.path.join(trainer.results_dir, 'test_results.json')
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'num_test_samples': len(test_dataset)
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 Test results saved to: {test_results_path}")
    
    # 12. 训练完成总结
    end_time = time.time()
    training_time = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ Training Completed!")
    print("="*60)
    print(f"⏱️ Total Training Time: {training_time//60:.0f} min {training_time%60:.0f} sec")
    print(f"📁 Model Save Path: {args.model_save_path}")
    print(f"🏆 Best Validation Accuracy: {max(trainer.history['val_accuracy']):.4f}")
    print(f"📈 Best F1 Score: {max(trainer.history['val_f1']):.4f}")
    print("="*60)
    
    return history


if __name__ == "__main__":
    history = main()
