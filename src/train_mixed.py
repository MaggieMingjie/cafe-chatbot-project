# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\train_mixed.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import time
import json
from collections import Counter
import numpy as np
from src.paths import MODELS_DIR, RAW_DATA_DIR

from preprocessing.data_loader import CafeDataLoader
from preprocessing.data_processor import CafeDataset, DataCollator
from model.cafe_model import CafeDialogueModel
from training.trainer import CafeTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Train Code-Switching Bilingual Coffee Shop Dialogue Model (English + Chinese Mixed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python train_mixed.py --data_path ../data/cafe_dialogue_mixed.json
  python train_mixed.py --base_model google/mt5-small --num_epochs 15
  python train_mixed.py --joint_loss_lambda 1.2
        """
    )
    
    # 数据路径
    parser.add_argument('--data_path', type=str, 
                        default=str(RAW_DATA_DIR / "cafe_dialogue_mixed.json"),
                        help='Path to code-switching bilingual dialogue dataset')
    parser.add_argument('--model_save_path', type=str,
                        default=str(MODELS_DIR / "mt5" / "mixed"),
                        help='Directory to save trained models')
    
    # 模型参数
    parser.add_argument('--base_model', type=str, default="google/mt5-small",
                        choices=["google/mt5-small", "google/mt5-base", "google/mt5-large"],
                        help='Base multilingual model')
    parser.add_argument('--num_slot_labels', type=int, default=20,
                        help='Number of slot labels for BIO annotation')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for slot filling predictions')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate for optimization')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--joint_loss_lambda', type=float, default=1.2,
                        help='Weight for slot filling loss in joint optimization')
    
    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of data for validation')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use Low-Rank Adaptation (LoRA)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='Rank of LoRA adaptation matrices')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='Scaling factor for LoRA adaptations')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test_run', action='store_true',
                        help='Run in test mode with single batch')
    parser.add_argument('--save_final_model', action='store_true', default=True,
                        help='Save final model after training')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Random seed set to: {seed}")


def is_chinese_char(char):
    """判断字符是否为中文字符"""
    return '\u4e00' <= char <= '\u9fff'


def analyze_code_switching_dataset(data_path):
    """分析代码混合数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("📊 Code-Switching Dataset Analysis")
    print("="*60)
    
    print(f"Total dialogues: {len(data)}")
    print("\n🌐 Language Distribution:")
    print(f"  Code-Switched (中英混合): {len(data)} dialogues (100.0%)")
    
    # 分析代码混合情况
    code_switching_stats = {
        'total_utterances': 0,
        'mixed_utterances': 0
    }
    
    for dialogue in data:
        for turn in dialogue.get('turns', []):
            if turn.get('speaker') == 'user':
                code_switching_stats['total_utterances'] += 1
                utterance = turn['utterance']
                
                has_english = any(c.isalpha() and ord(c) < 128 for c in utterance)
                has_chinese = any(is_chinese_char(c) for c in utterance)
                
                if has_english and has_chinese:
                    code_switching_stats['mixed_utterances'] += 1
    
    print(f"\n📝 Code-Switching Statistics:")
    print(f"  Total user utterances: {code_switching_stats['total_utterances']}")
    print(f"  Mixed utterances (中英混合): {code_switching_stats['mixed_utterances']}")
    
    # 意图分布
    all_intents = []
    for dialogue in data:
        for turn in dialogue.get('turns', []):
            if turn.get('speaker') == 'user' and 'intent' in turn:
                all_intents.append(turn['intent'])
    
    intent_counter = Counter(all_intents)
    print("\n🎯 Intent Distribution:")
    for intent, count in intent_counter.most_common():
        print(f"  {intent}: {count} ({count/len(all_intents)*100:.1f}%)")
    
    return {
        'total_dialogues': len(data),
        'total_utterances': code_switching_stats['total_utterances'],
        'mixed_utterances': code_switching_stats['mixed_utterances'],
        'intent_distribution': dict(intent_counter)
    }


def ensure_directories(model_save_path):
    """确保模型目录存在"""
    os.makedirs(model_save_path, exist_ok=True)
    print(f"📁 Model save path: {model_save_path}")


def save_training_config(args, model_save_path):
    """保存训练配置"""
    # 配置保存在模型目录，trainer会自动处理results目录
    config_path = os.path.join(model_save_path, 'training_config_mixed.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"📝 Training configuration saved to: {config_path}")


def main():
    """主训练函数"""
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保模型目录存在
    ensure_directories(args.model_save_path)
    
    # 保存训练配置
    save_training_config(args, args.model_save_path)
    
    # 记录开始时间
    start_time = time.time()
    
    print("\n" + "="*70)
    print("🍵 A Domain-Specific Bilingual Conversational Model")
    print("   for Task-Oriented Dialogue in Café-Based Micro-Domain")
    print("   (Code-Switching: English + Chinese Mixed)")
    print("="*70)
    print("\nTraining Configuration:")
    print(f"  - Base Model: {args.base_model}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Max Sequence Length: {args.max_length}")
    print(f"  - Joint Loss λ: {args.joint_loss_lambda}")
    print(f"  - LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    print("="*70)
    
    # ========== Step 1: 加载数据集 ==========
    print("\n" + "="*50)
    print("📂 Step 1: Loading Code-Switching Dataset")
    print("="*50)
    
    loader = CafeDataLoader(args.data_path)
    dataset_stats = analyze_code_switching_dataset(args.data_path)
    
    # ========== Step 2: 初始化分词器 ==========
    print("\n" + "="*50)
    print("🔤 Step 2: Initializing Tokenizer")
    print("="*50)
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # ========== Step 3: 创建意图标签映射 ==========
    print("\n" + "="*50)
    print("🏷️ Step 3: Creating Intent Labels")
    print("="*50)
    
    intent2id = loader.create_intent_labels()
    print(f"Intent labels: {list(intent2id.keys())}")
    print(f"Number of intents: {len(intent2id)}")
    
    # ========== Step 4: 数据划分 ==========
    print("\n" + "="*50)
    print("📊 Step 4: Splitting Dataset")
    print("="*50)
    
    test_ratio = 1 - args.train_ratio - args.val_ratio
    print(f"Train: {args.train_ratio*100:.0f}%, Val: {args.val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    
    train_data, val_data, test_data = loader.split_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )
    
    print(f"\n📈 Data Split Complete:")
    print(f"  Training: {len(train_data)} dialogues")
    print(f"  Validation: {len(val_data)} dialogues")
    print(f"  Test: {len(test_data)} dialogues")
    
    # ========== Step 5: 创建数据集 ==========
    print("\n" + "="*50)
    print("📦 Step 5: Creating PyTorch Datasets")
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # ========== Step 6: 创建数据加载器 ==========
    print("\n" + "="*50)
    print("🔄 Step 6: Creating DataLoaders")
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
    
    # 测试模式
    if args.test_run:
        print("\n🧪 Test mode: Using only one batch")
        train_dataloader = [next(iter(train_dataloader))]
        val_dataloader = [next(iter(val_dataloader))]
    
    # ========== Step 7: 创建模型 ==========
    print("\n" + "="*50)
    print("🤖 Step 7: Initializing Bilingual Dialogue Model")
    print("="*50)
    
    model = CafeDialogueModel(
        model_name=args.base_model,
        num_intents=len(intent2id),
        num_slot_labels=args.num_slot_labels,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        confidence_threshold=args.confidence_threshold
    )
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
    
    # ========== Step 8: 创建训练器 ==========
    print("\n" + "="*50)
    print("🏋️ Step 8: Initializing Trainer")
    print("="*50)
    
    # 使用 "mixed" 作为语言标识，这样结果会保存在 results/mixed/
    trainer = CafeTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        intent2id=intent2id,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        model_save_path=args.model_save_path,
        language="mixed"  # 使用 "mixed" 以匹配 results/mixed 目录
    )
    
    print(f"📁 Results will be saved to: {trainer.results_dir}")
    
    # ========== Step 9: 开始训练 ==========
    print("\n" + "="*50)
    print("🚀 Step 9: Starting Training")
    print("="*50)
    
    history = trainer.train()
    
    # ========== Step 10: 测试集评估 ==========
    print("\n" + "="*50)
    print("📈 Step 10: Test Set Evaluation")
    print("="*50)
    
    best_model_path = os.path.join(args.model_save_path, 'best_model_mixed.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        trainer.load_model('best_model_mixed.pt')
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0
        )
        
        # 临时替换验证集为测试集进行评估
        original_val_dataloader = trainer.val_dataloader
        trainer.val_dataloader = test_dataloader
        test_metrics = trainer.evaluate()
        trainer.val_dataloader = original_val_dataloader
        
        print("\n📊 Overall Test Results (Code-Switched Dataset):")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Intent Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Slot Filling F1: {test_metrics['f1']:.4f}")
        
        # 保存测试结果到results目录
        test_results_path = os.path.join(trainer.results_dir, 'test_results.json')
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'overall': test_metrics,
                'num_test_samples': len(test_dataset),
                'dataset_type': 'code_switched'
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 Test results saved to: {test_results_path}")
        
        # 保存代码混合分析
        analysis_path = os.path.join(trainer.results_dir, 'code_switching_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_metrics': test_metrics,
                'dataset_stats': dataset_stats
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 Code-switching analysis saved to: {analysis_path}")
    
    # ========== Step 11: 训练完成总结 ==========
    end_time = time.time()
    training_time = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ Training Completed Successfully!")
    print("="*70)
    print(f"⏱️  Total Training Time: {training_time//60:.0f} min {training_time%60:.0f} sec")
    print(f"📁 Model Save Path: {args.model_save_path}")
    print(f"📁 Results Save Path: {trainer.results_dir}")
    print(f"🏆 Best Validation Accuracy: {max(trainer.history['val_accuracy']):.4f}")
    print(f"📈 Best Slot Filling F1: {max(trainer.history['val_f1']):.4f}")
    
    # 添加实验摘要信息到已有的summary文件
    summary_path = os.path.join(trainer.results_dir, 'experiment_summary.txt')
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f"\nAdditional Info:\n")
        f.write(f"Total Training Time: {training_time//60:.0f} min {training_time%60:.0f} sec\n")
        f.write(f"Dataset: Code-Switched (中英混合)\n")
        f.write(f"Mixed Utterances: {dataset_stats['mixed_utterances']}\n")
    
    print("\n" + "="*70)
    print("🎉 Model is ready for code-switched English-Chinese conversations!")
    
    return history


if __name__ == "__main__":
    history = main()
