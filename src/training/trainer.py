# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\training\trainer.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import json
from typing import Dict
import matplotlib.pyplot as plt
from datetime import datetime
from src.paths import RESULTS_DIR


class CafeTrainer:
    """咖啡店模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        intent2id: Dict[str, int],
        learning_rate: float = 2e-4,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        device: str = None,
        model_save_path: str = "artifacts/models/mt5",
        language: str = "en"
    ):
        """
        初始化训练器
        """

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.intent2id = intent2id
        self.id2intent = {v: k for k, v in intent2id.items()}
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.model_save_path = model_save_path
        self.language = language

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🚀 Using device: {self.device}")

        # 移动模型到设备
        self.model.to(self.device)

        # 优化器
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 总训练步数
        total_steps = len(train_dataloader) * num_epochs

        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        # 创建模型目录
        os.makedirs(model_save_path, exist_ok=True)

        # Keep model checkpoints and derived metrics in separate artifact trees.
        self.results_dir = str(RESULTS_DIR / "mt5" / self.language)
        os.makedirs(self.results_dir, exist_ok=True)

    def train_epoch(self) -> float:
        """训练一个epoch"""

        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch in progress_bar:

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            intent_labels = batch['intent_id'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intent_labels=intent_labels,
                slot_labels=None
            )

            loss = outputs['loss']

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_dataloader)

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""

        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():

            for batch in tqdm(self.val_dataloader, desc="Evaluating"):

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent_id'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    intent_labels=intent_labels
                )

                loss = outputs['loss']
                total_loss += loss.item()

                intent_probs = torch.softmax(outputs['intent_logits'], dim=-1)
                preds = torch.argmax(intent_probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(intent_labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return {
            'loss': total_loss / len(self.val_dataloader),
            'accuracy': accuracy,
            'f1': f1
        }

    def train(self):

        print(f"\n{'='*50}")
        print(f"Start Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"{'='*50}\n")

        best_val_accuracy = 0

        for epoch in range(self.num_epochs):

            print(f"\n📅 Epoch {epoch + 1}/{self.num_epochs}")

            train_loss = self.train_epoch()

            val_metrics = self.evaluate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            if val_metrics['accuracy'] > best_val_accuracy:

                best_val_accuracy = val_metrics['accuracy']
                self.save_model(f'best_model_{self.language}.pt')

                print(f"✅ Best model saved (Accuracy: {best_val_accuracy:.4f})")

        print(f"\nTraining Finished! Best Accuracy: {best_val_accuracy:.4f}")

        self.save_model(f'final_model_{self.language}.pt')

        self.plot_training_history()
        self.save_training_metrics()
        self.save_training_metrics_csv()
        self.save_experiment_summary()

        return self.history

    def save_model(self, filename: str):

        save_path = os.path.join(self.model_save_path, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'intent2id': self.intent2id,
            'history': self.history
        }, save_path)

        print(f"Model saved to: {save_path}")

    def save_training_metrics(self):

        save_path = os.path.join(self.results_dir, 'training_metrics.json')

        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"Metrics saved to: {save_path}")

    def save_training_metrics_csv(self):

        import pandas as pd

        df = pd.DataFrame(self.history)
        df.index = df.index + 1
        df.index.name = "epoch"

        save_path = os.path.join(self.results_dir, 'training_metrics.csv')
        df.to_csv(save_path)

        print(f"CSV metrics saved to: {save_path}")

    def save_experiment_summary(self):

        best_acc = max(self.history['val_accuracy'])
        best_f1 = max(self.history['val_f1'])
        best_epoch = self.history['val_accuracy'].index(best_acc) + 1

        summary = f"""
==============================
Cafe Dialogue Model Experiment
==============================

Language: {self.language}

Total Epochs: {self.num_epochs}
Learning Rate: {self.learning_rate}

Best Validation Accuracy: {best_acc:.4f}
Best Validation F1: {best_f1:.4f}
Best Epoch: {best_epoch}

Final Train Loss: {self.history['train_loss'][-1]:.4f}
Final Val Loss: {self.history['val_loss'][-1]:.4f}

==============================
"""

        save_path = os.path.join(self.results_dir, 'experiment_summary.txt')

        with open(save_path, 'w') as f:
            f.write(summary)

        print(f"Experiment summary saved to: {save_path}")

    def plot_training_history(self):

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history['val_accuracy'], label='Accuracy', marker='o')
        axes[1].plot(self.history['val_f1'], label='F1 Score', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150)

        print(f"Training curve saved to: {save_path}")

        plt.show()

    def load_model(self, filename: str):
        """
        加载已保存的模型
        """

        load_path = os.path.join(self.model_save_path, filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'intent2id' in checkpoint:
            self.intent2id = checkpoint['intent2id']
            self.id2intent = {v: k for k, v in self.intent2id.items()}

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"✅ Model loaded from: {load_path}")
