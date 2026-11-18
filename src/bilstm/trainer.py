# -*- coding: utf-8 -*-
"""
BiLSTM训练器 - 修复版（添加类别权重）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, config, device, num_slots):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.num_slots = num_slots
        
        self.criterion_intent = nn.CrossEntropyLoss()
        
        # 计算类别权重 - 给槽位标签更高权重
        # 索引0是O标签，索引1+是B-和I-标签
        class_weights = torch.ones(num_slots)
        # 给所有非O标签（索引1及以后）更高的权重
        class_weights[1:] = 10.0  # 权重设为10，让模型更关注槽位
        self.criterion_slot = nn.CrossEntropyLoss(
            weight=class_weights.to(device), 
            ignore_index=0
        )
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_intent_correct = 0
        total_intent_samples = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            token_ids = batch["token_ids"].to(self.device)
            slot_ids = batch["slot_ids"].to(self.device)
            intent_ids = batch["intent_ids"].to(self.device)
            lengths = batch["lengths"]
            
            # 前向传播
            intent_logits, slot_logits = self.model(token_ids, lengths)
            
            # 计算意图损失
            intent_loss = self.criterion_intent(intent_logits, intent_ids)
            
            # 计算槽位损失
            batch_size, seq_len, num_slots = slot_logits.shape
            slot_logits_flat = slot_logits.reshape(-1, num_slots)
            slot_ids_flat = slot_ids.reshape(-1)
            slot_loss = self.criterion_slot(slot_logits_flat, slot_ids_flat)
            
            # 总损失（提高槽位损失的权重）
            loss = intent_loss + 1.5 * slot_loss  # 增加槽位损失权重
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算意图准确率
            _, predicted = torch.max(intent_logits, 1)
            total_intent_correct += (predicted == intent_ids).sum().item()
            total_intent_samples += intent_ids.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        intent_acc = total_intent_correct / total_intent_samples
        
        return avg_loss, intent_acc
    
    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_intent_correct = 0
        total_intent_samples = 0
        
        all_intent_preds = []
        all_intent_labels = []
        all_slot_preds = []
        all_slot_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                token_ids = batch["token_ids"].to(self.device)
                slot_ids = batch["slot_ids"].to(self.device)
                intent_ids = batch["intent_ids"].to(self.device)
                lengths = batch["lengths"]
                
                intent_logits, slot_logits = self.model(token_ids, lengths)
                
                # 计算损失
                intent_loss = self.criterion_intent(intent_logits, intent_ids)
                
                batch_size, seq_len, num_slots = slot_logits.shape
                slot_logits_flat = slot_logits.reshape(-1, num_slots)
                slot_ids_flat = slot_ids.reshape(-1)
                slot_loss = self.criterion_slot(slot_logits_flat, slot_ids_flat)
                
                loss = intent_loss + 1.5 * slot_loss
                total_loss += loss.item()
                
                # 意图预测
                _, intent_pred = torch.max(intent_logits, 1)
                total_intent_correct += (intent_pred == intent_ids).sum().item()
                total_intent_samples += intent_ids.size(0)
                
                # 收集预测和标签
                all_intent_preds.extend(intent_pred.cpu().numpy())
                all_intent_labels.extend(intent_ids.cpu().numpy())
                
                # 槽位预测（只取有效长度）
                slot_pred = torch.argmax(slot_logits, dim=-1)
                for i, length in enumerate(lengths):
                    actual_len = min(length.item(), seq_len)
                    all_slot_preds.extend(slot_pred[i, :actual_len].cpu().numpy())
                    all_slot_labels.extend(slot_ids[i, :actual_len].cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        intent_acc = total_intent_correct / total_intent_samples
        
        # 计算F1（槽位用micro平均更合理）
        intent_f1 = f1_score(all_intent_labels, all_intent_preds, average='macro', zero_division=0)
        slot_f1 = f1_score(all_slot_labels, all_slot_preds, average='micro', zero_division=0)
        overall_f1 = (intent_f1 + slot_f1) / 2
        
        return avg_loss, intent_acc, intent_f1, slot_f1, overall_f1
    
    def train(self):
        """完整训练流程"""
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 
            'val_intent_f1': [], 'val_slot_f1': [], 'val_overall_f1': []
        }
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_intent_f1, val_slot_f1, val_overall_f1 = self.evaluate(self.val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_intent_f1'].append(val_intent_f1)
            history['val_slot_f1'].append(val_slot_f1)
            history['val_overall_f1'].append(val_overall_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"Intent F1: {val_intent_f1:.4f}, Slot F1: {val_slot_f1:.4f}, Overall F1: {val_overall_f1:.4f}")
            
            # 保存最佳模型
            if val_overall_f1 > self.best_val_f1:
                self.best_val_f1 = val_overall_f1
                torch.save(self.model.state_dict(), "best_model.pt")
                self.patience_counter = 0
                print(f"  ✓ Best model saved (Overall F1: {val_overall_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return history