# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\preprocessing\data_processor.py

import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import numpy as np
from src.paths import RAW_DATA_DIR

class CafeDataset(Dataset):
    """咖啡店对话数据集类"""
    
    def __init__(
        self,
        dialogues: List[Dict],
        tokenizer: AutoTokenizer,
        intent2id: Dict[str, int],
        max_length: int = 128,
        is_training: bool = True
    ):
        """
        初始化数据集
        
        Args:
            dialogues: 对话列表
            tokenizer: 分词器
            intent2id: 意图到ID的映射
            max_length: 最大序列长度
            is_training: 是否为训练集
        """
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.intent2id = intent2id
        self.max_length = max_length
        self.is_training = is_training
        
        # 构建样本
        self.samples = self._build_samples()
        print(f"✅ Created {len(self.samples)} training samples")
    
    def _build_samples(self) -> List[Dict]:
        """构建训练样本"""
        samples = []
        
        for dialog in self.dialogues:
            context = []
            
            for turn_idx, turn in enumerate(dialog['turns']):
                if turn['speaker'] == 'user':
                    # 构建输入：将历史上下文和当前用户语句拼接
                    input_text = self._build_context(context, turn['utterance'])
                    
                    # 获取当前轮次的意图和槽位
                    intent = turn['intent']
                    slots = turn.get('slots', {})
                    
                    # 构建槽位标签（这里简化处理，实际需要BIO标注）
                    slot_labels = self._create_slot_labels(turn['utterance'], slots)
                    
                    samples.append({
                        'input_text': input_text,
                        'intent': intent,
                        'slots': slots,
                        'slot_labels': slot_labels,
                        'dialog_id': dialog['dialog_id'],
                        'turn_idx': turn_idx
                    })
                
                # 更新上下文（包括系统回复）
                context.append(f"{turn['speaker']}: {turn['utterance']}")
        
        return samples
    
    def _build_context(self, context: List[str], current_utterance: str) -> str:
        """构建对话历史上下文"""
        # 只保留最近的3轮对话作为上下文
        recent_context = context[-6:] if len(context) > 6 else context
        context_str = " ".join(recent_context)
        
        # 使用特殊标记分隔上下文和当前语句
        return f"{context_str} [SEP] {current_utterance}" if context_str else current_utterance
    
    def _create_slot_labels(self, utterance: str, slots: Dict) -> List[str]:
        """
        创建简单的槽位标签（简化版，实际需要BIO标注）
        这里我们只标记每个词是否属于某个槽位
        """
        words = utterance.split()
        labels = ['O'] * len(words)  # O表示非槽位
        
        # 简化处理：如果某个词出现在槽位值中，标记为对应槽位
        for slot_name, slot_value in slots.items():
            if isinstance(slot_value, str) and slot_value in utterance:
                # 找到槽位值的位置
                value_words = slot_value.split()
                for i, word in enumerate(words):
                    if word in value_words and len(value_words) > 0:
                        labels[i] = f"B-{slot_name}" if i == 0 or labels[i-1] == 'O' else f"I-{slot_name}"
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 对输入文本进行分词
        encoding = self.tokenizer(
            sample['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除批次维度
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent_id': torch.tensor(self.intent2id[sample['intent']], dtype=torch.long),
            'dialog_id': sample['dialog_id'],
            'turn_idx': sample['turn_idx']
        }
        
        return item


class DataCollator:
    """动态批处理数据整理器"""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        batch = {}
        
        # 获取所有键
        keys = features[0].keys()
        
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'intent_id']:
                # 张量类型的键，堆叠处理
                batch[key] = torch.stack([f[key] for f in features])
            else:
                # 其他类型的键，保持为列表
                batch[key] = [f[key] for f in features]
        
        return batch


# 测试代码
if __name__ == "__main__":
    from data_loader import CafeDataLoader
    from transformers import AutoTokenizer  # 或 T5Tokenizer
    
    # 加载数据
    data_path = str(RAW_DATA_DIR / "cafe_dialogue_en.json")
    loader = CafeDataLoader(data_path)
    
    # 创建分词器 - 使用AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    
    # 意图标签
    intent2id = loader.create_intent_labels()
    
    # 划分数据
    train_data, val_data, test_data = loader.split_data()
    
    # 创建数据集
    train_dataset = CafeDataset(train_data, tokenizer, intent2id)
    
    print(f"\n📝 First sample example:")
    sample = train_dataset[0]
    print(f"Input ID shape: {sample['input_ids'].shape}")
    print(f"Intent ID: {sample['intent_id']}")
