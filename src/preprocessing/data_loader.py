# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\preprocessing\data_loader.py

import json
import os
from typing import List, Dict, Any, Tuple
import random
from collections import Counter
from src.paths import RAW_DATA_DIR

class CafeDataLoader:
    """咖啡店数据集加载器"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        Args:
            data_path: JSON数据文件路径
        """
        self.data_path = data_path
        self.dialogues = self._load_data()
        self._analyze_data()
        
    def _load_data(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} dialogues")
        return data
    
    def _analyze_data(self):
        """分析数据集统计信息"""
        # 意图统计
        intents = []
        for dialog in self.dialogues:
            for turn in dialog['turns']:
                if turn['speaker'] == 'user':
                    intents.append(turn['intent'])
        
        intent_counts = Counter(intents)
        
        print("\n📊 Dataset Statistics:")
        print(f"Total dialogues: {len(self.dialogues)}")
        print(f"Total user utterances: {len(intents)}")
        print("\nIntent distribution:")
        for intent, count in intent_counts.most_common():
            print(f"  {intent}: {count} ({count/len(intents)*100:.1f}%)")
        
        # 来源模型统计
        sources = Counter([d['source_model'] for d in self.dialogues])
        print("\nSource model distribution:")
        for source, count in sources.most_common():
            print(f"  {source}: {count}")
    
    def get_all_slots(self) -> Dict[str, List[str]]:
        """获取所有槽位的可能取值"""
        slots_dict = {}
        
        for dialog in self.dialogues:
            for turn in dialog['turns']:
                if turn['speaker'] == 'user' and 'slots' in turn:
                    slots = turn['slots']
                    for slot_name, slot_value in slots.items():
                        if slot_name not in slots_dict:
                            slots_dict[slot_name] = set()
                        slots_dict[slot_name].add(str(slot_value))
        
        # 转换为列表
        for key in slots_dict:
            slots_dict[key] = list(slots_dict[key])
        
        return slots_dict
    
    def create_intent_labels(self) -> Dict[str, int]:
        """创建意图标签到ID的映射"""
        intents = set()
        for dialog in self.dialogues:
            for turn in dialog['turns']:
                if turn['speaker'] == 'user':
                    intents.add(turn['intent'])
        
        return {intent: idx for idx, intent in enumerate(sorted(intents))}
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        划分训练集、验证集、测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        random.seed(seed)
        dialogues_copy = self.dialogues.copy()
        random.shuffle(dialogues_copy)
        
        n = len(dialogues_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = dialogues_copy[:train_end]
        val_data = dialogues_copy[train_end:val_end]
        test_data = dialogues_copy[val_end:]
        
        print(f"\n📊 Dataset Split:")
        print(f"Training set: {len(train_data)} dialogues")
        print(f"Validation set: {len(val_data)} dialogues")
        print(f"Test set: {len(test_data)} dialogues")
        
        return train_data, val_data, test_data


# 测试代码
if __name__ == "__main__":
    data_path = str(RAW_DATA_DIR / "cafe_dialogue_en.json")
    loader = CafeDataLoader(data_path)
    
    # 获取槽位信息
    slots = loader.get_all_slots()
    print("\n📦 Slot types:")
    for slot, values in slots.items():
        print(f"  {slot}: {values[:5]}...")
    
    # 意图标签
    intent_labels = loader.create_intent_labels()
    print(f"\n🎯 Intent labels: {intent_labels}")
    
    # 划分数据
    train, val, test = loader.split_data()
