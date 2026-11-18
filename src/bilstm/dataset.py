# -*- coding: utf-8 -*-
"""
数据集类
"""

import torch
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    """对话数据集"""
    
    def __init__(self, samples, word_to_id, tag_to_id, intent_to_id, max_len=50):
        """
        Args:
            samples: 预处理后的样本列表
            word_to_id: 词到ID的映射
            tag_to_id: BIO标签到ID的映射
            intent_to_id: 意图到ID的映射
            max_len: 最大序列长度
        """
        self.samples = samples
        self.word_to_id = word_to_id
        self.tag_to_id = tag_to_id
        self.intent_to_id = intent_to_id
        self.max_len = max_len
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        tokens = sample["tokens"]
        intent = sample["intent"]
        slot_labels = sample["slot_labels"]
        
        # 转换为ID
        token_ids = [self.word_to_id.get(token, self.word_to_id.get("<UNK>", 1)) 
                     for token in tokens]
        slot_ids = [self.tag_to_id.get(tag, 0) for tag in slot_labels]
        intent_id = self.intent_to_id[intent]
        
        # 截断或填充
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            slot_ids = slot_ids[:self.max_len]
        else:
            pad_len = self.max_len - len(token_ids)
            token_ids = token_ids + [0] * pad_len
            slot_ids = slot_ids + [0] * pad_len
        
        return {
            "token_ids": torch.LongTensor(token_ids),
            "slot_ids": torch.LongTensor(slot_ids),
            "intent_id": torch.LongTensor([intent_id]),
            "length": len(tokens[:self.max_len])
        }


def collate_fn(batch):
    """批处理函数"""
    token_ids = torch.stack([item["token_ids"] for item in batch])
    slot_ids = torch.stack([item["slot_ids"] for item in batch])
    intent_ids = torch.cat([item["intent_id"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])
    
    return {
        "token_ids": token_ids,
        "slot_ids": slot_ids,
        "intent_ids": intent_ids,
        "lengths": lengths
    }