# -*- coding: utf-8 -*-
"""
XLM-R 数据集类
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class IntentDataset(Dataset):
    """意图分类数据集"""
    
    def __init__(self, samples, tokenizer, intent_to_id, max_len=128):
        """
        Args:
            samples: 预处理后的样本列表
            tokenizer: XLM-R tokenizer
            intent_to_id: 意图到ID的映射
            max_len: 最大序列长度
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.max_len = max_len
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 获取文本和标签
        text = sample["utterance"]
        intent = sample["intent"]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_id": torch.LongTensor([self.intent_to_id[intent]]).squeeze(0)
        }


def collate_fn(batch):
    """批处理函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    intent_ids = torch.stack([item["intent_id"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "intent_ids": intent_ids
    }