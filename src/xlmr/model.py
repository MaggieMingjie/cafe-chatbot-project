# -*- coding: utf-8 -*-
"""
XLM-R 模型封装
"""

import torch.nn as nn
from transformers import AutoModel, AutoConfig


class XLMRIntentClassifier(nn.Module):
    """XLM-R 意图分类器"""
    
    def __init__(self, model_name, num_intents, dropout=0.1):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_intents
        self.config.hidden_dropout_prob = dropout
        
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_intents)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_intents]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取 [CLS] token 的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits