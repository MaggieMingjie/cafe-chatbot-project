# -*- coding: utf-8 -*-
"""
BiLSTM模型定义（参考Paper1）
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """BiLSTM模型，联合训练意图分类和槽位填充"""
    
    def __init__(self, vocab_size, num_intents, num_slots, 
                 embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 意图分类头
        self.intent_classifier = nn.Linear(hidden_dim * 2, num_intents)
        
        # 槽位填充头
        self.slot_classifier = nn.Linear(hidden_dim * 2, num_slots)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch_size, seq_len]
            lengths: [batch_size] 实际长度（可选）
        
        Returns:
            intent_logits: [batch_size, num_intents]
            slot_logits: [batch_size, seq_len, num_slots]
        """
        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)
        
        # 如果提供了lengths，使用pack_padded_sequence
        if lengths is not None:
            # 确保lengths在CPU上且类型正确
            lengths = lengths.cpu().long()
            # 按长度降序排序
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sorted_idx]
            
            # Pack序列
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            packed_out, (hidden, cell) = self.lstm(packed)
            
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            
            # 恢复原始顺序
            _, unsorted_idx = torch.sort(sorted_idx)
            lstm_out = lstm_out[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一层双向隐藏状态用于意图分类
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        # 取最后一层（-2和-1）的双向状态
        last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # 意图分类
        intent_logits = self.intent_classifier(last_hidden)
        
        # 槽位填充
        slot_logits = self.slot_classifier(lstm_out)
        
        return intent_logits, slot_logits