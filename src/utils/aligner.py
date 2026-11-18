# -*- coding: utf-8 -*-
"""
槽位对齐工具：将槽位值匹配到句子中的位置
修复版：支持多词槽位、大小写、英文缩写
"""

import re
from difflib import SequenceMatcher


def simple_tokenize(text, lang="mixed"):
    """
    改进的分词函数
    - 英文：按空格分词，保留缩写（如 I'd, don't）
    - 中文：按字分词
    - 混合：英文按词，中文按字
    """
    if lang == "en":
        # 英文：按空格分词，保留标点但合并缩写
        tokens = []
        for token in text.split():
            # 保留缩写形式（如 I'd, don't, can't）
            # 但去掉其他标点
            token = re.sub(r'[^\w\']', '', token)
            if token:
                tokens.append(token)
        return tokens
    
    elif lang == "zh":
        # 中文：按字分词
        return list(text)
    
    else:  # mixed
        tokens = []
        for token in text.split():
            # 检查是否包含中文
            if any('\u4e00' <= c <= '\u9fff' for c in token):
                # 有中文，按字分
                tokens.extend(list(token))
            else:
                # 英文，清理标点
                token = re.sub(r'[^\w\']', '', token)
                if token:
                    tokens.append(token)
        return tokens


def normalize_text(text):
    """文本标准化：小写、去空格"""
    return text.lower().strip()


def fuzzy_match(word1, word2, threshold=0.8):
    """模糊匹配"""
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio() >= threshold


def align_slots(utterance, slots, lang="mixed", fuzzy=False):
    """
    改进的槽位对齐函数
    
    Args:
        utterance: 用户输入句子
        slots: 槽位字典 {"slot_name": "value"}
        lang: 语言类型
        fuzzy: 是否启用模糊匹配
    
    Returns:
        tokens: 分词后的token列表
        slot_labels: BIO标签列表，长度与token数相同
    """
    tokens = simple_tokenize(utterance, lang)
    slot_labels = ["O"] * len(tokens)
    
    # 记录已匹配的位置，避免重复
    matched_positions = set()
    
    for slot_name, slot_value in slots.items():
        if not slot_value:
            continue
        
        # 标准化槽位值
        value_norm = normalize_text(slot_value)
        
        # 将槽位值也分词
        value_tokens = simple_tokenize(slot_value, lang)
        
        # 方法1：多词精确匹配（滑动窗口）
        if len(value_tokens) > 0:
            for i in range(len(tokens) - len(value_tokens) + 1):
                # 检查是否已被匹配
                if any(pos in matched_positions for pos in range(i, i + len(value_tokens))):
                    continue
                
                # 精确匹配
                match = True
                for j in range(len(value_tokens)):
                    if normalize_text(tokens[i + j]) != normalize_text(value_tokens[j]):
                        match = False
                        break
                
                if match:
                    # 标记第一个词
                    slot_labels[i] = f"B-{slot_name}"
                    matched_positions.add(i)
                    # 标记后续词
                    for j in range(1, len(value_tokens)):
                        slot_labels[i + j] = f"I-{slot_name}"
                        matched_positions.add(i + j)
                    break
        
        # 方法2：单词精确匹配（如果多词没匹配到）
        if slot_labels.count(f"B-{slot_name}") == 0:
            for i, token in enumerate(tokens):
                if i in matched_positions:
                    continue
                if normalize_text(token) == value_norm:
                    slot_labels[i] = f"B-{slot_name}"
                    matched_positions.add(i)
                    break
        
        # 方法3：模糊匹配（可选）
        if fuzzy and slot_labels.count(f"B-{slot_name}") == 0:
            for i, token in enumerate(tokens):
                if i in matched_positions:
                    continue
                if fuzzy_match(token, value_norm):
                    slot_labels[i] = f"B-{slot_name}"
                    matched_positions.add(i)
                    break
    
    return tokens, slot_labels


def build_tag_to_id(slots):
    """
    构建BIO标签到ID的映射
    格式: O, B-drink_name, I-drink_name, B-size, I-size, ...
    """
    tag_to_id = {"O": 0}
    idx = 1
    for slot in slots:
        tag_to_id[f"B-{slot}"] = idx
        idx += 1
        tag_to_id[f"I-{slot}"] = idx
        idx += 1
    return tag_to_id


def get_bio_id(tag, tag_to_id):
    """将BIO标签转换为ID"""
    return tag_to_id.get(tag, 0)  # 0 = O