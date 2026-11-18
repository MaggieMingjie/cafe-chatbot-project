# analyze_slots.py
import json
import sys
import os
from collections import Counter
from experiments.bilstm.config import PROCESSED_DIR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 加载数据
with open(os.path.join(PROCESSED_DIR, "en", "train_samples.json"), 'r', encoding='utf-8') as f:
    samples = json.load(f)

print("="*60)
print("槽位详细分析")
print("="*60)

# 统计每个槽位的出现次数
slot_counts = Counter()
for sample in samples:
    for tag in sample['slot_labels']:
        if tag != "O":
            slot_counts[tag] += 1

print("\n各槽位标签出现次数:")
for tag, count in slot_counts.most_common():
    print(f"  {tag}: {count}")

# 统计每个样本的槽位数
slot_per_sample = []
for sample in samples:
    slot_count = sum(1 for tag in sample['slot_labels'] if tag != "O")
    slot_per_sample.append(slot_count)

print(f"\n每个样本的平均槽位数: {sum(slot_per_sample)/len(slot_per_sample):.2f}")
print(f"有槽位的样本数: {sum(1 for c in slot_per_sample if c > 0)}/{len(slot_per_sample)}")
print(f"槽位最多的样本: {max(slot_per_sample)} 个槽位")

# 找出有多个槽位的样本示例
print("\n示例：有多个槽位的样本")
count = 0
for sample in samples:
    non_o = [t for t in sample['slot_labels'] if t != "O"]
    if len(non_o) >= 5 and count < 5:
        print(f"  句子: {' '.join(sample['tokens'])}")
        print(f"  槽位: {non_o}")
        print()
        count += 1

# 检查槽位是否被正确标注
print("\n检查槽位标注质量:")
correct = 0
total_slot_values = 0
for sample in samples:
    tokens = sample['tokens']
    slot_labels = sample['slot_labels']
    # 检查是否有连续的 I-标签
    for i in range(len(slot_labels)-1):
        if slot_labels[i].startswith("B-") and slot_labels[i+1].startswith("I-"):
            correct += 1
        if slot_labels[i].startswith("I-"):
            total_slot_values += 1

print(f"  连续的 B-I 对数量: {correct}")
print(f"  I-标签总数: {total_slot_values}")
