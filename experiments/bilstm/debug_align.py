# debug_align.py
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.aligner import align_slots, simple_tokenize
from experiments.bilstm.config import PROCESSED_DIR

# 加载预处理后的样本
try:
    with open(os.path.join(PROCESSED_DIR, "en", "train_samples.json"), 'r', encoding='utf-8') as f:
        samples = json.load(f)
except FileNotFoundError:
    print("Please run preprocess.py first!")
    sys.exit(1)

print("="*60)
print("槽位对齐检查")
print("="*60)

# 检查前10个样本
for i, sample in enumerate(samples[:10]):
    print(f"\n样本 {i+1}:")
    print(f"  原句: {' '.join(sample['tokens'])}")
    print(f"  意图: {sample['intent']}")
    print(f"  槽位标签: {sample['slot_labels']}")
    
    # 统计非O标签
    non_o = [t for t in sample['slot_labels'] if t != "O"]
    if non_o:
        print(f"  ✓ 有 {len(non_o)} 个非O标签: {non_o}")
    else:
        print(f"  ❌ 没有非O标签")
    print("-"*40)

# 统计槽位标签分布
tag_counter = Counter()
for sample in samples:
    for tag in sample['slot_labels']:
        tag_counter[tag] += 1

print("\n槽位标签分布:")
for tag, count in tag_counter.most_common(20):
    print(f"  {tag}: {count}")

# 统计非O标签的比例
total_tags = sum(tag_counter.values())
non_o_tags = total_tags - tag_counter.get("O", 0)
print(f"\n非O标签比例: {non_o_tags/total_tags*100:.2f}%")

# 统计I-标签数量
i_tags = sum(v for k, v in tag_counter.items() if k.startswith("I-"))
print(f"I-标签数量: {i_tags}")

print("\n" + "="*60)
if non_o_tags/total_tags*100 > 5 and i_tags > 0:
    print("✅ 槽位对齐效果良好，可以开始训练")
else:
    print("⚠️ 槽位对齐可能有问题，请检查数据格式")
