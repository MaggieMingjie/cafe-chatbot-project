# -*- coding: utf-8 -*-
"""
BiLSTM模型配置文件
"""

# 模型参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 5  # 早停耐心值
LAMBDA = 1.2  # Slot Loss权重

# 数据参数
MAX_SEQ_LEN = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR

# Portable project paths
DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DIR = str(PROCESSED_DATA_DIR / "bilstm")
RESULTS_DIR = str(RESULTS_DIR / "bilstm")

# 意图和槽位映射
INTENTS = [
    "order", "modify_order", "inquire_price", 
    "inquire_menu", "recommend", "cancel_order", "greeting"
]

SLOTS = [
    "drink_name", "size", "temperature", "milk_type", 
    "sweetness", "add_on", "quantity"
]

# BIO标签
BIO_TAGS = ["O"] + [f"B-{slot}" for slot in SLOTS] + [f"I-{slot}" for slot in SLOTS]
