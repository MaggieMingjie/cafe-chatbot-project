# -*- coding: utf-8 -*-
"""
XLM-R 配置文件
"""

# 模型参数
MODEL_NAME = "xlm-roberta-base"  # 或 xlm-roberta-large
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
PATIENCE = 3  # 早停耐心值

# 数据参数
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 意图列表
INTENTS = [
    "order",
    "modify_order",
    "inquire_price",
    "inquire_menu",
    "recommend",
    "cancel_order",
    "greeting"
]

from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR

# Portable project paths
DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DIR = str(PROCESSED_DATA_DIR / "xlmr")
RESULTS_DIR = str(RESULTS_DIR / "xlmr")
