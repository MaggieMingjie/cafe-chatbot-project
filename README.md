# 咖啡店双语对话理解系统

本项目面向咖啡店点单场景，开展中文、英文及中英混合任务型对话的自然语言理解实验。系统支持意图识别与槽位填充，并实现了 mT5、BiLSTM、XLM-R 三类模型的训练、评估与消融实验流程。

## 主要功能

- 支持中文、英文和中英混合三类对话数据
- 联合完成意图分类与槽位填充任务
- 提供 mT5、BiLSTM、XLM-R 三种可复现实验流程
- 提供 mT5 消融实验及结果汇总脚本
- 通过统一路径配置避免依赖本机绝对路径

## 项目结构

```text
cafe_chatbot_project/
├── data/
│   ├── raw/                       # 原始对话与标注数据
│   └── processed/
│       ├── bilstm/                # BiLSTM 使用的预处理数据、词表和标签映射
│       └── xlmr/                  # XLM-R 使用的预处理数据和标签映射
├── src/                           # 可复用的核心实现
│   ├── bilstm/                    # BiLSTM 模型、数据集和训练器
│   ├── xlmr/                      # XLM-R 模型、数据集和训练器
│   ├── model/                     # mT5 联合模型
│   ├── preprocessing/             # 数据加载与处理模块
│   ├── training/                  # mT5 训练工具
│   └── paths.py                   # 全项目统一路径定义
├── experiments/                   # 各模型的预处理、训练和评估入口
├── scripts/                       # 自动化与结果收集脚本
├── artifacts/
│   ├── models/                    # 检查点与训练配置
│   └── results/                   # 评估指标、训练曲线和测试结果
├── reports/generated/             # 自动生成的表格、图表和汇总报告
├── requirements.txt               # Python 依赖
└── pyproject.toml                 # 项目工具配置
```

`data/raw/` 是原始数据来源。`data/processed/`、`artifacts/` 和 `reports/generated/` 中的内容均可通过项目脚本重新生成。

## 环境安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行实验

以下命令均应在项目根目录执行。

```powershell
# mT5 训练
python src/train_en.py
python src/train_zh.py
python src/train_mixed.py

# BiLSTM：预处理、训练与评估
python -m experiments.bilstm.preprocess
python -m experiments.bilstm.train --lang en
python -m experiments.bilstm.evaluate --lang en

# XLM-R：预处理、训练与评估
python -m experiments.xlmr.preprocess
python -m experiments.xlmr.train --lang en
python -m experiments.xlmr.evaluate --lang en

# 消融实验与结果汇总
.\scripts\run_ablation.bat
python scripts\collect_ablation_results.py
python -m src.summarize_results
```

语言参数可选值为 `en`（英文）、`zh`（中文）和 `mixed`（中英混合）。

## 产物命名规范

- 模型检查点：`artifacts/models/<模型>/<语言>/`
- 独立评估指标：`artifacts/results/<模型>/<语言>/evaluation_metrics.json`
- 测试结果：`artifacts/results/<模型>/<语言>/test_results.json`
- 训练过程记录：`training_metrics.json`、`training_metrics.csv`、`training_history.png`
- 可用于论文或汇报的汇总材料：`reports/generated/`

## 开发约定

- 新增路径请从 `src.paths` 导入，禁止写入本机绝对路径。
- 可复用实现放入 `src/`；模型专属命令入口放入 `experiments/`；自动化脚本放入 `scripts/`。
- 不提交 `.venv/`、Python 缓存、模型检查点或自动生成的实验产物。
