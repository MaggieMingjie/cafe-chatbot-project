# Coffee Shop Bilingual Task-Oriented Dialogue Dataset & System

## Project Overview

This project is developed as part of the undergraduate thesis:

**“A Domain-Specific Bilingual Conversational Model for Task-Oriented Dialogue in Café-Based Micro-Domain Service Environments.”**

The project builds a **domain-specific bilingual task-oriented dialogue dataset and an experimental dialogue understanding system** for coffee shop service scenarios.

It aims to study **Natural Language Understanding (NLU)** and dialogue modeling in **micro-domain service environments**, where conversational systems assist users in completing specific tasks such as ordering drinks or modifying orders.

The project contains two main components:

1. **Coffee Shop Dialogue Dataset**
2. **Transformer-based Dialogue Understanding System**

The project demonstrates a complete workflow from:

**dataset construction → annotation → model training → dialogue understanding**

It can be used for research on:

- Task-oriented dialogue systems
- Natural Language Understanding (NLU)
- Domain-specific conversational AI
- Low-resource domain adaptation

---

# Coffee Shop Dialogue Dataset

The **Coffee Shop Dialogue Dataset** is a bilingual task-oriented dialogue dataset designed for **coffee ordering scenarios**.

It is created to support research and experimentation on **intent recognition and slot filling tasks** in task-oriented dialogue systems.

The dialogues simulate common coffee shop interactions, including:

- Ordering drinks
- Modifying orders
- Menu inquiries
- Price inquiries
- Drink recommendations
- Canceling orders
- Greeting interactions

Each dialogue contains:

- **Intent labels**
- **Slot annotations**
- **System responses**
- **Structured JSON format**

The dataset supports the following Natural Language Understanding tasks:

- Intent Classification
- Slot Filling
- Joint NLU Modeling

---

## Dataset Statistics

The Coffee Shop Dialogue Dataset contains **1000 task-oriented dialogues** covering common coffee shop interaction scenarios.

To improve **linguistic diversity** and reduce **model bias**, dialogues were generated using multiple large language models and then **manually reviewed and structured** into the final dataset.

| Source Model | Dialogue Count |
|--------------|---------------|
| ChatGPT | 300 |
| DeepSeek | 300 |
| Doubao (豆包) | 200 |
| Qianwen (千问) | 200 |

**Total Dialogues: 1000**

Each dialogue follows a **3–4 turn structure** and includes:

- User utterances  
- System responses  
- Intent labels  
- Slot annotations  
- Structured JSON formatting  

The dataset is **bilingual**, containing both:

- **English dialogues**
- **Chinese dialogues**

This enables experiments on:

- Multilingual NLU
- Cross-lingual dialogue understanding

---

# Intent Definitions

| Intent | Description |
|------|-------------|
| order | User orders drinks |
| modify_order | Modify an existing order |
| inquire_price | Ask for price |
| inquire_menu | Ask for menu |
| recommend | Request drink recommendation |
| cancel_order | Cancel an order |
| greeting | Greeting |

---

# Slot Definitions

| Slot Name | Description | Meaning |
|-----------|-------------|--------|
| drink_name | Type of drink | Drink name |
| size | Drink size | Cup size |
| temperature | Hot or Iced | Drink temperature |
| milk_type | Type of milk | Milk type |
| sweetness | Sugar level | Sweetness level |
| add_on | Additional topping | Extra toppings |
| quantity | Number of drinks | Quantity |

---

# Data Format (3–4 Turns)

```json
{
  "dialog_id": "en_XXX",
  "language": "en",
  "source_model": "model_name",
  "turns": [
    {
      "speaker": "user",
      "utterance": "User utterance",
      "intent": "intent_type",
      "slots": {
        "drink_name": "Latte",
        "size": "Medium",
        "temperature": "Hot",
        "milk_type": "Oat",
        "sweetness": "Half Sugar",
        "add_on": "Extra Shot"
      }
    },
    {
      "speaker": "system",
      "utterance": "System response"
    }
  ]
}
```

---

# Dialogue Understanding System

To evaluate the usability of the dataset in task-oriented dialogue systems, this project implements a **Transformer-based dialogue understanding system**.

The system performs **joint learning of intent recognition and slot filling** for bilingual task-oriented conversations.

---

## Main Capabilities

- Intent Classification
- Slot Filling
- Joint NLU Modeling

The model is built on the **mT5 Transformer encoder**, combined with **parameter-efficient fine-tuning techniques**.

This design allows the model to effectively learn **semantic intent and slot information simultaneously**.

---

# System Architecture

```text
     User Utterance
           ↓
       mT5 Encoder
           ↓
  Intent Classification Head
      Slot Filling Head
           ↓
   Confidence Threshold
           ↓
 Final Dialogue Understanding
```

The architecture enables **joint intent-slot learning**, which improves dialogue understanding performance in task-oriented scenarios.

The system can also be extended to other **micro-domain service environments**, such as:

- Restaurant ordering
- Hotel reservation
- Customer service assistants

---

# Project Structure

```text
cafe_chatbot_project/
│
├── data/
│   ├── cafe_dialogue_en.json
│   └── cafe_dialogue_zh.json
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   └── data_processor.py
│   │
│   ├── model/
│   │   └── cafe_model.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── train_en.py
│   └── train_zh.py
│
├── models/
│   ├── best_model_en.pt
│   ├── final_model_en.pt
│   ├── best_model_zh.pt
│   └── final_model_zh.pt
│
├── results/
│   ├── en/
│   │   ├── training_history.png
│   │   ├── training_metrics.json
│   │   ├── training_metrics.csv
│   │   └── experiment_summary.txt
│   │
│   └── zh/
│       ├── training_history.png
│       ├── training_metrics.json
│       ├── training_metrics.csv
│       └── experiment_summary.txt
│
└── README.md
```

---

# Environment Setup

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

# Training

Train the English dataset

```bash
python src/train_en.py
```

Train the Chinese dataset

```bash
python src/train_zh.py
```

---

# Project Highlights

- Domain-specific task-oriented dialogue dataset
- Bilingual dialogue data (English + Chinese)
- Joint modeling of **Intent Classification and Slot Filling**
- Structured **JSON dialogue format**
- Transformer-based dialogue understanding system
- Extensible to other **service-oriented dialogue domains**

---

# Future Work

Possible future extensions include:

- Expanding the dataset size
- Introducing **mixed bilingual dialogues (Chinese + English)**
- Applying **larger Transformer models**
- Integrating **dialogue state tracking and policy learning**
