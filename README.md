# Coffee Shop Bilingual Task-Oriented Dialogue Dataset & System

## Project Overview

This project builds a **task-oriented dialogue dataset and experimental system** for the coffee shop domain, aiming to study **Natural Language Understanding (NLU)** and dialogue management in micro-domain service scenarios.

The project contains two core components:

1. **Coffee Shop Dialogue Dataset**
2. **Transformer-based Dialogue Understanding System**

The project demonstrates the complete workflow from **dataset construction → annotation → model training → dialogue understanding**, and can be used for research on **task-oriented dialogue systems, NLU tasks, and model adaptation in low-resource domains**.

---

# Coffee Shop Dialogue Dataset

The **Coffee Shop Dialogue Dataset** is a task-oriented dialogue dataset designed for coffee ordering scenarios.  
It is used to train and evaluate Natural Language Understanding models.

The dialogues simulate real coffee shop interactions such as:

- Ordering drinks
- Modifying orders
- Menu inquiry
- Price inquiry
- Drink recommendation
- Canceling orders
- Greeting

Each dialogue contains:

- **Intent labels**
- **Slot annotations**
- **Structured JSON format**

The dataset can be used for the following tasks:

- Intent Classification
- Slot Filling
- Joint NLU Modeling

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

## Dialogue Understanding System

To evaluate the usability of the dataset in task-oriented dialogue systems, this project implements a **Transformer-based dialogue understanding system**.

### Main Capabilities

- Intent Classification
- Slot Filling
- Joint NLU Modeling

The model is built on the **mT5 Transformer encoder**, combined with **parameter-efficient fine-tuning techniques**.

---

## System Architecture
```
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

This architecture performs **joint learning of intent recognition and slot filling**, and can be extended to other task-oriented dialogue scenarios.

---

## Project Structure
```
cafe_chatbot_project/
│
├── data/
│ ├── cafe_dialogue_en.json
│ └── cafe_dialogue_zh.json
│
├── src/
│ ├── preprocessing/
│ │ ├── data_loader.py
│ │ └── data_processor.py
│ │
│ ├── model/
│ │ └── cafe_model.py
│ │
│ ├── training/
│ │ └── trainer.py
│ │
│ ├── train_en.py
│ └── train_zh.py
│
├── models/
|   ├── best_model_en.pt
|   ├── final_model_en.pt
|   ├── best_model_zh.pt
|   └── final_model_zh.pt
│
├── results/
|   ├── en/
|   │   ├── training_history.png
|   │   ├── training_metrics.json
|   │   ├── training_metrics.csv
|   │   └── experiment_summary.txt
|   └── zh/
|       ├── training_history.png
|       ├── training_metrics.json
|       ├── training_metrics.csv
|       └── experiment_summary.txt
│
└── README.md
```

---

## Environment Setup
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Training
### Train the English dataset
python src/train_en.py
### Train the Chinese dataset
python src/train_zh.py

## Project Highlights
Domain-specific task-oriented dialogue dataset
Joint modeling of Intent + Slot
Structured JSON dialogue format
Transformer-based dialogue understanding system
Easily extendable to other service domains (restaurant, hotel, etc.)

## Future Work
Future extensions may include:
Expanding dataset size
Introducing mixed bilingual dialogue (Chinese + English)
Applying larger Transformer models
