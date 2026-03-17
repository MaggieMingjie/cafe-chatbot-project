# Coffee Shop Bilingual Task-Oriented Dialogue Dataset & System

## Project Overview
This project is developed as part of the undergraduate thesis:
“A Domain-Specific Bilingual Conversational Model for Task-Oriented Dialogue in Café-Based Micro-Domain Service Environments.”

The project builds a domain-specific bilingual task-oriented dialogue dataset and an experimental dialogue understanding system for coffee shop service scenarios.

It aims to study Natural Language Understanding (NLU) and dialogue modeling in micro-domain service environments, where conversational systems assist users in completing specific tasks such as ordering drinks or modifying orders.

The project contains two main components:
    Coffee Shop Dialogue Dataset (English, Chinese, and Code-Switched versions)
    Transformer-based Dialogue Understanding System

## Coffee Shop Dialogue Dataset
The Coffee Shop Dialogue Dataset is a bilingual task-oriented dialogue dataset designed for coffee ordering scenarios, with three versions:
    English-only dialogues (cafe_dialogue_en.json)
    Chinese-only dialogues (cafe_dialogue_zh.json)
    Code-Switched dialogues (cafe_dialogue_mixed.json) - 中英混合对话

The dialogues simulate common coffee shop interactions, including:
    Ordering drinks
    Modifying orders
    Menu inquiries
    Price inquiries
    Drink recommendations
    Canceling orders
    Greeting interactions

Each dialogue contains:
    Intent labels
    Slot annotations (BIO format)
    System responses
    Structured JSON format
    Language tags (en/zh/mixed)

## Dataset Statistics
Dataset     	Dialogues	User     Utterances    Language
English	        1000	    1487	 100%          English
Chinese	        1000	    1487	 100%          Chinese
Code-Switched 	1000	    1487	 78.8%         mixed utterances

## Source Model Distribution
Source Model	  Dialogue Count
ChatGPT	          300
DeepSeek	        300
Doubao (豆包)     200
Qianwen (千问)	  200
Total Dialogues: 1000 per language version

## Intent Definitions
Intent         	Description
order	         User orders drinks
modify_order	 Modify an existing order
inquire_price	 Ask for price
inquire_menu	 Ask for menu
recommend	     Request drink recommendation
cancel_order	 Cancel an order
greeting	     Greeting

## Slot Definitions
Slot Name	        Description	          Example Values
drink_name	      Type of drink	        Latte, 美式, 拿铁, Mocha
size	            Drink size	          Small, Medium, Large, 大杯, 中杯
temperature	      Hot or Iced	          Hot, Iced, 热, 冰
milk_type	        Type of milk	        Oat, Whole, 燕麦奶, 全脂
sweetness      	  Sugar level	          No sugar, Half sugar, 无糖, 半糖
add_on	          Additional toppings   Extra shot, Vanilla syrup, 香草糖浆
quantity	        Number of drinks      1, 2, 3, one, two

## Data Format Examples
### English Example
{
  "dialog_id": "en_001",
  "language": "en",
  "source_model": "chatgpt",
  "turns": [
    {
      "speaker": "user",
      "utterance": "I'd like a medium vanilla latte",
      "intent": "order",
      "slots": {
        "drink_name": "latte",
        "size": "medium",
        "add_on": "vanilla syrup"
      }
    },
    {
      "speaker": "system",
      "utterance": "Sure. Do you want it hot or iced?"
    }
  ]
}

### Code-Switched Example
{
  "dialog_id": "mixed_001",
  "language": "mixed",
  "source_model": "chatgpt",
  "turns": [
    {
      "speaker": "user",
      "utterance": "我想要一杯Medium冰拿铁，Half Sugar。",
      "intent": "order",
      "slots": {
        "drink_name": "拿铁",
        "size": "Medium",
        "temperature": "冰",
        "sweetness": "Half Sugar"
      }
    },
    {
      "speaker": "system",
      "utterance": "好的，一杯Medium冰拿铁，Half Sugar。还需要别的吗？"
    }
  ]
}

## Dialogue Understanding System
The system performs joint learning of intent recognition and slot filling for bilingual and code-switched task-oriented conversations.

## Main Capabilities
Intent Classification (7 intents)
Slot Filling (20 slot labels, BIO format)
Joint NLU Modeling
Code-Switched Dialogue Understanding (中英混合)
Confidence-based prediction with rule-based fallback

## Model Architecture
Base Model: mT5-small (multilingual T5 encoder)
Fine-tuning: LoRA (Low-Rank Adaptation) with rank=16, alpha=32
Joint Loss: L = L_intent + 1.2 × L_slot
Confidence Threshold: 0.7 for rule-based fallback

     User Utterance (English/Chinese/Code-Switched)
           ↓
       mT5 Encoder (Multilingual)
           ↓
  ┌──────────┴──────────┐
  ↓                     ↓
Intent Head        Slot Head
(Classification)   (BIO Tagging)
  ↓                     ↓
  └──────────┬──────────┘
           ↓
   Joint Loss Function
   (L_intent + 1.2 × L_slot)
           ↓
   Confidence-based Prediction
           ↓
 Final Dialogue Understanding

## Experimental Results
### Model Performance Comparison
Model	          Dataset	Test Accuracy	     Test       F1	      Key Feature
English      	  cafe_dialogue_en.json	     95.89%	    95.88%	  Pure English
Code-Switched	  cafe_dialogue_mixed.json   94.98%	    94.94%	  中英混合理解
Chinese	        cafe_dialogue_zh.json	     91.78%	    91.11%	  Pure Chinese

## Key Findings
English model achieves highest accuracy (95.89%) with strong generalization
Code-Switched model successfully handles 78.8% mixed utterances at 94.98% accuracy
Chinese model reaches 91.78% accuracy with stable training
All models converge within 10 epochs with no overfitting
Mixed model demonstrates strong cross-lingual generalization

## Project Structure
cafe_chatbot_project/
│
├── data/
│   ├── cafe_dialogue_en.json
│   ├── cafe_dialogue_zh.json
│   └── cafe_dialogue_mixed.json
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   └── data_processor.py
│   ├── model/
│   │   └── cafe_model.py
│   ├── training/
│   │   └── trainer.py
│   ├── train_en.py
│   ├── train_zh.py
│   └── train_mixed.py
│
├── models/
│   ├── best_model_en.pt
│   ├── final_model_en.pt
│   ├── best_model_zh.pt
│   ├── final_model_zh.pt
│   ├── best_model_mixed.pt
│   └── final_model_mixed.pt
│
├── results/
│   ├── en/
│   ├── zh/
│   └── mixed/
│
└── README.md

## Environment Setup
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Training
### Train English model
    python src/train_en.py

### Train Chinese model
    python src/train_zh.py

### Train Code-Switched model
    python src/train_mixed.py

All training scripts support configurable hyperparameters:
    python src/train_mixed.py --batch_size 16 --num_epochs 10 --learning_rate 2e-4

## Project Highlights
Three dataset versions: English, Chinese, and Code-Switched
Joint intent-slot modeling with mT5 and LoRA
Code-switched dialogue support (中英混合)
State-of-the-art results: 95.89% (EN), 94.98% (Mixed), 91.78% (ZH)
Confidence-based hybrid architecture with rule fallback
Comprehensive evaluation and visualization

## Future Work
Expand dataset to 5000+ dialogues
Integrate multi-turn dialogue state tracking
Apply larger Transformer models (mT5-base, mT5-large)
Add dialogue policy learning for end-to-end systems
Conduct real-world deployment and user studies
