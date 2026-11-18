import argparse
import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from src.paths import MODELS_DIR, RAW_DATA_DIR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/mt5-small")
    parser.add_argument("--language", default="mixed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--joint_loss_lambda", type=float, default=0.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--ablation_name", required=True)
    args = parser.parse_args()

    set_seed(42)
    print(f"🎲 Random seed set to: 42")

    model_dir = str(MODELS_DIR / "ablation" / args.ablation_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"📁 Model save path: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = load_dataset("json", data_files=str(RAW_DATA_DIR / "cafe_dialogue_mixed.json"))
    dataset = data["train"]

    intent_labels = ['cancel_order', 'greeting', 'inquire_menu', 'inquire_price', 'modify_order', 'order', 'recommend']
    intent2id = {i: idx for idx, i in enumerate(intent_labels)}
    num_intents = len(intent_labels)
    use_lora = not args.no_lora

    from src.model.cafe_model import CafeDialogueModel
    model = CafeDialogueModel(
        model_name=args.model_name,
        num_intents=num_intents,
        use_lora=use_lora
    )

    train_conf = {
        "base_model": args.model_name,
        "use_lora": use_lora,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs
    }
    conf_path = os.path.join(model_dir, "training_config_ablation.json")
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump(train_conf, f, indent=2, ensure_ascii=False)

    from src.training.trainer_ablation import CafeTrainerAblation
    trainer = CafeTrainerAblation(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        intent2id=intent2id,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        ablation_name=args.ablation_name,
        model_dir=model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.run()

if __name__ == "__main__":
    main()
