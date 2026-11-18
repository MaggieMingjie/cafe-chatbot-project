# C:\Users\Maggi\Desktop\cafe_chatbot_project\collect_ablation_results.py

import os
import json
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import REPORTS_DIR, RESULTS_DIR

def collect_results(base_path):
    """收集所有消融实验结果"""
    
    variants = {
        'full_model': 'Full model',
        'variant1_no_lora': 'Without LoRA',
        'variant2_no_joint_loss': 'Without joint loss',
        'variant3_no_threshold': 'Without confidence threshold'
    }
    
    results = []
    
    print("\n" + "="*60)
    print("Collecting Ablation Study Results")
    print("="*60)
    
    for folder_name, description in variants.items():
        results_path = os.path.join(base_path, folder_name, 'mixed', 'test_results.json')
        
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            overall = data.get('overall', {})
            ablation_config = data.get('ablation_config', {})
            
            results.append({
                'Model Variant': description,
                'Component Removed/Modified': folder_name.replace('_', ' '),
                'Test Accuracy (%)': round(overall.get('accuracy', data.get('test_intent_accuracy', 0)) * 100, 2),
                'Test F1 (%)': round(overall.get('f1', data.get('test_intent_f1', 0)) * 100, 2),
                'Slot F1 (%)': round(overall.get('slot_f1', data.get('test_slot_f1', 0)) * 100, 2),
                'use_lora': ablation_config.get('use_lora', 'N/A'),
                'joint_loss_lambda': ablation_config.get('joint_loss_lambda', 'N/A'),
                'confidence_threshold': ablation_config.get('confidence_threshold', 'N/A')
            })
            print(f"[OK] Collected: {description}")
        else:
            print(f"[WARN] Results not found: {results_path}")
    
    if len(results) == 0:
        print("\n[ERROR] No results found. Please run ablation experiments first.")
        return None
    
    # 创建表格
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Ablation Study Results Summary")
    print("="*60)
    print(df[['Model Variant', 'Test Accuracy (%)', 'Test F1 (%)', 'Slot F1 (%)']].to_string(index=False))
    
    # 计算性能变化
    full_model = df[df['Model Variant'] == 'Full model']
    if len(full_model) > 0:
        baseline_acc = full_model['Test Accuracy (%)'].values[0]
        baseline_f1 = full_model['Test F1 (%)'].values[0]
        
        print("\n" + "="*60)
        print("Performance Change Relative to Full Model")
        print("="*60)
        
        for _, row in df.iterrows():
            if row['Model Variant'] != 'Full model':
                acc_change = row['Test Accuracy (%)'] - baseline_acc
                f1_change = row['Test F1 (%)'] - baseline_f1
                print(f"{row['Model Variant']}:")
                print(f"  Accuracy: {row['Test Accuracy (%)']:.2f}% ({acc_change:+.2f}%)")
                print(f"  F1: {row['Test F1 (%)']:.2f}% ({f1_change:+.2f}%)")
    
    output_dir = REPORTS_DIR / "generated" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save publication-ready summaries outside the raw experiment artifacts.
    output_path = output_dir / 'ablation_results.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Results saved to: {output_path}")
    
    # 保存到JSON
    json_path = output_dir / 'ablation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to: {json_path}")
    
    # 生成LaTeX表格代码
    latex_table = generate_latex_table(df)
    latex_path = output_dir / 'ablation_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"[OK] LaTeX table saved to: {latex_path}")
    
    return df


def generate_latex_table(df):
    """生成LaTeX三线表代码"""
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{消融实验结果 (mT5-Bilingual模型)}
\\label{tab:ablation}
\\begin{tabular}{lccc}
\\hline
模型变体 & 测试准确率 & 测试F1 & 槽位F1 \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        variant = row['Model Variant']
        acc = f"{row['Test Accuracy (%)']:.2f}\\%"
        f1 = f"{row['Test F1 (%)']:.2f}\\%"
        slot_f1 = f"{row['Slot F1 (%)']:.2f}\\%"
        latex += f"{variant} & {acc} & {f1} & {slot_f1} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


if __name__ == "__main__":
    base_path = str(RESULTS_DIR / "ablation")
    df = collect_results(base_path)
    
    if df is not None:
        print("\n" + "="*60)
        print("Next steps:")
        print("1. Check reports/generated/ablation/ablation_results.csv")
        print("2. Copy reports/generated/ablation/ablation_table.tex into the paper")
        print("3. Paste into your paper's Results and Discussion section")
        print("="*60)
    
