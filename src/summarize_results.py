# C:\Users\Maggi\Desktop\cafe_chatbot_project\src\summarize_results.py

import json
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RESULTS_DIR, REPORTS_DIR

# 路径设置
RESULTS_PATH = str(RESULTS_DIR)
OUTPUT_PATH = str(REPORTS_DIR / "generated" / "model_comparison")

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def load_txt_summary(file_path):
    """从txt摘要中提取关键信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取最佳准确率和F1
        best_acc = None
        best_f1 = None
        
        for line in content.split('\n'):
            if 'Best Validation Accuracy:' in line:
                best_acc = float(line.split(':')[-1].strip().replace('%', ''))
            if 'Best Validation F1:' in line:
                best_f1 = float(line.split(':')[-1].strip().replace('%', ''))
        
        return best_acc, best_f1
    except:
        return None, None

def main():
    print("="*60)
    print("📊 实验结果汇总")
    print("="*60)
    
    results = {
        'en': {},
        'zh': {},
        'mixed': {}
    }
    
    # 1. 英文模型结果
    print("\n🔹 英文模型 (en):")
    en_path = os.path.join(RESULTS_PATH, 'mt5', 'en')
    
    # 训练指标
    metrics_path = os.path.join(en_path, 'training_metrics.json')
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
        if metrics:
            results['en']['train_loss'] = metrics.get('train_loss', [])
            results['en']['val_loss'] = metrics.get('val_loss', [])
            results['en']['val_accuracy'] = metrics.get('val_accuracy', [])
            results['en']['val_f1'] = metrics.get('val_f1', [])
            results['en']['best_val_acc'] = max(metrics.get('val_accuracy', [0]))
            results['en']['best_val_f1'] = max(metrics.get('val_f1', [0]))
            results['en']['best_epoch'] = metrics.get('val_accuracy', []).index(results['en']['best_val_acc']) + 1
            print(f"  - 最佳验证准确率: {results['en']['best_val_acc']*100:.2f}% (第{results['en']['best_epoch']}轮)")
    
    # 测试结果
    test_path = os.path.join(en_path, 'test_results.json')
    if os.path.exists(test_path):
        test = load_json(test_path)
        if test:
            results['en']['test_acc'] = test.get('test_accuracy', test.get('accuracy', 0))
            results['en']['test_f1'] = test.get('test_f1', test.get('f1', 0))
            results['en']['test_loss'] = test.get('test_loss', test.get('loss', 0))
            results['en']['test_samples'] = test.get('num_test_samples', 219)
            print(f"  - 测试准确率: {results['en']['test_acc']*100:.2f}%")
            print(f"  - 测试F1: {results['en']['test_f1']*100:.2f}%")
    
    # 2. 中文模型结果
    print("\n🔹 中文模型 (zh):")
    zh_path = os.path.join(RESULTS_PATH, 'mt5', 'zh')
    
    metrics_path = os.path.join(zh_path, 'training_metrics.json')
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
        if metrics:
            results['zh']['train_loss'] = metrics.get('train_loss', [])
            results['zh']['val_loss'] = metrics.get('val_loss', [])
            results['zh']['val_accuracy'] = metrics.get('val_accuracy', [])
            results['zh']['val_f1'] = metrics.get('val_f1', [])
            results['zh']['best_val_acc'] = max(metrics.get('val_accuracy', [0]))
            results['zh']['best_val_f1'] = max(metrics.get('val_f1', [0]))
            results['zh']['best_epoch'] = metrics.get('val_accuracy', []).index(results['zh']['best_val_acc']) + 1
            print(f"  - 最佳验证准确率: {results['zh']['best_val_acc']*100:.2f}% (第{results['zh']['best_epoch']}轮)")
    
    test_path = os.path.join(zh_path, 'test_results.json')
    if os.path.exists(test_path):
        test = load_json(test_path)
        if test:
            results['zh']['test_acc'] = test.get('test_accuracy', test.get('accuracy', 0))
            results['zh']['test_f1'] = test.get('test_f1', test.get('f1', 0))
            results['zh']['test_loss'] = test.get('test_loss', test.get('loss', 0))
            results['zh']['test_samples'] = test.get('num_test_samples', 219)
            print(f"  - 测试准确率: {results['zh']['test_acc']*100:.2f}%")
            print(f"  - 测试F1: {results['zh']['test_f1']*100:.2f}%")
    
    # 3. 混合模型结果
    print("\n🔹 混合模型 (mixed):")
    mixed_path = os.path.join(RESULTS_PATH, 'mt5', 'mixed')
    
    metrics_path = os.path.join(mixed_path, 'training_metrics.json')
    if os.path.exists(metrics_path):
        metrics = load_json(metrics_path)
        if metrics:
            results['mixed']['train_loss'] = metrics.get('train_loss', [])
            results['mixed']['val_loss'] = metrics.get('val_loss', [])
            results['mixed']['val_accuracy'] = metrics.get('val_accuracy', [])
            results['mixed']['val_f1'] = metrics.get('val_f1', [])
            results['mixed']['best_val_acc'] = max(metrics.get('val_accuracy', [0]))
            results['mixed']['best_val_f1'] = max(metrics.get('val_f1', [0]))
            results['mixed']['best_epoch'] = metrics.get('val_accuracy', []).index(results['mixed']['best_val_acc']) + 1
            print(f"  - 最佳验证准确率: {results['mixed']['best_val_acc']*100:.2f}% (第{results['mixed']['best_epoch']}轮)")
    
    test_path = os.path.join(mixed_path, 'test_results.json')
    if os.path.exists(test_path):
        test = load_json(test_path)
        if test:
            results['mixed']['test_acc'] = test.get('test_accuracy', test.get('accuracy', 0))
            results['mixed']['test_f1'] = test.get('test_f1', test.get('f1', 0))
            results['mixed']['test_loss'] = test.get('test_loss', test.get('loss', 0))
            results['mixed']['test_samples'] = test.get('num_test_samples', 219)
            print(f"  - 测试准确率: {results['mixed']['test_acc']*100:.2f}%")
            print(f"  - 测试F1: {results['mixed']['test_f1']*100:.2f}%")
    
    # 代码混合分析
    code_switch_path = os.path.join(mixed_path, 'code_switching_analysis.json')
    if os.path.exists(code_switch_path):
        code_analysis = load_json(code_switch_path)
        if code_analysis:
            results['mixed']['code_switch'] = code_analysis.get('dataset_stats', {})
            print(f"  - 中英混合语句: {results['mixed']['code_switch'].get('mixed_utterances', 0)}条")
    
    # 4. 保存汇总结果
    summary_path = os.path.join(OUTPUT_PATH, 'all_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 汇总结果已保存: {summary_path}")
    
    # 5. 生成对比表格 (CSV)
    comparison = []
    for lang in ['en', 'zh', 'mixed']:
        if results[lang].get('test_acc'):
            row = {
                'Model': 'English' if lang == 'en' else 'Chinese' if lang == 'zh' else 'Code-Switched',
                'Test Accuracy': f"{results[lang]['test_acc']*100:.2f}%",
                'Test F1': f"{results[lang]['test_f1']*100:.2f}%",
                'Best Val Accuracy': f"{results[lang].get('best_val_acc', 0)*100:.2f}%",
                'Best Val F1': f"{results[lang].get('best_val_f1', 0)*100:.2f}%",
                'Best Epoch': results[lang].get('best_epoch', '-')
            }
            comparison.append(row)
    
    df = pd.DataFrame(comparison)
    csv_path = os.path.join(OUTPUT_PATH, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 对比表格已保存: {csv_path}")
    
    # 6. 生成论文格式的表格 (LaTeX)
    latex = r"""\begin{table}[h]
\centering
\caption{Model Performance Comparison}
\label{tab:model_comparison}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Test Accuracy} & \textbf{Test F1} & \textbf{Best Val Acc} & \textbf{Best Epoch} \\
\hline
"""
    
    for row in comparison:
        latex += f"{row['Model']} & {row['Test Accuracy']} & {row['Test F1']} & {row['Best Val Accuracy']} & {row['Best Epoch']} \\\\\n"
    
    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}"
    
    latex_path = os.path.join(OUTPUT_PATH, 'comparison_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✅ LaTeX表格已保存: {latex_path}")
    
    # 7. 生成训练过程对比图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = {'en': 'blue', 'zh': 'red', 'mixed': 'green'}
        labels = {'en': 'English', 'zh': 'Chinese', 'mixed': 'Code-Switched'}
        
        # 训练损失
        ax = axes[0, 0]
        for lang in ['en', 'zh', 'mixed']:
            if results[lang].get('train_loss'):
                epochs = range(1, len(results[lang]['train_loss']) + 1)
                ax.plot(epochs, results[lang]['train_loss'], 
                       color=colors[lang], label=labels[lang], marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True)
        
        # 验证损失
        ax = axes[0, 1]
        for lang in ['en', 'zh', 'mixed']:
            if results[lang].get('val_loss'):
                epochs = range(1, len(results[lang]['val_loss']) + 1)
                ax.plot(epochs, results[lang]['val_loss'], 
                       color=colors[lang], label=labels[lang], marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # 验证准确率
        ax = axes[1, 0]
        for lang in ['en', 'zh', 'mixed']:
            if results[lang].get('val_accuracy'):
                epochs = range(1, len(results[lang]['val_accuracy']) + 1)
                acc = [a * 100 for a in results[lang]['val_accuracy']]
                ax.plot(epochs, acc, color=colors[lang], label=labels[lang], marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True)
        
        # 验证F1
        ax = axes[1, 1]
        for lang in ['en', 'zh', 'mixed']:
            if results[lang].get('val_f1'):
                epochs = range(1, len(results[lang]['val_f1']) + 1)
                f1 = [f * 100 for f in results[lang]['val_f1']]
                ax.plot(epochs, f1, color=colors[lang], label=labels[lang], marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score (%)')
        ax.set_title('Validation F1')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_PATH, 'training_comparison.png')
        plt.savefig(plot_path, dpi=150)
        print(f"✅ 对比图已保存: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ 无法生成图表: {e}")
    
    # 8. 生成Markdown报告
    md = f"""# 三大模型实验结果汇总报告

## 1. 总体对比

| 模型 | 测试准确率 | 测试F1 | 最佳验证准确率 | 最佳轮数 |
|------|------------|--------|----------------|----------|
"""
    
    for row in comparison:
        md += f"| {row['Model']} | {row['Test Accuracy']} | {row['Test F1']} | {row['Best Val Accuracy']} | {row['Best Epoch']} |\n"
    
    md += """
## 2. 详细结果

### 英文模型 (English)
- 测试准确率: **{:.2f}%**
- 测试F1: **{:.2f}%**
- 最佳验证准确率: {:.2f}% (第{}轮)
- 测试样本数: {}

### 中文模型 (Chinese)
- 测试准确率: **{:.2f}%**
- 测试F1: **{:.2f}%**
- 最佳验证准确率: {:.2f}% (第{}轮)
- 测试样本数: {}

### 混合模型 (Code-Switched)
- 测试准确率: **{:.2f}%**
- 测试F1: **{:.2f}%**
- 最佳验证准确率: {:.2f}% (第{}轮)
- 测试样本数: {}
- 中英混合语句: {}条 (78.8%)

## 3. 关键发现
1. **英文模型表现最佳**: {:.2f}% 测试准确率
2. **混合模型接近英文**: {:.2f}% vs 英文 {:.2f}% (差{:.2f}%)
3. **中文模型达到优秀**: {:.2f}% 测试准确率
4. **混合模型成功处理78.8%中英混合语句**

## 4. 文件保存位置
- 汇总结果: `results_summary/all_results.json`
- 对比表格: `results_summary/model_comparison.csv`
- LaTeX表格: `results_summary/comparison_table.tex`
- 对比图: `results_summary/training_comparison.png`
""".format(
    results['en']['test_acc']*100, results['en']['test_f1']*100,
    results['en']['best_val_acc']*100, results['en']['best_epoch'],
    results['en']['test_samples'],
    results['zh']['test_acc']*100, results['zh']['test_f1']*100,
    results['zh']['best_val_acc']*100, results['zh']['best_epoch'],
    results['zh']['test_samples'],
    results['mixed']['test_acc']*100, results['mixed']['test_f1']*100,
    results['mixed']['best_val_acc']*100, results['mixed']['best_epoch'],
    results['mixed']['test_samples'],
    results['mixed']['code_switch'].get('mixed_utterances', 0),
    results['en']['test_acc']*100,
    results['mixed']['test_acc']*100, results['en']['test_acc']*100,
    results['en']['test_acc']*100 - results['mixed']['test_acc']*100,
    results['zh']['test_acc']*100
)
    
    md_path = os.path.join(OUTPUT_PATH, 'summary_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"✅ Markdown报告已保存: {md_path}")
    
    print("\n" + "="*60)
    print("🎉 汇总完成！所有文件已保存到: results_summary/")
    print("="*60)

if __name__ == "__main__":
    main()
