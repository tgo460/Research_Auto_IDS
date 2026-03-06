import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_metric_comparison(report_data, output_dir):
    """Compares Light vs Cascade Metrics"""
    metrics = ['f1', 'precision', 'recall', 'fpr']
    
    data = []
    # Light
    for m in metrics:
        if m in report_data['light_only']:
            data.append({
                'Model': 'Light Only',
                'Metric': m.upper(),
                'Score': report_data['light_only'][m]
            })
    
    # Cascade
    for m in metrics:
        if m in report_data['cascade']:
            data.append({
                'Model': 'Cascade (Final)',
                'Metric': m.upper(),
                'Score': report_data['cascade'][m]
            })
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Performance Comparison: Light vs Cascade Architecture', fontsize=15)
    plt.ylim(0, 1.1)
    
    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metric_comparison.png')
    plt.savefig(save_path)
    print(f"Saved metric comparison to {save_path}")
    plt.close()

def plot_confusion_matrices(report_data, output_dir):
    """Plots Confusion Matrices for Light and Cascade"""
    
    cm_light = np.array(report_data['light_only'].get('confusion_matrix'))
    cm_cascade = np.array(report_data['cascade'].get('confusion_matrix'))
    
    if cm_light.size == 0 or cm_cascade.size == 0:
        print("Skipping Confusion Matrix plot: No matrix data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Labels
    labels = ['Normal', 'Attack']
    
    # Light Plot
    sns.heatmap(cm_light, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title('Light Model Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Cascade Plot
    sns.heatmap(cm_cascade, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title('Cascade System Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(save_path)
    print(f"Saved confusion matrices to {save_path}")
    plt.close()

def plot_ablation_study(ablation_data, output_dir):
    """Plots Baseline vs Engineered Features comparison"""
    if not ablation_data:
        return

    metrics = ['f1', 'precision', 'recall', 'roc_auc']
    data = []
    
    # Baseline
    for m in metrics:
        val = ablation_data['baseline_metrics'].get(m)
        if val is not None:
            data.append({'Configuration': 'Baseline (Raw CAN)', 'Metric': m.upper(), 'Score': val})
            
    # Engineered
    for m in metrics:
        val = ablation_data['engineered_metrics'].get(m)
        if val is not None:
            data.append({'Configuration': 'Engineered Features', 'Metric': m.upper(), 'Score': val})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Metric', y='Score', hue='Configuration', palette='magma')
    plt.title('Ablation Study: Feature Engineering Impact', fontsize=15)
    plt.ylim(0, 1.1)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_study_b1.png')
    plt.savefig(save_path)
    print(f"Saved ablation study to {save_path}")
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, 'logs')
    output_dir = os.path.join(base_dir, 'reports', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main Evaluation Report
    eval_report_path = os.path.join(logs_dir, 'cascade_eval_replica_report.json')
    eval_data = load_json(eval_report_path)
    
    if eval_data:
        print("Generating Cascade Evaluation Plots...")
        plot_metric_comparison(eval_data, output_dir)
        plot_confusion_matrices(eval_data, output_dir)
    else:
        print(f"Skipping Cascade plots. File missing: {eval_report_path}")
        
    # 2. Ablation Report
    ablation_report_path = os.path.join(logs_dir, 'b1_ablation_report_latest.json')
    ablation_data = load_json(ablation_report_path)
    
    if ablation_data:
        print("Generating Ablation Study Plots...")
        plot_ablation_study(ablation_data, output_dir)
    else:
        print(f"Skipping Ablation plots. File missing: {ablation_report_path}")

if __name__ == "__main__":
    main()
