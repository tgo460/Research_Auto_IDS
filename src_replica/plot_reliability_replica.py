import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif"
    })

def plot_reliability_diagram(bins_data, ece, brier, title, ax):
    """
    Plots a standard reliability diagram (Calibration curve) with confidence histograms.
    """
    confidences = np.array([b["avg_confidence"] for b in bins_data if b["count"] > 0])
    accuracies = np.array([b["empirical_positive_rate"] for b in bins_data if b["count"] > 0])
    counts = np.array([b["count"] for b in bins_data if b["count"] > 0])
    
    if len(confidences) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # Plot empirical calibration
    ax.plot(confidences, accuracies, "s-", label="Model Calibration")
    
    # Error bars (binomial proportion confidence intervals)
    z = 1.96 # 95% CI
    ci = z * np.sqrt(accuracies * (1 - accuracies) / counts)
    ci = np.nan_to_num(ci, nan=0.0)
    ax.fill_between(confidences, accuracies - ci, accuracies + ci, alpha=0.2, color='C0')
    
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(f"{title}\nECE: {ece:.4f} | Brier: {brier:.4f}")
    ax.legend(loc="upper left")

def plot_histogram(bins_data, ax):
    """
    Plots the count of samples in each bin.
    """
    bin_centers = np.array([(b["lower"] + b["upper"]) / 2 for b in bins_data])
    counts = np.array([b["count"] for b in bins_data])
    widths = np.array([b["upper"] - b["lower"] for b in bins_data])
    
    ax.bar(bin_centers, counts, width=widths * 0.9, align='center', alpha=0.7, color='gray', edgecolor='black')
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of Samples")
    ax.set_yscale("log") # Log scale helps see small bins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default="logs/cascade_eval_replica_report.json")
    parser.add_argument("--out_dir", type=str, default="reports/paper_figures")
    args = parser.parse_args()

    if not os.path.exists(args.report):
        print(f"Error: Report not found at {args.report}")
        return

    with open(args.report, "r") as f:
        data = json.load(f)

    if "calibration" not in data:
        print("Error: 'calibration' key not found in report.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    set_style()

    models = ["light_only", "cascade"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    for i, model_key in enumerate(models):
        if model_key not in data["calibration"]:
            continue
            
        cal_data = data["calibration"][model_key]
        bins_data = cal_data.get("bins", [])
        ece = cal_data.get("ece", 0.0)
        brier = cal_data.get("brier", 0.0)
        
        title = "Light Model (No Router)" if model_key == "light_only" else "Cascade Model (Router + Heavy)"
        
        # Reliability Curve
        plot_reliability_diagram(bins_data, ece, brier, title, axes[0, i])
        
        # Histogram
        plot_histogram(bins_data, axes[1, i])

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "reliability_diagrams_replica.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Reliability diagram to {out_path}")

if __name__ == "__main__":
    main()