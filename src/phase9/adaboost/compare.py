"""
CargoSense — Phase 9: AdaBoost Comparative Evaluation
Compares Run 5 (Baseline AdaBoost) vs. Run 6 (Custom AdaBoost).
Generates comparative diagnostic figures and prints benchmark summaries.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.evaluation import (
    plot_pr_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_calibration_curves
)

OUTPUT_COMPARISON_FIGURES = PROJECT_ROOT / "reports" / "phase9" / "adaboost" / "figures"
BASELINE_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "adaboost" / "baseline"
CUSTOM_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "adaboost" / "custom"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def generate_comparative_figures() -> None:
    """Generate all side-by-side diagnostic figures for Baseline vs. Custom AdaBoost."""
    OUTPUT_COMPARISON_FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    val_df = pd.read_csv(DATA_DIR / "phase8_val.csv")
    test_df = pd.read_csv(DATA_DIR / "phase8_test.csv")
    y_val = val_df["Became_FTL"].values
    y_test = test_df["Became_FTL"].values

    base_val_preds = pd.read_csv(BASELINE_REPORT_DIR / "val_predictions.csv")
    base_test_preds = pd.read_csv(BASELINE_REPORT_DIR / "predictions.csv")
    cust_val_preds = pd.read_csv(CUSTOM_REPORT_DIR / "val_predictions.csv")
    cust_test_preds = pd.read_csv(CUSTOM_REPORT_DIR / "predictions.csv")

    with open(BASELINE_REPORT_DIR / "metrics.json", "r", encoding="utf-8") as f:
        base_metrics = json.load(f)
    with open(CUSTOM_REPORT_DIR / "metrics.json", "r", encoding="utf-8") as f:
        cust_metrics = json.load(f)

    base_th = base_metrics["optimal_threshold"]
    cust_th = cust_metrics["optimal_threshold"]

    # --- Figure 1: PR Curves ---
    plot_pr_curves(
        {
            "Baseline Val": (y_val, base_val_preds["predicted_prob"].values),
            "Baseline Test": (y_test, base_test_preds["predicted_prob"].values),
            "Custom Val": (y_val, cust_val_preds["predicted_prob"].values),
            "Custom Test": (y_test, cust_test_preds["predicted_prob"].values),
        },
        output_path=OUTPUT_COMPARISON_FIGURES / "fig01_pr_curves.png",
        title="AdaBoost Comparison: Precision-Recall Curves (Run 5 vs. Run 6)"
    )

    # --- Figure 2: ROC Curves ---
    plot_roc_curves(
        {
            "Baseline Val": (y_val, base_val_preds["predicted_prob"].values),
            "Baseline Test": (y_test, base_test_preds["predicted_prob"].values),
            "Custom Val": (y_val, cust_val_preds["predicted_prob"].values),
            "Custom Test": (y_test, cust_test_preds["predicted_prob"].values),
        },
        output_path=OUTPUT_COMPARISON_FIGURES / "fig02_roc_curves.png",
        title="AdaBoost Comparison: ROC Curves (Run 5 vs. Run 6)"
    )

    # --- Figure 3: Calibration Curves ---
    plot_calibration_curves(
        {
            "Baseline AdaBoost (Test)": (y_test, base_test_preds["predicted_prob"].values),
            "Custom AdaBoost (Test)": (y_test, cust_test_preds["predicted_prob"].values),
        },
        output_path=OUTPUT_COMPARISON_FIGURES / "fig03_calibration_curves.png",
        n_bins=8
    )

    # --- Figure 4: Side-by-side Confusion Matrices ---
    base_cm = base_metrics["test_metrics_optimal_th"]["test_opt_confusion_matrix"]
    cust_cm = cust_metrics["test_metrics_optimal_th"]["test_opt_confusion_matrix"]

    plot_confusion_matrices(
        {
            f"Run 5 Baseline AdaBoost (th={base_th:.2f})": np.array(base_cm),
            f"Run 6 Custom AdaBoost (th={cust_th:.2f})": np.array(cust_cm),
        },
        output_path=OUTPUT_COMPARISON_FIGURES / "fig04_confusion_matrices.png"
    )

    # --- Figure 5: Feature Importances Comparison ---
    base_top = pd.DataFrame(base_metrics["top_features"][:10])
    cust_top = pd.DataFrame(cust_metrics["top_features"][:10])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    axes[0].barh(base_top["feature"][::-1], base_top["importance"][::-1], color="#2b5c8f", edgecolor="#1a365d")
    axes[0].set_title(f"Run 5 Baseline AdaBoost Top 10 Features", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Importance (Ensemble Weight)", fontsize=10, fontweight="bold")
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_width():.4f}", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2),
                         va="center", fontsize=9)

    axes[1].barh(cust_top["feature"][::-1], cust_top["importance"][::-1], color="#2a9d8f", edgecolor="#1b635b")
    axes[1].set_title(f"Run 6 Custom AdaBoost Top 10 Features", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Importance (Ensemble Weight)", fontsize=10, fontweight="bold")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.4f}", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2),
                         va="center", fontsize=9)

    plt.suptitle("AdaBoost Feature Importance Comparison (Run 5 Baseline vs. Run 6 Custom)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARISON_FIGURES / "fig05_feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparative figures generated successfully at: {OUTPUT_COMPARISON_FIGURES}")


if __name__ == "__main__":
    generate_comparative_figures()
