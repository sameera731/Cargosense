"""
CargoSense — Phase 9 Evaluation Protocol & Metrics Utilities
Common evaluation module designed for both Sameera (Random Forest) and Member 2 (XGBoost).
Provides standard definitions of:
  - PR-AUC (Primary Metric for class imbalance)
  - ROC-AUC
  - Precision, Recall, F1, Accuracy, Specificity
  - Confusion Matrix & Classification Threshold Optimization
  - Calibration Curve & Brier Score
  - Production Publication Plots (PR curves, ROC curves, Confusion Matrices, Calibration, Feature Importances)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics for operational models.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for positive class (Became_FTL = 1).
    threshold : float
        Decision threshold for converting probabilities to binary classes.
    prefix : str
        Prefix for metric keys in returned dictionary.

    Returns:
    --------
    metrics : dict
        Dictionary of all primary and secondary evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred_class = (y_pred_proba >= threshold).astype(int)

    # Threshold-independent metrics
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    # Threshold-dependent metrics
    prec = float(precision_score(y_true, y_pred_class, zero_division=0))
    rec = float(recall_score(y_true, y_pred_class, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_class, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred_class))

    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}pr_auc": round(pr_auc, 5),
        f"{pfx}roc_auc": round(roc_auc, 5),
        f"{pfx}brier_score": round(brier, 5),
        f"{pfx}threshold": round(threshold, 4),
        f"{pfx}precision": round(prec, 5),
        f"{pfx}recall": round(rec, 5),
        f"{pfx}f1": round(f1, 5),
        f"{pfx}accuracy": round(acc, 5),
        f"{pfx}specificity": round(specificity, 5),
        f"{pfx}tp": tp,
        f"{pfx}fp": fp,
        f"{pfx}tn": tn,
        f"{pfx}fn": fn,
        f"{pfx}confusion_matrix": [[tn, fp], [fn, tp]],
        f"{pfx}total_samples": len(y_true),
        f"{pfx}positive_samples": int(y_true.sum()),
        f"{pfx}positive_rate": round(float(y_true.mean()), 5)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    n_steps: int = 91
) -> Tuple[float, float, pd.DataFrame]:
    """
    Search for classification threshold that maximizes specified metric on validation data.
    Default metric is F1, providing optimal operational trade-off between precision and recall.

    Returns:
    --------
    best_threshold : float
    best_score : float
    sweep_df : pd.DataFrame
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    thresholds = np.linspace(min_threshold, max_threshold, n_steps)
    records = []

    for th in thresholds:
        y_pred = (y_pred_proba >= th).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        records.append({
            "threshold": float(th),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f)
        })

    sweep_df = pd.DataFrame(records)

    if metric == "f1":
        best_idx = sweep_df["f1"].idxmax()
    elif metric == "precision":
        best_idx = sweep_df["precision"].idxmax()
    elif metric == "recall":
        best_idx = sweep_df["recall"].idxmax()
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    best_row = sweep_df.iloc[best_idx]
    best_threshold = float(best_row["threshold"])
    best_score = float(best_row[metric])

    return best_threshold, best_score, sweep_df


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def plot_pr_curves(
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves — CargoSense Validation & Test"
) -> None:
    """
    Plot PR curves for one or more experiments with baseline prevalence line.
    experiments: Dict[name, (y_true, y_pred_proba)]
    """
    plt.figure(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", len(experiments))

    for idx, (name, (y_true, y_probs)) in enumerate(experiments.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        plt.plot(
            rec, prec,
            label=f"{name} (PR-AUC = {ap:.3f})",
            color=palette[idx],
            linewidth=2.2
        )

    # Reference baseline line: positive prevalence from first dataset
    sample_y_true = next(iter(experiments.values()))[0]
    prevalence = np.mean(sample_y_true)
    plt.axhline(
        y=prevalence,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline Prevalence ({prevalence:.1%})"
    )

    plt.xlabel("Recall (Coverage of Successful Opportunities)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (FTL Recommendation Accuracy)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curves"
) -> None:
    """
    Plot ROC curves for one or more experiments.
    experiments: Dict[name, (y_true, y_pred_proba)]
    """
    plt.figure(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", len(experiments))

    for idx, (name, (y_true, y_probs)) in enumerate(experiments.items()):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.plot(
            fpr, tpr,
            label=f"{name} (ROC-AUC = {auc:.3f})",
            color=palette[idx],
            linewidth=2.2
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrices(
    matrices: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    title_suffix: str = ""
) -> None:
    """
    Plot side-by-side confusion matrices with counts and normalized row percentages.
    matrices: Dict[title, 2x2 confusion matrix]
    """
    n_plots = len(matrices)
    fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, matrices.items()):
        cm = np.asarray(cm)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        annot = np.empty_like(cm).astype(str)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"{name} {title_suffix}".strip(), fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class (0=No FTL, 1=Became FTL)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight="bold")
        ax.set_xticklabels(["Did Not Convert", "Became FTL"])
        ax.set_yticklabels(["Did Not Convert", "Became FTL"], rotation=0)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    output_path: Optional[Path] = None,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Plot horizontal bar chart of top N feature importances.
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_df = df.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, max(5, top_n * 0.35)))
    bars = plt.barh(top_df["feature"], top_df["importance"], color="#1f77b4", edgecolor="#0e4b75")
    plt.xlabel("Importance (Gain / Weight)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + (max(df["importance"]) * 0.01), bar.get_y() + bar.get_height() / 2,
                 f"{w:.4f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return df


def plot_calibration_curves(
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    n_bins: int = 8
) -> None:
    """
    Plot reliability diagram (calibration curves).
    experiments: Dict[name, (y_true, y_pred_proba)]
    """
    plt.figure(figsize=(8.5, 6.5))
    palette = sns.color_palette("tab10", len(experiments))

    for idx, (name, (y_true, y_probs)) in enumerate(experiments.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_probs)
        plt.plot(
            prob_pred, prob_true,
            marker="o",
            linewidth=2,
            label=f"{name} (Brier = {brier:.4f})",
            color=palette[idx]
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram (Probability Calibration)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()
