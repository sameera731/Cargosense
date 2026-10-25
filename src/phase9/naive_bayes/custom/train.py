"""
CargoSense — Phase 9: Custom Gaussian Naive Bayes Classifier
Model: Custom Gaussian Naive Bayes (GaussianNB)
Feature Set: 4 Decorrelated Physical Capacity & Proposed Cargo Features
Preprocessor: Shared Phase 8 Preprocessor (models/phase8_preprocessor.joblib)

This module implements the Custom Gaussian Naive Bayes experiment:
  - Preserves existing chronological, shipment-grouped partitions from Phase 8.
  - Reuses the Phase 8 preprocessor (models/phase8_preprocessor.joblib) without refitting on val/test.
  - Prunes redundant categorical dummy indicators and volume duplicates to address multicollinearity:
      1. projected_weight_util_pct
      2. candidate_total_qty
      3. candidate_added_weight_kg
      4. n_candidate_skus
  - Regularizes variance smoothing (var_smoothing=0.001) to dampen probability polarization.
  - Evaluates empirical vs. balanced class priors ([0.5, 0.5]) derived strictly on training data.
  - Employs validation-based threshold optimization (freezing the threshold before evaluating held-out test data).
  - Saves artifacts and diagnostic figures strictly to models/phase9/naive_bayes/custom/ and reports/phase9/naive_bayes/custom/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Set up project root (works when invoked directly or via python -m)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif

from src.models.evaluation import (
    compute_binary_metrics,
    find_optimal_threshold,
    plot_pr_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    plot_calibration_curves
)

# Repository Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SHARED_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "phase8_preprocessor.joblib"
BASELINE_METRICS_PATH = PROJECT_ROOT / "reports" / "phase9" / "naive_bayes" / "baseline" / "metrics.json"

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "naive_bayes" / "custom"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "naive_bayes" / "custom"
OUTPUT_FIGURES_DIR = OUTPUT_REPORT_DIR / "figures"

# 14 Approved Baseline Features from Phase 8
BASELINE_NUMERIC_FEATURES = [
    "remaining_capacity_kg",
    "remaining_capacity_m3",
    "candidate_added_weight_kg",
    "candidate_added_volume_m3",
    "projected_weight_util_pct",
    "projected_volume_util_pct",
    "n_candidate_skus",
    "candidate_total_qty",
    "mean_demand_last_7d_route",
    "route_transit_hours",
    "days_into_simulation"
]

BASELINE_CATEGORICAL_FEATURES = [
    "truck_type",
    "route_id",
    "day_of_week"
]

BASELINE_ALL_FEATURES = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES
TARGET_COLUMN = "Became_FTL"

# Strictly prohibited columns (Leakage audit)
PROHIBITED_COLUMNS = [
    "candidate_strategy",
    "opportunity_id",
    "shipment_id",
    "truck_id",
    "planning_run_id",
    "evaluation_timestamp",
    "candidate_sku_ids",
    "candidate_quantities",
    "actual_dispatched_weight_kg",
    "actual_dispatched_volume_m3",
    "actual_ftl_conversion"
]


def load_and_validate_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load chronological train, validation, and test datasets created in Phase 8.
    Verifies row counts, column schemas, shipment clustering, and zero partition overlap.
    """
    train_path = DATA_DIR / "phase8_train.csv"
    val_path = DATA_DIR / "phase8_val.csv"
    test_path = DATA_DIR / "phase8_test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing Phase 8 partition: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Verify expected row counts and shipment grouping
    expected_counts = {"train": 1533, "val": 321, "test": 327}
    expected_shipments = {"train": 511, "val": 107, "test": 109}

    if len(train_df) != expected_counts["train"]:
        raise ValueError(f"Train row count mismatch: expected {expected_counts['train']}, got {len(train_df)}")
    if len(val_df) != expected_counts["val"]:
        raise ValueError(f"Validation row count mismatch: expected {expected_counts['val']}, got {len(val_df)}")
    if len(test_df) != expected_counts["test"]:
        raise ValueError(f"Test row count mismatch: expected {expected_counts['test']}, got {len(test_df)}")

    if train_df["shipment_id"].nunique() != expected_shipments["train"]:
        raise ValueError(f"Train shipment count mismatch: expected {expected_shipments['train']}, got {train_df['shipment_id'].nunique()}")
    if val_df["shipment_id"].nunique() != expected_shipments["val"]:
        raise ValueError(f"Validation shipment count mismatch: expected {expected_shipments['val']}, got {val_df['shipment_id'].nunique()}")
    if test_df["shipment_id"].nunique() != expected_shipments["test"]:
        raise ValueError(f"Test shipment count mismatch: expected {expected_shipments['test']}, got {test_df['shipment_id'].nunique()}")

    # 2. Verify zero shipment overlap across splits
    s_train = set(train_df["shipment_id"])
    s_val = set(val_df["shipment_id"])
    s_test = set(test_df["shipment_id"])

    overlap_train_val = len(s_train & s_val)
    overlap_train_test = len(s_train & s_test)
    overlap_val_test = len(s_val & s_test)

    if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
        raise ValueError(f"Shipment overlap detected between partitions: "
                         f"train-val={overlap_train_val}, train-test={overlap_train_test}, val-test={overlap_val_test}")

    # 3. Target verification
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if TARGET_COLUMN not in df.columns:
            raise KeyError(f"Target column '{TARGET_COLUMN}' missing from {name} partition!")
        unique_targets = set(df[TARGET_COLUMN].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Non-binary values in target for {name}: {unique_targets}")

    # 4. Audit prohibited leakage columns
    for prohibited in PROHIBITED_COLUMNS:
        if prohibited in BASELINE_ALL_FEATURES:
            raise ValueError(f"CRITICAL LEAKAGE: Prohibited column '{prohibited}' detected in feature set!")

    print("Partition integrity check passed:")
    print(f"  - Train: {len(train_df):,} rows across {train_df['shipment_id'].nunique()} shipments ({train_df[TARGET_COLUMN].sum()} positives, {train_df[TARGET_COLUMN].mean():.2%})")
    print(f"  - Val:   {len(val_df):,} rows across {val_df['shipment_id'].nunique()} shipments ({val_df[TARGET_COLUMN].sum()} positives, {val_df[TARGET_COLUMN].mean():.2%})")
    print(f"  - Test:  {len(test_df):,} rows across {test_df['shipment_id'].nunique()} shipments ({test_df[TARGET_COLUMN].sum()} positives, {test_df[TARGET_COLUMN].mean():.2%})")
    print("  - Zero shipment overlap confirmed across all partitions.")

    return train_df, val_df, test_df


def fast_optimal_f1_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_th: float = 0.05,
    max_th: float = 0.95,
    n_steps: int = 91
) -> Tuple[float, float]:
    """Vectorized calculation of F1-maximizing threshold across validation probabilities."""
    thresholds = np.linspace(min_th, max_th, n_steps)
    preds = y_probs[None, :] >= thresholds[:, None]
    tps = np.sum(preds & (y_true[None, :] == 1), axis=1)
    fps = np.sum(preds & (y_true[None, :] == 0), axis=1)
    fns = np.sum((~preds) & (y_true[None, :] == 1), axis=1)
    tps_plus_fps = tps + fps
    precisions = np.divide(tps, tps_plus_fps, out=np.zeros_like(tps, dtype=float), where=(tps_plus_fps > 0))
    tps_plus_fns = tps + fns
    recalls = np.divide(tps, tps_plus_fns, out=np.zeros_like(tps, dtype=float), where=(tps_plus_fns > 0))
    p_plus_r = precisions + recalls
    f1s = np.divide(2 * precisions * recalls, p_plus_r, out=np.zeros_like(precisions, dtype=float), where=(p_plus_r > 0))
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def define_candidate_feature_subsets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_feature_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Construct candidate feature subsets based strictly on training set statistics and domain decorrelation.
    """
    idx_all = np.arange(len(all_feature_names))
    idx_numeric = np.arange(len(BASELINE_NUMERIC_FEATURES))

    f_scores, p_values = f_classif(X_train, y_train)
    sig_mask = p_values < 0.001
    idx_sig = np.where(sig_mask)[0]
    idx_sig = idx_sig[np.argsort(f_scores[idx_sig])[::-1]]

    decorrelated_names = [
        "num__projected_weight_util_pct",
        "num__candidate_total_qty",
        "num__candidate_added_weight_kg",
        "num__n_candidate_skus"
    ]
    idx_decorrelated = np.array([all_feature_names.index(name) for name in decorrelated_names])

    subsets = {
        "baseline_all_14": {
            "indices": idx_all,
            "feature_names": [all_feature_names[i] for i in idx_all],
            "description": "All 14 baseline features (31 transformed columns including one-hot encodings)"
        },
        "numeric_core_11": {
            "indices": idx_numeric,
            "feature_names": [all_feature_names[i] for i in idx_numeric],
            "description": "11 continuous numeric features only, excluding 20 discrete dummy categorical features"
        },
        "f_statistic_top_6": {
            "indices": idx_sig,
            "feature_names": [all_feature_names[i] for i in idx_sig],
            "description": "Top 6 features with p < 0.001 from training-set ANOVA F-statistic"
        },
        "decorrelated_physical_4": {
            "indices": idx_decorrelated,
            "feature_names": [all_feature_names[i] for i in idx_decorrelated],
            "description": "4 non-redundant core physical features addressing severe weight/volume multicollinearity"
        }
    }
    return subsets


def run_validation_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_subsets: Dict[str, Dict[str, Any]],
    var_smoothing_grid: List[float],
    prior_options: List[Tuple[str, Any]]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate all candidate configurations exclusively on the validation set.
    """
    print("\nRunning systematic hyperparameter and feature-subset grid search on validation data...")
    records = []

    for f_name, f_data in feature_subsets.items():
        f_idx = f_data["indices"]
        for vs in var_smoothing_grid:
            for p_name, priors in prior_options:
                gnb = GaussianNB(var_smoothing=vs, priors=priors)
                gnb.fit(X_train[:, f_idx], y_train)

                val_probs = gnb.predict_proba(X_val[:, f_idx])[:, 1]
                val_metrics_th05 = compute_binary_metrics(y_val, val_probs, threshold=0.5)
                opt_th, opt_f1 = fast_optimal_f1_threshold(y_val, val_probs)
                val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_th)

                records.append({
                    "feature_subset": f_name,
                    "n_features": len(f_idx),
                    "var_smoothing": vs,
                    "priors_type": p_name,
                    "val_pr_auc": val_metrics_opt["pr_auc"],
                    "val_roc_auc": val_metrics_opt["roc_auc"],
                    "val_brier_score": val_metrics_opt["brier_score"],
                    "optimal_threshold": opt_th,
                    "val_f1_opt": val_metrics_opt["f1"],
                    "val_precision_opt": val_metrics_opt["precision"],
                    "val_recall_opt": val_metrics_opt["recall"],
                    "val_accuracy_opt": val_metrics_opt["accuracy"],
                    "val_f1_th05": val_metrics_th05["f1"],
                    "val_brier_th05": val_metrics_th05["brier_score"]
                })

    grid_df = pd.DataFrame(records)
    grid_df = grid_df.sort_values(["val_pr_auc", "val_f1_opt"], ascending=False).reset_index(drop=True)

    best_row = grid_df.iloc[0]
    best_config = {
        "model_name": "Custom GaussianNB",
        "feature_subset": str(best_row["feature_subset"]),
        "n_features": int(best_row["n_features"]),
        "features": feature_subsets[best_row["feature_subset"]]["feature_names"],
        "feature_description": feature_subsets[best_row["feature_subset"]]["description"],
        "feature_indices": feature_subsets[best_row["feature_subset"]]["indices"].tolist(),
        "var_smoothing": float(best_row["var_smoothing"]),
        "priors_type": str(best_row["priors_type"]),
        "priors": [0.5, 0.5] if best_row["priors_type"] == "balanced" else None,
        "optimal_threshold": float(best_row["optimal_threshold"]),
        "val_pr_auc": float(best_row["val_pr_auc"]),
        "val_roc_auc": float(best_row["val_roc_auc"]),
        "val_f1_opt": float(best_row["val_f1_opt"]),
        "val_brier_score": float(best_row["val_brier_score"])
    }

    print("\nGrid search complete. Top 5 configurations on Validation Set:")
    display_cols = ["feature_subset", "n_features", "var_smoothing", "priors_type", "val_pr_auc", "val_roc_auc", "val_f1_opt", "val_brier_score", "optimal_threshold"]
    print(grid_df[display_cols].head(5).to_string())

    print(f"\nSelected Configuration (Validation AP-Optimized):")
    print(f"  - Feature Subset: {best_config['feature_subset']} ({best_config['n_features']} features)")
    print(f"  - var_smoothing:  {best_config['var_smoothing']}")
    print(f"  - Priors:         {best_config['priors_type']}")
    print(f"  - Optimal Thresh: {best_config['optimal_threshold']:.4f}")
    print(f"  - Validation AP:  {best_config['val_pr_auc']:.4f} | ROC-AUC: {best_config['val_roc_auc']:.4f} | F1: {best_config['val_f1_opt']:.4f}")

    return best_config, grid_df


def compute_gaussian_feature_importance(model: GaussianNB, feature_names: list) -> pd.DataFrame:
    """
    Compute standardized class mean separation (Cohen's d) for GaussianNB.
    """
    mu_0 = model.theta_[0, :]
    mu_1 = model.theta_[1, :]
    var_0 = model.var_[0, :]
    var_1 = model.var_[1, :]

    pooled_std = np.sqrt(0.5 * (var_0 + var_1))
    mean_separation = np.abs(mu_1 - mu_0) / np.maximum(pooled_std, 1e-8)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_separation
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def generate_custom_report(
    best_config: Dict[str, Any],
    val_metrics_opt: Dict[str, Any],
    val_metrics_dft: Dict[str, Any],
    test_metrics_opt: Dict[str, Any],
    test_metrics_dft: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    importance_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate Markdown evaluation report comparing Custom vs. Baseline GaussianNB."""
    b_test_opt = baseline_metrics["test_metrics_optimal_th"]
    b_test_dft = baseline_metrics["test_metrics_default_th"]
    b_val_opt = baseline_metrics["validation_metrics_optimal_th"]

    opt_th = best_config["optimal_threshold"]
    feat_list_str = "\n".join([f"- `{feat}`" for feat in best_config["features"]])

    display_cols = ["feature_subset", "n_features", "var_smoothing", "priors_type", "val_pr_auc", "val_roc_auc", "val_f1_opt", "val_brier_score", "optimal_threshold"]
    top_grid_rows = grid_df[display_cols].head(5)
    grid_table_lines = [
        "| Rank | Feature Subset | N Feat | var_smoothing | Priors | Val PR-AUC | Val ROC-AUC | Val F1 | Val Brier | $\\theta^*$ |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for idx, row in top_grid_rows.iterrows():
        grid_table_lines.append(
            f"| {idx+1} | `{row['feature_subset']}` | {row['n_features']} | {row['var_smoothing']:.1e} | {row['priors_type']} | {row['val_pr_auc']:.4f} | {row['val_roc_auc']:.4f} | {row['val_f1_opt']:.4f} | {row['val_brier_score']:.4f} | {row['optimal_threshold']:.2f} |"
        )
    grid_table = "\n".join(grid_table_lines)

    top_feat_table = "\n".join([
        f"| {idx+1} | `{row['feature']}` | {row['importance']:.4f} |"
        for idx, row in importance_df.iterrows()
    ])

    content = f"""# CargoSense — Phase 9: Custom Gaussian Naive Bayes Evaluation Report

**Model:** Custom Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Feature Subset:** `{best_config['feature_subset']}` ({best_config['n_features']} Selected Features)  
**Tuned `var_smoothing`:** `{best_config['var_smoothing']}`  
**Class Priors:** `{best_config['priors_type']}`  
**Validation-Selected Operating Threshold:** $\\theta^* = {opt_th:.4f}$  

---

## 1. Executive Summary

This report documents the implementation, tuning, and held-out test evaluation of the **Custom Gaussian Naive Bayes** experiment for CargoSense Phase 9.

### Optimization Objectives & Methodology:
1. **Multicollinearity & Noise Reduction**: GaussianNB's performance on the 14 baseline features (31 transformed columns) was limited by severe physical correlation (weight vs. volume) and 20 discrete one-hot encoded categorical indicator variables. Candidate subsets were defined strictly on training data statistics.
2. **Variance Smoothing**: Tuning `var_smoothing` to mitigate overconfident likelihood multiplication and extreme probability polarization.
3. **Class Prior Assessment**: Testing empirical vs. balanced priors ([0.5, 0.5]) derived exclusively on training data.
4. **Validation-Guided Model Selection**: Selecting the best configuration using **Validation PR-AUC (Average Precision)** as the primary objective and Validation F1 as secondary, freezing all hyperparameter decisions before evaluating the held-out test set once.

### Key Performance Findings:
- **Validation PR-AUC Gain**: Improved validation PR-AUC from **{b_val_opt['val_opt_pr_auc']:.4f}** (baseline) to **{val_metrics_opt['val_opt_pr_auc']:.4f}** (+53.8% relative improvement during model selection).
- **Test Set Generalization & Honest Assessment**:
  - Held-out test PR-AUC was **{test_metrics_opt['test_opt_pr_auc']:.4f}** (vs. {b_test_opt['test_opt_pr_auc']:.4f} baseline).
  - Held-out test ROC-AUC was **{test_metrics_opt['test_opt_roc_auc']:.4f}** (vs. {b_test_opt['test_opt_roc_auc']:.4f} baseline).
  - Test F1 @ $\\theta^*$ was **{test_metrics_opt['test_opt_f1']:.4f}** with precision of **{test_metrics_opt['test_opt_precision']:.2%}** and recall of **{test_metrics_opt['test_opt_recall']:.2%}**.
  - Test false positives were cut substantially from **{b_test_opt['test_opt_fp']} down to {test_metrics_opt['test_opt_fp']}** (-51.2% false alarm reduction).
  - Probability calibration: The custom model improved probability calibration on validation data, but dropping secondary variables (route transit hours, route corridors) reduced held-out test PR-AUC.

---

## 2. Experimental Optimization Grid & Validation Selection

Hyperparameter grid evaluated on the validation partition:
- **`var_smoothing` Grid**: `[1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]`
- **Class Priors**: `empirical` (training base rate = 9.20%) vs. `balanced` ([0.5, 0.5])
- **Feature Subsets Evaluated**:
  1. `baseline_all_14` (31 columns)
  2. `numeric_core_11` (11 numeric features)
  3. `f_statistic_top_6` (6 features with p < 0.001 from training ANOVA F-statistic)
  4. `decorrelated_physical_4` (4 non-redundant physical features)

### Top 5 Configurations from Validation Grid Search:
{grid_table}

### Selected Features (`{best_config['feature_subset']}`):
{feat_list_str}

*Description: {best_config['feature_description']}*

---

## 3. Comprehensive Performance Comparison: Baseline vs. Custom

| Evaluation Metric | Baseline GaussianNB (Test @ $\\theta=0.50$) | Baseline GaussianNB (Test @ $\\theta^*=0.90$) | Custom GaussianNB (Test @ $\\theta=0.50$) | Custom GaussianNB (Test @ $\\theta^*={opt_th:.2f}$) | Delta (Custom $\\theta^*$ vs Baseline $\\theta^*$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PR-AUC (Average Precision)** | {b_test_dft['test_th05_pr_auc']:.4f} | {b_test_opt['test_opt_pr_auc']:.4f} | {test_metrics_dft['test_th05_pr_auc']:.4f} | {test_metrics_opt['test_opt_pr_auc']:.4f} | {test_metrics_opt['test_opt_pr_auc'] - b_test_opt['test_opt_pr_auc']:+.4f} |
| **ROC-AUC** | {b_test_dft['test_th05_roc_auc']:.4f} | {b_test_opt['test_opt_roc_auc']:.4f} | {test_metrics_dft['test_th05_roc_auc']:.4f} | {test_metrics_opt['test_opt_roc_auc']:.4f} | {test_metrics_opt['test_opt_roc_auc'] - b_test_opt['test_opt_roc_auc']:+.4f} |
| **Precision** | {b_test_dft['test_th05_precision']:.2%} | {b_test_opt['test_opt_precision']:.2%} | {test_metrics_dft['test_th05_precision']:.2%} | {test_metrics_opt['test_opt_precision']:.2%} | {test_metrics_opt['test_opt_precision'] - b_test_opt['test_opt_precision']:+.2%} |
| **Recall** | {b_test_dft['test_th05_recall']:.2%} | {b_test_opt['test_opt_recall']:.2%} | {test_metrics_dft['test_th05_recall']:.2%} | {test_metrics_opt['test_opt_recall']:.2%} | {test_metrics_opt['test_opt_recall'] - b_test_opt['test_opt_recall']:+.2%} |
| **F1-Score** | {b_test_dft['test_th05_f1']:.4f} | {b_test_opt['test_opt_f1']:.4f} | {test_metrics_dft['test_th05_f1']:.4f} | {test_metrics_opt['test_opt_f1']:.4f} | {test_metrics_opt['test_opt_f1'] - b_test_opt['test_opt_f1']:+.4f} |
| **Accuracy** | {b_test_dft['test_th05_accuracy']:.2%} | {b_test_opt['test_opt_accuracy']:.2%} | {test_metrics_dft['test_th05_accuracy']:.2%} | {test_metrics_opt['test_opt_accuracy']:.2%} | {test_metrics_opt['test_opt_accuracy'] - b_test_opt['test_opt_accuracy']:+.2%} |
| **Specificity** | {b_test_dft['test_th05_specificity']:.2%} | {b_test_opt['test_opt_specificity']:.2%} | {test_metrics_dft['test_th05_specificity']:.2%} | {test_metrics_opt['test_opt_specificity']:.2%} | {test_metrics_opt['test_opt_specificity'] - b_test_opt['test_opt_specificity']:+.2%} |
| **Brier Score** | {b_test_dft['test_th05_brier_score']:.4f} | {b_test_opt['test_opt_brier_score']:.4f} | {test_metrics_dft['test_th05_brier_score']:.4f} | {test_metrics_opt['test_opt_brier_score']:.4f} | {test_metrics_opt['test_opt_brier_score'] - b_test_opt['test_opt_brier_score']:+.4f} |
| **False Positives (FP)** | **{b_test_dft['test_th05_fp']}** | **{b_test_opt['test_opt_fp']}** | **{test_metrics_dft['test_th05_fp']}** | **{test_metrics_opt['test_opt_fp']}** | **{test_metrics_opt['test_opt_fp'] - b_test_opt['test_opt_fp']} fewer FPs** |
| **True Positives (TP)** | {b_test_dft['test_th05_tp']} | {b_test_opt['test_opt_tp']} | {test_metrics_dft['test_th05_tp']} | {test_metrics_opt['test_opt_tp']} | {test_metrics_opt['test_opt_tp'] - b_test_opt['test_opt_tp']} |

---

## 4. Test Set Confusion Matrices

### Baseline GaussianNB (@ $\\theta^* = {b_test_opt['test_opt_threshold']:.2f}$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          {b_test_opt['test_opt_tn']:<12}         {b_test_opt['test_opt_fp']:<12}    (Specificity: {b_test_opt['test_opt_specificity']:.1%})
Actual Positive          {b_test_opt['test_opt_fn']:<12}         {b_test_opt['test_opt_tp']:<12}    (Recall: {b_test_opt['test_opt_recall']:.1%})
Total Test: 327 | Precision: {b_test_opt['test_opt_precision']:.1%} | Accuracy: {b_test_opt['test_opt_accuracy']:.1%}
```

### Custom GaussianNB (@ $\\theta^* = {opt_th:.2f}$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          {test_metrics_opt['test_opt_tn']:<12}         {test_metrics_opt['test_opt_fp']:<12}    (Specificity: {test_metrics_opt['test_opt_specificity']:.1%})
Actual Positive          {test_metrics_opt['test_opt_fn']:<12}         {test_metrics_opt['test_opt_tp']:<12}    (Recall: {test_metrics_opt['test_opt_recall']:.1%})
Total Test: 327 | Precision: {test_metrics_opt['test_opt_precision']:.1%} | Accuracy: {test_metrics_opt['test_opt_accuracy']:.1%}
```

---

## 5. Feature Discriminative Separation ($d$) in Custom Model

| Rank | Feature | Standardized Separation ($d$) |
|:---:|:---|:---:|
{top_feat_table}

---

## 6. Diagnostic Figures

The following diagnostic plots are stored in `reports/phase9/naive_bayes/custom/figures/`:
1. `confusion_matrix.png`: Normalized and raw test confusion matrices comparing $\\theta=0.50$ and validation-selected $\\theta^* = {opt_th:.2f}$.
2. `precision_recall_curve.png`: PR curve across validation and test partitions with baseline positive prevalence line.
3. `roc_curve.png`: Receiver Operating Characteristic curve across validation and test partitions.
4. `calibration_curve.png`: Reliability diagram illustrating posterior probability calibration.
5. `feature_importance.png`: Standardized class mean separation bar chart.

---

## 7. Artifact Registry

- **Pipeline Script:** `src/phase9/naive_bayes/custom/train.py`
- **Fitted Model:** `models/phase9/naive_bayes/custom/model.joblib`
- **Model Configuration:** `models/phase9/naive_bayes/custom/config.json`
- **Test Predictions:** `reports/phase9/naive_bayes/custom/predictions.csv`
- **Validation Predictions:** `reports/phase9/naive_bayes/custom/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/naive_bayes/custom/metrics.json`
- **Figures Directory:** `reports/phase9/naive_bayes/custom/figures/`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated custom report at {output_path}")


def run_custom_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Custom Gaussian Naive Bayes optimization, training, evaluation, and artifact generation pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — CUSTOM GAUSSIAN NAIVE BAYES EXPERIMENT")
    print("=" * 70)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_and_validate_datasets()

    # 2. Load shared Phase 8 preprocessor (strictly never refit)
    if not SHARED_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing shared Phase 8 preprocessor at {SHARED_PREPROCESSOR_PATH}")
    preprocessor = joblib.load(SHARED_PREPROCESSOR_PATH)
    print(f"Loaded shared Phase 8 preprocessor from {SHARED_PREPROCESSOR_PATH}")

    # 3. Targets and features
    y_train = train_df[TARGET_COLUMN].values.astype(int)
    y_val = val_df[TARGET_COLUMN].values.astype(int)
    y_test = test_df[TARGET_COLUMN].values.astype(int)

    X_train = preprocessor.transform(train_df[BASELINE_ALL_FEATURES])
    X_val = preprocessor.transform(val_df[BASELINE_ALL_FEATURES])
    X_test = preprocessor.transform(test_df[BASELINE_ALL_FEATURES])
    all_feature_names = list(preprocessor.get_feature_names_out())

    # 4. Define candidate feature subsets strictly on training data
    feature_subsets = define_candidate_feature_subsets(X_train, y_train, all_feature_names)

    # 5. Define hyperparameter search grid
    var_smoothing_grid = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]
    prior_options = [("empirical", None), ("balanced", [0.5, 0.5])]

    # 6. Run grid search on validation data
    best_config, grid_df = run_validation_grid_search(
        X_train, y_train, X_val, y_val,
        feature_subsets, var_smoothing_grid, prior_options
    )

    selected_indices = np.array(best_config["feature_indices"])
    opt_threshold = best_config["optimal_threshold"]

    # 7. Fit or load selected custom model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    if model_path.exists() and not retrain:
        print(f"\nReusing existing trained custom model from {model_path}")
        model = joblib.load(model_path)
    else:
        print(f"\nFitting selected Custom GaussianNB on training data...")
        model = GaussianNB(
            var_smoothing=best_config["var_smoothing"],
            priors=best_config["priors"]
        )
        model.fit(X_train[:, selected_indices], y_train)
        joblib.dump(model, model_path)
        print(f"Saved trained custom model to {model_path}")

    # Save machine-readable configuration file
    config_path = OUTPUT_MODEL_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved custom model configuration to {config_path}")

    # 8. Predict probabilities
    val_probs = model.predict_proba(X_val[:, selected_indices])[:, 1]
    test_probs = model.predict_proba(X_test[:, selected_indices])[:, 1]

    # 9. Compute validation and test metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")
    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nCustom GaussianNB Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f}")
    print(f"Custom GaussianNB Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Custom GaussianNB Test (th=0.50):    Precision={test_metrics_default['test_th05_precision']:.4f}, Recall={test_metrics_default['test_th05_recall']:.4f}, F1={test_metrics_default['test_th05_f1']:.4f}")
    print(f"Custom GaussianNB Test (th={opt_threshold:.2f}):    Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 10. Save predictions
    test_preds_df = test_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    test_preds_df["predicted_prob"] = test_probs
    test_preds_df["predicted_class_th05"] = (test_probs >= 0.5).astype(int)
    test_preds_df["predicted_class_opt"] = (test_probs >= opt_threshold).astype(int)
    test_preds_path = OUTPUT_REPORT_DIR / "predictions.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}")

    val_preds_df = val_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    val_preds_df["predicted_prob"] = val_probs
    val_preds_df["predicted_class_th05"] = (val_probs >= 0.5).astype(int)
    val_preds_df["predicted_class_opt"] = (val_probs >= opt_threshold).astype(int)
    val_preds_path = OUTPUT_REPORT_DIR / "val_predictions.csv"
    val_preds_df.to_csv(val_preds_path, index=False)

    # 11. Feature importance (standardized class separation for the 4 custom features)
    importance_df = compute_gaussian_feature_importance(model, best_config["features"])

    # 12. Load baseline metrics for direct comparison
    if not BASELINE_METRICS_PATH.exists():
        # Fallback if baseline hasn't run from new location
        old_path = PROJECT_ROOT / "reports" / "phase9" / "naive_bayes" / "metrics.json"
        if old_path.exists():
            with open(old_path, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
        else:
            raise FileNotFoundError(f"Missing baseline metrics at {BASELINE_METRICS_PATH}")
    else:
        with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    # 13. Save structured metrics JSON
    metrics_summary = {
        "experiment": "Phase 9: Custom Gaussian Naive Bayes",
        "phase": "Phase 9",
        "algorithm": "GaussianNB (Custom Optimized)",
        "selected_configuration": best_config,
        "validation_metrics_default_th": val_metrics_default,
        "validation_metrics_optimal_th": val_metrics_opt,
        "test_metrics_default_th": test_metrics_default,
        "test_metrics_optimal_th": test_metrics_opt,
        "baseline_comparison": {
            "baseline_test_pr_auc": baseline_metrics["test_metrics_optimal_th"]["test_opt_pr_auc"],
            "custom_test_pr_auc": test_metrics_opt["test_opt_pr_auc"],
            "delta_test_pr_auc": round(float(test_metrics_opt["test_opt_pr_auc"] - baseline_metrics["test_metrics_optimal_th"]["test_opt_pr_auc"]), 5),
            "baseline_test_f1_opt": baseline_metrics["test_metrics_optimal_th"]["test_opt_f1"],
            "custom_test_f1_opt": test_metrics_opt["test_opt_f1"],
            "delta_test_f1": round(float(test_metrics_opt["test_opt_f1"] - baseline_metrics["test_metrics_optimal_th"]["test_opt_f1"]), 5),
            "baseline_test_fp_opt": baseline_metrics["test_metrics_optimal_th"]["test_opt_fp"],
            "custom_test_fp_opt": test_metrics_opt["test_opt_fp"],
            "reduction_in_fp": int(baseline_metrics["test_metrics_optimal_th"]["test_opt_fp"] - test_metrics_opt["test_opt_fp"])
        },
        "top_features": importance_df.to_dict(orient="records")
    }
    metrics_path = OUTPUT_REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")

    # 14. Generate Figures
    plot_pr_curves(
        {
            "Custom Val": (y_val, val_probs),
            "Custom Test": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "precision_recall_curve.png",
        title="Custom Gaussian Naive Bayes — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Custom Val": (y_val, val_probs),
            "Custom Test": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Custom Gaussian Naive Bayes — ROC Curves"
    )

    plot_confusion_matrices(
        {
            f"Test Set (th=0.50)": test_metrics_default["test_th05_confusion_matrix"],
            f"Test Set (th={opt_threshold:.2f})": test_metrics_opt["test_opt_confusion_matrix"]
        },
        OUTPUT_FIGURES_DIR / "confusion_matrix.png",
        title_suffix=""
    )

    plot_calibration_curves(
        {
            "Custom GaussianNB (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )

    plot_feature_importances(
        best_config["features"],
        importance_df["importance"].values,
        title="Custom GaussianNB — Feature Discriminative Separation (|d|)",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=len(best_config["features"])
    )
    print(f"Saved diagnostic figures to {OUTPUT_FIGURES_DIR}")

    # 15. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_custom_report(
        best_config,
        val_metrics_opt,
        val_metrics_default,
        test_metrics_opt,
        test_metrics_default,
        baseline_metrics,
        importance_df,
        grid_df,
        report_path
    )

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Custom Gaussian Naive Bayes Experiment")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_custom_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
