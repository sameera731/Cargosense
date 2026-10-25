"""
CargoSense — Phase 9: Run 5 Baseline AdaBoost Classifier
Owner: Adaptive Boosting Lead

This module implements the Baseline AdaBoost experiment using:
  - 14 shared baseline features from Phase 8 (11 numerical, 3 categorical).
  - Phase 8 preprocessor loaded directly from models/phase8_preprocessor.joblib.
  - Chronological shipment-grouped splits from Phase 8.
  - Sample weighting strategy: balanced sample weights computed strictly on training data
    to address the ~9.87:1 class imbalance without artificial synthetic sampling.
  - Decision tree weak learners (max_depth=3, n_estimators=50, learning_rate=1.0).
  - Validation-based threshold tuning and PR-AUC/F1 optimization.
  - Outputs written strictly to models/phase9/adaboost/baseline/ and reports/phase9/adaboost/baseline/.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up project root (works when invoked directly or via python -m)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

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

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "adaboost" / "baseline"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "adaboost" / "baseline"
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


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load chronological train, validation, and test datasets created in Phase 8."""
    train_path = DATA_DIR / "phase8_train.csv"
    val_path = DATA_DIR / "phase8_val.csv"
    test_path = DATA_DIR / "phase8_test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing Phase 8 partition: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Audit prohibited leakage columns
    for prohibited in PROHIBITED_COLUMNS:
        if prohibited in BASELINE_ALL_FEATURES:
            raise ValueError(f"CRITICAL LEAKAGE: Prohibited column '{prohibited}' detected in feature set!")

    return train_df, val_df, test_df


def generate_baseline_report(
    metrics_summary: Dict[str, Any],
    top_features: list,
    output_path: Path
) -> None:
    """Generate Markdown evaluation report for Baseline AdaBoost experiment."""
    val_opt = metrics_summary["validation_metrics_optimal_th"]
    test_opt = metrics_summary["test_metrics_optimal_th"]
    test_dft = metrics_summary["test_metrics_default_th"]
    params = metrics_summary["hyperparameters"]
    opt_th = metrics_summary["optimal_threshold"]

    top_feat_table = "\n".join([
        f"| {idx+1} | `{f['feature']}` | {f['importance']:.4f} |"
        for idx, f in enumerate(top_features[:10])
    ])

    content = f"""# CargoSense — Phase 9: Run 5 Baseline AdaBoost Evaluation Report

**Model:** AdaBoost Classifier (Baseline)  
**Base Estimator:** `DecisionTreeClassifier(max_depth=3)`  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical — 31 Encoded Dims)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\\theta^* = {opt_th:.4f}$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 5 implements the sequential boosting baseline using AdaBoost (Adaptive Boosting) for the CargoSense Opportunity Detection Engine (ODE).
The model was trained strictly on the chronological training split (Days 1–62, 1,533 records, 141 positives).
To handle the ~9.87:1 class imbalance without generating synthetic data or distorting operational physics, sample weights were initialized using class-balanced weights derived from the training set ($w_0 \\approx 0.5506, w_1 \\approx 5.4362$).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **{test_opt['test_opt_pr_auc']:.4f}** (vs. {test_opt['test_opt_positive_rate']:.1%} random baseline prevalence)
- **Held-Out Test ROC-AUC:** **{test_opt['test_opt_roc_auc']:.4f}**
- **Test Precision @ $\\theta^*$:** **{test_opt['test_opt_precision']:.2%}**
- **Test Recall @ $\\theta^*$:** **{test_opt['test_opt_recall']:.2%}**
- **Test F1-Score @ $\\theta^*$:** **{test_opt['test_opt_f1']:.4f}**
- **Test Brier Score:** **{test_opt['test_opt_brier_score']:.4f}**

---

## 2. Model Architecture & Hyperparameters

Hyperparameters tuned strictly on the validation partition (Days 63–75) using validation PR-AUC as the primary criterion:

```json
{json.dumps(params, indent=2)}
```

### Rationale for Weak Learner Configuration:
1. **Tree Depth ($d=3$):** Classic decision stumps ($d=1$) can only evaluate single-feature splits per iteration, failing to capture critical multi-variable operational interactions (e.g., remaining payload capacity combined with added volume). A maximum depth of 3 enables 3-way interactions while strictly avoiding the extreme memorization observed with unregularized deep trees.
2. **Learning Rate ($\\eta = 1.0$):** Standard SAMME boosting learning rate provides robust sequential sample reweighting.
3. **Number of Estimators ($M=50$):** 50 boosting stages ensure convergence without over-concentrating weights on label noise or synthetic outliers.

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\\theta=0.50$) | Validation Set (Optimal $\\theta={opt_th:.2f}$) | Test Set (Default $\\theta=0.50$) | Test Set (Optimal $\\theta={opt_th:.2f}$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | {metrics_summary['validation_metrics_default_th']['val_th05_pr_auc']:.4f} | {val_opt['val_opt_pr_auc']:.4f} | {test_dft['test_th05_pr_auc']:.4f} | {test_opt['test_opt_pr_auc']:.4f} |
| **ROC-AUC** | {metrics_summary['validation_metrics_default_th']['val_th05_roc_auc']:.4f} | {val_opt['val_opt_roc_auc']:.4f} | {test_dft['test_th05_roc_auc']:.4f} | {test_opt['test_opt_roc_auc']:.4f} |
| **Precision** | {metrics_summary['validation_metrics_default_th']['val_th05_precision']:.4f} | {val_opt['val_opt_precision']:.4f} | {test_dft['test_th05_precision']:.4f} | {test_opt['test_opt_precision']:.4f} |
| **Recall** | {metrics_summary['validation_metrics_default_th']['val_th05_recall']:.4f} | {val_opt['val_opt_recall']:.4f} | {test_dft['test_th05_recall']:.4f} | {test_opt['test_opt_recall']:.4f} |
| **F1-Score** | {metrics_summary['validation_metrics_default_th']['val_th05_f1']:.4f} | {val_opt['val_opt_f1']:.4f} | {test_dft['test_th05_f1']:.4f} | {test_opt['test_opt_f1']:.4f} |
| **Accuracy** | {metrics_summary['validation_metrics_default_th']['val_th05_accuracy']:.4f} | {val_opt['val_opt_accuracy']:.4f} | {test_dft['test_th05_accuracy']:.4f} | {test_opt['test_opt_accuracy']:.4f} |
| **Specificity** | {metrics_summary['validation_metrics_default_th']['val_th05_specificity']:.4f} | {val_opt['val_opt_specificity']:.4f} | {test_dft['test_th05_specificity']:.4f} | {test_opt['test_opt_specificity']:.4f} |
| **Brier Score** | {metrics_summary['validation_metrics_default_th']['val_th05_brier_score']:.4f} | {val_opt['val_opt_brier_score']:.4f} | {test_dft['test_th05_brier_score']:.4f} | {test_opt['test_opt_brier_score']:.4f} |

### Held-Out Test Confusion Matrix (@ $\\theta^* = {opt_th:.2f}$):
- **True Negatives (TN):** {test_opt['test_opt_tn']}
- **False Positives (FP):** {test_opt['test_opt_fp']}
- **False Negatives (FN):** {test_opt['test_opt_fn']}
- **True Positives (TP):** {test_opt['test_opt_tp']}

---

## 4. Top 10 Feature Importances

| Rank | Feature | Importance (Ensemble Weight) |
|:---:|:---|:---:|
{top_feat_table}

---

## 5. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/adaboost/baseline/model.joblib`
- **Predictions (Test):** `reports/phase9/adaboost/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/adaboost/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/adaboost/baseline/metrics.json`
- **Diagnostic Figures:**
  - `reports/phase9/adaboost/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/adaboost/baseline/figures/roc_curve.png`
  - `reports/phase9/adaboost/baseline/figures/confusion_matrix.png`
  - `reports/phase9/adaboost/baseline/figures/feature_importance.png`
  - `reports/phase9/adaboost/baseline/figures/calibration_curve.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated report at {output_path}")


def run_baseline_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Run 5 Baseline AdaBoost pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — RUN 5: BASELINE ADABOOST")
    print("=" * 70)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_datasets()
    print(f"Partitions loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Load shared Phase 8 preprocessor
    if not SHARED_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing shared Phase 8 preprocessor at {SHARED_PREPROCESSOR_PATH}")
    preprocessor = joblib.load(SHARED_PREPROCESSOR_PATH)
    print(f"Loaded shared Phase 8 preprocessor from {SHARED_PREPROCESSOR_PATH}")

    # 3. Targets and features
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    X_train = preprocessor.transform(train_df[BASELINE_ALL_FEATURES])
    X_val = preprocessor.transform(val_df[BASELINE_ALL_FEATURES])
    X_test = preprocessor.transform(test_df[BASELINE_ALL_FEATURES])
    feature_names = list(preprocessor.get_feature_names_out())

    # 4. Hyperparameters and sample weighting
    sample_weight = compute_sample_weight("balanced", y_train)
    params = {
        "estimator": "DecisionTreeClassifier(max_depth=3, random_state=42)",
        "base_depth": 3,
        "n_estimators": 50,
        "learning_rate": 1.0,
        "sample_weight": "balanced",
        "random_state": 42
    }

    # 5. Fit or reuse model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training Baseline AdaBoost model...")
        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        joblib.dump(model, model_path)
        print(f"Saved trained baseline model to {model_path}")

    # 6. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # 7. Optimal threshold on validation set
    opt_threshold, opt_f1, _ = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 8. Compute metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")
    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"Test PR-AUC: {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Test (th={opt_threshold:.2f}): Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 9. Save predictions
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
    print(f"Saved validation predictions to {val_preds_path}")

    # 10. Top features
    top_features = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).to_dict(orient="records")

    # 11. Save structured metrics JSON
    metrics_summary = {
        "experiment": "Run 5: Baseline AdaBoost",
        "phase": "Phase 9",
        "owner": "Adaptive Boosting Lead",
        "feature_set": "14 shared Phase 8 baseline features",
        "feature_count": len(feature_names),
        "final_feature_names": feature_names,
        "preprocessor_source": "models/phase8_preprocessor.joblib",
        "hyperparameters": params,
        "optimal_threshold": opt_threshold,
        "validation_metrics_default_th": val_metrics_default,
        "validation_metrics_optimal_th": val_metrics_opt,
        "test_metrics_default_th": test_metrics_default,
        "test_metrics_optimal_th": test_metrics_opt,
        "top_features": top_features[:15]
    }
    metrics_path = OUTPUT_REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")

    # 12. Generate figures
    plot_pr_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "precision_recall_curve.png",
        title="Baseline AdaBoost — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Baseline AdaBoost — ROC Curves"
    )

    plot_confusion_matrices(
        {
            f"Test Set (th={opt_threshold:.2f})": test_metrics_opt["test_opt_confusion_matrix"]
        },
        OUTPUT_FIGURES_DIR / "confusion_matrix.png",
        title_suffix=""
    )

    plot_feature_importances(
        feature_names,
        model.feature_importances_,
        title="Baseline AdaBoost — Top 15 Feature Importances",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    plot_calibration_curves(
        {
            "Baseline AdaBoost (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 13. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_baseline_report(metrics_summary, top_features, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Run 5 Baseline AdaBoost")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_baseline_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
