"""
CargoSense — Phase 9: Baseline Gaussian Naive Bayes Classifier
Model: Gaussian Naive Bayes (Baseline)
Feature Set: 14 Shared Phase 8 Baseline Features (11 Numerical, 3 Categorical)
Preprocessor: Shared Phase 8 Preprocessor (models/phase8_preprocessor.joblib)

This module implements the Baseline Gaussian Naive Bayes experiment:
  - Reuses the existing chronological, shipment-grouped partitions from Phase 8.
  - Excludes candidate_strategy, surrogate IDs, and post-cutoff operational metrics.
  - Transforms inputs via the pre-fitted Phase 8 ColumnTransformer pipeline.
  - Fits GaussianNB using default hyperparameters (var_smoothing=1e-9, empirical priors).
  - Evaluates performance using standard PR-AUC, ROC-AUC, F1, Precision, Recall, Accuracy, Brier score.
  - Employs validation-based threshold optimization (freezing the threshold before evaluating held-out test data).
  - Saves artifacts and diagnostic figures strictly to models/phase9/naive_bayes/baseline/ and reports/phase9/naive_bayes/baseline/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Set up project root (works when invoked directly or via python -m)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

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

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "naive_bayes" / "baseline"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "naive_bayes" / "baseline"
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


def compute_gaussian_feature_importance(model: GaussianNB, feature_names: list) -> pd.DataFrame:
    """
    Compute standardized class mean separation (Cohen's d) for GaussianNB.
    Quantifies feature discriminative separation between converted (1) and non-converted (0) opportunities:
        d_j = |mu_1j - mu_0j| / sqrt(0.5 * (sigma_1j^2 + sigma_0j^2))
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


def generate_baseline_report(
    metrics_summary: Dict[str, Any],
    importance_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate comprehensive Markdown evaluation report for Baseline Gaussian Naive Bayes."""
    val_opt = metrics_summary["validation_metrics_optimal_th"]
    test_opt = metrics_summary["test_metrics_optimal_th"]
    val_dft = metrics_summary["validation_metrics_default_th"]
    test_dft = metrics_summary["test_metrics_default_th"]
    params = metrics_summary["hyperparameters"]
    opt_th = metrics_summary["optimal_threshold"]

    top_feat_table = "\n".join([
        f"| {idx+1} | `{row['feature']}` | {row['importance']:.4f} |"
        for idx, row in importance_df.head(10).iterrows()
    ])

    content = f"""# CargoSense — Phase 9: Baseline Gaussian Naive Bayes Evaluation Report

**Model:** Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Experiment:** Baseline GaussianNB (14 Shared Phase 8 Features, 31 Encoded Dims)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Operating Thresholds:** Default $\\theta = 0.50$ | Validation-Optimized $\\theta^* = {opt_th:.4f}$  

---

## 1. Executive Summary

This report documents the implementation and evaluation of **Baseline Gaussian Naive Bayes** for Phase 9 of the CargoSense Opportunity Detection Engine (ODE).

Gaussian Naive Bayes serves as a classical probabilistic generative benchmark. It assumes that features are conditionally independent given the FTL outcome class and follow class-conditional univariate Gaussian distributions.

### Key Performance Metrics:
- **Held-Out Test PR-AUC:** **{test_opt['test_opt_pr_auc']:.4f}** (vs. 6.73% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **{test_opt['test_opt_roc_auc']:.4f}**
- **Test Precision @ $\\theta=0.50$:** **{test_dft['test_th05_precision']:.2%}** ({test_dft['test_th05_tp']}/{test_dft['test_th05_tp'] + test_dft['test_th05_fp']} predictions)
- **Test Recall @ $\\theta=0.50$:** **{test_dft['test_th05_recall']:.2%}** ({test_dft['test_th05_tp']}/{test_dft['test_th05_positive_samples']} positives identified)
- **Test F1-Score @ $\\theta=0.50$:** **{test_dft['test_th05_f1']:.4f}**
- **Test Precision @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_precision']:.2%}** ({test_opt['test_opt_tp']}/{test_opt['test_opt_tp'] + test_opt['test_opt_fp']} predictions)
- **Test Recall @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_recall']:.2%}** ({test_opt['test_opt_tp']}/{test_opt['test_opt_positive_samples']} positives identified)
- **Test F1-Score @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_f1']:.4f}**
- **Test Brier Score:** **{test_opt['test_opt_brier_score']:.4f}**

---

## 2. Theoretical Assumptions & Operational Limitations

Gaussian Naive Bayes relies on two mathematical assumptions that conflict with operational logistics data:

1. **Conditional Independence**:
   GaussianNB assumes features are conditionally independent:
   $$P(X_1, \\dots, X_d \\mid Y = c) = \\prod_{{i=1}}^d P(X_i \\mid Y = c)$$
   In freight consolidation, weight, volume, pieces, and utilization percentages are strongly correlated. Multiplying likelihoods across 31 features results in redundant evidence overcounting, pushing posterior probabilities to extreme values (high Brier score of {test_opt['test_opt_brier_score']:.4f}).

2. **Univariate Gaussian Distribution**:
   Bounded operational metrics (utilization percentages capped near 85–100%, non-negative piece counts, one-hot encoded route indicators) deviate significantly from continuous unbounded normal distributions.

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\\theta=0.50$) | Validation Set (Optimal $\\theta={opt_th:.2f}$) | Test Set (Default $\\theta=0.50$) | Test Set (Optimal $\\theta={opt_th:.2f}$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC (Average Precision)** | {val_dft['val_th05_pr_auc']:.4f} | {val_opt['val_opt_pr_auc']:.4f} | {test_dft['test_th05_pr_auc']:.4f} | {test_opt['test_opt_pr_auc']:.4f} |
| **ROC-AUC** | {val_dft['val_th05_roc_auc']:.4f} | {val_opt['val_opt_roc_auc']:.4f} | {test_dft['test_th05_roc_auc']:.4f} | {test_opt['test_opt_roc_auc']:.4f} |
| **Precision** | {val_dft['val_th05_precision']:.4f} | {val_opt['val_opt_precision']:.4f} | {test_dft['test_th05_precision']:.4f} | {test_opt['test_opt_precision']:.4f} |
| **Recall** | {val_dft['val_th05_recall']:.4f} | {val_opt['val_opt_recall']:.4f} | {test_dft['test_th05_recall']:.4f} | {test_opt['test_opt_recall']:.4f} |
| **F1-Score** | {val_dft['val_th05_f1']:.4f} | {val_opt['val_opt_f1']:.4f} | {test_dft['test_th05_f1']:.4f} | {test_opt['test_opt_f1']:.4f} |
| **Accuracy** | {val_dft['val_th05_accuracy']:.4f} | {val_opt['val_opt_accuracy']:.4f} | {test_dft['test_th05_accuracy']:.4f} | {test_opt['test_opt_accuracy']:.4f} |
| **Specificity** | {val_dft['val_th05_specificity']:.4f} | {val_opt['val_opt_specificity']:.4f} | {test_dft['test_th05_specificity']:.4f} | {test_opt['test_opt_specificity']:.4f} |
| **Brier Score** | {val_dft['val_th05_brier_score']:.4f} | {val_opt['val_opt_brier_score']:.4f} | {test_dft['test_th05_brier_score']:.4f} | {test_opt['test_opt_brier_score']:.4f} |
| **Positive Predictions Count** | {val_dft['val_th05_tp'] + val_dft['val_th05_fp']} | {val_opt['val_opt_tp'] + val_opt['val_opt_fp']} | {test_dft['test_th05_tp'] + test_dft['test_th05_fp']} | {test_opt['test_opt_tp'] + test_opt['test_opt_fp']} |

### Held-Out Test Confusion Matrices:
- **Default ($\\theta=0.50$):** TN={test_dft['test_th05_tn']}, FP={test_dft['test_th05_fp']}, FN={test_dft['test_th05_fn']}, TP={test_dft['test_th05_tp']} (Precision: {test_dft['test_th05_precision']:.1%}, Recall: {test_dft['test_th05_recall']:.1%})
- **Optimal ($\\theta^*={opt_th:.2f}$):** TN={test_opt['test_opt_tn']}, FP={test_opt['test_opt_fp']}, FN={test_opt['test_opt_fn']}, TP={test_opt['test_opt_tp']} (Precision: {test_opt['test_opt_precision']:.1%}, Recall: {test_opt['test_opt_recall']:.1%})

---

## 4. Feature Discriminative Power (Standardized Mean Separation)

| Rank | Feature | Standardized Separation ($d$) |
|:---:|:---|:---:|
{top_feat_table}

---

## 5. Artifact Registry

- **Model Artifact:** `models/phase9/naive_bayes/baseline/model.joblib`
- **Model Configuration:** `models/phase9/naive_bayes/baseline/config.json`
- **Predictions (Test):** `reports/phase9/naive_bayes/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/naive_bayes/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/naive_bayes/baseline/metrics.json`
- **Figures:**
  - `reports/phase9/naive_bayes/baseline/figures/confusion_matrix.png`
  - `reports/phase9/naive_bayes/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/roc_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/calibration_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/feature_importance.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated baseline report at {output_path}")


def run_baseline_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Baseline Gaussian Naive Bayes pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — BASELINE GAUSSIAN NAIVE BAYES EXPERIMENT")
    print("=" * 70)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_and_validate_datasets()

    # 2. Load shared Phase 8 preprocessor (never refit)
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
    feature_names = list(preprocessor.get_feature_names_out())

    # 4. Fit or load model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    params = {
        "priors": None,
        "var_smoothing": 1e-9
    }

    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Fitting Baseline GaussianNB model on training split...")
        model = GaussianNB(**params)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Saved trained model to {model_path}")

    # 5. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # 6. Optimal threshold on validation set
    opt_threshold, opt_f1, _ = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 7. Compute metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")
    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"Baseline Test PR-AUC: {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Baseline Test (th=0.50): Precision={test_metrics_default['test_th05_precision']:.4f}, Recall={test_metrics_default['test_th05_recall']:.4f}, F1={test_metrics_default['test_th05_f1']:.4f}")
    print(f"Baseline Test (th={opt_threshold:.2f}): Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 8. Save predictions
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

    # 9. Compute feature importance (standardized class separation)
    importance_df = compute_gaussian_feature_importance(model, feature_names)

    # 10. Save config
    config = {
        "model_name": "Baseline GaussianNB",
        "feature_set": "14 shared Phase 8 baseline features",
        "n_features": len(feature_names),
        "features": feature_names,
        "hyperparameters": params,
        "classes": [int(c) for c in model.classes_],
        "class_prior": [float(p) for p in model.class_prior_],
        "optimal_threshold": float(opt_threshold)
    }
    config_path = OUTPUT_MODEL_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")

    # 11. Save structured metrics JSON
    metrics_summary = {
        "experiment": "Phase 9: Baseline Gaussian Naive Bayes",
        "phase": "Phase 9",
        "algorithm": "GaussianNB",
        "feature_set": "14 shared Phase 8 baseline features",
        "preprocessor_source": "models/phase8_preprocessor.joblib",
        "hyperparameters": params,
        "classes": [int(c) for c in model.classes_],
        "class_prior": [float(p) for p in model.class_prior_],
        "optimal_threshold": float(opt_threshold),
        "validation_metrics_default_th": val_metrics_default,
        "validation_metrics_optimal_th": val_metrics_opt,
        "test_metrics_default_th": test_metrics_default,
        "test_metrics_optimal_th": test_metrics_opt,
        "top_features": importance_df.head(15).to_dict(orient="records")
    }
    metrics_path = OUTPUT_REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")

    # 12. Generate Figures
    plot_pr_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "precision_recall_curve.png",
        title="Baseline Gaussian Naive Bayes — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Baseline Gaussian Naive Bayes — ROC Curves"
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
            "Baseline GaussianNB (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )

    plot_feature_importances(
        feature_names,
        importance_df["importance"].values,
        title="Baseline GaussianNB — Feature Discriminative Separation (|d|)",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=15
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 13. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_baseline_report(metrics_summary, importance_df, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Baseline Gaussian Naive Bayes")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_baseline_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
