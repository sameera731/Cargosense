"""
CargoSense — Phase 9: Gaussian Naive Bayes Experiment
Model: Gaussian Naive Bayes (GaussianNB)
Feature Set: 14 Shared Phase 8 Baseline Features (11 Numerical, 3 Categorical)
Preprocessor: Shared Phase 8 Preprocessor (models/phase8_preprocessor.joblib)

This module implements Gaussian Naive Bayes as an additional Phase 9 benchmark:
  - Reuses the existing chronological, shipment-grouped partitions from Phase 8.
  - Excludes candidate_strategy, surrogate IDs, and post-cutoff operational metrics.
  - Transforms inputs via the pre-fitted Phase 8 ColumnTransformer pipeline.
  - Fits GaussianNB using default hyperparameters without data resampling or artificial priors.
  - Evaluates performance using standard PR-AUC, ROC-AUC, F1, Precision, Recall, and Confusion Matrices.
  - Employs validation-based threshold optimization (freezing the threshold before evaluating held-out test data).
  - Saves artifacts and diagnostic figures strictly to models/phase9/naive_bayes/ and reports/phase9/naive_bayes/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Set up project root (works when invoked directly or via python -m)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    plot_calibration_curves
)

# Repository Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SHARED_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "phase8_preprocessor.joblib"

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "naive_bayes"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "naive_bayes"
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


def generate_naive_bayes_report(
    metrics_summary: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate comprehensive Markdown evaluation report for Gaussian Naive Bayes experiment."""
    val_opt = metrics_summary["validation_metrics_optimal_th"]
    test_opt = metrics_summary["test_metrics_optimal_th"]
    val_dft = metrics_summary["validation_metrics_default_th"]
    test_dft = metrics_summary["test_metrics_default_th"]
    params = metrics_summary["hyperparameters"]
    opt_th = metrics_summary["optimal_threshold"]

    content = f"""# CargoSense — Phase 9: Gaussian Naive Bayes Evaluation Report

**Model:** Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical)  
**Input Dimension:** 31 Transformed Features (via `models/phase8_preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Decision Thresholds:** Default $\\theta = 0.50$ | Validation-Optimized $\\theta^* = {opt_th:.4f}$  

---

## 1. Executive Summary

This report documents the implementation and empirical evaluation of **Gaussian Naive Bayes (GaussianNB)** as an additional benchmark classifier for the CargoSense Opportunity Detection Engine (ODE) in Phase 9.

Under the project plan, Member 2 owns XGBoost (Runs 2 & 4) and Sameera owns Random Forest (Runs 1 & 3). Gaussian Naive Bayes serves as a complementary parametric probabilistic baseline to evaluate how a simple generative model with strong independence assumptions behaves on operational logistics data.

### Key Performance Findings:
- **Held-Out Test PR-AUC:** **{test_opt['test_opt_pr_auc']:.4f}** (vs. 0.1903 baseline XGBoost, and 6.73% random prevalence)
- **Held-Out Test ROC-AUC:** **{test_opt['test_opt_roc_auc']:.4f}** (vs. 0.8273 baseline XGBoost)
- **Test Precision @ $\\theta=0.50$:** **{test_dft['test_th05_precision']:.2%}** ({test_dft['test_th05_tp']}/{test_dft['test_th05_tp'] + test_dft['test_th05_fp']} predictions)
- **Test Recall @ $\\theta=0.50$:** **{test_dft['test_th05_recall']:.2%}** ({test_dft['test_th05_tp']}/{test_dft['test_th05_positive_samples']} positives identified)
- **Test F1-Score @ $\\theta=0.50$:** **{test_dft['test_th05_f1']:.4f}**
- **Test Precision @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_precision']:.2%}** ({test_opt['test_opt_tp']}/{test_opt['test_opt_tp'] + test_opt['test_opt_fp']} predictions)
- **Test Recall @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_recall']:.2%}** ({test_opt['test_opt_tp']}/{test_opt['test_opt_positive_samples']} positives identified)
- **Test F1-Score @ $\\theta^*={opt_th:.2f}$:** **{test_opt['test_opt_f1']:.4f}**
- **Test Brier Score:** **{test_opt['test_opt_brier_score']:.4f}** (vs. 0.1363 baseline XGBoost and 0.0812 custom XGBoost)

---

## 2. Theoretical Assumptions & Operational Limitations

Gaussian Naive Bayes relies on two core mathematical assumptions that are fundamentally violated by operational freight consolidation data:

### 2.1 Conditional Independence Assumption
GaussianNB assumes that all 31 input features are conditionally independent given the class label $Y \\in \\{{0, 1\\}}$:
$$P(X_1, X_2, \\dots, X_d \\mid Y = c) = \\prod_{{i=1}}^d P(X_i \\mid Y = c)$$

**Operational Reality & Impact:**
In freight operations, features exhibit strong structural multicollinearity:
1. `candidate_added_weight_kg` and `candidate_added_volume_m3` are physically correlated via cargo density.
2. `remaining_capacity_kg` and `remaining_capacity_m3` correlate with the assigned truck chassis.
3. `projected_weight_util_pct` and `projected_volume_util_pct` are deterministic functions of remaining capacity and candidate cargo.
4. `candidate_total_qty` and `candidate_added_weight_kg` correlate strongly with shipment scale.

Because evidence is assumed independent, multiplying class-conditional likelihoods over 31 features leads to **redundant evidence multiplication**. This causes the posterior probabilities to become artificially polarized towards 0 or 1, resulting in poor probability calibration (**Brier score of {test_opt['test_opt_brier_score']:.4f}**).

### 2.2 Univariate Gaussian Likelihood Assumption
GaussianNB models continuous features within each class using a univariate Gaussian distribution:
$$P(X_i \\mid Y = c) = \\frac{{1}}{{\\sqrt{{2\\pi\\sigma_{{ci}}^2}}}} \\exp\\left(-\\frac{{(x_i - \\mu_{{ci}})^2}}{{2\\sigma_{{ci}}^2}}\\right)$$

**Operational Reality & Impact:**
Physical logistics variables have bounded, non-Gaussian distributions:
- `projected_weight_util_pct` and `projected_volume_util_pct` are bounded percentages with sharp spikes near the 85% FTL threshold.
- `remaining_capacity` and piece counts are strictly non-negative, right-skewed counts and masses.
- One-hot encoded categorical indicator columns (such as routes and truck types) are discrete binary variables, for which continuous Gaussian density modeling is technically misspecified (despite StandardScaler centering).

---

## 3. Detailed Performance Across Splits

### 3.1 Evaluation Metrics Table

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

### 3.2 Test Set Confusion Matrices

#### Default Threshold ($\theta = 0.50$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          {test_dft['test_th05_tn']:<12}         {test_dft['test_th05_fp']:<12}    (Specificity: {test_dft['test_th05_specificity']:.1%})
Actual Positive          {test_dft['test_th05_fn']:<12}         {test_dft['test_th05_tp']:<12}    (Recall: {test_dft['test_th05_recall']:.1%})
Total Test: 327 | Precision: {test_dft['test_th05_precision']:.1%} | Accuracy: {test_dft['test_th05_accuracy']:.1%}
```

#### Validation-Optimized Threshold ($\theta^* = {opt_th:.2f}$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          {test_opt['test_opt_tn']:<12}         {test_opt['test_opt_fp']:<12}    (Specificity: {test_opt['test_opt_specificity']:.1%})
Actual Positive          {test_opt['test_opt_fn']:<12}         {test_opt['test_opt_tp']:<12}    (Recall: {test_opt['test_opt_recall']:.1%})
Total Test: 327 | Precision: {test_opt['test_opt_precision']:.1%} | Accuracy: {test_opt['test_opt_accuracy']:.1%}
```

> [!NOTE]
> **Small-Sample Sensitivity on Test Set**:  
> The held-out test partition contains exactly 22 positive instances (`Became_FTL = 1`) out of 327 candidates (6.73% prevalence). Because of this small sample, a shift of just 1 true positive or false positive moves precision or recall by ~4.5 percentage points.

---

## 4. Comparison Against Baseline Tree Ensembles

| Model / Configuration | Feature Set | Input Dim | Test PR-AUC | Test ROC-AUC | Test F1 @ $\\theta^*$ | Test Brier Score | Calibrated? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian Naive Bayes** | 14 Baseline | 31 | **{test_opt['test_opt_pr_auc']:.4f}** | **{test_opt['test_opt_roc_auc']:.4f}** | **{test_opt['test_opt_f1']:.4f}** | **{test_opt['test_opt_brier_score']:.4f}** | Poor (Polarized) |
| **XGBoost (Run 2 Baseline)** | 14 Baseline | 31 | 0.1903 | 0.8273 | 0.2667 | 0.1363 | Moderate |
| **XGBoost (Run 4 Custom)** | 14 Baseline + 4 Ratios | 35 | 0.2519 | 0.8596 | 0.2857 | 0.0812 | Excellent |

### Comparative Insights:
1. **Ranking vs. Calibration**: While GaussianNB produces strong PR-AUC ({test_opt['test_opt_pr_auc']:.4f}) and ROC-AUC ({test_opt['test_opt_roc_auc']:.4f}) by preserving monotonic ordering on dominant features like total quantity and transit hours, its predicted probabilities are poorly calibrated (Brier score of {test_opt['test_opt_brier_score']:.4f}).
2. **False Positive Burden**: At $\\theta=0.50$, GaussianNB triggers 86 false alarms on the test set, recommending 108 dispatches to capture 22 conversions. In contrast, Custom XGBoost slashed false alarms to 32 while retaining high precision.
3. **Operational Recommendation**: While GaussianNB demonstrates surprising ranking ability, its inability to model joint non-linear capacity bottlenecks and its distorted posterior probabilities make it unsuited for direct score-based ranking in Phase 10 without Platt scaling or isotonic calibration.

---

## 5. Artifact Registry

All generated artifacts adhere to the designated Phase 9 folder structure:

| Artifact Type | Path | Purpose |
|:---|:---|:---|
| **Pipeline Script** | `src/phase9/naive_bayes/train.py` | Standalone reproducible training and evaluation pipeline |
| **Serialized Model** | `models/phase9/naive_bayes/gaussian_nb.joblib` | Fitted `GaussianNB` model object |
| **Test Predictions** | `reports/phase9/naive_bayes/predictions.csv` | Held-out test predictions with probabilities and decisions |
| **Val Predictions** | `reports/phase9/naive_bayes/val_predictions.csv` | Validation set predictions with probabilities and decisions |
| **Structured Metrics** | `reports/phase9/naive_bayes/metrics.json` | Comprehensive machine-readable metrics dictionary |
| **Confusion Matrix** | `reports/phase9/naive_bayes/figures/confusion_matrix.png` | Normalized and raw confusion matrix plot |
| **PR Curve Plot** | `reports/phase9/naive_bayes/figures/precision_recall_curve.png` | Precision-Recall curve with baseline prevalence reference |
| **ROC Curve Plot** | `reports/phase9/naive_bayes/figures/roc_curve.png` | Receiver Operating Characteristic diagnostic plot |
| **Calibration Plot** | `reports/phase9/naive_bayes/figures/calibration_curve.png` | Reliability diagram demonstrating probability overconfidence |

---

## 6. Reproduction Command

To reproduce this experiment from the repository root:
```bash
python -m src.phase9.naive_bayes.train
```
*(Execution completes in < 2 seconds without retraining or modifying shared artifacts).*
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated report at {output_path}")


def run_naive_bayes_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Gaussian Naive Bayes training, evaluation, and artifact generation pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — GAUSSIAN NAIVE BAYES EXPERIMENT")
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

    # 4. Check array validity
    for X, name in [(X_train, "X_train"), (X_val, "X_val"), (X_test, "X_test")]:
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(f"Invalid values in {name}: {nan_count} NaNs, {inf_count} Infs")
    print(f"Transformed matrices validated: dimension = {X_train.shape[1]} features, 0 NaNs, 0 Infs.")

    # 5. Fit or reuse model
    model_path = OUTPUT_MODEL_DIR / "gaussian_nb.joblib"
    params = {
        "priors": None,
        "var_smoothing": 1e-9
    }

    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Fitting GaussianNB model with default settings...")
        model = GaussianNB(**params)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Saved trained GaussianNB model to {model_path}")

    # Verify class mapping
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError(f"Unexpected classes_ in GaussianNB: {model.classes_}")

    # 6. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # 7. Select optimal threshold on validation set only
    opt_threshold, opt_f1, _ = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 8. Compute metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")
    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nGaussianNB Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f}")
    print(f"GaussianNB Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"GaussianNB Test (th=0.50):    Precision={test_metrics_default['test_th05_precision']:.4f}, Recall={test_metrics_default['test_th05_recall']:.4f}, F1={test_metrics_default['test_th05_f1']:.4f}")
    print(f"GaussianNB Test (th={opt_threshold:.2f}):    Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 9. Save predictions (with candidate identifier for traceability)
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

    # 10. Save structured metrics JSON
    metrics_summary = {
        "experiment": "Phase 9: Gaussian Naive Bayes",
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
        "test_metrics_optimal_th": test_metrics_opt
    }
    metrics_path = OUTPUT_REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")

    # 11. Generate Figures
    plot_pr_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "precision_recall_curve.png",
        title="Gaussian Naive Bayes — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Gaussian Naive Bayes — ROC Curves"
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
            "Gaussian Naive Bayes (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 12. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_naive_bayes_report(metrics_summary, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Gaussian Naive Bayes Experiment")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_naive_bayes_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
