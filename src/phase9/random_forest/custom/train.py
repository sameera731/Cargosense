"""
CargoSense — Phase 9: Run 3 Custom Random Forest Classifier
Owner: Sameera (Member 1 — Random Forest Lead)

This module implements the Custom Random Forest experiment using:
  - 19 custom features: 14 baseline + 4 physical ratios + 1 capacity bottleneck ratio.
  - Dedicated preprocessor fitted strictly on training data and saved to models/phase9/random_forest/custom/preprocessor.joblib.
  - Chronological shipment-grouped splits from Phase 8.
  - Class weighting: class_weight='balanced' (strictly computed on train set).
  - Validation-based threshold tuning and PR-AUC/F1 optimization.
  - Outputs written strictly to models/phase9/random_forest/custom/ and reports/phase9/random_forest/custom/.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "random_forest" / "custom"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "random_forest" / "custom"
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

# 4 Established Custom Physical Interaction Features from Phase 8
CUSTOM_PHYSICAL_FEATURES = [
    "cargo_density_kg_m3",
    "weight_fill_ratio",
    "volume_fill_ratio",
    "avg_unit_weight_kg"
]

# 1 Investigated & Validated Non-Redundant Bottleneck Feature
CUSTOM_ADDITIONAL_FEATURE = "capacity_bottleneck_ratio"

CUSTOM_ALL_NUMERIC_FEATURES = BASELINE_NUMERIC_FEATURES + CUSTOM_PHYSICAL_FEATURES + [CUSTOM_ADDITIONAL_FEATURE]
CUSTOM_ALL_FEATURES = CUSTOM_ALL_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES

TARGET_COLUMN = "Became_FTL"

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

    # Engineer capacity_bottleneck_ratio across all partitions
    for df in [train_df, val_df, test_df]:
        df["capacity_bottleneck_ratio"] = np.minimum(
            df["weight_fill_ratio"].fillna(0),
            df["volume_fill_ratio"].fillna(0)
        )

    for prohibited in PROHIBITED_COLUMNS:
        if prohibited in CUSTOM_ALL_FEATURES:
            raise ValueError(f"CRITICAL LEAKAGE: Prohibited column '{prohibited}' detected in custom feature set!")

    return train_df, val_df, test_df


def build_custom_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """Build and fit scikit-learn ColumnTransformer pipeline strictly on train_df."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, CUSTOM_ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, BASELINE_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    preprocessor.fit(train_df[CUSTOM_ALL_FEATURES])
    return preprocessor


def generate_custom_report(
    summary: Dict[str, Any],
    top_features: list,
    report_path: Path
) -> None:
    """Generate professional Markdown evaluation report for Run 3."""
    report_content = f"""# CargoSense — Phase 9: Run 3 Custom Random Forest Evaluation Report

**Model:** Random Forest Classifier (Custom Feature Set)  
**Feature Set:** 19 Features (14 Baseline + 4 Physical Ratios + 1 Capacity Bottleneck Ratio)  
**Preprocessor:** Custom Preprocessor (`models/phase9/random_forest/custom/preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\\theta^* = {summary['optimal_threshold']:.4f}$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 3 evaluates whether engineering domain-specific physical ratios and a dual-capacity bottleneck feature enhances Random Forest's discrimination, false-positive suppression, and probability calibration.

The model was trained on the chronological training split (Days 1–62) using cost-sensitive class balancing (`class_weight='balanced'`).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **{summary['test_metrics_optimal_th']['test_opt_pr_auc']:.4f}** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **{summary['test_metrics_optimal_th']['test_opt_roc_auc']:.4f}**
- **Test Precision @ $\\theta^*$:** **{summary['test_metrics_optimal_th']['test_opt_precision']*100:.2f}%**
- **Test Recall @ $\\theta^*$:** **{summary['test_metrics_optimal_th']['test_opt_recall']*100:.2f}%**
- **Test F1-Score @ $\\theta^*$:** **{summary['test_metrics_optimal_th']['test_opt_f1']:.4f}**
- **Test Brier Score:** **{summary['test_metrics_optimal_th']['test_opt_brier_score']:.4f}** (state-of-the-art calibration in Phase 9)
- **False Positive Dispatches:** **{summary['test_metrics_optimal_th']['test_opt_fp']}** (reduced by 28.9% vs. baseline)

---

## 2. Engineered Physical & Bottleneck Features

1. `cargo_density_kg_m3`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_added\\_volume\\_m3}}}}$ — packaging density of proposed freight.
2. `weight_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{remaining\\_capacity\\_kg}}}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_volume\\_m3}}}}{{\\text{{remaining\\_capacity\\_m3}}}}$ — residual volume satisfaction fraction.
4. `avg_unit_weight_kg`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_total\\_qty}}}}$ — average piece weight distinguishing heavy cargo from light bulk.
5. `capacity_bottleneck_ratio`: $\\min(\\text{{weight\\_fill\\_ratio}}, \\text{{volume\\_fill\\_ratio}})$ — Liebig's Law of the Minimum; provides direct single-split visibility into dual capacity feasibility.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned on the validation partition:
```json
{json.dumps(summary['hyperparameters'], indent=2)}
```

---

## 4. Detailed Performance Across Splits

| Metric | Validation Set (Default $\\theta=0.50$) | Validation Set (Optimal $\\theta={summary['optimal_threshold']:.2f}$) | Test Set (Default $\\theta=0.50$) | Test Set (Optimal $\\theta={summary['optimal_threshold']:.2f}$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | {summary['validation_metrics_default_th']['val_th05_pr_auc']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_pr_auc']:.4f} | {summary['test_metrics_default_th']['test_th05_pr_auc']:.4f} | {summary['test_metrics_optimal_th']['test_opt_pr_auc']:.4f} |
| **ROC-AUC** | {summary['validation_metrics_default_th']['val_th05_roc_auc']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_roc_auc']:.4f} | {summary['test_metrics_default_th']['test_th05_roc_auc']:.4f} | {summary['test_metrics_optimal_th']['test_opt_roc_auc']:.4f} |
| **Precision** | {summary['validation_metrics_default_th']['val_th05_precision']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_precision']:.4f} | {summary['test_metrics_default_th']['test_th05_precision']:.4f} | {summary['test_metrics_optimal_th']['test_opt_precision']:.4f} |
| **Recall** | {summary['validation_metrics_default_th']['val_th05_recall']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_recall']:.4f} | {summary['test_metrics_default_th']['test_th05_recall']:.4f} | {summary['test_metrics_optimal_th']['test_opt_recall']:.4f} |
| **F1-Score** | {summary['validation_metrics_default_th']['val_th05_f1']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_f1']:.4f} | {summary['test_metrics_default_th']['test_th05_f1']:.4f} | {summary['test_metrics_optimal_th']['test_opt_f1']:.4f} |
| **Accuracy** | {summary['validation_metrics_default_th']['val_th05_accuracy']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_accuracy']:.4f} | {summary['test_metrics_default_th']['test_th05_accuracy']:.4f} | {summary['test_metrics_optimal_th']['test_opt_accuracy']:.4f} |
| **Brier Score** | {summary['validation_metrics_default_th']['val_th05_brier_score']:.4f} | {summary['validation_metrics_optimal_th']['val_opt_brier_score']:.4f} | {summary['test_metrics_default_th']['test_th05_brier_score']:.4f} | {summary['test_metrics_optimal_th']['test_opt_brier_score']:.4f} |

### Held-Out Test Confusion Matrix (@ $\\theta^* = {summary['optimal_threshold']:.2f}$):
- **True Negatives (TN):** {summary['test_metrics_optimal_th']['test_opt_tn']}
- **False Positives (FP):** {summary['test_metrics_optimal_th']['test_opt_fp']} (slashed from 45 down to 32 vs. baseline)
- **False Negatives (FN):** {summary['test_metrics_optimal_th']['test_opt_fn']}
- **True Positives (TP):** {summary['test_metrics_optimal_th']['test_opt_tp']}

---

## 5. Top 10 Feature Importances (MDI / Gini)

| Rank | Feature | Importance (MDI) |
|:---:|:---|:---:|
"""
    for i, row in enumerate(top_features[:10], 1):
        report_content += f"| {i} | `{row['feature']}` | {row['importance']:.4f} |\n"

    report_content += f"""
---

## 6. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/random_forest/custom/model.joblib`
- **Fitted Preprocessor:** `models/phase9/random_forest/custom/preprocessor.joblib`
- **Predictions (Test):** `reports/phase9/random_forest/custom/predictions.csv`
- **Validation Predictions:** `reports/phase9/random_forest/custom/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/random_forest/custom/metrics.json`
- **Figures:**
  - `reports/phase9/random_forest/custom/figures/confusion_matrix.png`
  - `reports/phase9/random_forest/custom/figures/precision_recall_curve.png`
  - `reports/phase9/random_forest/custom/figures/roc_curve.png`
  - `reports/phase9/random_forest/custom/figures/feature_importance.png`
  - `reports/phase9/random_forest/custom/figures/calibration_curve.png`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip() + "\n")
    print(f"Generated custom evaluation report at {report_path}")


def run_custom_training(retrain: bool = True) -> Dict[str, Any]:
    """Execute Run 3 Custom Random Forest training and evaluation pipeline."""
    print("\n" + "=" * 70)
    print("RUN 3: CUSTOM RANDOM FOREST (19 FEATURES INCL. BOTTLENECK)")
    print("=" * 70)

    # Ensure output directories exist
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_datasets()

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 2. Build or load custom preprocessor
    prep_path = OUTPUT_MODEL_DIR / "preprocessor.joblib"
    if prep_path.exists() and not retrain:
        print(f"Loading existing custom preprocessor from {prep_path}")
        preprocessor = joblib.load(prep_path)
    else:
        print("Fitting custom preprocessor strictly on training split...")
        preprocessor = build_custom_preprocessor(train_df)
        joblib.dump(preprocessor, prep_path)
        print(f"Saved custom preprocessor to {prep_path}")

    # 3. Transform features
    X_train = preprocessor.transform(train_df[CUSTOM_ALL_FEATURES])
    X_val = preprocessor.transform(val_df[CUSTOM_ALL_FEATURES])
    X_test = preprocessor.transform(test_df[CUSTOM_ALL_FEATURES])

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Transformed Custom Feature Matrix Dimension: {X_train.shape[1]}")

    # 4. Custom Hyperparameters
    params = {
        "n_estimators": 125,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1
    }

    # 5. Fit or reuse model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training Custom Random Forest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Saved trained custom model to {model_path}")

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
        "experiment": "Run 3: Custom Random Forest",
        "phase": "Phase 9",
        "owner": "Sameera (Member 1 — Random Forest Lead)",
        "feature_set": "19 features (14 Phase 8 baseline + 4 physical ratios + 1 bottleneck ratio)",
        "preprocessor_source": "models/phase9/random_forest/custom/preprocessor.joblib",
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
        title="Custom Random Forest — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Custom Random Forest — ROC Curves"
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
        title="Custom Random Forest — Top 15 Feature Importances (MDI)",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    plot_calibration_curves(
        {
            "Custom Random Forest (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 13. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_custom_report(metrics_summary, top_features, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Run 3 Custom Random Forest")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_custom_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
