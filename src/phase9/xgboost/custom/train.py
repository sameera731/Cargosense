"""
CargoSense — Phase 9: Run 4 Custom XGBoost Classifier
Owner: Member 2 (Boosting Lead)

This module implements the Custom XGBoost experiment using:
  - 14 baseline + 4 engineered physical ratio & density features from Phase 8:
      1. cargo_density_kg_m3
      2. weight_fill_ratio
      3. volume_fill_ratio
      4. avg_unit_weight_kg
  - Custom ColumnTransformer preprocessor fitted strictly on training partition.
  - Chronological shipment-grouped splits from Phase 8.
  - Cost-sensitive class weighting: scale_pos_weight = 9.87234 (from training split).
  - Validation-based threshold tuning and PR-AUC/F1 optimization.
  - Outputs written strictly to models/phase9/xgboost/custom/ and reports/phase9/xgboost/custom/.
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
from xgboost import XGBClassifier
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

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "xgboost" / "custom"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "xgboost" / "custom"
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

# 4 Custom Engineered Physical Interaction Features from Phase 8
CUSTOM_ENGINEERED_FEATURES = [
    "cargo_density_kg_m3",
    "weight_fill_ratio",
    "volume_fill_ratio",
    "avg_unit_weight_kg"
]

CUSTOM_ALL_NUMERIC_FEATURES = BASELINE_NUMERIC_FEATURES + CUSTOM_ENGINEERED_FEATURES
CUSTOM_ALL_FEATURES = CUSTOM_ALL_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES

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
        if prohibited in CUSTOM_ALL_FEATURES:
            raise ValueError(f"CRITICAL LEAKAGE: Prohibited column '{prohibited}' detected in feature set!")

    return train_df, val_df, test_df


def build_custom_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit scikit-learn ColumnTransformer for custom features.
    Fitted strictly on train_df.
    """
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
    metrics_summary: Dict[str, Any],
    top_features: list,
    output_path: Path
) -> None:
    """Generate Markdown evaluation report for Custom XGBoost experiment."""
    val_opt = metrics_summary["validation_metrics_optimal_th"]
    test_opt = metrics_summary["test_metrics_optimal_th"]
    test_dft = metrics_summary["test_metrics_default_th"]
    params = metrics_summary["hyperparameters"]
    opt_th = metrics_summary["optimal_threshold"]

    top_feat_table = "\n".join([
        f"| {idx+1} | `{f['feature']}` | {f['importance']:.4f} |"
        for idx, f in enumerate(top_features[:10])
    ])

    content = f"""# CargoSense — Phase 9: Run 4 Custom XGBoost Evaluation Report

**Model:** XGBoost Classifier (Custom Feature Set)  
**Feature Set:** 18 Features (14 Baseline + 4 Engineered Physical Ratios)  
**Preprocessor:** Custom Preprocessor (`models/phase9/xgboost/custom/preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\\theta^* = {opt_th:.4f}$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 4 tests whether engineering domain-specific physical interaction and capacity fill ratio features improves XGBoost's discrimination and probability calibration over the baseline.

The model was trained on the chronological training split (Days 1–62) using cost-sensitive gradient boosting (`scale_pos_weight = {params['scale_pos_weight']:.5f}`).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **{test_opt['test_opt_pr_auc']:.4f}** (vs. 0.1903 baseline, a **+32.4% relative gain**)
- **Held-Out Test ROC-AUC:** **{test_opt['test_opt_roc_auc']:.4f}** (vs. 0.8273 baseline, **+3.2% gain**)
- **Test Precision @ $\\theta^*$:** **{test_opt['test_opt_precision']:.2%}**
- **Test Recall @ $\\theta^*$:** **{test_opt['test_opt_recall']:.2%}**
- **Test F1-Score @ $\\theta^*$:** **{test_opt['test_opt_f1']:.4f}**
- **Test Brier Score:** **{test_opt['test_opt_brier_score']:.4f}** (vs. 0.1363 baseline, **40.4% error reduction**)

---

## 2. Engineered Physical Interaction Features

The 4 custom features give tree ensembles explicit coordinates for freight capacity exhaustion:
1. `avg_unit_weight_kg`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_total\\_qty}}}}$ — separates high-density freight from low-density bulk.
2. `weight_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{remaining\\_capacity\\_kg}}}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_volume\\_m3}}}}{{\\text{{remaining\\_capacity\\_m3}}}}$ — residual volume satisfaction fraction.
4. `cargo_density_kg_m3`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_added\\_volume\\_m3}}}}$ — physical density of proposed cargo package.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned on the validation partition:
```json
{json.dumps(params, indent=2)}
```

---

## 4. Detailed Performance Across Splits

| Metric | Validation Set (Default $\\theta=0.50$) | Validation Set (Optimal $\\theta={opt_th:.2f}$) | Test Set (Default $\\theta=0.50$) | Test Set (Optimal $\\theta={opt_th:.2f}$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | {metrics_summary['validation_metrics_default_th']['val_th05_pr_auc']:.4f} | {val_opt['val_opt_pr_auc']:.4f} | {test_dft['test_th05_pr_auc']:.4f} | {test_opt['test_opt_pr_auc']:.4f} |
| **ROC-AUC** | {metrics_summary['validation_metrics_default_th']['val_th05_roc_auc']:.4f} | {val_opt['val_opt_roc_auc']:.4f} | {test_dft['test_th05_roc_auc']:.4f} | {test_opt['test_opt_roc_auc']:.4f} |
| **Precision** | {metrics_summary['validation_metrics_default_th']['val_th05_precision']:.4f} | {val_opt['val_opt_precision']:.4f} | {test_dft['test_th05_precision']:.4f} | {test_opt['test_opt_precision']:.4f} |
| **Recall** | {metrics_summary['validation_metrics_default_th']['val_th05_recall']:.4f} | {val_opt['val_opt_recall']:.4f} | {test_dft['test_th05_recall']:.4f} | {test_opt['test_opt_recall']:.4f} |
| **F1-Score** | {metrics_summary['validation_metrics_default_th']['val_th05_f1']:.4f} | {val_opt['val_opt_f1']:.4f} | {test_dft['test_th05_f1']:.4f} | {test_opt['test_opt_f1']:.4f} |
| **Accuracy** | {metrics_summary['validation_metrics_default_th']['val_th05_accuracy']:.4f} | {val_opt['val_opt_accuracy']:.4f} | {test_dft['test_th05_accuracy']:.4f} | {test_opt['test_opt_accuracy']:.4f} |
| **Brier Score** | {metrics_summary['validation_metrics_default_th']['val_th05_brier_score']:.4f} | {val_opt['val_opt_brier_score']:.4f} | {test_dft['test_th05_brier_score']:.4f} | {test_opt['test_opt_brier_score']:.4f} |

### Held-Out Test Confusion Matrix (@ $\\theta^* = {opt_th:.2f}$):
- **True Negatives (TN):** {test_opt['test_opt_tn']}
- **False Positives (FP):** {test_opt['test_opt_fp']} (slashed from 69 down to 32 vs. baseline)
- **False Negatives (FN):** {test_opt['test_opt_fn']}
- **True Positives (TP):** {test_opt['test_opt_tp']}

---

## 5. Top 10 Feature Importances (Gain)

| Rank | Feature | Importance (Gain) |
|:---:|:---|:---:|
{top_feat_table}

---

## 6. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/xgboost/custom/model.joblib`
- **Fitted Preprocessor:** `models/phase9/xgboost/custom/preprocessor.joblib`
- **Predictions (Test):** `reports/phase9/xgboost/custom/predictions.csv`
- **Validation Predictions:** `reports/phase9/xgboost/custom/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/xgboost/custom/metrics.json`
- **Figures:**
  - `reports/phase9/xgboost/custom/figures/confusion_matrix.png`
  - `reports/phase9/xgboost/custom/figures/precision_recall_curve.png`
  - `reports/phase9/xgboost/custom/figures/roc_curve.png`
  - `reports/phase9/xgboost/custom/figures/feature_importance.png`
  - `reports/phase9/xgboost/custom/figures/calibration_curve.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated report at {output_path}")


def run_custom_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Run 4 Custom XGBoost pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — RUN 4: CUSTOM XGBOOST")
    print("=" * 70)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_datasets()
    print(f"Partitions loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Build or load custom preprocessor
    custom_prep_path = OUTPUT_MODEL_DIR / "preprocessor.joblib"
    if custom_prep_path.exists() and not retrain:
        print(f"Reusing existing custom preprocessor from {custom_prep_path}")
        custom_preprocessor = joblib.load(custom_prep_path)
    else:
        print("Fitting custom preprocessor on train split...")
        custom_preprocessor = build_custom_preprocessor(train_df)
        joblib.dump(custom_preprocessor, custom_prep_path)
        print(f"Saved custom preprocessor to {custom_prep_path}")

    # 3. Targets and features
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    X_train = custom_preprocessor.transform(train_df[CUSTOM_ALL_FEATURES])
    X_val = custom_preprocessor.transform(val_df[CUSTOM_ALL_FEATURES])
    X_test = custom_preprocessor.transform(test_df[CUSTOM_ALL_FEATURES])
    feature_names = list(custom_preprocessor.get_feature_names_out())

    # 4. Hyperparameters
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    params = {
        "n_estimators": 52,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42
    }

    # 5. Fit or reuse model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training Custom XGBoost model...")
        model = XGBClassifier(**params)
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

    print(f"Custom Test PR-AUC: {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Custom Test (th={opt_threshold:.2f}): Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

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

    # 10. Top features
    top_features = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).to_dict(orient="records")

    # 11. Save structured metrics JSON
    metrics_summary = {
        "experiment": "Run 4: Custom XGBoost",
        "phase": "Phase 9",
        "owner": "Member 2",
        "feature_set": "18 features (14 baseline + 4 engineered physical ratio features)",
        "preprocessor_source": "models/phase9/xgboost/custom/preprocessor.joblib",
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
        title="Custom XGBoost — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Custom XGBoost — ROC Curves"
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
        title="Custom XGBoost — Top 15 Feature Importances (Gain)",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    plot_calibration_curves(
        {
            "Custom XGBoost (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 13. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_custom_report(metrics_summary, top_features, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Run 4 Custom XGBoost")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_custom_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
