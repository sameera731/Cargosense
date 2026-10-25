"""
CargoSense — Phase 9: Run 6 Custom AdaBoost Classifier
Owner: Adaptive Boosting Lead

This module implements the Custom AdaBoost experiment using:
  - 14 baseline + exactly 4 engineered physical ratio & density features from Phase 8:
      1. cargo_density_kg_m3
      2. weight_fill_ratio
      3. volume_fill_ratio
      4. avg_unit_weight_kg
  - Explicit exclusions: Does NOT include capacity_bottleneck_ratio, capacity_utilization_imbalance,
    or any newly engineered features.
  - Custom ColumnTransformer preprocessor fitted strictly on the training partition.
  - Chronological shipment-grouped splits from Phase 8.
  - Sample weighting strategy: balanced sample weights computed strictly on training data
    to address the ~9.87:1 class imbalance without artificial synthetic sampling.
  - Decision tree weak learners (max_depth=3, n_estimators=70, learning_rate=0.8).
  - Validation-based threshold tuning and PR-AUC/F1 optimization.
  - Outputs written strictly to models/phase9/adaboost/custom/ and reports/phase9/adaboost/custom/.
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "phase9" / "adaboost" / "custom"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase9" / "adaboost" / "custom"
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

# Exactly 4 Approved Custom Physical Interaction Features from Phase 8
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
    "actual_ftl_conversion",
    "capacity_bottleneck_ratio",
    "capacity_utilization_imbalance"
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

    # Ensure the 4 custom physical features exist and are safely formatted
    for df in [train_df, val_df, test_df]:
        if "cargo_density_kg_m3" not in df.columns:
            df["cargo_density_kg_m3"] = df["candidate_added_weight_kg"] / df["candidate_added_volume_m3"].replace(0, np.nan)
        if "weight_fill_ratio" not in df.columns:
            df["weight_fill_ratio"] = df["candidate_added_weight_kg"] / df["remaining_capacity_kg"].replace(0, np.nan)
        if "volume_fill_ratio" not in df.columns:
            df["volume_fill_ratio"] = df["candidate_added_volume_m3"] / df["remaining_capacity_m3"].replace(0, np.nan)
        if "avg_unit_weight_kg" not in df.columns:
            df["avg_unit_weight_kg"] = df["candidate_added_weight_kg"] / df["candidate_total_qty"].replace(0, np.nan)

    # Audit prohibited leakage columns
    for prohibited in PROHIBITED_COLUMNS:
        if prohibited in CUSTOM_ALL_FEATURES:
            raise ValueError(f"CRITICAL LEAKAGE: Prohibited column '{prohibited}' detected in custom feature set!")

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
    """Generate Markdown evaluation report for Custom AdaBoost experiment."""
    val_opt = metrics_summary["validation_metrics_optimal_th"]
    test_opt = metrics_summary["test_metrics_optimal_th"]
    test_dft = metrics_summary["test_metrics_default_th"]
    params = metrics_summary["hyperparameters"]
    opt_th = metrics_summary["optimal_threshold"]

    top_feat_table = "\n".join([
        f"| {idx+1} | `{f['feature']}` | {f['importance']:.4f} |"
        for idx, f in enumerate(top_features[:10])
    ])

    content = f"""# CargoSense — Phase 9: Run 6 Custom AdaBoost Evaluation Report

**Model:** AdaBoost Classifier (Custom Feature Set)  
**Base Estimator:** `DecisionTreeClassifier(max_depth=3)`  
**Feature Set:** 18 Features (14 Baseline + Exactly 4 Engineered Physical Ratios — 35 Encoded Dims)  
**Preprocessor:** Custom Fitted Preprocessor (`models/phase9/adaboost/custom/preprocessor.joblib`)  
**Target Variable:** `{TARGET_COLUMN}` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\\theta^* = {opt_th:.4f}$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 6 evaluates whether supplementing the 14 baseline features with 4 domain-specific physical interaction ratios enhances AdaBoost's classification and ranking capability.
The model was trained strictly on the chronological training split (Days 1–62, 1,533 records, 141 positives) with training-derived balanced sample weights ($w_0 \\approx 0.5506, w_1 \\approx 5.4362$) to address the 9.87:1 class imbalance.

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **{test_opt['test_opt_pr_auc']:.4f}** (vs. {test_opt['test_opt_positive_rate']:.1%} random baseline prevalence)
- **Held-Out Test ROC-AUC:** **{test_opt['test_opt_roc_auc']:.4f}**
- **Test Precision @ $\\theta^*$:** **{test_opt['test_opt_precision']:.2%}**
- **Test Recall @ $\\theta^*$:** **{test_opt['test_opt_recall']:.2%}**
- **Test F1-Score @ $\\theta^*$:** **{test_opt['test_opt_f1']:.4f}**
- **Test Brier Score:** **{test_opt['test_opt_brier_score']:.4f}**

---

## 2. Engineered Physical Interaction Features

Exactly four domain features were incorporated, mirroring the custom XGBoost definitions:
1. `avg_unit_weight_kg`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_total\\_qty}}}}$ — separates high-density freight from small-parcel goods.
2. `weight_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{remaining\\_capacity\\_kg}}}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\\frac{{\\text{{candidate\\_added\\_volume\\_m3}}}}{{\\text{{remaining\\_capacity\\_m3}}}}$ — residual cube satisfaction fraction.
4. `cargo_density_kg_m3`: $\\frac{{\\text{{candidate\\_added\\_weight\\_kg}}}}{{\\text{{candidate\\_added\\_volume\\_m3}}}}$ — physical mass-to-volume ratio of proposed cargo package.

*Explicit Exclusions:* Per strict experiment design, `capacity_bottleneck_ratio` and `capacity_utilization_imbalance` were explicitly excluded.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned strictly on the validation partition (Days 63–75) using validation PR-AUC as the selection criterion:

```json
{json.dumps(params, indent=2)}
```

### Rationale:
- **Base Estimator:** Shallow decision trees with `max_depth=3` to model feature combinations without memorizing rare positive patterns.
- **Ensemble Scale & Shrinkage:** 70 boosting rounds with learning rate $\\eta = 0.8$ demonstrated the highest validation PR-AUC among evaluated configurations.

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
| **Specificity** | {metrics_summary['validation_metrics_default_th']['val_th05_specificity']:.4f} | {val_opt['val_opt_specificity']:.4f} | {test_dft['test_th05_specificity']:.4f} | {test_opt['test_opt_specificity']:.4f} |
| **Brier Score** | {metrics_summary['validation_metrics_default_th']['val_th05_brier_score']:.4f} | {val_opt['val_opt_brier_score']:.4f} | {test_dft['test_th05_brier_score']:.4f} | {test_opt['test_opt_brier_score']:.4f} |

### Held-Out Test Confusion Matrix (@ $\\theta^* = {opt_th:.2f}$):
- **True Negatives (TN):** {test_opt['test_opt_tn']}
- **False Positives (FP):** {test_opt['test_opt_fp']}
- **False Negatives (FN):** {test_opt['test_opt_fn']}
- **True Positives (TP):** {test_opt['test_opt_tp']}

---

## 5. Top 10 Feature Importances

| Rank | Feature | Importance (Ensemble Weight) |
|:---:|:---|:---:|
{top_feat_table}

---

## 6. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/adaboost/custom/model.joblib`
- **Fitted Preprocessor:** `models/phase9/adaboost/custom/preprocessor.joblib`
- **Predictions (Test):** `reports/phase9/adaboost/custom/predictions.csv`
- **Validation Predictions:** `reports/phase9/adaboost/custom/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/adaboost/custom/metrics.json`
- **Diagnostic Figures:**
  - `reports/phase9/adaboost/custom/figures/precision_recall_curve.png`
  - `reports/phase9/adaboost/custom/figures/roc_curve.png`
  - `reports/phase9/adaboost/custom/figures/confusion_matrix.png`
  - `reports/phase9/adaboost/custom/figures/feature_importance.png`
  - `reports/phase9/adaboost/custom/figures/calibration_curve.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Generated report at {output_path}")


def run_custom_training(retrain: bool = False) -> Dict[str, Any]:
    """Execute Run 6 Custom AdaBoost pipeline."""
    print("=" * 70)
    print("CARGOSENSE PHASE 9 — RUN 6: CUSTOM ADABOOST")
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

    # 4. Hyperparameters and sample weighting
    sample_weight = compute_sample_weight("balanced", y_train)
    params = {
        "estimator": "DecisionTreeClassifier(max_depth=3, random_state=42)",
        "base_depth": 3,
        "n_estimators": 70,
        "learning_rate": 0.8,
        "sample_weight": "balanced",
        "random_state": 42
    }

    # 5. Fit or reuse model
    model_path = OUTPUT_MODEL_DIR / "model.joblib"
    if model_path.exists() and not retrain:
        print(f"Reusing existing trained model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Training Custom AdaBoost model...")
        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
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
        "experiment": "Run 6: Custom AdaBoost",
        "phase": "Phase 9",
        "owner": "Adaptive Boosting Lead",
        "feature_set": "18 features (14 Baseline + 4 Engineered Physical Ratios)",
        "feature_count": len(feature_names),
        "final_feature_names": feature_names,
        "preprocessor_source": "models/phase9/adaboost/custom/preprocessor.joblib",
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
        title="Custom AdaBoost — Precision-Recall Curves"
    )

    plot_roc_curves(
        {
            "Validation Set": (y_val, val_probs),
            "Held-Out Test Set": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "roc_curve.png",
        title="Custom AdaBoost — ROC Curves"
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
        title="Custom AdaBoost — Top 15 Feature Importances",
        output_path=OUTPUT_FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    plot_calibration_curves(
        {
            "Custom AdaBoost (Test)": (y_test, test_probs)
        },
        OUTPUT_FIGURES_DIR / "calibration_curve.png"
    )
    print(f"Saved figures to {OUTPUT_FIGURES_DIR}")

    # 13. Generate Markdown Report
    report_path = OUTPUT_REPORT_DIR / "report.md"
    generate_custom_report(metrics_summary, top_features, report_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="CargoSense Phase 9 — Run 6 Custom AdaBoost")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model artifact exists")
    args = parser.parse_args()

    run_custom_training(retrain=args.retrain)


if __name__ == "__main__":
    main()
