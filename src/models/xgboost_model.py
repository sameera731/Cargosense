"""
CargoSense — Phase 9: XGBoost Baseline & Custom Model Experiments
Owner: Member 2 (Boosting Lead)
Collaborator: Sameera (Member 1 — Random Forest Lead)

This module implements:
  - Run 2: Baseline XGBoost (14 shared baseline features, Phase 8 preprocessor)
  - Run 4: Custom XGBoost (14 baseline + 4 engineered features, fitted custom preprocessor)
Both experiments use:
  - Identical chronological shipment-grouped splits from Phase 8.
  - Cost-sensitive class weighting: scale_pos_weight ≈ 9.8723 (computed strictly on train set).
  - Validation-based model tuning and classification threshold selection.
  - Held-out test set untouched until final evaluation.
  - Strict leakage prevention: target Became_FTL excluded, candidate_strategy excluded,
    identities/tracking IDs excluded, post-cutoff metrics excluded.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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
    train_path = PROCESSED_DATA_DIR / "phase8_train.csv"
    val_path = PROCESSED_DATA_DIR / "phase8_val.csv"
    test_path = PROCESSED_DATA_DIR / "phase8_test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing Phase 8 partition: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded partitions:")
    print(f"  - Train: {train_df.shape[0]:,} rows ({train_df[TARGET_COLUMN].sum()} positives, {train_df[TARGET_COLUMN].mean():.2%})")
    print(f"  - Val:   {val_df.shape[0]:,} rows ({val_df[TARGET_COLUMN].sum()} positives, {val_df[TARGET_COLUMN].mean():.2%})")
    print(f"  - Test:  {test_df.shape[0]:,} rows ({test_df[TARGET_COLUMN].sum()} positives, {test_df[TARGET_COLUMN].mean():.2%})")

    # Audit prohibited leakage columns
    for prohibited in PROHIBITED_COLUMNS:
        if prohibited in BASELINE_ALL_FEATURES or prohibited in CUSTOM_ALL_FEATURES:
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


def run_experiment_2_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run 2: Baseline XGBoost Classifier.
    Uses 14 baseline features and pre-fitted Phase 8 preprocessor.
    """
    print("\n" + "="*70)
    print("RUN 2: BASELINE XGBOOST EXPERIMENT (14 BASELINE FEATURES)")
    print("="*70)

    # 1. Load Phase 8 preprocessor
    prep_path = MODELS_DIR / "phase8_preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing Phase 8 preprocessor at {prep_path}")
    preprocessor = joblib.load(prep_path)

    # 2. Extract targets
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 3. Transform features
    X_train = preprocessor.transform(train_df[BASELINE_ALL_FEATURES])
    X_val = preprocessor.transform(val_df[BASELINE_ALL_FEATURES])
    X_test = preprocessor.transform(test_df[BASELINE_ALL_FEATURES])

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Transformed feature matrix dimension: {X_train.shape[1]} features")

    # 4. Imbalance handling
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    print(f"Class imbalance scale_pos_weight (Train): {scale_pos_weight:.5f}")

    # 5. Tuned Baseline Hyperparameters (optimized on validation set)
    params = {
        "n_estimators": 21,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_lambda": 3.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # 6. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # 7. Select optimal threshold on validation set
    opt_threshold, opt_f1, sweep_df = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 8. Compute evaluation metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")

    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nBaseline Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f}")
    print(f"Baseline Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Baseline Test (th={opt_threshold:.2f}): Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 9. Save predictions
    val_preds_df = val_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    val_preds_df["predicted_prob"] = val_probs
    val_preds_df["predicted_class_th05"] = (val_probs >= 0.5).astype(int)
    val_preds_df["predicted_class_opt"] = (val_probs >= opt_threshold).astype(int)
    val_preds_path = PROCESSED_DATA_DIR / "phase9_run2_baseline_val_preds.csv"
    val_preds_df.to_csv(val_preds_path, index=False)

    test_preds_df = test_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    test_preds_df["predicted_prob"] = test_probs
    test_preds_df["predicted_class_th05"] = (test_probs >= 0.5).astype(int)
    test_preds_df["predicted_class_opt"] = (test_probs >= opt_threshold).astype(int)
    test_preds_path = PROCESSED_DATA_DIR / "phase9_run2_baseline_test_preds.csv"
    test_preds_df.to_csv(test_preds_path, index=False)

    # 10. Save model artifact
    model_save_path = MODELS_DIR / "phase9_xgboost_baseline.joblib"
    joblib.dump(model, model_save_path)
    print(f"Saved baseline model to {model_save_path}")

    return {
        "run_name": "Run 2: Baseline XGBoost",
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "importances": model.feature_importances_,
        "params": params,
        "optimal_threshold": opt_threshold,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_metrics_default": val_metrics_default,
        "val_metrics_opt": val_metrics_opt,
        "test_metrics_default": test_metrics_default,
        "test_metrics_opt": test_metrics_opt,
        "sweep_df": sweep_df
    }


def run_experiment_4_custom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run 4: Custom XGBoost Classifier.
    Uses 14 baseline features + 4 engineered physical ratio & density features.
    Fitted custom preprocessor strictly on train_df.
    """
    print("\n" + "="*70)
    print("RUN 4: CUSTOM XGBOOST EXPERIMENT (18 FEATURES INCL. 4 ENGINEERED)")
    print("="*70)

    # 1. Fit custom preprocessor on train_df
    custom_preprocessor = build_custom_preprocessor(train_df)
    custom_prep_path = MODELS_DIR / "phase9_custom_preprocessor.joblib"
    joblib.dump(custom_preprocessor, custom_prep_path)
    print(f"Fitted & saved custom preprocessor to {custom_prep_path}")

    # 2. Extract targets
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 3. Transform features
    X_train = custom_preprocessor.transform(train_df[CUSTOM_ALL_FEATURES])
    X_val = custom_preprocessor.transform(val_df[CUSTOM_ALL_FEATURES])
    X_test = custom_preprocessor.transform(test_df[CUSTOM_ALL_FEATURES])

    feature_names = list(custom_preprocessor.get_feature_names_out())
    print(f"Transformed custom feature matrix dimension: {X_train.shape[1]} features")

    # 4. Imbalance handling
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    # 5. Tuned Custom Hyperparameters (optimized on validation set)
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

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # 6. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # 7. Select optimal threshold on validation set
    opt_threshold, opt_f1, sweep_df = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 8. Compute evaluation metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")

    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nCustom Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f}")
    print(f"Custom Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f}")
    print(f"Custom Test (th={opt_threshold:.2f}):   Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}")

    # 9. Save predictions
    val_preds_df = val_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    val_preds_df["predicted_prob"] = val_probs
    val_preds_df["predicted_class_th05"] = (val_probs >= 0.5).astype(int)
    val_preds_df["predicted_class_opt"] = (val_probs >= opt_threshold).astype(int)
    val_preds_path = PROCESSED_DATA_DIR / "phase9_run4_custom_val_preds.csv"
    val_preds_df.to_csv(val_preds_path, index=False)

    test_preds_df = test_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    test_preds_df["predicted_prob"] = test_probs
    test_preds_df["predicted_class_th05"] = (test_probs >= 0.5).astype(int)
    test_preds_df["predicted_class_opt"] = (test_probs >= opt_threshold).astype(int)
    test_preds_path = PROCESSED_DATA_DIR / "phase9_run4_custom_test_preds.csv"
    test_preds_df.to_csv(test_preds_path, index=False)

    # 10. Save model artifact
    model_save_path = MODELS_DIR / "phase9_xgboost_custom.joblib"
    joblib.dump(model, model_save_path)
    print(f"Saved custom model to {model_save_path}")

    return {
        "run_name": "Run 4: Custom XGBoost",
        "model": model,
        "preprocessor": custom_preprocessor,
        "feature_names": feature_names,
        "importances": model.feature_importances_,
        "params": params,
        "optimal_threshold": opt_threshold,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_metrics_default": val_metrics_default,
        "val_metrics_opt": val_metrics_opt,
        "test_metrics_default": test_metrics_default,
        "test_metrics_opt": test_metrics_opt,
        "sweep_df": sweep_df
    }


def generate_comparative_figures(
    run2_res: Dict[str, Any],
    run4_res: Dict[str, Any],
    y_val: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Generate high-resolution comparative diagnostic figures for reports."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. PR Curves
    val_experiments = {
        "Run 2: Baseline XGBoost (Val)": (y_val, run2_res["val_probs"]),
        "Run 4: Custom XGBoost (Val)": (y_val, run4_res["val_probs"]),
        "Run 2: Baseline XGBoost (Test)": (y_test, run2_res["test_probs"]),
        "Run 4: Custom XGBoost (Test)": (y_test, run4_res["test_probs"])
    }
    plot_pr_curves(val_experiments, FIGURES_DIR / "fig12_run2_run4_pr_curves.png",
                   title="Precision-Recall Curves — Run 2 (Baseline) vs Run 4 (Custom XGBoost)")
    print("Generated Fig 12: Precision-Recall Curves")

    # 2. ROC Curves
    plot_roc_curves(val_experiments, FIGURES_DIR / "fig13_run2_run4_roc_curves.png",
                    title="ROC Curves — Run 2 (Baseline) vs Run 4 (Custom XGBoost)")
    print("Generated Fig 13: ROC Curves")

    # 3. Confusion Matrices
    test_cms = {
        f"Run 2 Baseline (th={run2_res['optimal_threshold']:.2f})": run2_res["test_metrics_opt"]["test_opt_confusion_matrix"],
        f"Run 4 Custom (th={run4_res['optimal_threshold']:.2f})": run4_res["test_metrics_opt"]["test_opt_confusion_matrix"]
    }
    plot_confusion_matrices(test_cms, FIGURES_DIR / "fig14_run2_run4_confusion_matrices.png",
                            title_suffix="Held-Out Test Set")
    print("Generated Fig 14: Confusion Matrices")

    # 4. Feature Importance Comparison
    plot_feature_importances(
        run4_res["feature_names"],
        run4_res["importances"],
        title="Run 4 Custom XGBoost — Top 15 Feature Importances (Gain)",
        output_path=FIGURES_DIR / "fig15_run4_feature_importance.png",
        top_n=15
    )
    plot_feature_importances(
        run2_res["feature_names"],
        run2_res["importances"],
        title="Run 2 Baseline XGBoost — Top 15 Feature Importances (Gain)",
        output_path=FIGURES_DIR / "fig15_run2_feature_importance.png",
        top_n=15
    )
    print("Generated Fig 15: Feature Importances")

    # 5. Calibration Curves
    calib_experiments = {
        "Run 2: Baseline XGBoost": (y_test, run2_res["test_probs"]),
        "Run 4: Custom XGBoost": (y_test, run4_res["test_probs"])
    }
    plot_calibration_curves(calib_experiments, FIGURES_DIR / "fig16_run2_run4_calibration_curves.png")
    print("Generated Fig 16: Calibration Curves")


def save_metrics_summary(run2_res: Dict[str, Any], run4_res: Dict[str, Any]) -> None:
    """Save structured JSON metrics file for collaborative evaluation with Sameera."""
    summary = {
        "metadata": {
            "phase": "Phase 9",
            "owner": "Member 2",
            "models_evaluated": ["Run 2: Baseline XGBoost", "Run 4: Custom XGBoost"],
            "target": TARGET_COLUMN,
            "train_scale_pos_weight": run2_res["params"]["scale_pos_weight"],
            "seed": 42
        },
        "run2_baseline_xgboost": {
            "parameters": run2_res["params"],
            "optimal_threshold": run2_res["optimal_threshold"],
            "validation_metrics_default_th": run2_res["val_metrics_default"],
            "validation_metrics_optimal_th": run2_res["val_metrics_opt"],
            "test_metrics_default_th": run2_res["test_metrics_default"],
            "test_metrics_optimal_th": run2_res["test_metrics_opt"],
            "top_features": pd.DataFrame({
                "feature": run2_res["feature_names"],
                "importance": run2_res["importances"]
            }).sort_values("importance", ascending=False).head(10).to_dict(orient="records")
        },
        "run4_custom_xgboost": {
            "parameters": run4_res["params"],
            "optimal_threshold": run4_res["optimal_threshold"],
            "validation_metrics_default_th": run4_res["val_metrics_default"],
            "validation_metrics_optimal_th": run4_res["val_metrics_opt"],
            "test_metrics_default_th": run4_res["test_metrics_default"],
            "test_metrics_optimal_th": run4_res["test_metrics_opt"],
            "top_features": pd.DataFrame({
                "feature": run4_res["feature_names"],
                "importance": run4_res["importances"]
            }).sort_values("importance", ascending=False).head(10).to_dict(orient="records")
        },
        "comparison_summary": {
            "test_pr_auc_gain": round(float(run4_res["test_metrics_opt"]["test_opt_pr_auc"] - run2_res["test_metrics_opt"]["test_opt_pr_auc"]), 5),
            "test_pr_auc_relative_pct": round(float((run4_res["test_metrics_opt"]["test_opt_pr_auc"] - run2_res["test_metrics_opt"]["test_opt_pr_auc"]) / run2_res["test_metrics_opt"]["test_opt_pr_auc"] * 100), 2),
            "test_roc_auc_gain": round(float(run4_res["test_metrics_opt"]["test_opt_roc_auc"] - run2_res["test_metrics_opt"]["test_opt_roc_auc"]), 5),
            "test_brier_score_reduction": round(float(run2_res["test_metrics_opt"]["test_opt_brier_score"] - run4_res["test_metrics_opt"]["test_opt_brier_score"]), 5)
        }
    }

    metrics_path = MODELS_DIR / "phase9_xgboost_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved structured metrics summary to {metrics_path}")


def main():
    print("="*75)
    print("CargoSense Phase 9 — XGBoost Pipeline (Runs 2 & 4)")
    print("="*75)

    # 1. Load data
    train_df, val_df, test_df = load_datasets()
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 2. Run Baseline XGBoost
    run2_results = run_experiment_2_baseline(train_df, val_df, test_df)

    # 3. Run Custom XGBoost
    run4_results = run_experiment_4_custom(train_df, val_df, test_df)

    # 4. Generate Figures
    generate_comparative_figures(run2_results, run4_results, y_val, y_test)

    # 5. Save structured summary
    save_metrics_summary(run2_results, run4_results)

    print("\n" + "="*75)
    print("Phase 9 XGBoost Pipeline Executed Successfully!")
    print("="*75)


if __name__ == "__main__":
    main()
