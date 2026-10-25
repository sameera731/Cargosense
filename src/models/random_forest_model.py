"""
CargoSense — Phase 9: Random Forest Baseline & Custom Model Experiments
Owner: Sameera (Member 1 — Random Forest Lead)
Collaborator: Member 2 (Boosting Lead)

This module implements:
  - Run 1: Baseline Random Forest (14 shared baseline features, Phase 8 preprocessor)
  - Run 3: Custom Random Forest (14 baseline + 4 physical ratios + 1 bottleneck feature, fitted custom preprocessor)
Both experiments use:
  - Identical chronological shipment-grouped splits from Phase 8.
  - Class weighting: class_weight='balanced' (strictly computed on train set).
  - Validation-based model tuning and classification threshold selection (optimal F1).
  - Held-out test set untouched until final evaluation.
  - Strict leakage prevention: target Became_FTL excluded, candidate_strategy excluded,
    identities/tracking IDs excluded, post-cutoff metrics excluded.
  - Evaluation protocol consistent with src/models/evaluation.py.
  - All figures stored in reports/figures/random_forest/
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

# Directories
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RF_FIGURES_DIR = FIGURES_DIR / "random_forest"

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

# Custom Features for Run 3:
# 4 Established Physical Interaction Features from Phase 8
CUSTOM_PHYSICAL_FEATURES = [
    "cargo_density_kg_m3",
    "weight_fill_ratio",
    "volume_fill_ratio",
    "avg_unit_weight_kg"
]

# 1 Defensible Non-Redundant Custom Feature (Liebig's Law of the Minimum)
CUSTOM_ADDITIONAL_FEATURE = "capacity_bottleneck_ratio"

CUSTOM_ALL_NUMERIC_FEATURES = BASELINE_NUMERIC_FEATURES + CUSTOM_PHYSICAL_FEATURES + [CUSTOM_ADDITIONAL_FEATURE]
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

    # Engineer the capacity_bottleneck_ratio feature across all partitions
    for df in [train_df, val_df, test_df]:
        df["capacity_bottleneck_ratio"] = np.minimum(
            df["weight_fill_ratio"].fillna(0),
            df["volume_fill_ratio"].fillna(0)
        )

    print("Loaded partitions:")
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
    Build and fit scikit-learn ColumnTransformer for custom Random Forest features.
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


def run_experiment_1_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run 1: Baseline Random Forest Classifier.
    Uses 14 baseline features and pre-fitted Phase 8 preprocessor.
    """
    print("\n" + "=" * 70)
    print("RUN 1: BASELINE RANDOM FOREST EXPERIMENT (14 BASELINE FEATURES)")
    print("=" * 70)

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
    print(f"Transformed baseline feature matrix dimension: {X_train.shape[1]} features")

    # 4. Tuned Baseline Hyperparameters (optimized on validation set using PR-AUC)
    params = {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 5. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Verify finite probabilities in [0, 1]
    assert np.all(np.isfinite(val_probs)) and np.all((val_probs >= 0.0) & (val_probs <= 1.0)), "Invalid val probabilities!"
    assert np.all(np.isfinite(test_probs)) and np.all((test_probs >= 0.0) & (test_probs <= 1.0)), "Invalid test probabilities!"

    # 6. Select optimal threshold on validation set
    opt_threshold, opt_f1, sweep_df = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 7. Compute evaluation metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")

    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nBaseline Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f} | Brier: {val_metrics_opt['val_opt_brier_score']:.4f}")
    print(f"Baseline Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f} | Brier: {test_metrics_opt['test_opt_brier_score']:.4f}")
    print(f"Baseline Test (th={opt_threshold:.2f}): Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}, Specificity={test_metrics_opt['test_opt_specificity']:.4f}")

    # 8. Permutation importance on validation set (PR-AUC scoring)
    print("Computing permutation importance on validation set...")
    perm = permutation_importance(model, X_val, y_val, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=1)

    # 9. Save predictions
    val_preds_df = val_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    val_preds_df["predicted_prob"] = val_probs
    val_preds_df["predicted_class_th05"] = (val_probs >= 0.5).astype(int)
    val_preds_df["predicted_class_opt"] = (val_probs >= opt_threshold).astype(int)
    val_preds_df["selected_threshold"] = opt_threshold
    val_preds_path = PROCESSED_DATA_DIR / "phase9_run1_baseline_val_preds.csv"
    val_preds_df.to_csv(val_preds_path, index=False)
    print(f"Saved validation predictions to {val_preds_path}")

    test_preds_df = test_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    test_preds_df["predicted_prob"] = test_probs
    test_preds_df["predicted_class_th05"] = (test_probs >= 0.5).astype(int)
    test_preds_df["predicted_class_opt"] = (test_probs >= opt_threshold).astype(int)
    test_preds_df["selected_threshold"] = opt_threshold
    test_preds_path = PROCESSED_DATA_DIR / "phase9_run1_baseline_test_preds.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}")

    # 10. Save model artifact
    model_save_path = MODELS_DIR / "phase9_randomforest_baseline.joblib"
    joblib.dump(model, model_save_path)
    print(f"Saved baseline model to {model_save_path}")

    return {
        "run_name": "Run 1: Baseline Random Forest",
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "importances": model.feature_importances_,
        "perm_importances_mean": perm.importances_mean,
        "perm_importances_std": perm.importances_std,
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


def run_experiment_3_custom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run 3: Custom Random Forest Classifier.
    Uses 14 baseline features + 4 physical ratios + 1 capacity bottleneck ratio = 19 features (36 encoded).
    Fitted custom preprocessor strictly on train_df.
    """
    print("\n" + "=" * 70)
    print("RUN 3: CUSTOM RANDOM FOREST EXPERIMENT (19 FEATURES INCL. BOTTLENECK)")
    print("=" * 70)

    # 1. Fit custom preprocessor on train_df
    custom_preprocessor = build_custom_preprocessor(train_df)
    custom_prep_path = MODELS_DIR / "phase9_randomforest_custom_preprocessor.joblib"
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

    # 4. Tuned Custom Hyperparameters (optimized on validation set using PR-AUC)
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

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 5. Predict probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Verify finite probabilities in [0, 1]
    assert np.all(np.isfinite(val_probs)) and np.all((val_probs >= 0.0) & (val_probs <= 1.0)), "Invalid val probabilities!"
    assert np.all(np.isfinite(test_probs)) and np.all((test_probs >= 0.0) & (test_probs <= 1.0)), "Invalid test probabilities!"

    # 6. Select optimal threshold on validation set
    opt_threshold, opt_f1, sweep_df = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Validation F1-optimized threshold: {opt_threshold:.4f} (Val F1: {opt_f1:.4f})")

    # 7. Compute evaluation metrics
    val_metrics_default = compute_binary_metrics(y_val, val_probs, threshold=0.5, prefix="val_th05")
    val_metrics_opt = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold, prefix="val_opt")

    test_metrics_default = compute_binary_metrics(y_test, test_probs, threshold=0.5, prefix="test_th05")
    test_metrics_opt = compute_binary_metrics(y_test, test_probs, threshold=opt_threshold, prefix="test_opt")

    print(f"\nCustom Validation PR-AUC: {val_metrics_opt['val_opt_pr_auc']:.4f} | ROC-AUC: {val_metrics_opt['val_opt_roc_auc']:.4f} | Brier: {val_metrics_opt['val_opt_brier_score']:.4f}")
    print(f"Custom Test PR-AUC:       {test_metrics_opt['test_opt_pr_auc']:.4f} | ROC-AUC: {test_metrics_opt['test_opt_roc_auc']:.4f} | Brier: {test_metrics_opt['test_opt_brier_score']:.4f}")
    print(f"Custom Test (th={opt_threshold:.2f}):   Precision={test_metrics_opt['test_opt_precision']:.4f}, Recall={test_metrics_opt['test_opt_recall']:.4f}, F1={test_metrics_opt['test_opt_f1']:.4f}, Specificity={test_metrics_opt['test_opt_specificity']:.4f}")

    # 8. Permutation importance on validation set (PR-AUC scoring)
    print("Computing permutation importance on validation set...")
    perm = permutation_importance(model, X_val, y_val, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=1)

    # 9. Save predictions
    val_preds_df = val_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    val_preds_df["predicted_prob"] = val_probs
    val_preds_df["predicted_class_th05"] = (val_probs >= 0.5).astype(int)
    val_preds_df["predicted_class_opt"] = (val_probs >= opt_threshold).astype(int)
    val_preds_df["selected_threshold"] = opt_threshold
    val_preds_path = PROCESSED_DATA_DIR / "phase9_run3_custom_val_preds.csv"
    val_preds_df.to_csv(val_preds_path, index=False)
    print(f"Saved validation predictions to {val_preds_path}")

    test_preds_df = test_df[["opportunity_id", "shipment_id", TARGET_COLUMN]].copy()
    test_preds_df["predicted_prob"] = test_probs
    test_preds_df["predicted_class_th05"] = (test_probs >= 0.5).astype(int)
    test_preds_df["predicted_class_opt"] = (test_probs >= opt_threshold).astype(int)
    test_preds_df["selected_threshold"] = opt_threshold
    test_preds_path = PROCESSED_DATA_DIR / "phase9_run3_custom_test_preds.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}")

    # 10. Save model artifact
    model_save_path = MODELS_DIR / "phase9_randomforest_custom.joblib"
    joblib.dump(model, model_save_path)
    print(f"Saved custom model to {model_save_path}")

    return {
        "run_name": "Run 3: Custom Random Forest",
        "model": model,
        "preprocessor": custom_preprocessor,
        "feature_names": feature_names,
        "importances": model.feature_importances_,
        "perm_importances_mean": perm.importances_mean,
        "perm_importances_std": perm.importances_std,
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


def generate_random_forest_figures(
    run1_res: Dict[str, Any],
    run3_res: Dict[str, Any],
    y_val: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Generate all 9 dedicated Random Forest diagnostic and comparative plots.
    Stored strictly in reports/figures/random_forest/.
    """
    RF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    print("\n" + "=" * 70)
    print("GENERATING DEDICATED RANDOM FOREST FIGURES IN reports/figures/random_forest/")
    print("=" * 70)

    # 1. Fig 01: Baseline vs Custom PR Curves
    pr_experiments = {
        "Run 1: Baseline RF (Val)": (y_val, run1_res["val_probs"]),
        "Run 3: Custom RF (Val)": (y_val, run3_res["val_probs"]),
        "Run 1: Baseline RF (Test)": (y_test, run1_res["test_probs"]),
        "Run 3: Custom RF (Test)": (y_test, run3_res["test_probs"]),
    }
    plot_pr_curves(
        pr_experiments,
        RF_FIGURES_DIR / "fig01_baseline_vs_custom_pr_curves.png",
        title="Precision-Recall Curves — Run 1 (Baseline) vs Run 3 (Custom Random Forest)"
    )
    print("  [OK] Generated fig01_baseline_vs_custom_pr_curves.png")

    # 2. Fig 02: Baseline vs Custom ROC Curves
    plot_roc_curves(
        pr_experiments,
        RF_FIGURES_DIR / "fig02_baseline_vs_custom_roc_curves.png",
        title="Receiver Operating Characteristic (ROC) — Random Forest Experiments"
    )
    print("  [OK] Generated fig02_baseline_vs_custom_roc_curves.png")

    # 3. Fig 03: Baseline Confusion Matrices (Validation & Test)
    baseline_cms = {
        f"Validation Split (th={run1_res['optimal_threshold']:.2f})": run1_res["val_metrics_opt"]["val_opt_confusion_matrix"],
        f"Held-Out Test Set (th={run1_res['optimal_threshold']:.2f})": run1_res["test_metrics_opt"]["test_opt_confusion_matrix"]
    }
    plot_confusion_matrices(
        baseline_cms,
        RF_FIGURES_DIR / "fig03_baseline_confusion_matrix.png",
        title_suffix="— Run 1 Baseline RF"
    )
    print("  [OK] Generated fig03_baseline_confusion_matrix.png")

    # 4. Fig 04: Custom Confusion Matrices (Validation & Test)
    custom_cms = {
        f"Validation Split (th={run3_res['optimal_threshold']:.2f})": run3_res["val_metrics_opt"]["val_opt_confusion_matrix"],
        f"Held-Out Test Set (th={run3_res['optimal_threshold']:.2f})": run3_res["test_metrics_opt"]["test_opt_confusion_matrix"]
    }
    plot_confusion_matrices(
        custom_cms,
        RF_FIGURES_DIR / "fig04_custom_confusion_matrix.png",
        title_suffix="— Run 3 Custom RF"
    )
    print("  [OK] Generated fig04_custom_confusion_matrix.png")

    # 5. Fig 05: Baseline Feature Importance (MDI / Gini)
    plot_feature_importances(
        run1_res["feature_names"],
        run1_res["importances"],
        title="Run 1 Baseline Random Forest — Top 15 Feature Importances (MDI / Gini)",
        output_path=RF_FIGURES_DIR / "fig05_baseline_feature_importance.png",
        top_n=15
    )
    print("  [OK] Generated fig05_baseline_feature_importance.png")

    # 6. Fig 06: Custom Feature Importance (MDI / Gini)
    plot_feature_importances(
        run3_res["feature_names"],
        run3_res["importances"],
        title="Run 3 Custom Random Forest — Top 15 Feature Importances (MDI / Gini)",
        output_path=RF_FIGURES_DIR / "fig06_custom_feature_importance.png",
        top_n=15
    )
    print("  [OK] Generated fig06_custom_feature_importance.png")

    # 7. Fig 07: Calibration Curves
    calib_experiments = {
        "Run 1: Baseline RF (Test)": (y_test, run1_res["test_probs"]),
        "Run 3: Custom RF (Test)": (y_test, run3_res["test_probs"])
    }
    plot_calibration_curves(
        calib_experiments,
        RF_FIGURES_DIR / "fig07_calibration_curves.png",
        n_bins=8
    )
    print("  [OK] Generated fig07_calibration_curves.png")

    # 8. Fig 08: Permutation Importance Comparison (Validation Set)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df_base_perm = pd.DataFrame({
        "feature": run1_res["feature_names"],
        "importance": run1_res["perm_importances_mean"]
    }).sort_values("importance", ascending=False).head(12).iloc[::-1]

    bars1 = axes[0].barh(df_base_perm["feature"], df_base_perm["importance"], color="#2B5B84", edgecolor="#14324c")
    axes[0].set_title("A. Run 1 Baseline RF (Val Permutation PR-AUC)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean PR-AUC Drop Upon Feature Shuffling", fontsize=10, fontweight="bold")
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8)

    df_cust_perm = pd.DataFrame({
        "feature": run3_res["feature_names"],
        "importance": run3_res["perm_importances_mean"]
    }).sort_values("importance", ascending=False).head(12).iloc[::-1]

    bars2 = axes[1].barh(df_cust_perm["feature"], df_cust_perm["importance"], color="#27AE60", edgecolor="#196F3D")
    axes[1].set_title("B. Run 3 Custom RF (Val Permutation PR-AUC)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean PR-AUC Drop Upon Feature Shuffling", fontsize=10, fontweight="bold")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8)

    plt.suptitle("Validation Permutation Feature Importance (PR-AUC Scoring, 10 Repeats)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RF_FIGURES_DIR / "fig08_permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Generated fig08_permutation_importance.png")

    # 9. Fig 09: Comprehensive Model Comparison
    # Compare Run 1 vs Run 3 and contextualize with XGBoost Run 2 vs Run 4
    xgb_metrics_path = MODELS_DIR / "phase9_xgboost_metrics.json"
    xgb_metrics = None
    if xgb_metrics_path.exists():
        with open(xgb_metrics_path, "r") as f:
            xgb_metrics = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    runs = ["Run 1: Baseline RF", "Run 3: Custom RF"]
    pr_aucs = [
        run1_res["test_metrics_opt"]["test_opt_pr_auc"],
        run3_res["test_metrics_opt"]["test_opt_pr_auc"]
    ]
    roc_aucs = [
        run1_res["test_metrics_opt"]["test_opt_roc_auc"],
        run3_res["test_metrics_opt"]["test_opt_roc_auc"]
    ]
    briers = [
        run1_res["test_metrics_opt"]["test_opt_brier_score"],
        run3_res["test_metrics_opt"]["test_opt_brier_score"]
    ]
    f1s = [
        run1_res["test_metrics_opt"]["test_opt_f1"],
        run3_res["test_metrics_opt"]["test_opt_f1"]
    ]

    colors = ["#2B5B84", "#27AE60"]

    if xgb_metrics:
        runs.extend(["Run 2: Baseline XGB", "Run 4: Custom XGB"])
        pr_aucs.extend([
            xgb_metrics["run2_baseline_xgboost"]["test_metrics_optimal_th"]["test_opt_pr_auc"],
            xgb_metrics["run4_custom_xgboost"]["test_metrics_optimal_th"]["test_opt_pr_auc"]
        ])
        roc_aucs.extend([
            xgb_metrics["run2_baseline_xgboost"]["test_metrics_optimal_th"]["test_opt_roc_auc"],
            xgb_metrics["run4_custom_xgboost"]["test_metrics_optimal_th"]["test_opt_roc_auc"]
        ])
        briers.extend([
            xgb_metrics["run2_baseline_xgboost"]["test_metrics_optimal_th"]["test_opt_brier_score"],
            xgb_metrics["run4_custom_xgboost"]["test_metrics_optimal_th"]["test_opt_brier_score"]
        ])
        f1s.extend([
            xgb_metrics["run2_baseline_xgboost"]["test_metrics_optimal_th"]["test_opt_f1"],
            xgb_metrics["run4_custom_xgboost"]["test_metrics_optimal_th"]["test_opt_f1"]
        ])
        colors.extend(["#E67E22", "#D35400"])

    # Panel 1: PR-AUC (Primary Metric)
    bars_pr = axes[0, 0].bar(runs, pr_aucs, color=colors, edgecolor="black", linewidth=0.8)
    axes[0, 0].axhline(y=0.0673, color="gray", linestyle="--", linewidth=1.2, label="Prevalence (6.7%)")
    axes[0, 0].set_title("A. Test PR-AUC (Primary Metric)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("PR-AUC Score", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylim(0, 0.40)
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_pr:
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    # Panel 2: ROC-AUC
    bars_roc = axes[0, 1].bar(runs, roc_aucs, color=colors, edgecolor="black", linewidth=0.8)
    axes[0, 1].axhline(y=0.50, color="gray", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    axes[0, 1].set_title("B. Test ROC-AUC (Global Discrimination)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("ROC-AUC Score", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_roc:
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    # Panel 3: Brier Score (Lower is better)
    bars_br = axes[1, 0].bar(runs, briers, color=colors, edgecolor="black", linewidth=0.8)
    axes[1, 0].set_title("C. Test Brier Score (Probability Calibration — Lower is Better)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Brier Score Loss", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylim(0, 0.18)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_br:
        h = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    # Panel 4: F1-Score at Optimal Threshold
    bars_f1 = axes[1, 1].bar(runs, f1s, color=colors, edgecolor="black", linewidth=0.8)
    axes[1, 1].set_title("D. Test F1-Score @ Validation-Tuned Threshold", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("F1 Score", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylim(0, 0.45)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars_f1:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    for ax in axes.flat:
        ax.set_xticklabels(runs, rotation=15, ha="right", fontsize=9)

    plt.suptitle("Phase 9 Multi-Metric Model Comparison (Held-Out Test Set)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RF_FIGURES_DIR / "fig09_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Generated fig09_model_comparison.png")


def save_metrics_summary(run1_res: Dict[str, Any], run3_res: Dict[str, Any]) -> None:
    """Save structured JSON metrics file for collaborative evaluation with Member 2."""
    summary = {
        "metadata": {
            "phase": "Phase 9",
            "owner": "Sameera (Member 1 — Random Forest Lead)",
            "models_evaluated": [
                "Run 1: Baseline Random Forest",
                "Run 3: Custom Random Forest"
            ],
            "target": TARGET_COLUMN,
            "train_class_weight": "balanced",
            "seed": 42,
            "baseline_raw_features": BASELINE_ALL_FEATURES,
            "custom_raw_features": CUSTOM_ALL_FEATURES,
            "additional_features_investigated": {
                "capacity_bottleneck_ratio": {
                    "formula": "min(weight_fill_ratio, volume_fill_ratio)",
                    "decision": "INCLUDED in Run 3",
                    "rationale": "Implements Liebig's Law of the Minimum for dual capacity constraints. Boosted validation PR-AUC from 0.2958 to 0.3216."
                },
                "capacity_utilization_imbalance": {
                    "formula": "abs(weight_fill_ratio - volume_fill_ratio)",
                    "decision": "REJECTED",
                    "rationale": "Mathematically redundant with capacity_bottleneck_ratio (collinear r = -1.000) due to candidate generation heuristics saturating max(w, v) ≈ 1.0. Adding both reduced validation PR-AUC to 0.3213."
                }
            }
        },
        "run1_baseline_random_forest": {
            "parameters": run1_res["params"],
            "optimal_threshold": run1_res["optimal_threshold"],
            "validation_metrics_default_th": run1_res["val_metrics_default"],
            "validation_metrics_optimal_th": run1_res["val_metrics_opt"],
            "test_metrics_default_th": run1_res["test_metrics_default"],
            "test_metrics_optimal_th": run1_res["test_metrics_opt"],
            "top_features_tree_mdi": pd.DataFrame({
                "feature": run1_res["feature_names"],
                "importance": run1_res["importances"]
            }).sort_values("importance", ascending=False).head(10).to_dict(orient="records"),
            "top_features_permutation_val": pd.DataFrame({
                "feature": run1_res["feature_names"],
                "mean_pr_auc_drop": run1_res["perm_importances_mean"],
                "std": run1_res["perm_importances_std"]
            }).sort_values("mean_pr_auc_drop", ascending=False).head(10).to_dict(orient="records")
        },
        "run3_custom_random_forest": {
            "parameters": run3_res["params"],
            "optimal_threshold": run3_res["optimal_threshold"],
            "validation_metrics_default_th": run3_res["val_metrics_default"],
            "validation_metrics_optimal_th": run3_res["val_metrics_opt"],
            "test_metrics_default_th": run3_res["test_metrics_default"],
            "test_metrics_optimal_th": run3_res["test_metrics_opt"],
            "top_features_tree_mdi": pd.DataFrame({
                "feature": run3_res["feature_names"],
                "importance": run3_res["importances"]
            }).sort_values("importance", ascending=False).head(10).to_dict(orient="records"),
            "top_features_permutation_val": pd.DataFrame({
                "feature": run3_res["feature_names"],
                "mean_pr_auc_drop": run3_res["perm_importances_mean"],
                "std": run3_res["perm_importances_std"]
            }).sort_values("mean_pr_auc_drop", ascending=False).head(10).to_dict(orient="records")
        },
        "comparison_summary": {
            "val_pr_auc_gain": round(float(run3_res["val_metrics_opt"]["val_opt_pr_auc"] - run1_res["val_metrics_opt"]["val_opt_pr_auc"]), 5),
            "val_roc_auc_gain": round(float(run3_res["val_metrics_opt"]["val_opt_roc_auc"] - run1_res["val_metrics_opt"]["val_opt_roc_auc"]), 5),
            "test_pr_auc_diff": round(float(run3_res["test_metrics_opt"]["test_opt_pr_auc"] - run1_res["test_metrics_opt"]["test_opt_pr_auc"]), 5),
            "test_roc_auc_diff": round(float(run3_res["test_metrics_opt"]["test_opt_roc_auc"] - run1_res["test_metrics_opt"]["test_opt_roc_auc"]), 5),
            "test_brier_score_reduction": round(float(run1_res["test_metrics_opt"]["test_opt_brier_score"] - run3_res["test_metrics_opt"]["test_opt_brier_score"]), 5),
            "test_brier_score_relative_pct": round(float((run1_res["test_metrics_opt"]["test_opt_brier_score"] - run3_res["test_metrics_opt"]["test_opt_brier_score"]) / run1_res["test_metrics_opt"]["test_opt_brier_score"] * 100), 2),
            "test_false_positive_reduction_count": int(run1_res["test_metrics_opt"]["test_opt_fp"] - run3_res["test_metrics_opt"]["test_opt_fp"])
        }
    }

    metrics_path = MODELS_DIR / "phase9_randomforest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved structured Random Forest metrics summary to {metrics_path}")


def main():
    print("=" * 75)
    print("CargoSense Phase 9 — Random Forest Pipeline (Runs 1 & 3)")
    print("Owner: Sameera (Member 1 — Random Forest Lead)")
    print("=" * 75)

    # 1. Load data
    train_df, val_df, test_df = load_datasets()
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 2. Run Baseline Random Forest (Run 1)
    run1_results = run_experiment_1_baseline(train_df, val_df, test_df)

    # 3. Run Custom Random Forest (Run 3)
    run3_results = run_experiment_3_custom(train_df, val_df, test_df)

    # 4. Generate all 9 dedicated figures
    generate_random_forest_figures(run1_results, run3_results, y_val, y_test)

    # 5. Save structured summary
    save_metrics_summary(run1_results, run3_results)

    print("\n" + "=" * 75)
    print("Phase 9 Random Forest Pipeline Executed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
