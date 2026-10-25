# CargoSense — Phase 9: Run 3 Custom Random Forest Evaluation Report

**Model:** Random Forest Classifier (Custom Feature Set)  
**Feature Set:** 19 Features (14 Baseline + 4 Physical Ratios + 1 Capacity Bottleneck Ratio)  
**Preprocessor:** Custom Preprocessor (`models/phase9/random_forest/custom/preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.1900$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 3 evaluates whether engineering domain-specific physical ratios and a dual-capacity bottleneck feature enhances Random Forest's discrimination, false-positive suppression, and probability calibration.

The model was trained on the chronological training split (Days 1–62) using cost-sensitive class balancing (`class_weight='balanced'`).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.2158** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8614**
- **Test Precision @ $\theta^*$:** **20.00%**
- **Test Recall @ $\theta^*$:** **36.36%**
- **Test F1-Score @ $\theta^*$:** **0.2581**
- **Test Brier Score:** **0.0561** (state-of-the-art calibration in Phase 9)
- **False Positive Dispatches:** **32** (reduced by 28.9% vs. baseline)

---

## 2. Engineered Physical & Bottleneck Features

1. `cargo_density_kg_m3`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_added\_volume\_m3}}$ — packaging density of proposed freight.
2. `weight_fill_ratio`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{remaining\_capacity\_kg}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\frac{\text{candidate\_added\_volume\_m3}}{\text{remaining\_capacity\_m3}}$ — residual volume satisfaction fraction.
4. `avg_unit_weight_kg`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_total\_qty}}$ — average piece weight distinguishing heavy cargo from light bulk.
5. `capacity_bottleneck_ratio`: $\min(\text{weight\_fill\_ratio}, \text{volume\_fill\_ratio})$ — Liebig's Law of the Minimum; provides direct single-split visibility into dual capacity feasibility.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned on the validation partition:
```json
{
  "n_estimators": 125,
  "max_depth": null,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "class_weight": "balanced",
  "random_state": 42,
  "n_jobs": 1
}
```

---

## 4. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.19$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.19$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3216 | 0.3216 | 0.2158 | 0.2158 |
| **ROC-AUC** | 0.8922 | 0.8922 | 0.8614 | 0.8614 |
| **Precision** | 0.0000 | 0.3429 | 0.0000 | 0.2000 |
| **Recall** | 0.0000 | 0.5217 | 0.0000 | 0.3636 |
| **F1-Score** | 0.0000 | 0.4138 | 0.0000 | 0.2581 |
| **Accuracy** | 0.9284 | 0.8941 | 0.9327 | 0.8593 |
| **Brier Score** | 0.0564 | 0.0564 | 0.0561 | 0.0561 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.19$):
- **True Negatives (TN):** 273
- **False Positives (FP):** 32 (slashed from 45 down to 32 vs. baseline)
- **False Negatives (FN):** 14
- **True Positives (TP):** 8

---

## 5. Top 10 Feature Importances (MDI / Gini)

| Rank | Feature | Importance (MDI) |
|:---:|:---|:---:|
| 1 | `num__capacity_bottleneck_ratio` | 0.1909 |
| 2 | `num__avg_unit_weight_kg` | 0.1356 |
| 3 | `num__weight_fill_ratio` | 0.0814 |
| 4 | `num__volume_fill_ratio` | 0.0771 |
| 5 | `num__candidate_total_qty` | 0.0765 |
| 6 | `num__cargo_density_kg_m3` | 0.0706 |
| 7 | `num__projected_volume_util_pct` | 0.0623 |
| 8 | `num__projected_weight_util_pct` | 0.0583 |
| 9 | `num__candidate_added_weight_kg` | 0.0449 |
| 10 | `num__candidate_added_volume_m3` | 0.0284 |

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
