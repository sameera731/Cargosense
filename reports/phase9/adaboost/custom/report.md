# CargoSense — Phase 9: Run 6 Custom AdaBoost Evaluation Report

**Model:** AdaBoost Classifier (Custom Feature Set)  
**Base Estimator:** `DecisionTreeClassifier(max_depth=3)`  
**Feature Set:** 18 Features (14 Baseline + Exactly 4 Engineered Physical Ratios — 35 Encoded Dims)  
**Preprocessor:** Custom Fitted Preprocessor (`models/phase9/adaboost/custom/preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.4700$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 6 evaluates whether supplementing the 14 baseline features with 4 domain-specific physical interaction ratios enhances AdaBoost's classification and ranking capability.
The model was trained strictly on the chronological training split (Days 1–62, 1,533 records, 141 positives) with training-derived balanced sample weights ($w_0 \approx 0.5506, w_1 \approx 5.4362$) to address the 9.87:1 class imbalance.

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.2006** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8510**
- **Test Precision @ $\theta^*$:** **17.24%**
- **Test Recall @ $\theta^*$:** **45.45%**
- **Test F1-Score @ $\theta^*$:** **0.2500**
- **Test Brier Score:** **0.1077**

---

## 2. Engineered Physical Interaction Features

Exactly four domain features were incorporated, mirroring the custom XGBoost definitions:
1. `avg_unit_weight_kg`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_total\_qty}}$ — separates high-density freight from small-parcel goods.
2. `weight_fill_ratio`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{remaining\_capacity\_kg}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\frac{\text{candidate\_added\_volume\_m3}}{\text{remaining\_capacity\_m3}}$ — residual cube satisfaction fraction.
4. `cargo_density_kg_m3`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_added\_volume\_m3}}$ — physical mass-to-volume ratio of proposed cargo package.

*Explicit Exclusions:* Per strict experiment design, `capacity_bottleneck_ratio` and `capacity_utilization_imbalance` were explicitly excluded.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned strictly on the validation partition (Days 63–75) using validation PR-AUC as the selection criterion:

```json
{
  "estimator": "DecisionTreeClassifier(max_depth=3, random_state=42)",
  "base_depth": 3,
  "n_estimators": 70,
  "learning_rate": 0.8,
  "sample_weight": "balanced",
  "random_state": 42
}
```

### Rationale:
- **Base Estimator:** Shallow decision trees with `max_depth=3` to model feature combinations without memorizing rare positive patterns.
- **Ensemble Scale & Shrinkage:** 70 boosting rounds with learning rate $\eta = 0.8$ demonstrated the highest validation PR-AUC among evaluated configurations.

---

## 4. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.47$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.47$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3046 | 0.3046 | 0.2006 | 0.2006 |
| **ROC-AUC** | 0.8793 | 0.8793 | 0.8510 | 0.8510 |
| **Precision** | 0.3103 | 0.2745 | 0.2667 | 0.1724 |
| **Recall** | 0.3913 | 0.6087 | 0.3636 | 0.4546 |
| **F1-Score** | 0.3462 | 0.3784 | 0.3077 | 0.2500 |
| **Accuracy** | 0.8941 | 0.8567 | 0.8899 | 0.8165 |
| **Specificity** | 0.9329 | 0.8758 | 0.9279 | 0.8426 |
| **Brier Score** | 0.1079 | 0.1079 | 0.1077 | 0.1077 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.47$):
- **True Negatives (TN):** 257
- **False Positives (FP):** 48
- **False Negatives (FN):** 12
- **True Positives (TP):** 10

---

## 5. Top 10 Feature Importances

| Rank | Feature | Importance (Ensemble Weight) |
|:---:|:---|:---:|
| 1 | `num__candidate_total_qty` | 0.2086 |
| 2 | `num__avg_unit_weight_kg` | 0.1508 |
| 3 | `num__weight_fill_ratio` | 0.0995 |
| 4 | `num__cargo_density_kg_m3` | 0.0883 |
| 5 | `num__projected_volume_util_pct` | 0.0791 |
| 6 | `num__candidate_added_weight_kg` | 0.0695 |
| 7 | `num__volume_fill_ratio` | 0.0504 |
| 8 | `cat__day_of_week_6` | 0.0422 |
| 9 | `num__remaining_capacity_m3` | 0.0411 |
| 10 | `num__projected_weight_util_pct` | 0.0360 |

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
