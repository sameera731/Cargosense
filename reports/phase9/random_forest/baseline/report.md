# CargoSense — Phase 9: Run 1 Baseline Random Forest Evaluation Report

**Model:** Random Forest Classifier (Baseline Bagging)  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.3900$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 1 establishes the bagging baseline for the CargoSense Opportunity Detection Engine (ODE).
The model was trained on the chronological training split (Days 1–62) using cost-sensitive class balancing (`class_weight='balanced'`) to handle the 9.87:1 class imbalance.

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.2233** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8616**
- **Test Precision @ $\theta^*$:** **21.05%**
- **Test Recall @ $\theta^*$:** **54.55%**
- **Test F1-Score @ $\theta^*$:** **0.3038**
- **Test Brier Score:** **0.0711**

---

## 2. Model Architecture & Hyperparameters

Baseline hyperparameters tuned on the validation partition:
```json
{
  "n_estimators": 50,
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "class_weight": "balanced",
  "random_state": 42,
  "n_jobs": 1
}
```

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.39$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.39$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3042 | 0.3042 | 0.2233 | 0.2233 |
| **ROC-AUC** | 0.8783 | 0.8783 | 0.8616 | 0.8616 |
| **Precision** | 0.2903 | 0.2537 | 0.2632 | 0.2105 |
| **Recall** | 0.3913 | 0.7391 | 0.2273 | 0.5454 |
| **F1-Score** | 0.3333 | 0.3778 | 0.2439 | 0.3038 |
| **Accuracy** | 0.8879 | 0.8256 | 0.9052 | 0.8318 |
| **Brier Score** | 0.0725 | 0.0725 | 0.0711 | 0.0711 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.39$):
- **True Negatives (TN):** 260
- **False Positives (FP):** 45
- **False Negatives (FN):** 10
- **True Positives (TP):** 12

---

## 4. Top 10 Feature Importances (MDI / Gini)

| Rank | Feature | Importance (MDI) |
|:---:|:---|:---:|
| 1 | `num__projected_volume_util_pct` | 0.2162 |
| 2 | `num__projected_weight_util_pct` | 0.1985 |
| 3 | `num__candidate_total_qty` | 0.1637 |
| 4 | `num__candidate_added_weight_kg` | 0.1150 |
| 5 | `num__n_candidate_skus` | 0.0742 |
| 6 | `num__candidate_added_volume_m3` | 0.0594 |
| 7 | `num__remaining_capacity_kg` | 0.0341 |
| 8 | `num__remaining_capacity_m3` | 0.0289 |
| 9 | `num__mean_demand_last_7d_route` | 0.0286 |
| 10 | `num__days_into_simulation` | 0.0265 |

---

## 5. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/random_forest/baseline/model.joblib`
- **Predictions (Test):** `reports/phase9/random_forest/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/random_forest/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/random_forest/baseline/metrics.json`
- **Figures:**
  - `reports/phase9/random_forest/baseline/figures/confusion_matrix.png`
  - `reports/phase9/random_forest/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/random_forest/baseline/figures/roc_curve.png`
  - `reports/phase9/random_forest/baseline/figures/feature_importance.png`
  - `reports/phase9/random_forest/baseline/figures/calibration_curve.png`
