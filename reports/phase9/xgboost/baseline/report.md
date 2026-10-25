# CargoSense — Phase 9: Run 2 Baseline XGBoost Evaluation Report

**Model:** XGBoost Classifier (Baseline)  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.6400$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 2 establishes the boosting baseline for the CargoSense Opportunity Detection Engine (ODE).
The model was trained on the chronological training split (Days 1–62) using cost-sensitive gradient boosting (`scale_pos_weight = 9.87234`) to handle the 9.87:1 class imbalance without synthetic data corruption.

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.1903** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8273**
- **Test Precision @ $\theta^*$:** **16.87%**
- **Test Recall @ $\theta^*$:** **63.64%**
- **Test F1-Score @ $\theta^*$:** **0.2667**
- **Test Brier Score:** **0.1363**

---

## 2. Model Architecture & Hyperparameters

Baseline hyperparameters tuned on the validation partition:
```json
{
  "n_estimators": 21,
  "max_depth": 4,
  "learning_rate": 0.08,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "min_child_weight": 1,
  "reg_lambda": 3.0,
  "objective": "binary:logistic",
  "eval_metric": "aucpr",
  "scale_pos_weight": 9.872340425531915,
  "random_state": 42
}
```

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.64$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.64$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3330 | 0.3330 | 0.1903 | 0.1903 |
| **ROC-AUC** | 0.8775 | 0.8775 | 0.8273 | 0.8273 |
| **Precision** | 0.2190 | 0.2346 | 0.2018 | 0.1687 |
| **Recall** | 1.0000 | 0.8261 | 1.0000 | 0.6364 |
| **F1-Score** | 0.3594 | 0.3654 | 0.3359 | 0.2667 |
| **Accuracy** | 0.7446 | 0.7944 | 0.7339 | 0.7645 |
| **Brier Score** | 0.1300 | 0.1300 | 0.1363 | 0.1363 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.64$):
- **True Negatives (TN):** 236
- **False Positives (FP):** 69
- **False Negatives (FN):** 8
- **True Positives (TP):** 14

---

## 4. Top 10 Feature Importances (Gain)

| Rank | Feature | Importance (Gain) |
|:---:|:---|:---:|
| 1 | `num__candidate_total_qty` | 0.3325 |
| 2 | `num__projected_weight_util_pct` | 0.2222 |
| 3 | `num__candidate_added_weight_kg` | 0.1435 |
| 4 | `num__projected_volume_util_pct` | 0.1239 |
| 5 | `num__remaining_capacity_m3` | 0.0281 |
| 6 | `num__candidate_added_volume_m3` | 0.0231 |
| 7 | `num__mean_demand_last_7d_route` | 0.0212 |
| 8 | `cat__truck_type_20ft ICV` | 0.0201 |
| 9 | `cat__route_id_RT-MUM-HYD` | 0.0198 |
| 10 | `num__days_into_simulation` | 0.0184 |

---

## 5. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/xgboost/baseline/model.joblib`
- **Predictions (Test):** `reports/phase9/xgboost/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/xgboost/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/xgboost/baseline/metrics.json`
- **Figures:**
  - `reports/phase9/xgboost/baseline/figures/confusion_matrix.png`
  - `reports/phase9/xgboost/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/xgboost/baseline/figures/roc_curve.png`
  - `reports/phase9/xgboost/baseline/figures/feature_importance.png`
  - `reports/phase9/xgboost/baseline/figures/calibration_curve.png`
