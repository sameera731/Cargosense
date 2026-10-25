# CargoSense — Phase 9: Run 4 Custom XGBoost Evaluation Report

**Model:** XGBoost Classifier (Custom Feature Set)  
**Feature Set:** 18 Features (14 Baseline + 4 Engineered Physical Ratios)  
**Preprocessor:** Custom Preprocessor (`models/phase9/xgboost/custom/preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.4800$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 4 tests whether engineering domain-specific physical interaction and capacity fill ratio features improves XGBoost's discrimination and probability calibration over the baseline.

The model was trained on the chronological training split (Days 1–62) using cost-sensitive gradient boosting (`scale_pos_weight = 9.87234`).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.2519** (vs. 0.1903 baseline, a **+32.4% relative gain**)
- **Held-Out Test ROC-AUC:** **0.8596** (vs. 0.8273 baseline, **+3.2% gain**)
- **Test Precision @ $\theta^*$:** **21.95%**
- **Test Recall @ $\theta^*$:** **40.91%**
- **Test F1-Score @ $\theta^*$:** **0.2857**
- **Test Brier Score:** **0.0812** (vs. 0.1363 baseline, **40.4% error reduction**)

---

## 2. Engineered Physical Interaction Features

The 4 custom features give tree ensembles explicit coordinates for freight capacity exhaustion:
1. `avg_unit_weight_kg`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_total\_qty}}$ — separates high-density freight from low-density bulk.
2. `weight_fill_ratio`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{remaining\_capacity\_kg}}$ — residual payload satisfaction fraction.
3. `volume_fill_ratio`: $\frac{\text{candidate\_added\_volume\_m3}}{\text{remaining\_capacity\_m3}}$ — residual volume satisfaction fraction.
4. `cargo_density_kg_m3`: $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_added\_volume\_m3}}$ — physical density of proposed cargo package.

---

## 3. Model Architecture & Hyperparameters

Custom hyperparameters tuned on the validation partition:
```json
{
  "n_estimators": 52,
  "max_depth": 5,
  "learning_rate": 0.08,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 1,
  "reg_lambda": 1.0,
  "objective": "binary:logistic",
  "eval_metric": "aucpr",
  "scale_pos_weight": 9.872340425531915,
  "random_state": 42
}
```

---

## 4. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.48$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.48$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3867 | 0.3867 | 0.2519 | 0.2519 |
| **ROC-AUC** | 0.8944 | 0.8944 | 0.8596 | 0.8596 |
| **Precision** | 0.3077 | 0.3256 | 0.2162 | 0.2195 |
| **Recall** | 0.5217 | 0.6087 | 0.3636 | 0.4091 |
| **F1-Score** | 0.3871 | 0.4242 | 0.2712 | 0.2857 |
| **Accuracy** | 0.8816 | 0.8816 | 0.8685 | 0.8624 |
| **Brier Score** | 0.0751 | 0.0751 | 0.0812 | 0.0812 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.48$):
- **True Negatives (TN):** 273
- **False Positives (FP):** 32 (slashed from 69 down to 32 vs. baseline)
- **False Negatives (FN):** 13
- **True Positives (TP):** 9

---

## 5. Top 10 Feature Importances (Gain)

| Rank | Feature | Importance (Gain) |
|:---:|:---|:---:|
| 1 | `num__candidate_total_qty` | 0.1494 |
| 2 | `num__avg_unit_weight_kg` | 0.1359 |
| 3 | `num__projected_weight_util_pct` | 0.0804 |
| 4 | `num__projected_volume_util_pct` | 0.0705 |
| 5 | `num__weight_fill_ratio` | 0.0432 |
| 6 | `num__volume_fill_ratio` | 0.0395 |
| 7 | `cat__day_of_week_1` | 0.0354 |
| 8 | `cat__route_id_RT-MUM-HYD` | 0.0348 |
| 9 | `cat__day_of_week_6` | 0.0309 |
| 10 | `cat__day_of_week_2` | 0.0300 |

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
