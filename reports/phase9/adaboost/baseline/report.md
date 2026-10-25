# CargoSense — Phase 9: Run 5 Baseline AdaBoost Evaluation Report

**Model:** AdaBoost Classifier (Baseline)  
**Base Estimator:** `DecisionTreeClassifier(max_depth=3)`  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical — 31 Encoded Dims)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Evaluation Threshold:** $\theta^* = 0.4500$ (F1-optimized on validation set)  

---

## 1. Executive Summary

Run 5 implements the sequential boosting baseline using AdaBoost (Adaptive Boosting) for the CargoSense Opportunity Detection Engine (ODE).
The model was trained strictly on the chronological training split (Days 1–62, 1,533 records, 141 positives).
To handle the ~9.87:1 class imbalance without generating synthetic data or distorting operational physics, sample weights were initialized using class-balanced weights derived from the training set ($w_0 \approx 0.5506, w_1 \approx 5.4362$).

### Key Metrics Summary:
- **Held-Out Test PR-AUC:** **0.2639** (vs. 6.7% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8645**
- **Test Precision @ $\theta^*$:** **20.00%**
- **Test Recall @ $\theta^*$:** **50.00%**
- **Test F1-Score @ $\theta^*$:** **0.2857**
- **Test Brier Score:** **0.1090**

---

## 2. Model Architecture & Hyperparameters

Hyperparameters tuned strictly on the validation partition (Days 63–75) using validation PR-AUC as the primary criterion:

```json
{
  "estimator": "DecisionTreeClassifier(max_depth=3, random_state=42)",
  "base_depth": 3,
  "n_estimators": 50,
  "learning_rate": 1.0,
  "sample_weight": "balanced",
  "random_state": 42
}
```

### Rationale for Weak Learner Configuration:
1. **Tree Depth ($d=3$):** Classic decision stumps ($d=1$) can only evaluate single-feature splits per iteration, failing to capture critical multi-variable operational interactions (e.g., remaining payload capacity combined with added volume). A maximum depth of 3 enables 3-way interactions while strictly avoiding the extreme memorization observed with unregularized deep trees.
2. **Learning Rate ($\eta = 1.0$):** Standard SAMME boosting learning rate provides robust sequential sample reweighting.
3. **Number of Estimators ($M=50$):** 50 boosting stages ensure convergence without over-concentrating weights on label noise or synthetic outliers.

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.45$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.45$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3193 | 0.3193 | 0.2639 | 0.2639 |
| **ROC-AUC** | 0.8726 | 0.8726 | 0.8645 | 0.8645 |
| **Precision** | 0.3000 | 0.2500 | 0.1875 | 0.2000 |
| **Recall** | 0.1304 | 0.6522 | 0.1364 | 0.5000 |
| **F1-Score** | 0.1818 | 0.3614 | 0.1579 | 0.2857 |
| **Accuracy** | 0.9159 | 0.8349 | 0.9021 | 0.8318 |
| **Specificity** | 0.9765 | 0.8490 | 0.9574 | 0.8557 |
| **Brier Score** | 0.1091 | 0.1091 | 0.1090 | 0.1090 |

### Held-Out Test Confusion Matrix (@ $\theta^* = 0.45$):
- **True Negatives (TN):** 261
- **False Positives (FP):** 44
- **False Negatives (FN):** 11
- **True Positives (TP):** 11

---

## 4. Top 10 Feature Importances

| Rank | Feature | Importance (Ensemble Weight) |
|:---:|:---|:---:|
| 1 | `num__candidate_total_qty` | 0.2276 |
| 2 | `num__projected_volume_util_pct` | 0.2002 |
| 3 | `num__projected_weight_util_pct` | 0.1439 |
| 4 | `num__days_into_simulation` | 0.0744 |
| 5 | `num__candidate_added_weight_kg` | 0.0677 |
| 6 | `num__mean_demand_last_7d_route` | 0.0591 |
| 7 | `num__candidate_added_volume_m3` | 0.0575 |
| 8 | `num__remaining_capacity_m3` | 0.0517 |
| 9 | `num__remaining_capacity_kg` | 0.0384 |
| 10 | `cat__day_of_week_6` | 0.0254 |

---

## 5. Artifacts and Diagnostic Figures

- **Trained Model:** `models/phase9/adaboost/baseline/model.joblib`
- **Predictions (Test):** `reports/phase9/adaboost/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/adaboost/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/adaboost/baseline/metrics.json`
- **Diagnostic Figures:**
  - `reports/phase9/adaboost/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/adaboost/baseline/figures/roc_curve.png`
  - `reports/phase9/adaboost/baseline/figures/confusion_matrix.png`
  - `reports/phase9/adaboost/baseline/figures/feature_importance.png`
  - `reports/phase9/adaboost/baseline/figures/calibration_curve.png`
