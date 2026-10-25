# CargoSense — Phase 9: Baseline Gaussian Naive Bayes Evaluation Report

**Model:** Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Experiment:** Baseline GaussianNB (14 Shared Phase 8 Features, 31 Encoded Dims)  
**Preprocessor:** Shared Phase 8 Preprocessor (`models/phase8_preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Operating Thresholds:** Default $\theta = 0.50$ | Validation-Optimized $\theta^* = 0.9000$  

---

## 1. Executive Summary

This report documents the implementation and evaluation of **Baseline Gaussian Naive Bayes** for Phase 9 of the CargoSense Opportunity Detection Engine (ODE).

Gaussian Naive Bayes serves as a classical probabilistic generative benchmark. It assumes that features are conditionally independent given the FTL outcome class and follow class-conditional univariate Gaussian distributions.

### Key Performance Metrics:
- **Held-Out Test PR-AUC:** **0.2738** (vs. 6.73% random baseline prevalence)
- **Held-Out Test ROC-AUC:** **0.8474**
- **Test Precision @ $\theta=0.50$:** **20.37%** (22/108 predictions)
- **Test Recall @ $\theta=0.50$:** **100.00%** (22/22 positives identified)
- **Test F1-Score @ $\theta=0.50$:** **0.3385**
- **Test Precision @ $\theta^*=0.90$:** **20.00%** (21/105 predictions)
- **Test Recall @ $\theta^*=0.90$:** **95.45%** (21/22 positives identified)
- **Test F1-Score @ $\theta^*=0.90$:** **0.3307**
- **Test Brier Score:** **0.2606**

---

## 2. Theoretical Assumptions & Operational Limitations

Gaussian Naive Bayes relies on two mathematical assumptions that conflict with operational logistics data:

1. **Conditional Independence**:
   GaussianNB assumes features are conditionally independent:
   $$P(X_1, \dots, X_d \mid Y = c) = \prod_{i=1}^d P(X_i \mid Y = c)$$
   In freight consolidation, weight, volume, pieces, and utilization percentages are strongly correlated. Multiplying likelihoods across 31 features results in redundant evidence overcounting, pushing posterior probabilities to extreme values (high Brier score of 0.2606).

2. **Univariate Gaussian Distribution**:
   Bounded operational metrics (utilization percentages capped near 85–100%, non-negative piece counts, one-hot encoded route indicators) deviate significantly from continuous unbounded normal distributions.

---

## 3. Detailed Performance Across Splits

| Metric | Validation Set (Default $\theta=0.50$) | Validation Set (Optimal $\theta=0.90$) | Test Set (Default $\theta=0.50$) | Test Set (Optimal $\theta=0.90$) |
|:---|:---:|:---:|:---:|:---:|
| **PR-AUC (Average Precision)** | 0.2239 | 0.2239 | 0.2738 | 0.2738 |
| **ROC-AUC** | 0.8573 | 0.8573 | 0.8474 | 0.8474 |
| **Precision** | 0.2150 | 0.2170 | 0.2037 | 0.2000 |
| **Recall** | 1.0000 | 1.0000 | 1.0000 | 0.9546 |
| **F1-Score** | 0.3538 | 0.3566 | 0.3385 | 0.3307 |
| **Accuracy** | 0.7383 | 0.7414 | 0.7370 | 0.7401 |
| **Specificity** | 0.7181 | 0.7215 | 0.7180 | 0.7246 |
| **Brier Score** | 0.2599 | 0.2599 | 0.2606 | 0.2606 |
| **Positive Predictions Count** | 107 | 106 | 108 | 105 |

### Held-Out Test Confusion Matrices:
- **Default ($\theta=0.50$):** TN=219, FP=86, FN=0, TP=22 (Precision: 20.4%, Recall: 100.0%)
- **Optimal ($\theta^*=0.90$):** TN=221, FP=84, FN=1, TP=21 (Precision: 20.0%, Recall: 95.5%)

---

## 4. Feature Discriminative Power (Standardized Mean Separation)

| Rank | Feature | Standardized Separation ($d$) |
|:---:|:---|:---:|
| 1 | `num__projected_weight_util_pct` | 1.0738 |
| 2 | `num__projected_volume_util_pct` | 1.0535 |
| 3 | `num__candidate_total_qty` | 0.4624 |
| 4 | `num__candidate_added_weight_kg` | 0.4087 |
| 5 | `num__n_candidate_skus` | 0.3636 |
| 6 | `num__candidate_added_volume_m3` | 0.2998 |
| 7 | `num__mean_demand_last_7d_route` | 0.1422 |
| 8 | `cat__truck_type_14ft LCV` | 0.1171 |
| 9 | `cat__day_of_week_1` | 0.1103 |
| 10 | `cat__truck_type_20ft ICV` | 0.1102 |

---

## 5. Artifact Registry

- **Model Artifact:** `models/phase9/naive_bayes/baseline/model.joblib`
- **Model Configuration:** `models/phase9/naive_bayes/baseline/config.json`
- **Predictions (Test):** `reports/phase9/naive_bayes/baseline/predictions.csv`
- **Validation Predictions:** `reports/phase9/naive_bayes/baseline/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/naive_bayes/baseline/metrics.json`
- **Figures:**
  - `reports/phase9/naive_bayes/baseline/figures/confusion_matrix.png`
  - `reports/phase9/naive_bayes/baseline/figures/precision_recall_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/roc_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/calibration_curve.png`
  - `reports/phase9/naive_bayes/baseline/figures/feature_importance.png`
