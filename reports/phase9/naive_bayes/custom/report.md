# CargoSense — Phase 9: Custom Gaussian Naive Bayes Evaluation Report

**Model:** Custom Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Feature Subset:** `decorrelated_physical_4` (4 Selected Features)  
**Tuned `var_smoothing`:** `0.001`  
**Class Priors:** `empirical`  
**Validation-Selected Operating Threshold:** $\theta^* = 0.9200$  

---

## 1. Executive Summary

This report documents the implementation, tuning, and held-out test evaluation of the **Custom Gaussian Naive Bayes** experiment for CargoSense Phase 9.

### Optimization Objectives & Methodology:
1. **Multicollinearity & Noise Reduction**: GaussianNB's performance on the 14 baseline features (31 transformed columns) was limited by severe physical correlation (weight vs. volume) and 20 discrete one-hot encoded categorical indicator variables. Candidate subsets were defined strictly on training data statistics.
2. **Variance Smoothing**: Tuning `var_smoothing` to mitigate overconfident likelihood multiplication and extreme probability polarization.
3. **Class Prior Assessment**: Testing empirical vs. balanced priors ([0.5, 0.5]) derived exclusively on training data.
4. **Validation-Guided Model Selection**: Selecting the best configuration using **Validation PR-AUC (Average Precision)** as the primary objective and Validation F1 as secondary, freezing all hyperparameter decisions before evaluating the held-out test set once.

### Key Performance Findings:
- **Validation PR-AUC Gain**: Improved validation PR-AUC from **0.2239** (baseline) to **0.3444** (+53.8% relative improvement during model selection).
- **Test Set Generalization & Honest Assessment**:
  - Held-out test PR-AUC was **0.1537** (vs. 0.2738 baseline).
  - Held-out test ROC-AUC was **0.7948** (vs. 0.8474 baseline).
  - Test F1 @ $\theta^*$ was **0.2253** with precision of **16.33%** and recall of **36.36%**.
  - Test false positives were cut substantially from **84 down to 41** (-51.2% false alarm reduction).
  - Probability calibration: The custom model improved probability calibration on validation data, but dropping secondary variables (route transit hours, route corridors) reduced held-out test PR-AUC.

---

## 2. Experimental Optimization Grid & Validation Selection

Hyperparameter grid evaluated on the validation partition:
- **`var_smoothing` Grid**: `[1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]`
- **Class Priors**: `empirical` (training base rate = 9.20%) vs. `balanced` ([0.5, 0.5])
- **Feature Subsets Evaluated**:
  1. `baseline_all_14` (31 columns)
  2. `numeric_core_11` (11 numeric features)
  3. `f_statistic_top_6` (6 features with p < 0.001 from training ANOVA F-statistic)
  4. `decorrelated_physical_4` (4 non-redundant physical features)

### Top 5 Configurations from Validation Grid Search:
| Rank | Feature Subset | N Feat | var_smoothing | Priors | Val PR-AUC | Val ROC-AUC | Val F1 | Val Brier | $\theta^*$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `decorrelated_physical_4` | 4 | 1.0e-03 | empirical | 0.3444 | 0.8410 | 0.3218 | 0.2985 | 0.92 |
| 2 | `decorrelated_physical_4` | 4 | 1.0e-03 | balanced | 0.3444 | 0.8410 | 0.3046 | 0.4739 | 0.95 |
| 3 | `decorrelated_physical_4` | 4 | 1.0e-02 | empirical | 0.3430 | 0.8413 | 0.3261 | 0.1879 | 0.75 |
| 4 | `decorrelated_physical_4` | 4 | 1.0e-02 | balanced | 0.3430 | 0.8413 | 0.3067 | 0.4032 | 0.87 |
| 5 | `decorrelated_physical_4` | 4 | 5.0e-02 | empirical | 0.2855 | 0.8391 | 0.3261 | 0.1218 | 0.57 |

### Selected Features (`decorrelated_physical_4`):
- `num__projected_weight_util_pct`
- `num__candidate_total_qty`
- `num__candidate_added_weight_kg`
- `num__n_candidate_skus`

*Description: 4 non-redundant core physical features addressing severe weight/volume multicollinearity*

---

## 3. Comprehensive Performance Comparison: Baseline vs. Custom

| Evaluation Metric | Baseline GaussianNB (Test @ $\theta=0.50$) | Baseline GaussianNB (Test @ $\theta^*=0.90$) | Custom GaussianNB (Test @ $\theta=0.50$) | Custom GaussianNB (Test @ $\theta^*=0.92$) | Delta (Custom $\theta^*$ vs Baseline $\theta^*$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PR-AUC (Average Precision)** | 0.2738 | 0.2738 | 0.1537 | 0.1537 | -0.1202 |
| **ROC-AUC** | 0.8474 | 0.8474 | 0.7948 | 0.7948 | -0.0526 |
| **Precision** | 20.37% | 20.00% | 13.17% | 16.33% | -3.67% |
| **Recall** | 100.00% | 95.45% | 100.00% | 36.36% | -59.09% |
| **F1-Score** | 0.3385 | 0.3307 | 0.2328 | 0.2253 | -0.1054 |
| **Accuracy** | 73.70% | 74.01% | 55.66% | 83.18% | +9.17% |
| **Specificity** | 71.80% | 72.46% | 52.46% | 86.56% | +14.10% |
| **Brier Score** | 0.2606 | 0.2606 | 0.2861 | 0.2861 | +0.0255 |
| **False Positives (FP)** | **86** | **84** | **145** | **41** | **-43 fewer FPs** |
| **True Positives (TP)** | 22 | 21 | 22 | 8 | -13 |

---

## 4. Test Set Confusion Matrices

### Baseline GaussianNB (@ $\theta^* = 0.90$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          221                  84              (Specificity: 72.5%)
Actual Positive          1                    21              (Recall: 95.5%)
Total Test: 327 | Precision: 20.0% | Accuracy: 74.0%
```

### Custom GaussianNB (@ $\theta^* = 0.92$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          264                  41              (Specificity: 86.6%)
Actual Positive          14                   8               (Recall: 36.4%)
Total Test: 327 | Precision: 16.3% | Accuracy: 83.2%
```

---

## 5. Feature Discriminative Separation ($d$) in Custom Model

| Rank | Feature | Standardized Separation ($d$) |
|:---:|:---|:---:|
| 1 | `num__projected_weight_util_pct` | 1.0727 |
| 2 | `num__candidate_total_qty` | 0.4621 |
| 3 | `num__candidate_added_weight_kg` | 0.4085 |
| 4 | `num__n_candidate_skus` | 0.3634 |

---

## 6. Diagnostic Figures

The following diagnostic plots are stored in `reports/phase9/naive_bayes/custom/figures/`:
1. `confusion_matrix.png`: Normalized and raw test confusion matrices comparing $\theta=0.50$ and validation-selected $\theta^* = 0.92$.
2. `precision_recall_curve.png`: PR curve across validation and test partitions with baseline positive prevalence line.
3. `roc_curve.png`: Receiver Operating Characteristic curve across validation and test partitions.
4. `calibration_curve.png`: Reliability diagram illustrating posterior probability calibration.
5. `feature_importance.png`: Standardized class mean separation bar chart.

---

## 7. Artifact Registry

- **Pipeline Script:** `src/phase9/naive_bayes/custom/train.py`
- **Fitted Model:** `models/phase9/naive_bayes/custom/model.joblib`
- **Model Configuration:** `models/phase9/naive_bayes/custom/config.json`
- **Test Predictions:** `reports/phase9/naive_bayes/custom/predictions.csv`
- **Validation Predictions:** `reports/phase9/naive_bayes/custom/val_predictions.csv`
- **Structured Metrics:** `reports/phase9/naive_bayes/custom/metrics.json`
- **Figures Directory:** `reports/phase9/naive_bayes/custom/figures/`
