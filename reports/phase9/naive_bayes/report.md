# CargoSense — Phase 9: Gaussian Naive Bayes Evaluation Report

**Model:** Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Feature Set:** 14 Approved Shared Baseline Features (11 Numerical, 3 Categorical)  
**Input Dimension:** 31 Transformed Features (via `models/phase8_preprocessor.joblib`)  
**Target Variable:** `Became_FTL` (Binary: 1 = Became FTL, 0 = Did Not Convert)  
**Decision Thresholds:** Default $\theta = 0.50$ | Validation-Optimized $\theta^* = 0.9000$  

---

## 1. Executive Summary

This report documents the implementation and empirical evaluation of **Gaussian Naive Bayes (GaussianNB)** as an additional benchmark classifier for the CargoSense Opportunity Detection Engine (ODE) in Phase 9.

Under the project plan, Member 2 owns XGBoost (Runs 2 & 4) and Sameera owns Random Forest (Runs 1 & 3). Gaussian Naive Bayes serves as a complementary parametric probabilistic baseline to evaluate how a simple generative model with strong independence assumptions behaves on operational logistics data.

### Key Performance Findings:
- **Held-Out Test PR-AUC:** **0.2738** (vs. 0.1903 baseline XGBoost, and 6.73% random prevalence)
- **Held-Out Test ROC-AUC:** **0.8474** (vs. 0.8273 baseline XGBoost)
- **Test Precision @ $\theta=0.50$:** **20.37%** (22/108 predictions)
- **Test Recall @ $\theta=0.50$:** **100.00%** (22/22 positives identified)
- **Test F1-Score @ $\theta=0.50$:** **0.3385**
- **Test Precision @ $\theta^*=0.90$:** **20.00%** (21/105 predictions)
- **Test Recall @ $\theta^*=0.90$:** **95.45%** (21/22 positives identified)
- **Test F1-Score @ $\theta^*=0.90$:** **0.3307**
- **Test Brier Score:** **0.2606** (vs. 0.1363 baseline XGBoost and 0.0812 custom XGBoost)

---

## 2. Theoretical Assumptions & Operational Limitations

Gaussian Naive Bayes relies on two core mathematical assumptions that are fundamentally violated by operational freight consolidation data:

### 2.1 Conditional Independence Assumption
GaussianNB assumes that all 31 input features are conditionally independent given the class label $Y \in \{0, 1\}$:
$$P(X_1, X_2, \dots, X_d \mid Y = c) = \prod_{i=1}^d P(X_i \mid Y = c)$$

**Operational Reality & Impact:**
In freight operations, features exhibit strong structural multicollinearity:
1. `candidate_added_weight_kg` and `candidate_added_volume_m3` are physically correlated via cargo density.
2. `remaining_capacity_kg` and `remaining_capacity_m3` correlate with the assigned truck chassis.
3. `projected_weight_util_pct` and `projected_volume_util_pct` are deterministic functions of remaining capacity and candidate cargo.
4. `candidate_total_qty` and `candidate_added_weight_kg` correlate strongly with shipment scale.

Because evidence is assumed independent, multiplying class-conditional likelihoods over 31 features leads to **redundant evidence multiplication**. This causes the posterior probabilities to become artificially polarized towards 0 or 1, resulting in poor probability calibration (**Brier score of 0.2606**).

### 2.2 Univariate Gaussian Likelihood Assumption
GaussianNB models continuous features within each class using a univariate Gaussian distribution:
$$P(X_i \mid Y = c) = \frac{1}{\sqrt{2\pi\sigma_{ci}^2}} \exp\left(-\frac{(x_i - \mu_{ci})^2}{2\sigma_{ci}^2}\right)$$

**Operational Reality & Impact:**
Physical logistics variables have bounded, non-Gaussian distributions:
- `projected_weight_util_pct` and `projected_volume_util_pct` are bounded percentages with sharp spikes near the 85% FTL threshold.
- `remaining_capacity` and piece counts are strictly non-negative, right-skewed counts and masses.
- One-hot encoded categorical indicator columns (such as routes and truck types) are discrete binary variables, for which continuous Gaussian density modeling is technically misspecified (despite StandardScaler centering).

---

## 3. Detailed Performance Across Splits

### 3.1 Evaluation Metrics Table

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

### 3.2 Test Set Confusion Matrices

#### Default Threshold ($	heta = 0.50$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          219                  86              (Specificity: 71.8%)
Actual Positive          0                    22              (Recall: 100.0%)
Total Test: 327 | Precision: 20.4% | Accuracy: 73.7%
```

#### Validation-Optimized Threshold ($	heta^* = 0.90$):
```
                  Predicted Negative    Predicted Positive
Actual Negative          221                  84              (Specificity: 72.5%)
Actual Positive          1                    21              (Recall: 95.5%)
Total Test: 327 | Precision: 20.0% | Accuracy: 74.0%
```

> [!NOTE]
> **Small-Sample Sensitivity on Test Set**:  
> The held-out test partition contains exactly 22 positive instances (`Became_FTL = 1`) out of 327 candidates (6.73% prevalence). Because of this small sample, a shift of just 1 true positive or false positive moves precision or recall by ~4.5 percentage points.

---

## 4. Comparison Against Baseline Tree Ensembles

| Model / Configuration | Feature Set | Input Dim | Test PR-AUC | Test ROC-AUC | Test F1 @ $\theta^*$ | Test Brier Score | Calibrated? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian Naive Bayes** | 14 Baseline | 31 | **0.2738** | **0.8474** | **0.3307** | **0.2606** | Poor (Polarized) |
| **XGBoost (Run 2 Baseline)** | 14 Baseline | 31 | 0.1903 | 0.8273 | 0.2667 | 0.1363 | Moderate |
| **XGBoost (Run 4 Custom)** | 14 Baseline + 4 Ratios | 35 | 0.2519 | 0.8596 | 0.2857 | 0.0812 | Excellent |

### Comparative Insights:
1. **Ranking vs. Calibration**: While GaussianNB produces strong PR-AUC (0.2738) and ROC-AUC (0.8474) by preserving monotonic ordering on dominant features like total quantity and transit hours, its predicted probabilities are poorly calibrated (Brier score of 0.2606).
2. **False Positive Burden**: At $\theta=0.50$, GaussianNB triggers 86 false alarms on the test set, recommending 108 dispatches to capture 22 conversions. In contrast, Custom XGBoost slashed false alarms to 32 while retaining high precision.
3. **Operational Recommendation**: While GaussianNB demonstrates surprising ranking ability, its inability to model joint non-linear capacity bottlenecks and its distorted posterior probabilities make it unsuited for direct score-based ranking in Phase 10 without Platt scaling or isotonic calibration.

---

## 5. Artifact Registry

All generated artifacts adhere to the designated Phase 9 folder structure:

| Artifact Type | Path | Purpose |
|:---|:---|:---|
| **Pipeline Script** | `src/phase9/naive_bayes/train.py` | Standalone reproducible training and evaluation pipeline |
| **Serialized Model** | `models/phase9/naive_bayes/gaussian_nb.joblib` | Fitted `GaussianNB` model object |
| **Test Predictions** | `reports/phase9/naive_bayes/predictions.csv` | Held-out test predictions with probabilities and decisions |
| **Val Predictions** | `reports/phase9/naive_bayes/val_predictions.csv` | Validation set predictions with probabilities and decisions |
| **Structured Metrics** | `reports/phase9/naive_bayes/metrics.json` | Comprehensive machine-readable metrics dictionary |
| **Confusion Matrix** | `reports/phase9/naive_bayes/figures/confusion_matrix.png` | Normalized and raw confusion matrix plot |
| **PR Curve Plot** | `reports/phase9/naive_bayes/figures/precision_recall_curve.png` | Precision-Recall curve with baseline prevalence reference |
| **ROC Curve Plot** | `reports/phase9/naive_bayes/figures/roc_curve.png` | Receiver Operating Characteristic diagnostic plot |
| **Calibration Plot** | `reports/phase9/naive_bayes/figures/calibration_curve.png` | Reliability diagram demonstrating probability overconfidence |

---

## 6. Reproduction Command

To reproduce this experiment from the repository root:
```bash
python -m src.phase9.naive_bayes.train
```
*(Execution completes in < 2 seconds without retraining or modifying shared artifacts).*
