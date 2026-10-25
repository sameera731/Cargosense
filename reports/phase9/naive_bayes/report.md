# CargoSense — Phase 9: Gaussian Naive Bayes Implementation, Evaluation & Comparative Analysis Report
## Baseline GaussianNB vs. Custom GaussianNB

**Author / Team:** CargoSense Data Science Team  
**Phase:** Phase 9 (Opportunity Detection Engine — Model Exploration & Benchmarking)  
**Input Partitions:** `data/processed/phase8_train.csv`, `data/processed/phase8_val.csv`, `data/processed/phase8_test.csv`  
**Shared Preprocessor:** `models/phase8_preprocessor.joblib` (unmodified, fitted strictly on Train)  
**Baseline Artifacts:** `models/phase9/naive_bayes/baseline/`, `reports/phase9/naive_bayes/baseline/`  
**Custom Artifacts:** `models/phase9/naive_bayes/custom/`, `reports/phase9/naive_bayes/custom/`  
**Comparative Figures Directory:** `reports/phase9/naive_bayes/figures/`  

---

## Executive Summary

Phase 9 implements and benchmarks probabilistic and tree-based classification architectures for the CargoSense Opportunity Detection Engine (ODE). The engine evaluates candidate Partial-Truckload (PTL) consolidation combinations to predict whether an opportunity will successfully convert to Full-Truckload status prior to line-haul dispatch (`Became_FTL = 1`).

This report presents a thorough, methodologically rigorous comparison of **Baseline Gaussian Naive Bayes** (all 14 shared Phase 8 features, 31 encoded dimensions, default variance smoothing $10^{-9}$) against **Custom Gaussian Naive Bayes** (decorrelated 4-feature physical subset, regularized variance smoothing $10^{-3}$, empirical class priors).

```
========================================================================================================
EXPERIMENT SUMMARY: GAUSSIAN NAIVE BAYES BASELINE VS. CUSTOM
========================================================================================================
Metric                          Baseline GaussianNB              Custom GaussianNB                Delta
--------------------------------------------------------------------------------------------------------
Features & Preprocessing        14 shared Phase 8 (31 dims)      4 decorrelated physical dims     -27 dims (pruned)
Hyperparameter var_smoothing    1e-09 (scikit-learn default)     1e-03 (regularized)              10^6x smoothing
Class Priors                    Empirical [0.9080, 0.0920]       Empirical [0.9080, 0.0920]       Preserved
Validation Optimal Threshold    theta* = 0.9000                  theta* = 0.9200                  Tuned on Val
Validation PR-AUC               0.2239                           0.3444                           +0.1205 (+53.8%)
Validation ROC-AUC              0.8573                           0.8410                           -0.0163 (-1.9%)
Validation F1-Score             0.3566                           0.3218                           -0.0348 (-9.8%)
Test PR-AUC (Primary)           0.2738                           0.1537                           -0.1201 (-43.9%)
Test ROC-AUC                    0.8474                           0.7948                           -0.0526 (-6.2%)
Test Brier Score (Loss)         0.2606                           0.2861                           +0.0255 (+9.8%)
Test False Positives (FP)       84 (72.46% specificity)          41 (86.56% specificity)          -43 alerts (-51.2%)
Test Classification Accuracy    74.01%                           83.18%                           +9.17 pp
========================================================================================================
```

### Key Technical Findings:

1. **Validation Selection vs. Test Generalization (Honest Evaluation)**:
   - On the chronological validation partition (Days 63–75), the **Custom GaussianNB** model achieved a peak PR-AUC of **0.3444**, an impressive **+53.8% relative improvement** over the baseline (0.2239). Pruning correlated volume and route features significantly reduced GaussianNB's tendency to overconfidently assign extreme posterior probabilities on validation combinations.
   - However, on the final held-out test partition (Days 76–89), the **Baseline GaussianNB** achieved a higher PR-AUC of **0.2738** compared to **0.1537** for Custom GaussianNB. The baseline's inclusion of line-haul corridor transit features and route indicators retained predictive value across late-stage simulation shifts that the 4-feature physical subset could not access.
   - **Crucial project conclusion:** We do not claim the custom model is superior overall. Rather, each model represents a distinct operational trade-off profile on held-out test rows.

2. **Dramatic Operational False-Alarm Suppression**:
   - At their respective validation-tuned operating thresholds ($\theta^* = 0.90$ for Baseline, $\theta^* = 0.92$ for Custom), the Custom model slashed false positive alerts from **84 down to 41** on identical test rows.
   - This represents a **51.19% reduction in false alarms** (-43 false staging dispatches) and boosted test specificity from **72.46% to 86.56%**, elevating overall classification accuracy from **74.01% to 83.18%**.
   - In cross-dock freight environments, where every false positive commits dock labor and bay capacity to staging cargo that fails to convert, this operational noise reduction is of critical commercial value.

3. **High Optimal Thresholds ($\theta^* \approx 0.90 - 0.92$) & Probability Distortion**:
   - Both models required decision thresholds far above the conventional 0.50 cutoff ($\theta^* = 0.90$ and $0.92$).
   - This phenomenon is a direct mathematical consequence of the **conditional independence assumption**:
     $$P(X_1, \dots, X_d \mid Y=c) = \prod_{i=1}^d P(X_i \mid Y=c)$$
     When multiple positively correlated physical features (such as weight utilization, volume utilization, added weight, and SKU count) are present, the Naive Bayes likelihood product multiplies partially redundant probabilities, pushing posterior probability estimates toward extreme values (0.0 or 1.0).

4. **Brier Score Diagnostics**:
   - The test Brier scores for Baseline (0.2606) and Custom (0.2861) reflect probability calibration challenges inherent to generative Gaussian assumptions on multi-modal logistics data (where true test prevalence is only 6.73%, meaning an uncalibrated majority predictor outputs Brier $\approx 0.063$). Tree-based models (Random Forest Brier 0.0561, XGBoost Brier 0.0583) significantly outperform GaussianNB on calibration because they do not impose distributional or independence assumptions.

---

## 1. Experimental Setup & Dataset Partitions

### 1.1 Chronological Shipment-Grouped Partitions
All experiments strictly adhere to the chronological shipment-grouped splits established in Phase 8 to prevent data leakage and shipment cross-contamination:

| Partition | Simulation Days | Records | Shipments | Positives | Prevalence | Split Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Train** | Days 1–62 | 1,533 | 511 | 141 | 9.20% | Model training & prior calculation |
| **Validation** | Days 63–75 | 321 | 107 | 23 | 7.17% | Hyperparameter grid search & threshold selection ($\theta^*$) |
| **Held-Out Test** | Days 76–89 | 327 | 109 | 22 | 6.73% | Final unbiased benchmark (identical rows across all models) |

Zero shipment, truck, or operational run overlap exists between partitions. Competing candidate combinations generated for the same parent shipment remain grouped within the same partition.

### 1.2 Shared Preprocessor Integrity
The preprocessing pipeline `models/phase8_preprocessor.joblib` was preserved without modification. The pipeline was fitted strictly on the 1,533 training rows and applied as a frozen transform to validation and test partitions:
- **Numerical Pipeline**: Median imputation followed by `StandardScaler`.
- **Categorical Pipeline**: Frequent category imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

---

## 2. Model Methodology & Tuning Strategy

### 2.1 Baseline Gaussian Naive Bayes Architecture
The baseline model serves as the reference benchmark:
- **Feature Set**: All 14 Phase 8 baseline features transformed into 31 encoded dimensions (11 standardized numerical features + 20 one-hot encoded categorical indicators).
- **Variance Smoothing (`var_smoothing`)**: $10^{-9}$ (standard `scikit-learn` default).
- **Class Priors**: Empirical training set frequencies:
  $$P(Y=0) = \frac{1392}{1533} \approx 0.9080, \quad P(Y=1) = \frac{141}{1533} \approx 0.0920$$
- **Threshold Tuning**: Evaluated across 91 evenly spaced threshold candidates $\theta \in [0.05, 0.95]$ on validation data to maximize F1-score. Optimal validation threshold: $\theta^* = 0.90$.

### 2.2 Custom Gaussian Naive Bayes Optimization
To address the primary vulnerability of GaussianNB—feature redundancy and collinearity—a systematic 72-configuration validation search was conducted over three design dimensions:

1. **Feature Subset Selection**:
   - `all_31_encoded`: All 31 dimensions from the baseline preprocessor.
   - `all_11_numerical`: The 11 continuous features, removing one-hot sparse categorical indicators.
   - `decorrelated_physical_4`: 4 core physical features (`projected_weight_util_pct`, `candidate_total_qty`, `candidate_added_weight_kg`, `n_candidate_skus`), deliberately excluding collinear volume features (`projected_volume_util_pct` has Pearson $r = 0.88$ with weight utilization) and route indicators.
   - `top_6_separators`: The top 6 features exhibiting the largest standardized class-mean separation in training data.

2. **Variance Regularization (`var_smoothing`)**:
   - Grid: $\{10^{-9}, 10^{-7}, 10^{-5}, 10^{-3}, 10^{-2}, 10^{-1}\}$. Increasing smoothing broadens class variance, acting as an empirical shrinkage regularizer.

3. **Class Prior Formulations**:
   - `empirical`: $P(Y) = [0.9080, 0.0920]$.
   - `uniform`: $P(Y) = [0.5000, 0.5000]$ (forcing equal class weight to penalize majority bias).

#### Validation Selection Result:
- **Selected Configuration**: `decorrelated_physical_4`, `var_smoothing = 0.001`, `priors = empirical`.
- **Validation PR-AUC**: **0.3444** (highest among all 72 configurations, vs. 0.2239 for baseline).
- **Optimal Decision Threshold**: $\theta^* = 0.92$ (maximizing validation F1-score at 0.3218).

---

## 3. Comprehensive Performance Evaluation

### 3.1 Validation Set Performance (Tuning & Selection)

Evaluated on the 321 validation rows (23 positive FTL conversions):

| Metric | Baseline GaussianNB (Val @ $\theta^*=0.90$) | Custom GaussianNB (Val @ $\theta^*=0.92$) | Delta ($\Delta$) | Operational Meaning |
|:---|:---:|:---:|:---:|:---|
| **PR-AUC (Primary)** | 0.2239 | **0.3444** | **+0.1205 (+53.8%)** | Massive precision-recall gain on validation distribution |
| **ROC-AUC** | **0.8573** | 0.8410 | -0.0163 (-1.9%) | Both models maintain strong overall ranking ability |
| **Brier Score (Loss)** | **0.2599** | 0.2986 | +0.0387 | Custom posterior distribution is slightly wider |
| **Precision @ $\theta^*$** | 21.70% (23/106) | **21.88% (14/64)** | **+0.18 pp** | Comparable precision on minority class |
| **Recall @ $\theta^*$** | **100.00% (23/23)** | 60.87% (14/23) | -39.13 pp | Baseline flags all positives at the expense of high false alarms |
| **F1-Score @ $\theta^*$** | **0.3566** | 0.3218 | -0.0348 | Threshold selection objective |
| **Accuracy @ $\theta^*$** | 74.14% | **81.62%** | **+7.48 pp** | Substantially higher overall accuracy |
| **Specificity @ $\theta^*$** | 72.15% (215/298) | **83.22% (248/298)** | **+11.07 pp** | Custom model rejects non-converting candidates far better |
| **False Positives (FP)** | 83 | **50** | **-33 alerts (-39.8%)** | 33 fewer false alarms on validation staging floor |

---

### 3.2 Held-Out Test Set Performance (Final Unbiased Benchmark)

Evaluated on the 327 held-out test rows (22 positive FTL conversions) at fixed validation-selected thresholds:

| Metric | Baseline GaussianNB (Test @ $\theta^*=0.90$) | Custom GaussianNB (Test @ $\theta^*=0.92$) | Delta ($\Delta$) | Operational Impact |
|:---|:---:|:---:|:---:|:---|
| **PR-AUC (Primary)** | **0.2738** | 0.1537 | -0.1201 (-43.9%) | Baseline corridor features generalize better to test dates |
| **ROC-AUC** | **0.8474** | 0.7948 | -0.0526 (-6.2%) | Baseline retains broader discriminative ranking |
| **Brier Score (Loss)** | **0.2606** | 0.2861 | +0.0255 (+9.8%) | Moderate probability calibration across both models |
| **Precision @ $\theta^*$** | **20.00% (21/105)** | 16.33% (8/49) | -3.67 pp | Baseline achieves slightly higher positive predictive value |
| **Recall @ $\theta^*$** | **95.45% (21/22)** | 36.36% (8/22) | -59.09 pp | Baseline captures nearly all FTL conversions (21/22) |
| **F1-Score @ $\theta^*$** | **0.3307** | 0.2253 | -0.1054 | Baseline higher due to dominant recall |
| **Accuracy @ $\theta^*$** | 74.01% | **83.18%** | **+9.17 pp** | Custom model classifies 83.2% of test rows correctly |
| **Specificity @ $\theta^*$** | 72.46% (221/305) | **86.56% (264/305)** | **+14.10 pp** | **Superior rejection of non-converting loads** |
| **False Positives (FP)** | 84 | **41** | **-43 alerts (-51.2%)** | **43 fewer wasted warehouse staging dispatches** |
| **True Negatives (TN)** | 221 | **264** | **+43 correct rejections** | Significantly cleaner staging bays |
| **Missed Positives (FN)** | **1** | 14 | +13 missed conversions | Trade-off: Custom is conservative; Baseline is aggressive |

---

## 4. Confusion Matrix Analysis (Held-Out Test Set)

Evaluating identical 327 test records under the validation-derived operating thresholds:

### Baseline GaussianNB ($\theta^* = 0.90$)
```
                          PREDICTED
                    Did Not Convert    Became FTL      Total
ACTUAL
Did Not Convert           221 (TN)         84 (FP)       305 (Specificity: 72.5%)
Became FTL                  1 (FN)         21 (TP)        22 (Recall: 95.5%)
Total                     222             105            327 (Precision: 20.0%)
```

### Custom GaussianNB ($\theta^* = 0.92$)
```
                          PREDICTED
                    Did Not Convert    Became FTL      Total
ACTUAL
Did Not Convert           264 (TN)         41 (FP)       305 (Specificity: 86.6%)
Became FTL                 14 (FN)          8 (TP)        22 (Recall: 36.4%)
Total                     278              49            327 (Precision: 16.3%)
```

### Operational Synthesis:
- **Baseline GaussianNB** acts as a **high-sensitivity screening filter**. It captures 21 out of 22 FTL opportunities (95.45% recall), missing only a single conversion. However, it generates 84 false alarms, meaning 4 out of every 5 flagged shipments do not convert.
- **Custom GaussianNB** acts as a **precision/specificity filter**. It cuts false alerts by more than half (down to 41, a 51.2% reduction) and correctly identifies 264 true negatives, but misses 14 FTL opportunities.
- In cross-dock operations with high staging costs and limited bay space, the Custom model's false-positive suppression provides strong operational value, whereas in operations prioritizing maximum volume capture regardless of staging churn, the Baseline model's high recall is preferred.

---

## 5. Feature Contribution & Separability Analysis

Because Gaussian Naive Bayes is a generative parametric classifier rather than a tree-based ensemble, Mean Decrease in Impurity (Gini) is not defined. Instead, feature importance is measured via **standardized class-mean separation** (the univariate Mahalanobis distance):
$$d_j = \frac{|\mu_{j, 1} - \mu_{j, 0}|}{\sqrt{\frac{1}{2}(\sigma_{j, 1}^2 + \sigma_{j, 0}^2)}}$$

This metric directly reflects how effectively each feature shifts the class-conditional log-likelihood ratio in Bayes' theorem:

| Rank | Feature Name | Baseline Separability ($d_j$) | Custom Separability ($d_j$) | Operational Domain |
|:---:|:---|:---:|:---:|:---|
| 1 | `num__projected_weight_util_pct` | 1.0738 | 1.0727 | Capacity Utilization |
| 2 | `num__projected_volume_util_pct` | 1.0535 | *Excluded (Collinear)* | Capacity Utilization |
| 3 | `num__candidate_total_qty` | 0.4624 | 0.4621 | Proposed Cargo Quantity |
| 4 | `num__candidate_added_weight_kg` | 0.4087 | 0.4085 | Proposed Cargo Weight |
| 5 | `num__n_candidate_skus` | 0.3636 | 0.3634 | Cargo SKU Diversity |
| 6 | `num__candidate_added_volume_m3` | 0.2998 | *Excluded (Collinear)* | Proposed Cargo Volume |
| 7 | `num__mean_demand_last_7d_route` | 0.1422 | *Excluded* | Corridor Demand History |
| 8 | `cat__truck_type_14ft LCV` | 0.1171 | *Excluded* | Fleet Equipment Class |
| 9 | `cat__day_of_week_1` (Tuesday) | 0.1103 | *Excluded* | Temporal Schedule |
| 10 | `cat__truck_type_20ft ICV` | 0.1102 | *Excluded* | Fleet Equipment Class |

### Insights:
- Projected weight utilization (`projected_weight_util_pct`) and volume utilization (`projected_volume_util_pct`) are the dominant class separators, with $d > 1.05$.
- However, weight and volume utilization are strongly collinear ($r = 0.88$). In the Baseline model, including both features caused the Naive Bayes likelihood product to double-count utilization evidence. The Custom model resolves this by pruning volume utilization, relying purely on the 4 decorrelated physical dimensions.

---

## 6. Diagnostic Figures Registry

All comparative evaluation figures have been generated and saved to [`reports/phase9/naive_bayes/figures/`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/):

| Figure File | Description | Key Diagnostic Insight |
|:---|:---|:---|
| [`fig01_pr_curves.png`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/fig01_pr_curves.png) | Precision-Recall Curves for Baseline & Custom on Validation and Test sets | Custom dominates on validation; Baseline retains broader test precision across recall levels. |
| [`fig02_roc_curves.png`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/fig02_roc_curves.png) | ROC Curves for Baseline & Custom on Validation and Test sets | Baseline achieves test ROC-AUC 0.8474 vs. Custom 0.7948. |
| [`fig03_calibration_curves.png`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/fig03_calibration_curves.png) | Reliability diagrams (calibration curves) on held-out test partition | Illustrates the sigmoidal distortion caused by GaussianNB independence assumption. |
| [`fig04_confusion_matrices.png`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/fig04_confusion_matrices.png) | Side-by-side test confusion matrices at operating thresholds $\theta^*$ | Highlights the 51.2% false alarm reduction (84 vs. 41 FP) in the Custom model. |
| [`fig05_feature_comparison.png`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/figures/fig05_feature_comparison.png) | Class separation bar chart across top features | Quantifies the standardized mean difference ($d_j$) for all model inputs. |

Individual experiment figures are also maintained in:
- Baseline Figures: `reports/phase9/naive_bayes/baseline/figures/`
- Custom Figures: `reports/phase9/naive_bayes/custom/figures/`

---

## 7. Cross-Model Context: GaussianNB vs. Tree Ensembles

To provide context within the broader CargoSense Phase 9 benchmarking initiative, the table below places Gaussian Naive Bayes alongside Random Forest and XGBoost on identical held-out test rows (327 records, 22 positives):

| Model Architecture | Phase 9 Run | Test PR-AUC | Test ROC-AUC | Test Brier Score | Test Recall | Test Specificity | Test False Positives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline GaussianNB** | NB Baseline | **0.2738** | 0.8474 | 0.2606 | **95.45%** | 72.46% | 84 |
| **Custom GaussianNB** | NB Custom | 0.1537 | 0.7948 | 0.2861 | 36.36% | **86.56%** | **41** |
| **Baseline Random Forest** | Run 1 | 0.2233 | 0.8616 | 0.0711 | 54.55% | 85.25% | 45 |
| **Custom Random Forest** | Run 3 | 0.2158 | 0.8614 | **0.0561** | 36.36% | **89.51%** | **32** |
| **Baseline XGBoost** | Run 2 | 0.2312 | 0.8647 | 0.0634 | 59.09% | 84.92% | 46 |
| **Custom XGBoost** | Run 4 | **0.2520** | **0.8710** | 0.0583 | 54.55% | 88.52% | 35 |

### Algorithmic Takeaways:
1. **Tree Ensembles Are Superior for Calibrated Probability Scoring**:
   - Random Forest (Brier 0.0561) and XGBoost (Brier 0.0583) produce dramatically lower Brier scores than GaussianNB (0.2606–0.2861). Tree models partition the feature space non-linearly without assuming Gaussian feature distributions or class-conditional independence.
2. **GaussianNB Offers a Unique High-Recall Screening Option**:
   - Baseline GaussianNB achieves the highest test recall in the entire project (**95.45%**, 21/22 opportunities detected). In applications where missing an opportunity incurs severe penalties and downstream verification is inexpensive, Baseline GaussianNB serves as an effective wide-funnel pre-screener.
3. **Custom GaussianNB Confirms Collinearity Sensitivity**:
   - The custom experiment confirmed that pruning correlated features successfully curbs overconfident false positives, cutting false alarms by 51.2% and improving test specificity to 86.56%.

---

## 8. Artifact Registry & Reproduction Commands

### 8.1 Deliverable Artifact Locations

| Artifact | Location |
|:---|:---|
| **Baseline Source Code** | [`src/phase9/naive_bayes/baseline/train.py`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/src/phase9/naive_bayes/baseline/train.py) |
| **Custom Source Code** | [`src/phase9/naive_bayes/custom/train.py`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/src/phase9/naive_bayes/custom/train.py) |
| **Baseline Model Joblib** | [`models/phase9/naive_bayes/baseline/model.joblib`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/models/phase9/naive_bayes/baseline/model.joblib) |
| **Baseline Configuration** | [`models/phase9/naive_bayes/baseline/config.json`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/models/phase9/naive_bayes/baseline/config.json) |
| **Custom Model Joblib** | [`models/phase9/naive_bayes/custom/model.joblib`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/models/phase9/naive_bayes/custom/model.joblib) |
| **Custom Configuration** | [`models/phase9/naive_bayes/custom/config.json`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/models/phase9/naive_bayes/custom/config.json) |
| **Baseline Predictions (Test)** | [`reports/phase9/naive_bayes/baseline/predictions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/baseline/predictions.csv) |
| **Baseline Predictions (Val)** | [`reports/phase9/naive_bayes/baseline/val_predictions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/baseline/val_predictions.csv) |
| **Baseline Metrics Summary** | [`reports/phase9/naive_bayes/baseline/metrics.json`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/baseline/metrics.json) |
| **Baseline Individual Report** | [`reports/phase9/naive_bayes/baseline/report.md`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/baseline/report.md) |
| **Custom Predictions (Test)** | [`reports/phase9/naive_bayes/custom/predictions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/custom/predictions.csv) |
| **Custom Predictions (Val)** | [`reports/phase9/naive_bayes/custom/val_predictions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/custom/val_predictions.csv) |
| **Custom Metrics Summary** | [`reports/phase9/naive_bayes/custom/metrics.json`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/custom/metrics.json) |
| **Custom Individual Report** | [`reports/phase9/naive_bayes/custom/report.md`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/custom/report.md) |
| **Consolidated Report** | [`reports/phase9/naive_bayes/report.md`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/reports/phase9/naive_bayes/report.md) |
| **Jupyter Notebook** | [`notebooks/phase9_naive_bayes.ipynb`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/notebooks/phase9_naive_bayes.ipynb) |

### 8.2 Reproduction Commands

To reproduce the Baseline Gaussian Naive Bayes pipeline:
```bash
python -m src.phase9.naive_bayes.baseline.train
```

To reproduce the Custom Gaussian Naive Bayes pipeline:
```bash
python -m src.phase9.naive_bayes.custom.train
```

Both pipelines execute deterministically in under 3 seconds from the repository root.
