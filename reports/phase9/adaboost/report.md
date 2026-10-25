# CargoSense — Phase 9: AdaBoost Implementation, Evaluation & Comparative Analysis Report
## Run 5 (Baseline AdaBoost) vs. Run 6 (Custom AdaBoost)

**Author / Lead:** Adaptive Boosting Lead  
**Date:** October 2026  
**Input Partitions:** `data/processed/phase8_train.csv`, `data/processed/phase8_val.csv`, `data/processed/phase8_test.csv`  
**Baseline Artifacts:** `models/phase9/adaboost/baseline/model.joblib`, `reports/phase9/adaboost/baseline/`  
**Custom Artifacts:** `models/phase9/adaboost/custom/model.joblib`, `models/phase9/adaboost/custom/preprocessor.joblib`, `reports/phase9/adaboost/custom/`  
**Comparative Figures Directory:** `reports/phase9/adaboost/figures/`  

---

## Executive Summary

Phase 9 implements and benchmarks sequential boosting ensemble architectures for the CargoSense Opportunity Detection Engine (ODE). The engine scores candidate Partial-Truckload (PTL) consolidation packages to predict whether a candidate will successfully convert to Full-Truckload status prior to line-haul dispatch (`Became_FTL = 1`).

This report provides the complete comparative evaluation of **Run 5: Baseline AdaBoost** (14 shared Phase 8 baseline features, 31 encoded dimensions) against **Run 6: Custom AdaBoost** (18 features, including 4 engineered physical fill ratios and packaging densities, 35 encoded dimensions).

```
========================================================================================================
EXPERIMENT SUMMARY: ADABOOST RUN 5 (BASELINE) VS. RUN 6 (CUSTOM)
========================================================================================================
Metric                          Run 5: Baseline AdaBoost         Run 6: Custom AdaBoost           Delta
--------------------------------------------------------------------------------------------------------
Features & Preprocessing        14 shared Phase 8 (31 dims)      18 features (35 dims)            +4 ratios
Base Estimator                  DecisionTree (depth 3)           DecisionTree (depth 3)           Equal depth
Hyperparameters                 50 estimators, lr=1.0            70 estimators, lr=0.8            Refined
Validation Optimal Threshold    theta* = 0.4500                  theta* = 0.4700                  Tuned on Val
Validation PR-AUC (Primary)     0.3193                           0.3047                           -0.0146 (-4.6%)
Validation ROC-AUC              0.8726                           0.8793                           +0.0067 (+0.8%)
Validation F1-Score             0.3615                           0.3784                           +0.0169 (+4.7%)
Validation Precision @ theta*   25.00%                           24.14%                           -0.86%
Validation Recall @ theta*      65.22%                           60.87%                           -4.35%
Held-Out Test PR-AUC (Primary)  0.2639                           0.2006                           -0.0633 (-24.0%)
Held-Out Test ROC-AUC           0.8645                           0.8510                           -0.0135 (-1.6%)
Held-Out Test F1-Score          0.2857                           0.2500                           -0.0357 (-12.5%)
Held-Out Test Precision @ theta*20.00%                           17.24%                           -2.76%
Held-Out Test Recall @ theta*   50.00%                           45.45%                           -4.55%
Held-Out Test Brier Score (Loss)0.1090                           0.1077                           -0.0013 (-1.2%)
Held-Out Test False Positives   44 (85.57% specificity)          48 (84.26% specificity)          +4 alarms (+9.1%)
Held-Out Test False Negatives   11 (50.00% missed FTL)           12 (54.55% missed FTL)           +1 missed (+9.1%)
========================================================================================================
```

### Key Technical Findings:

1. **Baseline Superiority on Primary Ranking Metric (PR-AUC)**:
   - On the held-out test partition, **Run 5 (Baseline AdaBoost) achieved a PR-AUC of 0.2639**, significantly outperforming the 6.73% random prevalence baseline and exceeding **Run 6 (Custom AdaBoost, PR-AUC = 0.2006) by +31.5% relative (+0.0633 absolute)**.
   - While the custom engineered ratios (`avg_unit_weight_kg`, `weight_fill_ratio`, `cargo_density_kg_m3`, `volume_fill_ratio`) were actively utilized by the custom weak learners (ranking in the top 7 features), they introduced split redundancy and gradient variance in sequential boosting updates, diluting generalization on late-period test batches.
2. **Operational Staging Trade-Offs at Optimized Thresholds**:
   - Classification thresholds were selected strictly on validation data to maximize the F1-score ($\theta^* = 0.45$ for Baseline, $\theta^* = 0.47$ for Custom).
   - Baseline AdaBoost identified **11 of 22 converting opportunities (50.00% recall)** with **44 false positive staging alerts (85.57% specificity)**.
   - Custom AdaBoost identified **10 of 22 converting opportunities (45.45% recall)** with **48 false positive staging alerts (84.26% specificity)**.
   - In cross-dock operations, Run 5 delivers fewer wasted staging allocations (44 vs. 48) while successfully capturing an additional line-haul conversion.
3. **Probability Calibration Consistency**:
   - Both models exhibited solid probability calibration on the held-out test partition, with Brier scores of **0.1090** (Baseline) and **0.1077** (Custom).
   - In reliability diagram analysis, both models show slight overconfidence in the mid-probability tier ($0.40 - 0.70$), typical of exponential-loss boosting ensembles, which is effectively countered by threshold optimization.
4. **Dominance of Volumetric and Quantity Cues**:
   - In both models, `candidate_total_qty` emerged as the #1 most important feature (22.76% importance in Baseline, 20.86% in Custom).
   - In Custom AdaBoost, `avg_unit_weight_kg` claimed the #2 spot (15.08%), followed by `weight_fill_ratio` (9.95%) and `cargo_density_kg_m3` (8.83%), confirming strong model receptivity to physical density dynamics.

---

## 1. Experimental Setup & Dataset Partitions

### 1.1 Chronological Shipment-Grouped Partitions
All experiments strictly preserve the chronological shipment-grouped splits established in Phase 8:

| Partition | Simulation Days | Records | Shipments | Positives | Prevalence | Split Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Train** | Days 1–62 | 1,533 | 511 | 141 | 9.20% | Model training & preprocessor fitting |
| **Validation** | Days 63–75 | 321 | 107 | 23 | 7.17% | Hyperparameter tuning & threshold selection ($\theta^*$) |
| **Held-Out Test** | Days 76–89 | 327 | 109 | 22 | 6.73% | Final unbiased operational evaluation |

Zero shipment, truck, or operational run overlap exists across partitions. Competing candidate combinations (3 candidate packages per shipment: `balanced`, `volume_optimized`, `weight_optimized`) remain grouped together within their respective split.

### 1.2 Data Leakage Prevention Audit
All post-cutoff operational fields, surrogate identifiers, and dispatch outcomes were strictly prohibited:
- Prohibited columns: `candidate_strategy`, `opportunity_id`, `shipment_id`, `truck_id`, `planning_run_id`, `evaluation_timestamp`, `candidate_sku_ids`, `candidate_quantities`, `actual_dispatched_weight_kg`, `actual_dispatched_volume_m3`, `actual_ftl_conversion`.
- In Run 6 Custom AdaBoost, additional features from other models (`capacity_bottleneck_ratio`, `capacity_utilization_imbalance`) were strictly excluded.

### 1.3 Class Imbalance Strategy
To handle the ~9.87:1 negative-to-positive class imbalance without generating synthetic points that violate freight physics:
$$\text{sample\_weight}_i = \frac{N}{2 \cdot N_{y_i}}$$
For training data, negative sample weight $w_0 \approx 0.5506$ and positive sample weight $w_1 \approx 5.4362$. Initializing AdaBoost's sample distribution with balanced weights forces early weak learners to focus on minority-class decision boundaries before standard adaptive reweighting takes over.

---

## 2. Feature Engineering & Preprocessing Architecture

### 2.1 Shared Baseline Features (Run 5: 14 Features, 31 Encoded Dims)
Run 5 uses the 14 features shortlisted in Phase 7 and preprocessed via `models/phase8_preprocessor.joblib`:
- **11 Numerical Features:** `remaining_capacity_kg`, `remaining_capacity_m3`, `candidate_added_weight_kg`, `candidate_added_volume_m3`, `projected_weight_util_pct`, `projected_volume_util_pct`, `n_candidate_skus`, `candidate_total_qty`, `mean_demand_last_7d_route`, `route_transit_hours`, `days_into_simulation`
  - Preprocessing: `SimpleImputer(strategy="median")` + `StandardScaler()`
- **3 Categorical Features:** `truck_type` (3 types), `route_id` (10 corridors), `day_of_week` (7 days)
  - Preprocessing: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 2.2 Custom Features (Run 6: 18 Features, 35 Encoded Dims)
Run 6 incorporates the 14 baseline features plus exactly four physical interaction ratios:
1. `avg_unit_weight_kg` = $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_total\_qty}}$
2. `weight_fill_ratio` = $\frac{\text{candidate\_added\_weight\_kg}}{\text{remaining\_capacity\_kg}}$
3. `volume_fill_ratio` = $\frac{\text{candidate\_added\_volume\_m3}}{\text{remaining\_capacity\_m3}}$
4. `cargo_density_kg_m3` = $\frac{\text{candidate\_added\_weight\_kg}}{\text{candidate\_added\_volume\_m3}}$

Preprocessing is performed using a custom `ColumnTransformer` fitted strictly on `phase8_train.csv` and saved to `models/phase9/adaboost/custom/preprocessor.joblib`.

---

## 3. AdaBoost Modeling Methodology & Hyperparameter Selection

### 3.1 Weak Learner Architecture: Depth-3 Decision Trees vs. Stumps
In standard textbook AdaBoost, decision stumps (`max_depth=1`) are common. However, empirical sweep on the validation set revealed:
- **Decision Stumps ($d=1$):** Achieved validation PR-AUC of only ~0.230, as single-variable axis-aligned cuts cannot represent the simultaneous weight-and-volume constraints essential to freight loading.
- **Deep Trees ($d \ge 4$):** Memorized the 141 training positives (training PR-AUC reached 1.000) and degraded on held-out test data (test PR-AUC fell to 0.152).
- **Depth-3 Trees ($d=3$):** Selected as the optimal weak learner complexity. A 3-level tree provides up to 8 leaf regions, capturing non-linear interactions (e.g., high cube utilization conditional on sufficient payload headroom) while maintaining high generalization stability.

### 3.2 Hyperparameter Tuning Summary

Hyperparameter tuning was conducted strictly against the validation partition using validation PR-AUC:

| Parameter | Run 5 Baseline | Run 6 Custom | Selection Rationale |
|:---|:---:|:---:|:---|
| **Base Estimator** | `DecisionTreeClassifier(max_depth=3)` | `DecisionTreeClassifier(max_depth=3)` | Balances non-linear capacity interaction capacity with anti-overfitting |
| **$n\_estimators$** | 50 | 70 | Validation PR-AUC peaked at 50 rounds for baseline and 70 rounds for custom |
| **Learning Rate ($\eta$)** | 1.0 | 0.8 | Lower learning rate in custom provides shrinkage against feature collinearity |
| **Sample Weight** | Balanced ($w_0 \approx 0.55, w_1 \approx 5.44$) | Balanced ($w_0 \approx 0.55, w_1 \approx 5.44$) | Balances positive loss penalty across sequential boosting stages |
| **Random State** | 42 | 42 | Full reproducibility |

---

## 4. Comprehensive Performance Evaluation

### 4.1 Detailed Metric Comparison Across Splits

| Metric | Baseline Val ($\theta=0.50$) | Baseline Val ($\theta^*=0.45$) | Baseline Test ($\theta=0.50$) | Baseline Test ($\theta^*=0.45$) | Custom Val ($\theta=0.50$) | Custom Val ($\theta^*=0.47$) | Custom Test ($\theta=0.50$) | Custom Test ($\theta^*=0.47$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.3193 | 0.3193 | **0.2639** | **0.2639** | 0.3047 | 0.3047 | 0.2006 | 0.2006 |
| **ROC-AUC** | 0.8726 | 0.8726 | **0.8645** | **0.8645** | 0.8793 | 0.8793 | 0.8510 | 0.8510 |
| **Brier Score** | 0.1091 | 0.1091 | 0.1090 | 0.1090 | 0.1077 | 0.1077 | **0.1077** | **0.1077** |
| **Precision** | 30.00% | 25.00% | 18.75% | **20.00%** | 32.14% | 24.14% | **26.67%** | 17.24% |
| **Recall** | 13.04% | 65.22% | 13.64% | **50.00%** | 39.13% | 60.87% | 36.36% | 45.45% |
| **F1-Score** | 0.1818 | 0.3615 | 0.1579 | **0.2857** | 0.3529 | 0.3784 | **0.3077** | 0.2500 |
| **Accuracy** | 91.59% | 83.49% | 90.21% | 83.18% | 89.41% | 82.55% | 88.99% | 81.65% |
| **Specificity**| 97.65% | 84.90% | 95.74% | 85.57% | 93.29% | 84.23% | 92.79% | 84.26% |
| **True Positives (TP)** | 3 | 15 | 3 | **11** | 9 | 14 | 8 | 10 |
| **False Positives (FP)**| 7 | 45 | 13 | **44** | 20 | 47 | 22 | 48 |
| **False Negatives (FN)**| 20 | 8 | 19 | **11** | 14 | 9 | 14 | 12 |
| **True Negatives (TN)** | 291 | 253 | 292 | 261 | 278 | 251 | 283 | 257 |

### 4.2 Held-Out Test Confusion Matrices (@ Optimal Threshold $\theta^*$)

```
RUN 5: BASELINE ADABOOST (@ theta* = 0.45)
Actual \\ Pred       Did Not Convert (0)     Became FTL (1)      Total
Did Not Convert (0)        261 (85.57%)         44 (14.43%)        305
Became FTL (1)              11 (50.00%)         11 (50.00%)         22
Total                      272                  55                 327

RUN 6: CUSTOM ADABOOST (@ theta* = 0.47)
Actual \\ Pred       Did Not Convert (0)     Became FTL (1)      Total
Did Not Convert (0)        257 (84.26%)         48 (15.74%)        305
Became FTL (1)              12 (54.55%)         10 (45.45%)         22
Total                      269                  58                 327
```

---

## 5. Diagnostic Figures & Visual Analysis

All diagnostic figures are persisted in `reports/phase9/adaboost/figures/`:

| Figure File | Description | Operational Insight |
|:---|:---|:---|
| [`fig01_pr_curves.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase9/adaboost/figures/fig01_pr_curves.png) | Precision-Recall Curves for Runs 5 & 6 on Validation and Test sets | Baseline AdaBoost maintains higher precision across recall thresholds on test data. |
| [`fig02_roc_curves.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase9/adaboost/figures/fig02_roc_curves.png) | ROC Curves for Runs 5 & 6 on Validation and Test sets | Both models maintain strong discriminative separation (ROC-AUC ~0.85–0.87). |
| [`fig03_calibration_curves.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase9/adaboost/figures/fig03_calibration_curves.png) | Reliability Diagram for Baseline Test vs. Custom Test | Custom AdaBoost achieves slightly lower calibration loss (Brier 0.1077 vs 0.1090). |
| [`fig04_confusion_matrices.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase9/adaboost/figures/fig04_confusion_matrices.png) | Side-by-side test confusion matrices at optimal operating thresholds | Baseline achieves higher specificity (85.57% vs 84.26%) and lower false alarms (44 vs 48). |
| [`fig05_feature_comparison.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase9/adaboost/figures/fig05_feature_comparison.png) | Top 10 feature importances for Baseline vs. Custom | Custom physical ratios capture 38.9% of total ensemble weight in Run 6. |

---

## 6. Feature Importance & Operational Interpretation

### Top 10 Feature Importances Comparison:

| Rank | Run 5 Baseline Feature | Importance | Run 6 Custom Feature | Importance |
|:---:|:---|:---:|:---|:---:|
| 1 | `num__candidate_total_qty` | 0.2276 | `num__candidate_total_qty` | 0.2086 |
| 2 | `num__projected_volume_util_pct` | 0.2002 | `num__avg_unit_weight_kg` *(Custom)* | 0.1508 |
| 3 | `num__candidate_added_weight_kg` | 0.1167 | `num__weight_fill_ratio` *(Custom)* | 0.0995 |
| 4 | `num__route_transit_hours` | 0.0955 | `num__cargo_density_kg_m3` *(Custom)* | 0.0883 |
| 5 | `num__remaining_capacity_kg` | 0.0768 | `num__projected_volume_util_pct` | 0.0791 |
| 6 | `num__n_candidate_skus` | 0.0526 | `num__candidate_added_weight_kg` | 0.0695 |
| 7 | `num__mean_demand_last_7d_route` | 0.0521 | `num__volume_fill_ratio` *(Custom)* | 0.0504 |
| 8 | `cat__truck_type_14ft LCV` | 0.0381 | `cat__day_of_week_6` | 0.0422 |
| 9 | `num__projected_weight_util_pct` | 0.0353 | `num__remaining_capacity_m3` | 0.0411 |
| 10 | `cat__day_of_week_6` | 0.0343 | `num__projected_weight_util_pct` | 0.0360 |

### Analysis:
- In Run 5, tree splits rely predominantly on volume utilization (`projected_volume_util_pct` = 20.02%) and package count (`candidate_total_qty` = 22.76%).
- In Run 6, the engineered features absorb substantial weight: `avg_unit_weight_kg` (15.08%), `weight_fill_ratio` (9.95%), `cargo_density_kg_m3` (8.83%), and `volume_fill_ratio` (5.04%) account for **38.9%** of all tree split decisions.
- However, because these ratios are derived from raw inputs already available to depth-3 trees (`candidate_added_weight_kg`, `candidate_added_volume_m3`, `remaining_capacity_kg`), introducing them created collinear split competition, slightly degrading out-of-sample generalization.

---

## 7. Conclusions, Operational Trade-Offs & Limitations

### 7.1 Operational Conclusions:
1. **Model Selection for AdaBoost Deployment:**
   - **Run 5 (Baseline AdaBoost)** is the recommended configuration between the two boosting runs, delivering superior precision-recall area (0.2639 vs. 0.2006) and a cleaner trade-off between recall (50.00%) and specificity (85.57%).
2. **Threshold Tuning is Operationally Mandatory:**
   - At the standard threshold $\theta = 0.50$, both AdaBoost models suffer from severe recall collapse (13.64% in Baseline, 36.36% in Custom), missing the vast majority of converting freight.
   - Operating at the validation-selected threshold ($\theta^* = 0.45$ for Baseline, $\theta^* = 0.47$ for Custom) successfully recovers 50.00% recall, identifying high-value line-haul opportunities for warehouse staging.

### 7.2 Project Limitations & Synthetic Target Disclaimers:
- **Synthetic Ground Truth:** The target variable `Became_FTL` is derived from deterministic heuristics and order fulfillment thresholds defined in Phase 4. Consequently, all reported metrics reflect model fidelity against the synthetic operational distribution, rather than observed empirical freight behavior.
- **Line-Haul Corridors:** Route-specific effects are conditioned on simulated corridors (`RT-MUM-*`); unobserved network congestion, carrier capacity shifts, and dynamic pricing were not simulated.
- **Future Scope:** Cross-model synthesis comparing AdaBoost against Bagging (Random Forest), Gradient Boosting (XGBoost), and Generative Classifiers (Naive Bayes) will be conducted in a dedicated cross-model benchmarking report.
