# CargoSense — Phase 8: Feature Engineering & Baseline Modeling Preparation Report

**Primary Owner:** Member 2  
**Collaborator & Reviewer:** Sameera (Member 1 — Random Forest Lead)  
**Date:** October 2026  
**Input Data:** `data/processed/ode_opportunities.csv` (2,181 candidate opportunities across 727 shipments)  
**Output Partitions:** `data/processed/phase8_train.csv`, `data/processed/phase8_val.csv`, `data/processed/phase8_test.csv`  
**Serialized Pipeline:** `models/phase8_preprocessor.joblib`  
**Feature Dictionary:** `data/processed/phase8_feature_dictionary.csv`  

---

## Executive Summary

Phase 8 translates the operational Opportunity Discovery Engine (ODE) candidate dataset from Phase 6 and the empirical findings from Phase 7 into a **leakage-safe, reproducible machine learning dataset and preprocessing pipeline** for Phase 9 predictive modeling.

Phase 9 compares two tree-based ensemble paradigms: **Bagging** (Random Forest — Sameera) and **Boosting** (XGBoost — Member 2). To ensure an equitable and scientifically rigorous benchmark, Phase 8 establishes a strictly identical baseline feature set, an unbreached chronological train/validation/test split, and a unified preprocessing architecture fitted exclusively on training data.

### Key Deliverables & Outcomes:
1. **Schema Validation**: Validated the 14-feature shortlist from Phase 7 against `ode_opportunities.csv`. All 14 features exist, have 0 missing values, and are strictly available at the 18:00 planning cutoff.
2. **Strict Leakage Prevention**:
   - `candidate_strategy` was **completely excluded** from model inputs. In Phase 6 discrete event simulation, balanced candidates represented 100% of successful conversions while single-dimension heuristics yielded 0%. Including strategy would cause trivial shortcut learning rather than generalizable physical reasoning.
   - All post-cutoff operational fields (actual dock scan logs, actual dispatched weights/volumes, final utilization) and tracking IDs (`opportunity_id`, `shipment_id`, `truck_id`) were excluded.
3. **Chronological Shipment-Grouped Split**:
   - Partitions were formed chronologically by nightly planning runs:
     - **Train (Days 1–62)**: 62 runs, 511 shipments, 1533 candidates (141 positives, 9.20% positive rate).
     - **Validation (Days 63–75)**: 13 runs, 107 shipments, 321 candidates (23 positives, 7.17% positive rate).
     - **Test (Days 76–89)**: 14 runs, 109 shipments, 327 candidates (22 positives, 6.73% positive rate).
   - **Competing Candidate Integrity**: All 3 competing candidate combinations for each shipment remain strictly clustered within the same partition. There is **exactly 0 shipment overlap** and **0 run overlap** across splits.
4. **Scikit-Learn Preprocessing Pipeline**:
   - Implemented a unified `ColumnTransformer` applying `SimpleImputer(strategy='median')` and `StandardScaler()` to 11 numeric features, and `SimpleImputer(strategy='most_frequent')` with `OneHotEncoder(handle_unknown='ignore')` to 3 categorical features (`truck_type`, `route_id`, `day_of_week`).
   - Fitted **strictly on training data** and serialized to `models/phase8_preprocessor.joblib`. Transformed dimension: **31 features**.
5. **Class Imbalance Strategy**:
   - Addressed the ~9.87:1 imbalance ratio by computing standard cost-sensitive hyperparameters for Phase 9: `class_weight='balanced'` for Random Forest, and `scale_pos_weight=9.87` for XGBoost.

---

## 1. Schema Validation & Cutoff Availability Audit

The 14 features shortlisted in Phase 7 were audited against the actual columns in `data/processed/ode_opportunities.csv`:

| Feature Name | Shortlist Status | Found in Dataset | Data Type | Null Count | Pre-Cutoff Timing Verified? |
|:---|:---:|:---:|:---:|:---:|:---:|
| `remaining_capacity_kg` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `remaining_capacity_m3` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `candidate_added_weight_kg` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `candidate_added_volume_m3` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `projected_weight_util_pct` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `projected_volume_util_pct` | Yes | Yes | `float64` | 0 | Yes (18:00 Day d) |
| `n_candidate_skus` | Yes | Yes | `int64` | 0 | Yes (18:00 Day d) |
| `candidate_total_qty` | Yes | Yes | `int64` | 0 | Yes (18:00 Day d) |
| `mean_demand_last_7d_route` | Yes | Yes | `float64` | 0 | Yes (Day d-1 history) |
| `truck_type` | Yes | Yes | `object` | 0 | Yes (Truck master) |
| `route_id` | Yes | Yes | `object` | 0 | Yes (Route master) |
| `route_transit_hours` | Yes | Yes | `float64` | 0 | Yes (Route master) |
| `day_of_week` | Yes | Yes | `int64` | 0 | Yes (18:00 Day d) |
| `days_into_simulation` | Yes | Yes | `int64` | 0 | Yes (Planning date) |

### Findings & Resolutions:
- **No Schema Mismatches**: All 14 shortlisted columns are present under their exact names.
- **Completeness**: 0 missing values across all 2,181 rows.
- **Timing**: Every feature is derived strictly from data available at or before 18:00 on the day of the nightly planning run.
- **Zero-Variance Columns Noted**: Columns `avail_compatible_cargo_kg`, `avail_compatible_cargo_m3`, `n_compatible_pending_lines`, and `compatible_pending_qty` in `ode_opportunities.csv` contain all zeros. As mandated, these zero-variance columns were excluded from the baseline feature set without inventing synthetic placeholders.

---

## 2. Excluded Columns & Leakage Prevention Audit

To ensure the trained models predict real operational likelihood rather than memorizing dataset artifacts or exploiting future lookahead, the following columns are strictly excluded from the ML feature matrix:

| Excluded Field | Source / Type | Justification for Exclusion |
|:---|:---|:---|
| `candidate_strategy` | ODE Generation | **CRITICAL LEAKAGE**: In Phase 6, the `balanced` strategy achieved 100% of positive FTL conversions (186/186) while single-dimension strategies achieved 0%. Leaving this in would allow the tree models to split on a single categorical label and ignore all physical constraints. |
| `opportunity_id` | Identifier | Surrogate key; causes memorization of training instances. |
| `evaluation_timestamp` | Timestamp | Non-generalizable string timestamp. |
| `planning_run_id` | Operational Group | Identifier used solely to define chronological splits; excluded from model inputs. |
| `shipment_id` | Operational Group | Identifier used solely to enforce competing candidate clustering; excluded from inputs. |
| `truck_id` | Identifier | Chassis identifier; risk of memorizing specific vehicle operational quirks. |
| `candidate_sku_ids` | High-cardinality text | Semicolon-delimited string; adequately summarized by `n_candidate_skus`, weight, and volume. |
| `candidate_quantities` | High-cardinality text | Semicolon-delimited string; adequately summarized by `candidate_total_qty`. |
| `is_feasible` | Binary constant | Zero-variance constant (=1 for all candidates); uninformative. |
| `planned_weight_kg` | Load mass | Collinear with truck max payload and remaining capacity. |
| `planned_volume_m3` | Load cube | Collinear with truck max volume and remaining volume. |
| `projected_weight_kg` | Load mass | Linear sum of planned and added weight. |
| `projected_volume_m3` | Load cube | Linear sum of planned and added volume. |
| `route_distance_km` | Spatial distance | Highly collinear ($r > 0.98$) with `route_transit_hours` and captured by `route_id`. |
| `Became_FTL` | **Target Variable** | The ground-truth outcome being predicted; strictly held out as label vector $y$. |

---

## 3. Comprehensive Feature Dictionary

The following table documents the shared baseline and engineered custom features:

| feature_name | feature_group | type | pre_cutoff_availability | definition | missing_treatment | leakage_assessment | model_usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| remaining_capacity_kg | Truck Capacity | Numeric (Continuous) | Yes (Truck Master & Baseline Load at 18:00) | Residual payload weight allowance before consolidation: max_payload_kg - planned_weight_kg | SimpleImputer(strategy='median') | Safe: Derived strictly from pre-cutoff baseline cargo | Shared Baseline |
| remaining_capacity_m3 | Truck Capacity | Numeric (Continuous) | Yes (Truck Master & Baseline Load at 18:00) | Residual volume allowance before consolidation: max_volume_m3 - planned_volume_m3 (corrected 3x) | SimpleImputer(strategy='median') | Safe: Derived strictly from pre-cutoff baseline cargo | Shared Baseline |
| candidate_added_weight_kg | Proposed Cargo | Numeric (Continuous) | Yes (Candidate Proposal at 18:00) | Total aggregate mass of candidate order lines proposed for consolidation | SimpleImputer(strategy='median') | Safe: Proposed cargo mass known at planning cutoff | Shared Baseline |
| candidate_added_volume_m3 | Proposed Cargo | Numeric (Continuous) | Yes (Candidate Proposal at 18:00) | Total aggregate volume of candidate order lines proposed for consolidation (3x pack volume) | SimpleImputer(strategy='median') | Safe: Proposed cargo volume known at planning cutoff | Shared Baseline |
| projected_weight_util_pct | Utilization | Numeric (Continuous) | Yes (Calculated at 18:00) | Expected percentage payload utilization after adding candidate: (planned_weight + added_weight) / max_payload * 100 | SimpleImputer(strategy='median') | Safe: Prospective calculation; does not reflect actual dock loading | Shared Baseline |
| projected_volume_util_pct | Utilization | Numeric (Continuous) | Yes (Calculated at 18:00) | Expected percentage volume utilization after adding candidate: (planned_volume + added_volume) / max_volume * 100 | SimpleImputer(strategy='median') | Safe: Prospective calculation; does not reflect actual dock loading | Shared Baseline |
| n_candidate_skus | Product Complexity | Numeric (Discrete) | Yes (Candidate Proposal at 18:00) | Number of unique SKUs included in the candidate cargo combination | SimpleImputer(strategy='median') | Safe: Measures picking & packing complexity known at cutoff | Shared Baseline |
| candidate_total_qty | Proposed Cargo | Numeric (Discrete) | Yes (Candidate Proposal at 18:00) | Total count of individual piece units in the proposed candidate cargo | SimpleImputer(strategy='median') | Safe: Piece pick volume known at cutoff | Shared Baseline |
| mean_demand_last_7d_route | Commercial Velocity | Numeric (Continuous) | Yes (Historical Demand through Day d-1) | Rolling 7-day average daily units demanded for the destination route corridor | SimpleImputer(strategy='median') | Safe: Strictly uses prior 7 days of historical demand; no forward lookahead | Shared Baseline |
| route_transit_hours | Logistics & Route | Numeric (Continuous) | Yes (Route Master Table) | Standard line-haul transit time in hours from Bhiwandi Central Hub to destination | SimpleImputer(strategy='median') | Safe: Master static route operational parameter | Shared Baseline |
| days_into_simulation | Temporal | Numeric (Discrete) | Yes (Planning Date at 18:00) | Integer counter of days elapsed in the 90-day simulation timeline (1 to 89) | SimpleImputer(strategy='median') | Safe: Measures macro temporal progression; known at planning date | Shared Baseline |
| truck_type | Vehicle Class | Categorical (Nominal) | Yes (Truck Master Table) | Vehicle chassis category: '14ft LCV', '20ft ICV', '24ft MCV' | OneHotEncoder(handle_unknown='ignore') | Safe: Fixed physical truck attribute assigned to shipment | Shared Baseline |
| route_id | Logistics & Route | Categorical (Nominal) | Yes (Shipment Record at 18:00) | Unique line-haul corridor identifier (e.g., RT-MUM-PUN, RT-MUM-BLR) | OneHotEncoder(handle_unknown='ignore') | Safe: Destination route designated prior to planning run | Shared Baseline |
| day_of_week | Temporal | Categorical (Nominal / Discrete) | Yes (Planning Timestamp at 18:00) | Day of week of planning cutoff: 0 (Monday) to 6 (Sunday) | OneHotEncoder(handle_unknown='ignore') | Safe: Known exactly at cutoff; captures weekly operational rhythm | Shared Baseline |
| cargo_density_kg_m3 | Physical Density | Numeric (Continuous) | Yes (Derived at 18:00) | Density of proposed candidate cargo: candidate_added_weight_kg / candidate_added_volume_m3 | SimpleImputer(strategy='median') | Safe: Ratio of candidate properties known at cutoff | Custom Phase 9 (Runs 3 & 4) |
| weight_fill_ratio | Physical Fit | Numeric (Continuous) | Yes (Derived at 18:00) | Ratio of proposed cargo weight to remaining weight capacity | SimpleImputer(strategy='median') | Safe: Prospective fit ratio | Custom Phase 9 (Runs 3 & 4) |
| volume_fill_ratio | Physical Fit | Numeric (Continuous) | Yes (Derived at 18:00) | Ratio of proposed cargo volume to remaining volume capacity | SimpleImputer(strategy='median') | Safe: Prospective fit ratio | Custom Phase 9 (Runs 3 & 4) |
| avg_unit_weight_kg | Product Characteristics | Numeric (Continuous) | Yes (Derived at 18:00) | Average weight per unit in proposed cargo: candidate_added_weight_kg / candidate_total_qty | SimpleImputer(strategy='median') | Safe: Aggregate unit weight known at cutoff | Custom Phase 9 (Runs 3 & 4) |

---

## 4. Leakage-Safe Chronological Shipment-Grouped Split Strategy

### 4.1 Chronological Partitioning Logic
In industrial supply chain deployments, consolidation models are trained on historical warehouse dispatches and deployed to predict on future planning cycles. A random row-wise split would violate this real-world operational structure and produce overly optimistic evaluation.

Therefore, the 89 nightly planning runs were divided chronologically:
- **Training Set (Days 1–62: Jan 1 – Mar 2, 2024)**: Historical base for model fitting and cross-validation.
- **Validation Set (Days 63–75: Mar 3 – Mar 15, 2024)**: Model hyperparameter tuning and early stopping.
- **Test Set (Days 76–89: Mar 16 – Mar 29, 2024)**: Unseen held-out benchmark for final model comparison.

### 4.2 Competing Candidate Clustering Integrity
For every shipment, the ODE generates exactly 3 candidate SKU combinations (`balanced`, `volume_optimized`, `weight_optimized`). If one candidate were placed in the training set and another in the test set, information about that specific shipment's residual capacity and cargo availability would leak across partitions.

**Verification Results:**
- Shipments in Training: **511** (100% of 3 candidates = 1533 rows)
- Shipments in Validation: **107** (100% of 3 candidates = 321 rows)
- Shipments in Test: **109** (100% of 3 candidates = 327 rows)
- **Train $\cap$ Validation Shipments**: **0**
- **Train $\cap$ Test Shipments**: **0**
- **Validation $\cap$ Test Shipments**: **0**
- Competing candidate integrity is 100% preserved.

### 4.3 Class Prevalence Across Splits
- **Train**: 141 positives / 1533 rows = **9.20%**
- **Validation**: 23 positives / 321 rows = **7.17%**
- **Test**: 22 positives / 327 rows = **6.73%**
- **Overall**: 186 positives / 2181 rows = **8.53%**

The target prevalence remains consistent across the chronological horizon, confirming that temporal drift does not collapse the positive class in later observation periods.

---

## 5. Preprocessing Pipeline Architecture

The baseline pipeline is implemented using scikit-learn standard components:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, BASELINE_NUMERIC_FEATURES),
        ('cat', categorical_transformer, BASELINE_CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)
```

### Preprocessor Properties:
- **Fit Isolation**: The pipeline is fitted **strictly on `X_train`**. Validation and test matrices are transformed using the training parameters.
- **One-Hot Encoding**: Encodes 3 truck types, 10 route corridors, and 7 days of the week, expanding categorical inputs into 20 binary indicators.
- **Output Dimensions**: Total transformed feature dimension is **31 features** (11 scaled numeric features + 20 encoded categorical indicators).
- **Serialization**: Saved to `models/phase8_preprocessor.joblib` for zero-overhead loading in Phase 9.

---

## 6. Phase 9 Experiment Protocol & Model Loading Guide

In Phase 9, Sameera and Member 2 will execute the four designated experiments:

| Experiment | Model | Feature Set | Primary Metric | Hyperparameter Imbalance Treatment | Owner |
|:---|:---|:---|:---|:---|:---|
| **Run 1** | Random Forest | Shared Baseline (14 features) | PR-AUC, F1, Recall | `class_weight='balanced'` | Sameera |
| **Run 2** | XGBoost | Shared Baseline (14 features) | PR-AUC, F1, Recall | `scale_pos_weight=9.87` | Member 2 |
| **Run 3** | Random Forest | Baseline + Custom Features | PR-AUC, F1, Recall | `class_weight='balanced_subsample'` | Sameera |
| **Run 4** | XGBoost | Baseline + Custom Features | PR-AUC, F1, Recall | `scale_pos_weight=9.87` + tuning | Member 2 |

### Standard Code Snippet for Phase 9:
```python
import pandas as pd
import joblib

# 1. Load split partitions
train_df = pd.read_csv('data/processed/phase8_train.csv')
val_df   = pd.read_csv('data/processed/phase8_val.csv')
test_df  = pd.read_csv('data/processed/phase8_test.csv')

# 2. Extract targets
y_train = train_df['Became_FTL']
y_val   = val_df['Became_FTL']
y_test  = test_df['Became_FTL']

# 3. Load fitted preprocessor
preprocessor = joblib.load('models/phase8_preprocessor.joblib')

# 4. Transform baseline features
BASELINE_FEATURES = [
    'remaining_capacity_kg', 'remaining_capacity_m3',
    'candidate_added_weight_kg', 'candidate_added_volume_m3',
    'projected_weight_util_pct', 'projected_volume_util_pct',
    'n_candidate_skus', 'candidate_total_qty',
    'mean_demand_last_7d_route', 'route_transit_hours',
    'days_into_simulation', 'truck_type', 'route_id', 'day_of_week'
]

X_train = preprocessor.transform(train_df[BASELINE_FEATURES])
X_val   = preprocessor.transform(val_df[BASELINE_FEATURES])
X_test  = preprocessor.transform(test_df[BASELINE_FEATURES])

print(f"Transformed shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
```

---

## 7. Deliverable Verification Checklist

- [x] `data/processed/phase8_train.csv` (1,533 rows, 44 columns)
- [x] `data/processed/phase8_val.csv` (321 rows, 44 columns)
- [x] `data/processed/phase8_test.csv` (327 rows, 44 columns)
- [x] `data/processed/phase8_split_metadata.json` (complete partition metrics)
- [x] `data/processed/phase8_feature_dictionary.csv` (complete feature dictionary)
- [x] `models/phase8_preprocessor.joblib` (serialized fitted pipeline)
- [x] `reports/figures/fig09_train_val_test_split.png`
- [x] `reports/figures/fig10_baseline_feature_distributions.png`
- [x] `reports/figures/fig11_feature_correlation_matrix.png`
- [x] `reports/phase8_feature_engineering.md`
- [x] `notebooks/phase8_feature_engineering.ipynb`

Phase 8 is fully verified, complete, and ready for Phase 9 model training.
