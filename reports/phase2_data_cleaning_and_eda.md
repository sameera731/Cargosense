# CargoSense -- Phase 2: Data Cleaning and Exploratory Data Analysis (EDA) Report

**Project Title:** CargoSense: A Data-Driven Opportunity Detection Engine for PTL-to-FTL Conversion  
**Stage:** Phase 2 -- Data Cleaning and Exploratory Data Analysis (EDA)  
**Date:** October 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary & Objective

The objective of **Phase 2** is to rigorously examine, clean, and explore the two original datasets:
1. **Supply-Chain Dataset:** Product- and SKU-level operational attributes across 100 products.
2. **Smart-Logistics Dataset:** IoT telemetry, environmental conditions, and transit events across 10 fleet assets over 2024.

This phase was executed under strict non-destructive principles:
- Raw source files in `data/raw/` remain untouched.
- Cleaned and standardized datasets were exported to `data/processed/`.
- No synthetic records or final multi-table schemas (`products.csv`, `trucks.csv`, `orders.csv`, `ode_opportunities.csv`) were created.
- No predictive ML models were trained.

Through detailed statistical profiling, cross-tabulation, and 10 custom visual analyses, Phase 2 provides the empirical foundation needed to understand the strengths, severe anomalies, synthetic artifacts, and critical missing links in the data before designing the CargoSense Opportunity Detection Engine.

---

## 2. In-Depth Data Quality Analysis

### 2.1 Missing Values and Identifiers

#### Supply Chain Dataset (`supply_chain_data.csv`)
- **Missing Values:** Exactly `0` across all 24 columns (100% data completeness).
- **Duplicate Records:** `0` exact duplicate rows.
- **Identifier Audit:** The column `sku` contains 100 unique values (`SKU0` through `SKU99`) for 100 rows, confirming its integrity as a unique primary key for the catalog.

#### Smart Logistics Dataset (`smart_logistics_dataset.csv`)
- **Missing Values:** Only one column contains missing data:
  - `logistics_delay_reason`: 263 null values out of 1,000 rows (**26.3% missing**).
  - All other 15 columns have zero missing cells.
- **Duplicate Records:** `0` exact duplicate rows.
- **Identifier Audit:** Telemetry pings are grouped under `asset_id` (`Truck_1` through `Truck_10`). Each vehicle has between 89 and 109 pings. No single composite identifier `(asset_id, timestamp)` is duplicated.

---

### 2.2 Deep Investigation of Overlapping & Ambiguous Columns

#### A. `lead_times` vs `lead_time` vs `manufacturing_lead_time` (Supply Chain)
The raw dataset contains two nearly identical column names (`Lead times` and `Lead time`) alongside `Manufacturing lead time`.
- **Statistical Audit:**
  - `lead_times` (plural): Mean = 15.96 days, Median = 17.0 days, Std = 8.79 days, Range = [1, 30].
  - `lead_time` (singular): Mean = 17.08 days, Median = 18.0 days, Std = 8.85 days, Range = [1, 30].
  - `manufacturing_lead_time`: Mean = 14.77 days, Median = 14.0 days, Std = 8.91 days, Range = [1, 30].
  - **Correlation:** The Pearson correlation between `lead_times` and `lead_time` is **$-0.0028$** (statistically uncorrelated).
  - **Row Equality:** In only **1 out of 100 rows** do `lead_times` and `lead_time` have matching values.
- **Finding:** These are not duplicate encodings of the same measurement. They appear to be independently sampled integers from a discrete uniform distribution $\mathcal{U}\{1, 30\}$. Neither column is discarded during cleaning, but `lead_times` is documented as supplier fulfillment lead time, while `manufacturing_lead_time` covers factory production.

#### B. `costs` vs `manufacturing_costs` vs `shipping_costs` (Supply Chain)
- In logistics accounting, total unit cost is typically broken down into manufacturing, warehousing, and transportation costs.
- **Statistical Audit:**
  - $\text{costs} - (\text{manufacturing\_costs} + \text{shipping\_costs})$ is strictly positive for all 100 SKUs:
    - Minimum difference: $+\$28.32$
    - Maximum difference: $+\$949.53$
    - Mean difference: $+\$476.43$
- **Finding:** `costs` is **not** an aggregate sum of manufacturing and shipping costs. It represents a separate, broader total operational cost allocation (e.g., including inventory holding, customs, packaging, or marketing overhead).

#### C. `revenue_generated` vs $(\text{price} \times \text{number_of_products_sold})$ (Supply Chain)
- Calculated transaction volume: $\text{Calculated} = \text{price} \times \text{number\_of\_products\_sold}$.
- Recorded `revenue_generated`:
  - Mean absolute error: **$\$19,187.46$**.
  - Ratio $\frac{\text{revenue\_generated}}{\text{price} \times \text{sold}}$ spans from **$0.034$** ($3.4\%$) to **$105.76$** ($10,576\%$), with a median of $0.353$.
- **Finding:** `revenue_generated` does not correspond to single-period SKU product revenue. It was generated independently or reflects multi-channel/annual brand revenue.

---

### 2.3 Mathematical Proof: Deterministic Target Leakage in `logistics_delay`

In predictive modeling, **target leakage** occurs when a predictor variable contains information about the target that would not be available at the time of prediction, or when the target is directly constructed from feature columns.

We conducted a systematic cross-tabulation of `logistics_delay` against all categorical and numerical features:

| `shipment_status` | `traffic_status` | `logistics_delay == 0` (On-Time) | `logistics_delay == 1` (Delayed) | Total Pings |
| :--- | :--- | :---: | :---: | :---: |
| **Delayed** | Any (`Clear`, `Detour`, `Heavy`) | **0** | **350** | 350 |
| Any (`In Transit`, `Delivered`) | **Heavy** | **0** | **216** | 216 |
| `In Transit` | `Clear` or `Detour` | **214** | **0** | 214 |
| `Delivered` | `Clear` or `Detour` | **220** | **0** | 220 |
| **Total** | | **434** | **566** | **1,000** |

#### The Exact Deterministic Rule
$$\text{logistics\_delay} = 1 \iff (\text{shipment\_status} == \text{'Delayed'}) \lor (\text{traffic\_status} == \text{'Heavy'})$$

- When condition is `True`: **566 out of 566 records** have `logistics_delay == 1` (100.0% precision, 100.0% recall).
- When condition is `False`: **434 out of 434 records** have `logistics_delay == 0` (100.0% precision, 100.0% recall).
- **Rule Accuracy: 1,000 / 1,000 records (100.00%)**.

#### Critical Machine Learning Implications
1. Any machine learning model trained to predict `logistics_delay` while including `shipment_status` or `traffic_status` in its feature set will experience **100% target leakage**.
2. A trivial decision tree or logistic regression model will learn this exact boolean gate and achieve 1.00 ROC-AUC, creating an illusion of high predictive power while offering zero real-world utility.
3. In operational deployment, `shipment_status == 'Delayed'` is the **outcome** of a delay, not a prior predictive signal.

---

### 2.4 Anomaly: Independent Generation of `logistics_delay_reason`

In genuine logistics management systems, root causes (e.g., `Weather`, `Traffic`, `Mechanical Failure`) are documented only when an exception or delay occurs.

Cross-tabulating `logistics_delay` vs `logistics_delay_reason`:
| Reported Reason | `logistics_delay == 0` (On-Time) | `logistics_delay == 1` (Delayed) | Overall Total |
| :--- | :---: | :---: | :---: |
| `Weather` | 116 (26.7%) | 151 (26.7%) | 267 |
| `Traffic` | 101 (23.3%) | 135 (23.9%) | 236 |
| `Mechanical Failure` | 101 (23.3%) | 133 (23.5%) | 234 |
| `Missing / NaN` | 116 (26.7%) | 147 (26.0%) | 263 |
| **Total** | **434 (100%)** | **566 (100%)** | **1,000** |

**Finding:**  
- **318 out of 434 non-delayed shipments (73.3%)** are assigned active delay causes (`Mechanical Failure`, `Weather`, `Traffic`) despite suffering no delay.
- Conversely, **147 out of 566 delayed shipments (26.0%)** have missing reason tags (`NaN`).
- The four states are uniformly distributed at ~25% across both classes, proving that `logistics_delay_reason` was populated independently by a synthetic sampler.

---

### 2.5 Geographical & Trajectory Plausibility Audit

1. **Global Coordinates vs Localized Operations:**
   - Supply chain origins are located in 5 Indian commercial hubs (`Mumbai`, `Delhi`, `Kolkata`, `Bangalore`, `Chennai`).
   - In contrast, the smart logistics GPS coordinates span:
     - `latitude`: $[-89.79^\circ, +89.87^\circ]$ (South Pole to North Pole)
     - `longitude`: $[-179.82^\circ, +179.92^\circ]$ (Full global circumference)
2. **Consecutive Ping Jumps:**
   - For all 10 trucks, computing the distance between consecutive chronological pings reveals:
     - Mean consecutive latitude jump: **$55.0^\circ$ to $63.8^\circ$**.
     - Mean consecutive longitude jump: **$108.4^\circ$ to $128.2^\circ$**.
   - A truck reporting a ping in Antarctica at 10:00 AM reports its next ping 2 days later in the Arctic or mid-Pacific.
   - **Conclusion:** These coordinates are synthetically generated spatial noise. They cannot be used for road route routing or real geofenced transit distance calculation.

---

### 2.6 Asset ID Audit

- The fleet consists of 10 distinct entities: `Truck_1` through `Truck_10`.
- All identifiers follow the `Truck_{i}` naming convention.
- There are no mixed asset classes (e.g., vans, ships, drones).
- **Limitation:** The dataset does not specify vehicle configuration (e.g., 20ft container, 32ft multi-axle, refrigerated reefer, or dry van), nor does it state vehicle gross weight ratings (GVWR) or volumetric boundaries.

---

## 3. Justified Cleaning Decisions

All cleaning steps were executed deterministically and saved into `data/processed/`. No data rows were deleted.

### Summary of Cleaning Actions
| Dataset | Column / Operation | Action Taken | Operational Justification |
| :--- | :--- | :--- | :--- |
| **Both** | All column headers | Renamed to `snake_case` | Eliminates spaces, hyphens, and mixed-case parsing bugs across Python and SQL pipelines. |
| **Supply Chain** | Text fields | Stripped whitespace | Ensures string equality matching succeeds without silent whitespace mismatch. |
| **Supply Chain** | Ambiguous columns | Preserved as-is | Dropping `lead_times` or `costs` would destroy information; keeping them with clear documentation avoids false assumptions. |
| **Supply Chain** | Operational Screening | Added `ptl_road_candidate` | Non-destructive boolean flag: marks SKUs with `transportation_modes == 'Road'`, `stock_levels > 0`, and `inspection_results == 'Pass'` (7 out of 100 SKUs meet criteria). |
| **Logistics** | `timestamp` | Converted to `datetime64[ns]` | Enables time-series operations, chronological sorting, and rolling aggregations. |
| **Logistics** | Ordering | Sorted chronologically by `timestamp` | Resolves unordered telemetry pings; essential for time-series and state tracking. |
| **Logistics** | Temporal features | Extracted `month`, `day_of_week`, `hour` | Enhances feature space for diurnal and seasonal traffic analysis. |
| **Logistics** | `logistics_delay_reason` | Imputed `NaN` $\rightarrow$ `'Not Specified'` | Preserves 263 rows without losing record integrity. |
| **Logistics** | Reconciled Delay Reason | Added `logistics_delay_reason_cleaned` | Reconciles reporting logic: sets reason to `'None (On-Time)'` if `logistics_delay == 0`, and `'Unspecified Delay'` if delayed with missing tag. |
| **Logistics** | Operational Screening | Added `is_ptl_candidate` | Flags records where `asset_utilization < 80.0%` (512 out of 1,000 pings represent PTL opportunities). |

---

## 4. Extended Statistical Profiles

### 4.1 Cleaned Supply Chain Dataset
*Dataset dimensions: 100 rows, 25 columns (including candidate flag).*

| Column | Mean | Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min | Max | Range | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `price` ($) | 49.46 | 31.17 | 51.24 | 19.60 | 77.20 | 57.60 | 1.70 | 99.17 | 97.47 | -0.02 |
| `availability` (units) | 48.40 | 30.74 | 43.50 | 22.75 | 75.00 | 52.25 | 1.00 | 100.00 | 99.00 | +0.18 |
| `number_of_products_sold` | 460.99 | 303.78 | 392.50 | 184.25 | 704.25 | 520.00 | 8.00 | 996.00 | 988.00 | +0.28 |
| `revenue_generated` ($) | 5,776.05 | 2,732.84 | 6,006.35 | 2,812.85 | 8,253.98 | 5,441.13 | 1,061.62 | 9,866.47 | 8,804.85 | -0.17 |
| `stock_levels` (units) | 47.77 | 31.37 | 47.50 | 16.75 | 73.00 | 56.25 | 0.00 | 100.00 | 100.00 | +0.10 |
| `lead_times` (days) | 15.96 | 8.79 | 17.00 | 8.00 | 24.00 | 16.00 | 1.00 | 30.00 | 29.00 | -0.13 |
| `order_quantities` (units)| 49.22 | 26.78 | 52.00 | 26.00 | 71.25 | 45.25 | 1.00 | 96.00 | 95.00 | -0.11 |
| `shipping_times` (days) | 5.75 | 2.72 | 6.00 | 3.75 | 8.00 | 4.25 | 1.00 | 10.00 | 9.00 | -0.28 |
| `shipping_costs` ($) | 5.55 | 2.65 | 5.32 | 3.54 | 7.60 | 4.06 | 1.01 | 9.93 | 8.92 | -0.05 |
| `lead_time` (days) | 17.08 | 8.85 | 18.00 | 10.00 | 25.00 | 15.00 | 1.00 | 30.00 | 29.00 | -0.33 |
| `production_volumes` | 567.84 | 263.05 | 568.50 | 352.00 | 797.00 | 445.00 | 104.00 | 985.00 | 881.00 | -0.08 |
| `manufacturing_lead_time`| 14.77 | 8.91 | 14.00 | 7.00 | 23.00 | 16.00 | 1.00 | 30.00 | 29.00 | +0.18 |
| `manufacturing_costs` ($)| 47.27 | 28.98 | 45.91 | 22.98 | 68.62 | 45.64 | 1.09 | 99.47 | 98.38 | +0.19 |
| `defect_rates` (%) | 2.28 | 1.46 | 2.14 | 1.01 | 3.56 | 2.55 | 0.02 | 4.94 | 4.92 | +0.13 |
| `costs` ($) | 529.25 | 258.30 | 520.43 | 318.78 | 763.08 | 444.30 | 103.92 | 997.41 | 893.50 | +0.04 |

---

### 4.2 Cleaned Smart Logistics Dataset
*Dataset dimensions: 1,000 rows, 21 columns (including parsed date parts & flags).*

| Column | Mean | Std | Median | Q1 (25%) | Q3 (75%) | IQR | Min | Max | Range | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `asset_utilization` (%) | 79.60 | 11.63 | 79.25 | 69.48 | 89.43 | 19.95 | 60.00 | 100.00 | 40.00 | +0.04 |
| `inventory_level` (units) | 297.92 | 113.55 | 299.00 | 201.00 | 399.00 | 198.00 | 100.00 | 500.00 | 400.00 | +0.02 |
| `waiting_time` (min) | 35.06 | 14.48 | 35.00 | 23.00 | 49.00 | 26.00 | 10.00 | 60.00 | 50.00 | +0.01 |
| `demand_forecast` (units) | 199.28 | 59.92 | 202.00 | 144.00 | 251.25 | 107.25 | 100.00 | 300.00 | 200.00 | -0.02 |
| `temperature` (°C) | 23.89 | 3.32 | 23.80 | 21.20 | 26.60 | 5.40 | 18.00 | 30.00 | 12.00 | +0.02 |
| `humidity` (%) | 65.04 | 8.75 | 65.20 | 57.20 | 72.40 | 15.20 | 50.00 | 80.00 | 30.00 | -0.05 |
| `user_transaction_amount` | 299.06 | 117.79 | 301.50 | 191.75 | 405.00 | 213.25 | 100.00 | 500.00 | 400.00 | +0.00 |
| `user_purchase_frequency` | 5.51 | 2.94 | 6.00 | 3.00 | 8.00 | 5.00 | 1.00 | 10.00 | 9.00 | -0.00 |
| `logistics_delay` | 0.57 | 0.50 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | -0.27 |

*Statistical Note on Distributions:*  
Across virtually all numerical fields in both datasets, the skewness values are between $-0.33$ and $+0.28$, indicating near-perfect symmetry. In the logistics data, features like `inventory_level` (100–500), `demand_forecast` (100–300), and `waiting_time` (10–60) follow uniform or truncated normal distributions typical of synthetic benchmark datasets.

---

## 5. Visualizations & Analytical Interpretations

Ten high-resolution charts were produced and saved to `reports/figures/`.

### 5.1 Supply Chain Analysis

#### Figure 1: Key Numerical Distributions
- **File:** [`reports/figures/fig01_sc_numerical_distributions.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig01_sc_numerical_distributions.png)
- **Observations:** Prices ($1.70 to $99.17) and stock levels (0 to 100 units) are broadly dispersed without extreme isolated outliers. Total costs exhibit a bell-like dispersion centered at $520.
- **CargoSense Relevance:** The wide spread in stock levels confirms that some SKUs have zero or negligible safety stock, which must be filtered out when considering cargo additions to avoid fulfillment stockouts.

#### Figure 2: Categorical Frequencies & Consolidation Candidacy
- **File:** [`reports/figures/fig02_sc_categorical_distributions.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig02_sc_categorical_distributions.png)
- **Observations:**  
  - Transportation modes: `Road` accounts for **29 SKUs (29%)**, Rail (28%), Air (26%), and Sea (17%).
  - Quality audit: Only **23% of SKUs have Passed inspection**, while 41% are Pending and 36% have Failed.
- **CargoSense Relevance:** Road freight is the only transport mode suitable for truckload consolidation. Furthermore, combining `transportation_modes == 'Road'` with `inspection_results == 'Pass'` narrows the candidate pool to **7 ready-to-dispatch SKUs**.

#### Figure 3: Cost and Revenue Accounting Consistency
- **File:** [`reports/figures/fig03_sc_cost_and_revenue_analysis.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig03_sc_cost_and_revenue_analysis.png)
- **Observations:** Shows the complete separation of total costs from the sum of manufacturing and shipping costs, as well as the non-linear scatter between calculated sales value and recorded revenue.
- **CargoSense Relevance:** Prevents algorithms from relying on simplistic arithmetic cost identities.

#### Figure 4: Supply Chain Correlation Heatmap
- **File:** [`reports/figures/fig04_sc_correlation_heatmap.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig04_sc_correlation_heatmap.png)
- **Observations:** Pairwise correlations among all 15 numerical features are weak ($|r| < 0.25$). Notably, `lead_times` and `lead_time` show $r = -0.00$.

---

### 5.2 Smart Logistics Analysis

#### Figure 5: Operational Capacity & Inventory Distributions
- **File:** [`reports/figures/fig05_logistics_distributions.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig05_logistics_distributions.png)
- **Observations:** `asset_utilization` spans strictly between 60.0% and 100.0%. Mean utilization is 79.6%.
- **CargoSense Relevance:** In trucking operations, a load below 80%–85% is typically considered a Partial Truckload (PTL). Exactly **512 out of 1,000 pings (51.2%)** fall below 80% utilization, representing the primary target pool for opportunity detection.

#### Figure 6: Fleet Pings and Delay Reason Cleanup
- **File:** [`reports/figures/fig06_logistics_categoricals.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig06_logistics_categoricals.png)
- **Observations:** Telemetry pings are evenly distributed across all 10 assets (`Truck_1` through `Truck_10`), with each truck contributing roughly 10% of records.

#### Figure 7: ML Target Leakage Proof
- **File:** [`reports/figures/fig07_logistics_delay_leakage_proof.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig07_logistics_delay_leakage_proof.png)
- **Observations:** Visualizes the 100% delay rate whenever `shipment_status == 'Delayed'` or `traffic_status == 'Heavy'`, providing clear graphical proof of synthetic feature construction.

#### Figure 8: 2024 Time-Series Trends
- **File:** [`reports/figures/fig08_logistics_time_series.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig08_logistics_time_series.png)
- **Observations:** Weekly aggregated fleet utilization fluctuates between 76% and 83%, regularly crossing the 80% PTL opportunity boundary throughout the entire year.

#### Figure 9: Logistics Correlation Heatmap
- **File:** [`reports/figures/fig09_logistics_correlation_heatmap.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig09_logistics_correlation_heatmap.png)
- **Observations:** Confirms near-zero linear correlations between delay and environmental/sensor metrics (temperature, humidity, waiting time, utilization).

#### Figure 10: Geographic Dispersion (GPS Scatter)
- **File:** [`reports/figures/fig10_logistics_gps_dispersion.png`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures/fig10_logistics_gps_dispersion.png)
- **Observations:** Demonstrates the uniform global scatter of coordinates across all four quadrants of the Earth, confirming that coordinates cannot represent real road trucking paths.

---

## 6. What the Existing Datasets CAN and CANNOT Support

### 6.1 What They CAN Support
1. **Catalog & SKU Profiling:** Identifying product categories, price tiers, baseline order batch sizes, and warehouse stock levels.
2. **Quality & Transport Filtering:** Isolating SKUs that travel by road and pass inspection.
3. **PTL Vehicle State Identification:** Using `asset_utilization < 80.0%` to identify candidate underutilized trucks in need of consolidation.
4. **Dock Waiting Time Constraints:** Leveraging `waiting_time` (10–60 min) as an operational time window during which loading can take place before truck departure.

### 6.2 What They CANNOT Support (Critical Data Gaps)
1. **No Relational Linkage:** Zero common foreign keys connect the two datasets. We cannot tell which truck is carrying which SKU, or which warehouse a truck is currently visiting.
2. **No Physical Weight/Volume Capacities:**
   - Real PTL-to-FTL consolidation requires verifying that $\sum \text{weight} \le \text{Max Payload}$ and $\sum \text{volume} \le \text{Cargo Volume}$.
   - Neither dataset defines SKU physical dimensions/weights or truck volumetric limits.
3. **No Dynamic Order Backlog:**
   - The supply chain data provides only a static aggregate `order_quantities` number.
   - It lacks pending customer orders with specific order timestamps, destination addresses, and requested delivery deadlines.
4. **No Loading Manifests:** No trip manifests tracking cargo items loaded or unloaded over time.
5. **No Ground Truth Conversion Outcomes:** Neither dataset tracks whether a PTL was successfully filled to FTL prior to dispatch, or what cost savings were realized.

---

## 7. Strategic Synthesis: Data Needed in Later Phases

To build the end-to-end **CargoSense Opportunity Detection Engine** without misrepresenting the raw data, the following operational schema must be synthetically modeled in Phase 3:

```text
[Truck Assets] (from Smart Logistics)
       │
       ▼
[Truck Trips & Current Loads]  ◄─── (Capacities: Payload kg, Volume m³, Origin, Destination)
       │
       ▼
[Opportunity Detection Engine (ODE)] ◄─── Matches Available Capacity with Pending Demand
       ▲
       │
[Pending Customer Orders]      ◄─── (Order items, Quantities, Delivery Deadlines)
       ▲
       │
[Products & SKUs] (from Supply Chain Data + Weights/Volumes)
```

1. **Truck Capacity Schema:** Assigning realistic truck classes (e.g., 16-ton multi-axle, 40 $\text{m}^3$ volume) to `Truck_1` through `Truck_10`.
2. **SKU Physical Dimensions:** Enriching the 100 SKUs with weight (kg/unit) and box volume ($\text{m}^3/\text{unit}$) based on product types (cosmetics, haircare, skincare).
3. **Pending Orders Queue:** Simulating dynamic customer orders originating from the 5 Indian hub cities with specific destination routes and fulfillment deadlines.
4. **Relational Opportunity Ground Truth:** Tracking whether an identified PTL-to-FTL match was dispatched, delayed, or resulted in net freight cost savings.

---

## 8. Summary of Created Artifacts & Next Steps

### Deliverables Created in Phase 2
1. **Cleaned Datasets:**
   - [`data/processed/supply_chain_data_cleaned.csv`](file:///c:/Sameera-sem5/Projects/Cargosense/data/processed/supply_chain_data_cleaned.csv) (100 rows, 25 columns)
   - [`data/processed/smart_logistics_dataset_cleaned.csv`](file:///c:/Sameera-sem5/Projects/Cargosense/data/processed/smart_logistics_dataset_cleaned.csv) (1,000 rows, 21 columns)
2. **Analysis Script:**
   - [`src/phase2_clean_and_eda.py`](file:///c:/Sameera-sem5/Projects/Cargosense/src/phase2_clean_and_eda.py) (Modular, non-destructive cleaning & EDA pipeline)
3. **Visualization Figures (10 High-Resolution Plots):**
   - Located in [`reports/figures/`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/figures)
4. **Documentation:**
   - This comprehensive report: [`reports/phase2_data_cleaning_and_eda.md`](file:///c:/Sameera-sem5/Projects/Cargosense/reports/phase2_data_cleaning_and_eda.md)

---

*Phase 2 is fully complete. In accordance with project instructions, I will wait for your explicit review and approval before proceeding to any subsequent phase.*
