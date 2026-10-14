# CargoSense -- Phase 1: Dataset Inspection & Project Setup Report

**Project Title:** CargoSense: A Data-Driven Opportunity Detection Engine for PTL-to-FTL Conversion  
**Stage:** Phase 1 -- Dataset Inspection & Project Setup  
**Date:** October 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

In Phase 1 of **CargoSense**, the repository was structured for reproducible data science development, raw datasets were safely organized and cataloged, and a comprehensive non-destructive inspection was executed using Python and Pandas.

The primary objective of CargoSense is to identify opportunities to convert partially loaded trucks (**PTLs**) into full truckloads (**FTLs**) by matching available truck capacity with suitable pending orders and inventory. This report details the technical findings from inspecting the two primary datasets, evaluates their suitability for PTL-to-FTL opportunity detection, identifies critical data gaps, and outlines recommendations for Phase 2.

---

## 2. Repository Structure

The project has been organized into a clean, modular structure separating raw data, processed assets, scripts, notebooks, and reports:

```text
Cargosense/
├── data/
│   ├── raw/                           # Immutable raw source datasets
│   │   ├── smart_logistics_dataset.csv
│   │   └── supply_chain_data.csv
│   ├── processed/                     # Cleaned, standardized datasets (Phase 2+)
│   └── synthetic/                     # Documented operational simulation data (Phase 3+)
├── notebooks/                         # Jupyter exploratory notebooks
├── reports/                           # Technical documentation & inspection logs
│   ├── phase1_dataset_inspection.md   # This comprehensive report
│   └── phase1_output.txt              # Complete terminal inspection log (1,500+ lines)
└── src/                               # Modular Python source code
    └── phase1_inspect.py              # Beginner-friendly inspection pipeline
```

### Original CSV Relocation Details
Both original CSV files were moved from the root workspace directory into `data/raw/` without altering their contents, encodings, or row counts:

1. **Supply-Chain Dataset:**  
   - Original location: `supply_chain_data.csv`  
   - Current location: `data/raw/supply_chain_data.csv` (21,053 bytes, 100 rows, 24 columns)
2. **Logistics Dataset:**  
   - Original location: `smart_logistics_dataset.csv`  
   - Current location: `data/raw/smart_logistics_dataset.csv` (106,483 bytes, 1,000 rows, 16 columns)

---

## 3. Dataset Inspection & Technical Findings

### 3.1 Dataset 1: Supply-Chain Dataset (`supply_chain_data.csv`)

#### High-Level Metrics
- **Dimensions:** 100 rows, 24 columns
- **Missing Values:** 0 missing values across all 2,400 cells (100% complete)
- **Duplicate Rows:** 0 exact duplicate rows
- **Granularity:** SKU level (100 unique SKUs: `SKU0` through `SKU99`)

#### Schema & Column Data Types
| Column Name | Pandas Dtype | CargoSense Domain Role |
| :--- | :--- | :--- |
| `Product type` | `object` | Category classification (`skincare`, `haircare`, `cosmetics`) |
| `SKU` | `object` | Unique Stock Keeping Unit identifier (Primary Key) |
| `Price` | `float64` | Unit selling price ($) |
| `Availability` | `int64` | Current stock availability count |
| `Number of products sold` | `int64` | Historical unit sales volume |
| `Revenue generated` | `float64` | Aggregate monetary revenue recorded |
| `Customer demographics` | `object` | Target customer demographic |
| `Stock levels` | `int64` | Current warehouse inventory count |
| `Lead times` | `int64` | Supplier order lead time (days) |
| `Order quantities` | `int64` | Batch order size (units) |
| `Shipping times` | `int64` | Transit time to destination (days) |
| `Shipping carriers` | `object` | Logistics carrier (`Carrier A`, `Carrier B`, `Carrier C`) |
| `Shipping costs` | `float64` | Freight cost per unit/batch ($) |
| `Supplier name` | `object` | Manufacturing supplier entity (`Supplier 1` to `5`) |
| `Location` | `object` | Warehouse/Origin location (5 Indian hubs) |
| `Lead time` | `int64` | Secondary lead time column (days) |
| `Production volumes` | `int64` | Manufactured units per production cycle |
| `Manufacturing lead time`| `int64` | Production turnaround time (days) |
| `Manufacturing costs` | `float64` | Unit manufacturing cost ($) |
| `Inspection results` | `object` | QA check outcome (`Pending`, `Fail`, `Pass`) |
| `Defect rates` | `float64` | QA defect percentage (%) |
| `Transportation modes` | `object` | Freight transit mode (`Road`, `Rail`, `Air`, `Sea`) |
| `Routes` | `object` | Assigned transit corridor (`Route A`, `Route B`, `Route C`) |
| `Costs` | `float64` | Total operational/logistical cost allocation ($) |

#### Numerical Summary Statistics (Selected Metrics)
| Metric | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Price` | 49.46 | 31.17 | 1.70 | 19.60 | 47.60 | 77.20 | 99.17 |
| `Stock levels` | 47.77 | 31.37 | 0.00 | 16.75 | 47.50 | 76.25 | 100.00 |
| `Order quantities` | 49.22 | 26.78 | 1.00 | 26.00 | 52.00 | 70.00 | 96.00 |
| `Production volumes`| 567.84 | 263.05 | 104.00 | 352.00 | 568.50 | 797.00 | 998.00 |
| `Shipping costs` | 5.55 | 2.65 | 1.01 | 3.54 | 5.41 | 7.70 | 9.93 |
| `Costs` | 529.25 | 258.30 | 103.92 | 318.78 | 520.43 | 764.07 | 993.53 |
| `Defect rates` | 2.28 | 1.46 | 0.02 | 1.01 | 2.14 | 3.51 | 4.94 |

#### Categorical Distributions & Special Columns
- **`Product type`:** `skincare` (40%), `haircare` (34%), `cosmetics` (26%)
- **`Transportation modes`:** `Road` (29%), `Rail` (28%), `Air` (26%), `Sea` (17%)
  - *PTL-to-FTL Relevance:* The 29 `Road` transport records represent immediate candidates for road freight consolidation.
- **`Routes`:** `Route A` (43%), `Route B` (37%), `Route C` (20%)
- **`Shipping carriers`:** `Carrier B` (43%), `Carrier C` (29%), `Carrier A` (28%)
- **`Location`:** `Kolkata` (25%), `Mumbai` (22%), `Chennai` (20%), `Bangalore` (18%), `Delhi` (15%)
- **`Inspection results`:** `Pending` (41%), `Fail` (36%), `Pass` (23%)
  - *Quality Note:* Over 36% of SKUs have failed inspection; failed or pending SKUs cannot be safely loaded into trucks until certified.
- **`Asset_ID`, `Shipment_Status`, `Traffic_Status`:** Not present in this dataset.

---

### 3.2 Dataset 2: Smart-Logistics Dataset (`smart_logistics_dataset.csv`)

#### High-Level Metrics
- **Dimensions:** 1,000 rows, 16 columns
- **Missing Values:** 263 missing cells (all located in `Logistics_Delay_Reason`, 26.3% null)
- **Duplicate Rows:** 0 exact duplicate rows
- **Granularity:** Telemetry/event pings across 10 vehicle assets (`Truck_1` through `Truck_10`) over the calendar year 2024.

#### Schema & Column Data Types
| Column Name | Pandas Dtype | CargoSense Domain Role |
| :--- | :--- | :--- |
| `Timestamp` | `object` | Telemetry ping date/time (ISO 8601 string format) |
| `Asset_ID` | `object` | Vehicle fleet identifier (`Truck_1` to `Truck_10`) |
| `Latitude` | `float64` | GPS coordinate (-89.79 to +89.87) |
| `Longitude` | `float64` | GPS coordinate (-179.82 to +179.92) |
| `Inventory_Level` | `int64` | Cargo item/unit count inside truck (100 to 500) |
| `Shipment_Status` | `object` | Operational status (`Delayed`, `Delivered`, `In Transit`) |
| `Temperature` | `float64` | Ambient cargo compartment temperature (°C: 18.0 to 30.0) |
| `Humidity` | `float64` | Ambient relative humidity (%: 50.0 to 80.0) |
| `Traffic_Status` | `object` | Route traffic condition (`Clear`, `Detour`, `Heavy`) |
| `Waiting_Time` | `int64` | Terminal/dock wait time in minutes (10 to 60) |
| `User_Transaction_Amount`| `int64` | Consignment value indicator ($100 to $500) |
| `User_Purchase_Frequency` | `int64` | Order frequency index (1 to 10) |
| `Logistics_Delay_Reason`| `object` | Reported incident root cause (`Weather`, `Traffic`, `Mechanical Failure`, or `NaN`) |
| `Asset_Utilization` | `float64` | Truck capacity utilization percentage (60.0% to 100.0%) |
| `Demand_Forecast` | `int64` | Regional short-term demand forecast units (100 to 300) |
| `Logistics_Delay` | `int64` | Binary delay indicator (`0` = on time, `1` = delayed) |

#### Numerical Summary Statistics (Selected Metrics)
| Metric | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Asset_Utilization` (%) | 79.60 | 11.63 | 60.00 | 69.47 | 79.25 | 89.43 | 100.00 |
| `Inventory_Level` | 297.92 | 113.55 | 100.00 | 201.00 | 299.00 | 399.00 | 500.00 |
| `Demand_Forecast` | 199.28 | 59.92 | 100.00 | 144.00 | 202.00 | 251.25 | 300.00 |
| `Waiting_Time` (min) | 35.06 | 14.48 | 10.00 | 23.00 | 35.00 | 49.00 | 60.00 |
| `Logistics_Delay` | 0.57 | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

#### Categorical Distributions & Special Columns
- **`Asset_ID` (10 trucks):** Evenly distributed across the fleet:
  - `Truck_8`: 109 pings (10.9%), `Truck_4`: 107 (10.7%), `Truck_2`: 105 (10.5%), `Truck_10`: 105 (10.5%), `Truck_6`: 103 (10.3%), `Truck_7`: 102 (10.2%), `Truck_9`: 94 (9.4%), `Truck_3`: 93 (9.3%), `Truck_5`: 93 (9.3%), `Truck_1`: 89 (8.9%).
- **`Shipment_Status`:** `Delayed` (35.0%), `Delivered` (33.8%), `In Transit` (31.2%).
- **`Traffic_Status`:** `Detour` (34.5%), `Clear` (32.8%), `Heavy` (32.7%).
- **`Logistics_Delay`:** `1` (Delayed: 56.6%), `0` (On-time: 43.4%).
- **`Logistics_Delay_Reason`:** `Weather` (267), `Missing/NaN` (263), `Traffic` (236), `Mechanical Failure` (234).
- **`Timestamp` Range:** Earliest: `2024-01-01 11:37:57`, Latest: `2024-12-30 20:21:58` (1,000 distinct timestamps).

---

## 4. In-Depth Data Quality & Ambiguity Analysis

### 4.1 Ambiguous & Duplicate Columns in Supply Chain Data
1. **`Lead times` vs `Lead time` vs `Manufacturing lead time`:**
   - The dataset contains both `Lead times` (plural) and `Lead time` (singular).
   - *Statistical Verification:* The two columns have a correlation of `-0.0028` and identical values in only 1 out of 100 rows.
   - Neither column is documented with clear supply-chain definitions (e.g., supplier fulfillment vs replenishment lead time).
2. **`Costs` vs `Manufacturing costs` vs `Shipping costs`:**
   - In standard accounting, one might expect $\text{Total Costs} = \text{Manufacturing Costs} + \text{Shipping Costs}$.
   - *Statistical Verification:* $\text{Costs} - (\text{Manufacturing costs} + \text{Shipping costs})$ ranges from $+28.32$ to $+949.53$ (mean discrepancy $+476.43$).
   - `Costs` represents an independent overhead or composite cost figure rather than an explicit sum.
3. **`Revenue generated` vs $(\text{Price} \times \text{Number of products sold})$:**
   - For SKU0: $\text{Price} = 69.81$, $\text{Units Sold} = 802$, calculated product $= 55,986.02$, but recorded revenue $= 8,661.99$.
   - For SKU2: $\text{Price} = 11.32$, $\text{Units Sold} = 8$, calculated product $= 90.56$, but recorded revenue $= 9,577.75$.
   - *Finding:* `Revenue generated` cannot be derived from single-period price $\times$ units sold.

### 4.2 Inconsistencies & Synthetic Characteristics in Smart-Logistics Data
1. **`Logistics_Delay_Reason` vs `Logistics_Delay`:**
   - Out of 434 records where `Logistics_Delay == 0` (shipment is NOT delayed), 318 records still report an active delay reason:
     - 116 list `Weather`, 101 list `Traffic`, 101 list `Mechanical Failure`.
   - Conversely, out of 566 records where `Logistics_Delay == 1` (shipment IS delayed), 147 records have `NaN` as the delay reason.
   - *Domain Implication:* The delay reason was generated independently of the delay indicator. In real operational systems, a delay reason is only assigned when a delay flag is raised.
2. **Global Coordinates vs Local Operations:**
   - `Latitude` spans $-89.79^\circ$ to $+89.87^\circ$ and `Longitude` spans $-179.82^\circ$ to $+179.92^\circ$.
   - While mathematically valid coordinates, they represent random points across the entire globe (including Antarctica, mid-Pacific, and Sahara).
   - In contrast, `supply_chain_data.csv` specifies 5 Indian commercial metropolitan hubs (`Mumbai`, `Delhi`, `Kolkata`, `Bangalore`, `Chennai`).
3. **No Direct Relational Keys Between Datasets:**
   - The set intersection between column names of both datasets is **empty** ($\emptyset$).
   - There are no shared foreign keys (`Truck_ID`, `SKU_ID`, `Trip_ID`, or `Warehouse_ID`) connecting the products to the vehicles.

---

## 5. Suitability Assessment for CargoSense (PTL-to-FTL Conversion)

### 5.1 What the Current Data CAN Support
| Analytical Dimension | Usable Columns in Existing Datasets | Operational Capability for CargoSense |
| :--- | :--- | :--- |
| **Product & SKU Analysis** | `SKU`, `Product type`, `Price`, `Stock levels`, `Defect rates`, `Inspection results` | Classifying high-velocity SKUs, filtering out defective stock, ranking by profitability and stock safety buffers. |
| **Inventory & Demand Analysis** | `Stock levels`, `Order quantities`, `Inventory_Level`, `Demand_Forecast` | Assessing inventory replenishment urgency and comparing localized warehouse stock to regional demand forecasts. |
| **Truck Fleet & Logistics Analysis** | `Asset_ID`, `Shipment_Status`, `Traffic_Status`, `Waiting_Time`, `Logistics_Delay` | Tracking truck availability, idle waiting times at hubs, and congestion bottlenecks that threaten delivery schedules. |
| **Truck Utilization Analysis** | `Asset_Utilization` (60.0% to 100.0%) | Categorizing trucks as **PTL candidates** (e.g. utilization $< 80\%$ or $< 85\%$) versus near-FTL trucks ($\ge 85\%$). |
| **Candidate SKU Generation** | `Transportation modes == 'Road'`, `Stock levels > 0`, `Location`, `Routes` | Filtering SKUs eligible for road freight matching on compatible origin-destination routes. |
| **PTL-to-FTL Opportunity Prediction** | `Asset_Utilization`, `Demand_Forecast`, `Waiting_Time`, `Traffic_Status`, `Stock levels` | Building classification or heuristic rules to evaluate whether an underutilized truck has sufficient local inventory and waiting buffer to consolidate cargo into an FTL. |

---

### 5.2 Critical Information Missing from Original Datasets

To construct a realistic, end-to-end PTL-to-FTL Conversion Engine, the following fundamental data points are missing from the raw datasets:

1. **Explicit Truck Weight & Volume Capacities:**
   - The raw data provides only an abstract percentage (`Asset_Utilization`) and a generic unit count (`Inventory_Level`).
   - Missing: Maximum gross vehicle weight (e.g., 16,000 kg payload limit) and volumetric capacity (e.g., 40 $\text{m}^3$ cargo box).
2. **Current Cargo Load Weight & Volume:**
   - PTL-to-FTL consolidation requires checking whether additional cargo will breach either weight limits (weighing out) or volume limits (cubing out).
   - Missing: Item unit weight (kg) and dimensions/volume ($\text{m}^3$) per SKU.
3. **Pending Orders Linked to SKUs & Delivery Deadlines:**
   - The supply chain data has static aggregate `Order quantities`, but lacks a dynamic order backlog.
   - Missing: Order timestamp, destination hub/customer, requested delivery date, and order priority.
4. **SKU-Level Truck Manifests:**
   - Missing: Manifest tables specifying which exact SKUs and quantities are loaded onto which truck during a specific trip.
5. **Timestamped Warehouse Inventory Changes:**
   - `supply_chain_data.csv` is a static snapshot with no time dimension. Real opportunity matching requires tracking stock levels dynamically as orders arrive and trucks depart.
6. **Historical PTL-to-FTL Conversion Outcomes:**
   - Neither dataset records whether consolidation was attempted, whether dispatch was delayed, or what freight cost savings were achieved.

> [!NOTE]
> **Data Science Pedagogy Alignment:**  
> In later phases (specifically Phase 3), we will address these gaps by developing a documented synthetic operational simulation module that bridges the two datasets realistically without overwriting or misrepresenting raw data.

---

## 6. Educational Guide: Beginner-Friendly Pandas Operations

To support learning Data Science (Units I--VI), the inspection pipeline `src/phase1_inspect.py` demonstrates key Pandas fundamentals:

1. **Loading Data (`pd.read_csv`):**
   ```python
   df = pd.read_csv("data/raw/supply_chain_data.csv")
   ```
   *Concept:* Parses raw comma-separated text files into a two-dimensional tabular data structure called a `DataFrame`.

2. **Examining Dimensions and Types (`.shape`, `.dtypes`):**
   ```python
   n_rows, n_cols = df.shape
   print(df.dtypes)
   ```
   *Concept:* `.shape` returns an immutable tuple `(rows, columns)` to verify volume. `.dtypes` identifies whether Pandas inferred columns as integers, 64-bit floats, or generic text (`object`).

3. **Summary Statistics (`.describe()`):**
   ```python
   df.describe()
   ```
   *Concept:* Quickly computes central tendency (mean, median), dispersion (standard deviation, interquartile range), and extrema (min, max) for continuous features.

4. **Missing Value Auditing (`.isnull().sum()`):**
   ```python
   missing_counts = df.isnull().sum()
   ```
   *Concept:* Creates a boolean mask of missing cells (`True`/`False`) and sums them along axis 0 to pinpoint data collection gaps.

5. **Frequency Distributions (`.value_counts()`):**
   ```python
   df['Transportation modes'].value_counts()
   ```
   *Concept:* Tallies occurrences of discrete categories, critical for identifying class imbalances or filtering valid subsets.

6. **Outlier Auditing with Z-Score Boundaries:**
   ```python
   mean, std = df[col].mean(), df[col].std()
   outliers = df[(df[col] < mean - 5*std) | (df[col] > mean + 5*std)]
   ```
   *Concept:* Applying empirical rule principles to detect data anomalies located more than 5 standard deviations away from the mean.

---

## 7. Instructions: How to Run the Inspection

The inspection script is located in `src/phase1_inspect.py` and can be executed at any time from the terminal in the project root:

```bash
# Navigate to the project root directory
cd c:\Sameera-sem5\Projects\Cargosense

# Execute the Phase 1 inspection pipeline
python src/phase1_inspect.py
```

To regenerate or update the full text log file:
```powershell
python src/phase1_inspect.py | Out-File -FilePath reports/phase1_output.txt -Encoding utf8
```

The script runs completely non-destructively: it reads the raw files, runs all statistical and quality audits, and prints the findings without modifying or overwriting any CSV file.

---

## 8. Recommended Next Steps for Phase 2

With Phase 1 complete, here are the proposed priorities for **Phase 2 (Data Cleaning & Feature Assessment)**:

1. **Handling Supply Chain Inconsistencies:**
   - Standardize column names into snake_case (e.g., `product_type`, `stock_levels`).
   - Resolve the ambiguity between `lead_times` and `lead_time` by designating a primary operational lead time metric.
   - Filter or segment SKUs by `transportation_modes == 'Road'` and `inspection_results == 'Pass'` to establish a clean pool of consolidation-eligible candidate goods.
2. **Handling Logistics Dataset Inconsistencies:**
   - Clean `Logistics_Delay_Reason` by setting reasons to `'None'` where `Logistics_Delay == 0`.
   - Parse `Timestamp` into proper Pandas `datetime64` objects and extract temporal features (hour of day, day of week, month).
   - Define a formal operational PTL indicator: e.g., $\text{PTL Flag} = 1$ if $\text{Asset\_Utilization} < 80.0\%$.
3. **Data Schema Bridging Design (Preparation for Phase 3):**
   - Formulate a clean relational contract (linking Indian logistics hubs, routes, truck payloads, and SKU dimensions) so that synthetic operational matching records can be generated rigorously and reproducibly.
