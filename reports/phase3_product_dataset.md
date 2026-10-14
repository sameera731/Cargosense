# CargoSense -- Phase 3: Product Dataset Documentation Report

**Project Title:** CargoSense: A Data-Driven Opportunity Detection Engine for PTL-to-FTL Conversion  
**Stage:** Phase 3 -- Product Dataset Preparation & Review  
**Date:** October 2026  
**Status:** Reviewed, Validated & Approved for Review  

---

## 1. Executive Summary & Purpose

The objective of **Phase 3** is to prepare a finalized, domain-ready **master product catalog** ([`data/processed/products.csv`](file:///c:/Sameera-sem5/Projects/Cargosense/data/processed/products.csv)) to serve as the core product entity for the CargoSense Opportunity Detection Engine (ODE).

In freight consolidation, converting Partial Truckloads (**PTL**) into Full Truckloads (**FTL**) requires matching available vehicle capacity with candidate cargo. However, freight vehicles are physically constrained by two distinct thresholds:
1. **Weight Capacity (Payload in $kg$):** Axle and gross vehicle weight limits (**"weighing out"**).
2. **Volumetric Capacity (Cube in $m^3$):** Total usable interior space of the cargo box (**"cubing out"**).

The raw supply chain dataset (`supply_chain_data.csv`) contained commercial transactions and inventory counts, but lacked unit weights, packaging volumes, and warehouse facility identifiers. In Phase 3, we:
- Preserved all 100 original SKUs and observed commercial attributes.
- Established documented, realistic packaging assumptions for consumer packaged goods (CPG).
- Mapped origin cities deterministically to regional Indian distribution hubs.
- Separated transport eligibility, quality inspection, and consolidation candidacy into distinct logical rules.
- Validated the catalog against a comprehensive 10-point domain test suite.

---

## 2. Product Catalog Schema & Attribute Provenance

The finalized product catalog resides at [`data/processed/products.csv`](file:///c:/Sameera-sem5/Projects/Cargosense/data/processed/products.csv). It contains exactly **100 rows** (one per SKU) and **22 columns**.

| Column Name | Data Type | Provenance | Description & Operational Function |
| :--- | :---: | :---: | :--- |
| `sku` | `string` | **Observed** | Unique Stock Keeping Unit identifier (Primary Key: `SKU0`–`SKU99`). |
| `product_type` | `string` | **Observed** | Product category (`skincare`, `haircare`, `cosmetics`). |
| `unit_price` | `float64` | **Observed** | Unit commercial selling price ($). |
| `stock_level` | `int64` | **Observed** | Current on-hand inventory count at the warehouse. |
| `order_quantity_baseline`| `int64` | **Observed** | Historical customer order batch size. |
| `lead_time_days` | `int64` | **Observed** | Supplier replenishment lead time (days). |
| `manufacturing_cost` | `float64` | **Observed** | Unit factory manufacturing cost ($). |
| `shipping_cost_baseline`| `float64` | **Observed** | Base single-unit freight shipping cost ($). |
| `supplier_name` | `string` | **Observed** | Manufacturing vendor (`Supplier 1` to `Supplier 5`). |
| `origin_location` | `string` | **Observed** | Sourcing metropolitan hub in India (`Mumbai`, `Delhi`, `Kolkata`, `Bangalore`, `Chennai`). |
| `warehouse_id` | `string` | **Derived** | Deterministic regional fulfillment facility code (`WH-MUM-01`, etc.). |
| `warehouse_facility` | `string` | **Derived** | Realistic facility name (e.g., `Bhiwandi Western Hub`). |
| `warehouse_region` | `string` | **Derived** | Geographic zone (`West`, `North`, `East`, `South-Central`, `South-Coast`). |
| `transportation_mode` | `string` | **Observed** | Default transit modality (`Road`, `Rail`, `Air`, `Sea`). |
| `assigned_route` | `string` | **Observed** | Designated transit corridor (`Route A`, `Route B`, `Route C`). |
| `inspection_status` | `string` | **Observed** | QA status (`Pass`, `Pending`, `Fail`). Preserved without modification. |
| `defect_rate_pct` | `float64` | **Observed** | Quality assurance defect percentage (%). |
| `unit_weight_kg` | `float64` | **Assumed** | Packaging gross weight per unit ($kg$). |
| `unit_volume_m3` | `float64` | **Assumed** | External carton packaging volume per unit ($m^3$). |
| `density_kg_m3` | `float64` | **Derived** | Effective packaging density ($\text{unit\_weight\_kg} / \text{unit\_volume\_m3}$). |
| `is_road_eligible` | `boolean` | **Derived** | `True` strictly if `transportation_mode == 'Road'`. |
| `is_consolidation_candidate`| `boolean` | **Derived** | `True` strictly if `is_road_eligible` AND `inspection_status == 'Pass'` AND `stock_level > 0`. |

---

## 3. Physical Attributes & Modeling Assumptions

### 3.1 Rationale for Category-Based Packaging Models
In consumer packaged goods (CPG) distribution, products are shipped in protective corrugated secondary cartons. Because physical measurements were absent from the source data, we established **documented, plausible assumptions** based on typical commercial packaging standards for personal care items:

1. **Haircare (34 SKUs):**
   - *Form Factor:* 250ml to 500ml liquid bottles (shampoos, conditioners, hair oils) packaged in corrugated shipper cartons.
   - *Assumed Weight Model:* Gaussian distribution $\mathcal{N}(\mu=0.50, \sigma=0.06)$ clipped to $[0.35, 0.70]$ kg.
   - *Assumed Packaging Volume:* Gaussian distribution $\mathcal{N}(\mu=0.0010, \sigma=0.00012)$ clipped to $[0.0007, 0.0014] \text{ m}^3$ (approx. 1.0 liter box, e.g. $7 \times 7 \times 20$ cm).
   - *Resulting Empirical Density:* Mean **$510.1 \text{ kg/m}^3$** (Min: 386.2, Max: 701.9).
2. **Skincare (40 SKUs):**
   - *Form Factor:* 50ml to 200ml tubes, pump bottles, and jars (creams, lotions, cleansers) with internal air cushions and protective secondary cartons.
   - *Assumed Weight Model:* Gaussian distribution $\mathcal{N}(\mu=0.25, \sigma=0.04)$ clipped to $[0.15, 0.38]$ kg.
   - *Assumed Packaging Volume:* Gaussian distribution $\mathcal{N}(\mu=0.0005, \sigma=0.00008)$ clipped to $[0.0003, 0.0008] \text{ m}^3$ (approx. 0.5 liter box, e.g. $6 \times 6 \times 14$ cm).
   - *Resulting Empirical Density:* Mean **$510.6 \text{ kg/m}^3$** (Min: 265.0, Max: 749.3).
3. **Cosmetics (26 SKUs):**
   - *Form Factor:* Compact lightweight items (lipsticks, compact powders, eye shadow palettes, mascaras) with protective cardstock boxes.
   - *Assumed Weight Model:* Gaussian distribution $\mathcal{N}(\mu=0.12, \sigma=0.025)$ clipped to $[0.05, 0.20]$ kg.
   - *Assumed Packaging Volume:* Gaussian distribution $\mathcal{N}(\mu=0.00025, \sigma=0.00005)$ clipped to $[0.00012, 0.00040] \text{ m}^3$ (approx. 0.25 liter box, e.g. $5 \times 5 \times 10$ cm).
   - *Resulting Empirical Density:* Mean **$440.0 \text{ kg/m}^3$** (Min: 292.7, Max: 746.0).

### 3.2 Consistency and Reproducibility
- **Deterministic Random Seed:** Generation is fixed using `np.random.RandomState(42)`. Rerunning the code produces bit-for-bit identical results.
- **Density Formulation:** Density is calculated strictly as:
  $$\text{density\_kg\_m3} = \frac{\text{unit\_weight\_kg}}{\text{unit\_volume\_m3}}$$
- **Clean Representation:** The values in `products.csv` are kept clean and realistic (e.g., $0.53$ kg, $0.000983 \text{ m}^3$), without cluttering the CSV rows with repetitive synthetic labels.

---

## 4. Warehouse Facility Assignment Rule

The raw supply chain dataset specifies an origin `location` across 5 Indian commercial metropolitan centers. Rather than inventing disconnected random locations, we mapped each observed city deterministically (1-to-1) to an established regional logistics hub:

| Observed Origin City | Assigned Warehouse ID | Facility Name | Geographic Region | SKU Count |
| :--- | :--- | :--- | :--- | :---: |
| **Mumbai** | `WH-MUM-01` | Bhiwandi Western Hub | West | 22 |
| **Delhi** | `WH-DEL-01` | Kundli Northern Hub | North | 15 |
| **Kolkata** | `WH-KOL-01` | Dankuni Eastern Hub | East | 25 |
| **Bangalore** | `WH-BLR-01` | Nelamangala Central South Hub | South-Central | 18 |
| **Chennai** | `WH-MAA-01` | Sriperumbudur Coastal South Hub | South-Coast | 20 |

### Limitations of the Mapping Rule
1. **Model Assumption, Not Observed Truth:** The source dataset does not specify actual warehouse addresses or private logistics facilities. This mapping is an operational modeling construct designed to ground freight flows in realistic Indian trucking corridors.
2. **Single-Facility Assignment:** Each SKU is currently stored exclusively at its origin city's regional hub. The current model does not simulate multi-node inventory replication, cross-docking, or hub-to-hub transfers.

---

## 5. Transport Modality, Quality & Consolidation Eligibility

A critical review requirement in Phase 3 is ensuring that **transport modality**, **inspection quality**, and **consolidation candidacy** remain separate, explicit concepts.

### 5.1 Distinct Eligibility Rules
1. **Road Transport Eligibility (`is_road_eligible`):**
   - Rule: $\text{is\_road\_eligible} = (\text{transportation\_mode} == \text{'Road'})$
   - Distribution: **29 SKUs (29%)** are eligible for road freight; 71 SKUs use Rail, Air, or Sea.
2. **Inspection Quality Gate (`inspection_status`):**
   - Preserves raw inspection categories: `Pass` (23 SKUs), `Pending` (41 SKUs), `Fail` (36 SKUs).
   - *Policy:* No pending or failed SKU is altered or assumed to pass. Failed or uncertified goods cannot be legally loaded onto commercial trucks.
3. **Consolidation Candidacy (`is_consolidation_candidate`):**
   - Rule: An SKU is an active operational candidate for truckload consolidation **if and only if** it satisfies all three conditions:
     $$\text{is\_consolidation\_candidate} = \text{is\_road\_eligible} \land (\text{inspection\_status} == \text{'Pass'}) \land (\text{stock\_level} > 0)$$
   - Distribution: Exactly **7 SKUs** meet all three criteria in the current catalog.

### 5.2 Contingency Breakdown
| Transport Mode | Passed Inspection | Pending Inspection | Failed Inspection | Total SKUs |
| :--- | :---: | :---: | :---: | :---: |
| **Road** | **7** *(Candidates)* | 12 | 10 | **29** |
| **Air** | 7 | 10 | 9 | **26** |
| **Rail** | 6 | 11 | 11 | **28** |
| **Sea** | 3 | 8 | 6 | **17** |
| **Total** | **23** | **41** | **36** | **100** |

---

## 6. Comprehensive 10-Point Validation Results

The product catalog was evaluated against 10 automated assertion tests. All 10 checks passed:

| # | Validation Check | Target Specification | Observed Result | Status |
| :-: | :--- | :--- | :--- | :-: |
| 1 | **Exact 100 Rows** | Length of dataset equals 100 | Exactly 100 rows | **PASS** |
| 2 | **100 Unique SKUs** | Unique count of `sku` equals 100 | Exactly 100 unique values | **PASS** |
| 3 | **All SKU IDs Preserved** | Set equals `SKU0` through `SKU99` | 100% match, zero dropped/altered | **PASS** |
| 4 | **No Missing Values** | Null count across all 22 columns is 0 | 0 null cells | **PASS** |
| 5 | **Positive Weight & Volume** | `unit_weight_kg > 0` & `unit_volume_m3 > 0` | Min weight: 0.077 kg, Min vol: 0.00012 m³ | **PASS** |
| 6 | **Density Calculated Correctly** | `density_kg_m3 == round(weight / volume, 1)` | 100% exact match across all rows | **PASS** |
| 7 | **Non-Negative Stock** | `stock_level >= 0` for all SKUs | Min stock: 0, Max stock: 100 | **PASS** |
| 8 | **Valid Warehouse Mapping** | IDs start with `WH-` and cities in mapping | 100% valid mapped facilities | **PASS** |
| 9 | **Valid Category Ranges** | Modality $\in$ {Road, Rail, Air, Sea}, Status $\in$ {Pass, Pending, Fail} | 100% valid domain categories | **PASS** |
| 10 | **Eligibility Consistency** | Flags strictly match documented boolean rules | 100% consistent with logic | **PASS** |

---

## 7. Downstream Relational Integration (Roadmap for Phase 4+)

The prepared `products.csv` table does not exist in isolation. It forms the master product dimension table that will connect to future operational datasets in subsequent phases:

```text
       ┌────────────────────────┐
       │      products.csv      │
       │────────────────────────│
       │ * sku (PK)             │
       │   product_type         │
       │   unit_weight_kg       │
       │   unit_volume_m3       │
       │   warehouse_id (FK)    │
       │   stock_level          │
       └───────────┬────────────┘
                   │ 1
                   │
         ┌─────────┴─────────┐
         │ N                 │ N
         ▼                   ▼
┌──────────────────┐  ┌───────────────────────┐
│    orders.csv    │  │    truck_loads.csv    │
│──────────────────│  │───────────────────────│
│ * order_id       │  │ * trip_id             │
│   sku (FK)       │  │   truck_id            │
│   order_quantity │  │   sku (FK)            │
│   total_weight   │  │   loaded_units        │
│   total_volume   │  │   loaded_weight_kg    │
│   destination    │  │   loaded_volume_m3    │
└────────┬─────────┘  └───────────┬───────────┘
         │                        │
         └───────────┬────────────┘
                     ▼
       ┌────────────────────────────┐
       │   ode_opportunities.csv    │
       │────────────────────────────│
       │ * opportunity_id           │
       │   trip_id (FK)             │
       │   candidate_order_id (FK)  │
       │   remaining_payload_kg     │
       │   remaining_volume_m3      │
       │   weight_feasible (bool)   │
       │   volume_feasible (bool)   │
       │   conversion_status        │
       └────────────────────────────┘
```

1. **Connection to Customer Orders (`orders.csv`):**  
   Order line items will reference `sku`. Total order demand weight $= \text{order\_quantity} \times \text{unit\_weight\_kg}$, and order volume $= \text{order\_quantity} \times \text{unit\_volume\_m3}$.
2. **Connection to Truck Loads (`truck_loads.csv`):**  
   Current freight loaded on a vehicle trip will link to `products.csv` to calculate utilized payload and cubic volume.
3. **Connection to the Opportunity Detection Engine (`ode_opportunities.csv`):**  
   When a truck is underutilized ($< 80\%$), the ODE matches remaining capacity against candidate pending orders originating from the vehicle's current warehouse headed toward the same route corridor.

---

## 8. Summary of Limitations

1. **Not Real Industry Measurements:** Unit weights and packaging volumes are simulated approximations based on standard CPG categories. They are designed to teach freight capacity mechanics, not to reflect proprietary manufacturer blueprints.
2. **No Real Truck or Order Records Yet:** `products.csv` defines the static product catalog. It does not establish real truck assignments, active order backlogs, or observed PTL-to-FTL conversion outcomes. Those operational dynamics will be developed in subsequent approved phases.
