# CargoSense — Phase 5: Operational Data Audit Report

**Project Title:** CargoSense: A Data-Driven Opportunity Detection Engine for PTL-to-FTL Conversion  
**Stage:** Phase 5 — Operational Data Audit & ODE Readiness Assessment  
**Date:** October 2026  
**Status:** Completed

---

## 1. Executive Summary

This report documents a rigorous, independent audit of the Phase 4 synthetic operational dataset comprising 14 relational CSV tables in `data/synthetic/`. The audit was conducted against three objectives:

1. **Relational Integrity:** Verify primary keys, foreign keys, business invariants, and temporal consistency.
2. **Operational Feasibility:** Determine whether the data supports identifying, evaluating, and predicting PTL-to-FTL conversion opportunities.
3. **Target Derivation:** Assess whether a defensible `Became_FTL` prediction target can be constructed from operational evidence.

### Key Findings

| Area | Verdict |
|:-----|:--------|
| Primary-key uniqueness | ✅ All 14 tables pass |
| Foreign-key consistency | ✅ Zero orphan references across 25 FK checks |
| Order conservation invariant | ✅ Zero violations across 2,477 order lines |
| Truck capacity compliance | ✅ Zero over-weight or over-volume shipments |
| Route permission compliance | ✅ Zero violations across 737 shipments |
| Timestamp ordering | ✅ Zero temporal ordering violations |
| Inventory accounting | ✅ Zero mismatches, zero negative balances |
| Dock scan balance | ✅ Zero scan balance failures across 2,612 lines |
| **Truck utilization** | ⚠️ **Critical: ALL 691 dispatched shipments < 12% utilization** |
| **FTL outcomes** | ⚠️ **Zero shipments reach any FTL threshold (85%, 80%, 70%, or even 50%)** |
| **Post-planning consolidation** | ⚠️ **Not modeled: each shipment carries exactly the planned allocation** |
| **`Became_FTL` target** | ⚠️ **Cannot be derived from subsequent loading events** |

> [!IMPORTANT]
> The relational architecture and business invariants are excellent and fully validated. However, the simulation produces only extremely underutilized shipments (mean weight utilization 1.43%), never models post-planning cargo consolidation, and generates zero FTL outcomes. The `Became_FTL` target as originally conceived cannot be derived from this dataset. **Phase 6 can proceed, but requires an adapted target definition.**

---

## 2. Tables Inspected & Actual Dimensions

All 14 tables were loaded and inspected from [`data/synthetic/`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic):

| # | Table | Primary Key | Rows | Cols | Nulls (non-trivial) |
|:-:|:------|:------------|-----:|-----:|:--------------------|
| 1 | [`products.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/products.csv) | `sku_id` | 100 | 10 | None |
| 2 | [`trucks.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/trucks.csv) | `truck_id` | 30 | 7 | None |
| 3 | [`routes.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/routes.csv) | `route_id` | 10 | 6 | None |
| 4 | [`truck_route_permissions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/truck_route_permissions.csv) | `permission_id` | 300 | 5 | None |
| 5 | [`orders.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/orders.csv) | `order_id` | 1,454 | 8 | None |
| 6 | [`order_lines.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/order_lines.csv) | `order_line_id` | 2,477 | 8 | None |
| 7 | [`planning_snapshots.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/planning_snapshots.csv) | `planning_run_id` | 90 | 7 | None |
| 8 | [`planned_truck_allocations.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/planned_truck_allocations.csv) | `allocation_line_id` | 2,612 | 11 | None |
| 9 | [`shipments.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/shipments.csv) | `shipment_id` | 737 | 17 | `loading_start_time` (10), `actual_dispatch_time` (46), `actual_delivery_time` (47) |
| 10 | [`shipment_lines.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/shipment_lines.csv) | `shipment_line_id` | 2,612 | 10 | None |
| 11 | [`truck_load_events.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/truck_load_events.csv) | `event_id` | 2,720 | 10 | None |
| 12 | [`inventory_transactions.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/inventory_transactions.csv) | `transaction_id` | 5,257 | 10 | `order_line_id` (330 — supplier receipts), `planning_run_id` (2,807 — non-planning txns) |
| 13 | [`inventory_snapshots.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/inventory_snapshots.csv) | `snapshot_id` | 9,000 | 7 | None |
| 14 | [`demand_history.csv`](file:///c:/Users/sanch/OneDrive/Desktop/TY/DS/DSCP/Cargosense/data/synthetic/demand_history.csv) | `record_id` | 90,000 | 10 | None |

All null values are structurally expected (e.g., `PLANNED` shipments have no dispatch/delivery timestamps; `SUPPLIER_RECEIPT` transactions have no `order_line_id`).

---

## 3. Confirmed Integrity Findings

### 3.1 Primary-Key Uniqueness ✅

Every table's designated primary key is unique and non-null. Zero duplicates or missing identifiers in any of the 14 tables.

### 3.2 Foreign-Key Consistency ✅

**25 foreign-key relationships** were tested. All produce **zero orphan references**:

| Child Table | FK Column | Parent Table | Parent Column | Orphans |
|:------------|:----------|:-------------|:--------------|:-------:|
| `truck_route_permissions` | `truck_id` | `trucks` | `truck_id` | 0 |
| `truck_route_permissions` | `route_id` | `routes` | `route_id` | 0 |
| `orders` | `route_id` | `routes` | `route_id` | 0 |
| `order_lines` | `order_id` | `orders` | `order_id` | 0 |
| `order_lines` | `sku_id` | `products` | `sku_id` | 0 |
| `planned_truck_allocations` | `planning_run_id` | `planning_snapshots` | `planning_run_id` | 0 |
| `planned_truck_allocations` | `truck_id` | `trucks` | `truck_id` | 0 |
| `planned_truck_allocations` | `order_line_id` | `order_lines` | `order_line_id` | 0 |
| `planned_truck_allocations` | `sku_id` | `products` | `sku_id` | 0 |
| `planned_truck_allocations` | `route_id` | `routes` | `route_id` | 0 |
| `shipments` | `truck_id` | `trucks` | `truck_id` | 0 |
| `shipments` | `route_id` | `routes` | `route_id` | 0 |
| `shipments` | `planning_run_id` | `planning_snapshots` | `planning_run_id` | 0 |
| `shipment_lines` | `shipment_id` | `shipments` | `shipment_id` | 0 |
| `shipment_lines` | `order_line_id` | `order_lines` | `order_line_id` | 0 |
| `shipment_lines` | `sku_id` | `products` | `sku_id` | 0 |
| `truck_load_events` | `shipment_id` | `shipments` | `shipment_id` | 0 |
| `truck_load_events` | `order_line_id` | `order_lines` | `order_line_id` | 0 |
| `truck_load_events` | `sku_id` | `products` | `sku_id` | 0 |
| `truck_load_events` | `shipment_line_id` | `shipment_lines` | `shipment_line_id` | 0 |
| `inventory_transactions` | `sku_id` | `products` | `sku_id` | 0 |
| `inventory_transactions` *(non-null)* | `order_line_id` | `order_lines` | `order_line_id` | 0 |
| `inventory_snapshots` | `sku_id` | `products` | `sku_id` | 0 |
| `demand_history` | `sku_id` | `products` | `sku_id` | 0 |
| `demand_history` | `route_id` | `routes` | `route_id` | 0 |

### 3.3 Cancellation-Free Order Conservation ✅

Verified across all 2,477 order lines:
- `ordered_quantity = shipped_quantity + open_quantity`: **0 violations**
- Over-shipments (`shipped > ordered`): **0**
- Negative open quantity: **0**
- Global accounting: 136,020 = 134,640 + 1,380 ✅

### 3.4 Truck Capacity Compliance ✅

- Over-weight shipments: **0 / 737**
- Over-volume shipments: **0 / 737**

### 3.5 Route Permission Compliance ✅

- Shipments violating truck-route authorization: **0 / 737**
- Shipment lines with order-route mismatch: **0**

### 3.6 Dock Scan Balance & All-or-Nothing Dispatch ✅

- Net scan balance failures: **0 / 2,612** shipment lines
- Line-header status mismatches: **0**
- Held/planned lines with `shipped_quantity > 0`: **0**

### 3.7 Inventory Accounting ✅

- `available != on_hand - reserved`: **0** mismatches across 9,000 snapshots
- Negative `available_quantity`: **0**
- Negative `on_hand_quantity`: **0**
- Negative `reserved_quantity`: **0**
- Day 1 snapshot reconciliation: **0** mismatches
- Cross-day snapshot continuity (days 1-5 verified): **0** mismatches
- Transactions at/after March 30 23:00:00: **0**

### 3.8 Temporal Ordering ✅

- Loading start → end → dispatch ordering violations: **0 / 691** dispatched shipments
- Sequential truck trip overlaps: **0**

### 3.9 Demand History ✅

- Row count matches Cartesian grid (100 × 10 × 90 = 90,000): ✅
- `gross_units_demanded = net_units_demanded` for all rows: ✅
- Total demand units (136,020) matches total ordered units (136,020): ✅
- Zero-demand cells: 87,553 (97.3%) — expected for sparse SKU×city×day combinations

### 3.10 Duplicate & Reference ID Checks ✅

- Duplicate transaction `reference_id`: **0**
- Exact duplicate load event rows: **0**
- Duplicate `(order_line_id, planning_run_id)` in allocations: **0**
- Shipment line SKU vs order line SKU mismatches: **0**

---

## 4. Potential Risks & Observations

### 4.1 Missing Transaction Types (Low Risk)

The Phase 4 report documents 6 inventory transaction types, but only 3 appear in the actual data:

| Transaction Type | Count | Status |
|:-----------------|------:|:-------|
| `ORDER_RESERVATION` | 2,477 | ✅ Present |
| `DISPATCH_SHIPPED` | 2,450 | ✅ Present |
| `SUPPLIER_RECEIPT` | 330 | ✅ Present |
| `REPLAN_RELEASE` | 0 | ⚠️ Documented but absent |
| `REPLAN_RESERVATION` | 0 | ⚠️ Documented but absent |
| `CYCLE_COUNT_ADJUST` | 0 | ⚠️ Documented but absent |

**Assessment:** The simulation loop does not implement batch replanning or cycle count adjustments. Every reservation made at order placement persists until dispatch. This is internally consistent — the generator never needs to release and re-reserve stock because truck assignments are deterministic within each nightly run. However, the documentation implies a more complex replanning process than what was implemented.

### 4.2 Shipment Null Timestamps (Expected, No Risk)

- 10 `PLANNED` shipments have no loading/dispatch timestamps — expected for post-horizon trips
- 36 `HELD` shipments have no `actual_dispatch_time` — expected for dock-held trips
- 1 `IN_TRANSIT` shipment has no `actual_delivery_time` — expected for late-horizon dispatch

### 4.3 Validation Test Coverage Gaps (Medium Risk)

The existing 15-point suite does **not** test:

| Gap | Risk Level | Verified in This Audit |
|:----|:-----------|:----------------------|
| Route permission compliance for shipments | Medium | ✅ Verified — 0 violations |
| Shipment line SKU consistency with order line | Medium | ✅ Verified — 0 mismatches |
| Order line weight/volume recalculation accuracy | Low | ✅ Verified — 0.000 max error |
| Cross-day snapshot continuity (day N = day N-1 + deltas) | Medium | ✅ Verified days 1-5 |
| `REPLAN_*` transaction existence | Low | ⚠️ Confirmed absent |

---

## 5. Timestamp & Temporal-Leakage Assessment

### 5.1 Temporal Architecture

The simulation follows a strict daily cycle:

```
Day d:
  08:00-20:00  Customer orders placed, ORDER_RESERVATION transactions
  23:00        Nightly inventory snapshot captured
  23:00        Planning run cutoff (RUN-YYYYMMDD)
  
Day d+1:
  04:00        Planning run completed (5-hour batch gap)
  04:15        Loading starts
  05:45        Loading ends
  06:00+       Actual dispatch (+ 5-25 min jitter)
```

### 5.2 Leakage Risks for ODE Construction

| Risk | Description | Severity |
|:-----|:------------|:---------|
| **Post-dispatch features** | `actual_dispatch_time`, `actual_delivery_time`, `shipment_status` are outcomes, not predictors | **High** — must be excluded from candidate features |
| **Same-day shipment outcome** | If evaluation timestamp is the planning cutoff (23:00), and the outcome is the next-day dispatch, both are available in the same dataset | **Medium** — requires strict temporal partitioning |
| **Shipped quantity in order_lines** | `shipped_quantity` and `open_quantity` reflect final cumulative state, not the state at any evaluation point | **High** — must use inventory snapshots and allocation records at the evaluation timestamp |
| **Utilization percentages** | `weight_utilization_pct` and `volume_utilization_pct` in shipments are post-hoc calculations, not predictive features | **Medium** — must reconstruct from planned allocations |

### 5.3 Temporal Boundary Recommendation

The only safe evaluation timestamp for ODE candidates is the **planning run cutoff** (23:00 on day d). At this moment:
- The nightly inventory snapshot is available
- All day d orders are known
- Planned allocations have not yet been finalized (the solver runs 23:00-04:00)
- Dispatch has not occurred

Features available at 23:00 day d:
- Inventory snapshot for day d (all SKUs)
- Orders placed on day d and earlier
- Truck availability from prior trips
- Demand history through day d
- Planned allocations from prior runs (day d-1 and earlier)

Features **NOT** available:
- Day d planning run outputs (completed at 04:00 day d+1)
- Dispatch outcomes from day d+1

---

## 6. Assessment of `Became_FTL` Target Derivation

### 6.1 The Original Target Concept

The development plan describes `Became_FTL` as a label derived from "subsequent operational events" — whether a PTL truck successfully reached FTL capacity through cargo consolidation.

### 6.2 Why the Current Data Cannot Support This Target

> [!CAUTION]
> **Three structural barriers prevent deriving `Became_FTL` from subsequent loading events:**

**Barrier 1: Extreme Underutilization (No FTL Outcomes Exist)**

All 691 dispatched shipments have:
- Weight utilization: mean **1.43%**, max **11.30%**, min **0.02%**
- Volume utilization: mean **0.45%**, max **3.42%**, min **0.01%**

**Zero shipments** reach any FTL threshold — not 85%, not 70%, not even 50%. The positive class for `Became_FTL` is empty.

**Root Cause:** The generator allocates each shipment with only the order lines available for one route on one day. With ~17 orders/day spread across 10 routes and 30 trucks, each truck carries only 2-5 order lines per trip. Individual order lines contain 20-100 units of lightweight CPG products (0.05-0.95 kg per unit), producing total shipment weights of 5-40 kg on trucks with capacities of 3,500-16,000 kg.

**Barrier 2: No Post-Planning Consolidation**

The simulation does not model the business process that creates FTL outcomes. Each shipment carries **exactly** the planned allocation with zero weight discrepancy between planned and actual loads. There are no events where:
- Additional cargo is added to a truck between planning and dispatch
- Multiple planning runs contribute different cargo to the same trip
- A driver waits for additional orders before departing

Every trip is allocated once and departs with exactly that cargo.

**Barrier 3: No Counterfactual Comparison**

Even if some trucks were well-utilized, the data cannot distinguish:
- A truck that was **converted** from PTL to FTL (successful consolidation)
- A truck that was **always** planned as FTL (no consolidation needed)
- A truck that remained PTL despite a consolidation attempt

### 6.3 Feasible Alternative Target: `could_reach_FTL`

Despite the barriers above, the data **can** support a structurally sound prediction target with a different question:

> **Given a PTL shipment and the available inventory, pending orders, and route-compatible cargo at the planning cutoff, _could_ the truck's remaining capacity be filled to the FTL threshold?**

This target is derivable because:
1. Each shipment's planned load and remaining capacity are known from `planned_truck_allocations` and `trucks`
2. Available inventory at the cutoff is known from `inventory_snapshots`
3. Pending unfulfilled orders for the same route are known from `order_lines` and `orders`
4. Product weights and volumes are known from `products`

The label would be:

$$\text{could\_reach\_FTL} = \begin{cases} 1 & \text{if feasible subset of available same-route cargo fills remaining capacity to} \geq T\% \\ 0 & \text{otherwise} \end{cases}$$

Where T is the FTL threshold (e.g., 85%).

**This is not a prediction of what _did_ happen, but a feasibility assessment of what _could_ have happened.** It is appropriate for an educational project and produces a non-trivial binary classification task.

---

## 7. Proposed ODE Row Definition & Schema

### 7.1 What One Row Represents

One row in the ODE dataset represents a **candidate consolidation opportunity**: a specific PTL shipment evaluated at the planning cutoff timestamp, asking whether sufficient compatible cargo exists to fill it to FTL capacity.

### 7.2 Unique Candidate Identifier

```
opportunity_id = f"OPP-{shipment_id}-{planning_run_id}"
```

Since each shipment is associated with exactly one planning run, this is unique.

### 7.3 Proposed Schema

| Column | Source | Description | Notes |
|:-------|:-------|:------------|:------|
| `opportunity_id` | Derived | Unique candidate identifier | PK |
| `evaluation_timestamp` | `planning_snapshots.cutoff_timestamp` | Planning cutoff (23:00) | Temporal anchor |
| `planning_run_id` | `shipments.planning_run_id` | Planning run reference | FK |
| `shipment_id` | `shipments.shipment_id` | Candidate shipment | FK |
| `truck_id` | `shipments.truck_id` | Truck assigned | FK |
| `route_id` | `shipments.route_id` | Route corridor | FK |
| `truck_type` | `trucks.truck_type` | Vehicle class | Feature |
| `max_payload_kg` | `trucks.max_payload_kg` | Truck weight capacity | Feature |
| `max_volume_m3` | `trucks.max_volume_m3` | Truck volume capacity | Feature |
| `planned_weight_kg` | `shipments.total_weight_kg` | Planned cargo weight | Feature |
| `planned_volume_m3` | `shipments.total_volume_m3` | Planned cargo volume | Feature |
| `remaining_capacity_kg` | Derived | `max_payload_kg - planned_weight_kg` | Feature |
| `remaining_capacity_m3` | Derived | `max_volume_m3 - planned_volume_m3` | Feature |
| `weight_utilization_pct` | `shipments.weight_utilization_pct` | Current weight utilization | Feature |
| `volume_utilization_pct` | `shipments.volume_utilization_pct` | Current volume utilization | Feature |
| `n_order_lines_on_trip` | Derived | Count of allocated order lines | Feature |
| `n_distinct_skus_on_trip` | Derived | Count of distinct SKUs loaded | Feature |
| `route_distance_km` | `routes.distance_km` | Corridor distance | Feature |
| `route_transit_hours` | `routes.standard_transit_hours` | Transit duration | Feature |
| `avail_inventory_same_route_kg` | Derived | Available inventory (weight) for same-route pending orders at snapshot time | Feature |
| `avail_inventory_same_route_m3` | Derived | Available inventory (volume) for same-route pending orders at snapshot time | Feature |
| `pending_open_orders_same_route` | Derived | Count of unfulfilled order lines for same route | Feature |
| `pending_open_qty_same_route` | Derived | Total open units for same-route orders | Feature |
| `mean_demand_last_7d_route` | `demand_history` | Rolling 7-day mean demand for this route | Feature |
| `day_of_week` | Derived | Day of week (0-6) | Feature |
| `days_into_simulation` | Derived | Day index (1-90) | Feature |
| `could_reach_ftl` | Derived | Whether feasible cargo exists to reach FTL threshold | **Target** |

### 7.4 Fields That Must Be Excluded (Leakage Prevention)

| Excluded Field | Reason |
|:---------------|:-------|
| `shipment_status` | Post-dispatch outcome |
| `actual_dispatch_time` | Future event |
| `actual_delivery_time` | Future event |
| `expected_arrival_time` | Computed at dispatch time |
| `shipped_quantity` (from order_lines) | Final cumulative state, not snapshot state |
| `open_quantity` (from order_lines) | Final cumulative state, not snapshot state |
| Day d planning run allocations | Not available at 23:00 cutoff (solver runs 23:00-04:00) |

### 7.5 Handling Unobservable Outcomes

- **HELD shipments (36):** Include in ODE with `could_reach_ftl` computed normally. The hold reason (vehicle maintenance) is independent of consolidation feasibility.
- **PLANNED shipments (10):** These are post-horizon (scheduled after March 30). **Exclude** — they have no corresponding inventory snapshot or operational context.
- **IN_TRANSIT shipment (1):** Include — was dispatched and has full operational context.

### 7.6 Candidate Count Estimate

- Total dispatched + held shipments with full operational context: ~727
- Minus PLANNED shipments (no evaluatable context): ~717 candidates

---

## 8. Recommended Next Steps for Phase 6

### Step 1: Construct the Evaluation-Time Feature Matrix
For each candidate shipment, join to the **nightly inventory snapshot** and **order state** as of the planning cutoff timestamp. Do not use final-state columns from `order_lines`.

### Step 2: Compute Route-Compatible Available Cargo
For each candidate, identify open order lines destined for the same route, cross-reference with available inventory at the snapshot, and compute the total weight and volume of cargo that could feasibly be added.

### Step 3: Derive the `could_reach_ftl` Target
Apply a multi-dimensional knapsack feasibility check: can a subset of route-compatible pending cargo fill the truck's remaining capacity to the FTL threshold on either weight or volume?

### Step 4: Validate Class Balance
If the target is severely imbalanced (e.g., >95% one class), consider adjusting the FTL threshold or using a continuous utilization target instead.

### Step 5: Save `ode_opportunities.csv`
Export the flattened ODE dataset with all features and the target. Keep source table joins documented.

---

## 9. Go/No-Go Recommendation

### ✅ CONDITIONAL GO — Phase 6 can proceed with adaptations

**Rationale:**

The synthetic data ecosystem is **structurally excellent**:
- All 14 tables are internally consistent
- All 25 foreign-key relationships are intact
- All documented business invariants hold
- Temporal ordering and inventory accounting are verified
- The relational architecture supports rich feature engineering

However, Phase 6 **must** use an adapted target definition:

| Original Plan | Required Adaptation |
|:--------------|:-------------------|
| Derive `Became_FTL` from subsequent loading events | Use `could_reach_ftl` feasibility assessment instead |
| Observe actual FTL conversions | Compute hypothetical feasibility from available cargo |
| Binary outcome from operational history | Binary outcome from knapsack feasibility at evaluation time |

> [!WARNING]
> The adaptation is not a limitation of the audit — it is a structural property of the simulation. The generator does not model the consolidation process that would produce FTL outcomes. Any attempt to force a `Became_FTL` label from the current data would produce either an empty positive class or randomly assigned labels.

**The adapted approach is:**
- Defensible (based on observable data at evaluation time)
- Non-trivial (requires multi-dimensional feasibility computation)
- Educationally valuable (teaches feature engineering, temporal partitioning, and leakage prevention)
- Honest (does not fabricate outcomes that the simulation does not produce)

---

*Phase 5 audit is complete. Awaiting approval before proceeding to Phase 6.*
