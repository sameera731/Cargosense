"""
CargoSense -- Phase 3: Product Dataset Preparation Module
=========================================================
This module prepares the master product catalog for the CargoSense
Opportunity Detection Engine (ODE).

Core transformations:
  1. Preserves all observed commercial and supply-chain attributes.
  2. Generates category-specific physical packaging dimensions (weight, volume, density).
  3. Maps origin hub cities to regional distribution centers.
  4. Formulates explicit, distinct transport and consolidation eligibility flags.
  5. Validates catalog integrity across 10 distinct domain rules.

Outputs:
  - data/processed/products.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS & REPRODUCIBLE MODELING ASSUMPTIONS
# ---------------------------------------------------------------------------
# Packaging dimension benchmarks for personal care & cosmetics:
PHYSICAL_BENCHMARKS = {
    "haircare": {
        "desc": "Bottled liquids (shampoos, conditioners, hair oils, 250-500ml) in protective cartons",
        "weight_mean": 0.50, "weight_std": 0.06, "weight_min": 0.35, "weight_max": 0.70,
        "vol_mean": 0.0010, "vol_std": 0.00012, "vol_min": 0.0007, "vol_max": 0.0014,
    },
    "skincare": {
        "desc": "Tubes, jars, and pump dispensers (lotions, creams, 50-200ml) with protective cartons",
        "weight_mean": 0.25, "weight_std": 0.04, "weight_min": 0.15, "weight_max": 0.38,
        "vol_mean": 0.0005, "vol_std": 0.00008, "vol_min": 0.0003, "vol_max": 0.0008,
    },
    "cosmetics": {
        "desc": "Compact items (lipsticks, palettes, powders, mascaras) in lightweight carton packaging",
        "weight_mean": 0.12, "weight_std": 0.025, "weight_min": 0.05, "weight_max": 0.20,
        "vol_mean": 0.00025, "vol_std": 0.00005, "vol_min": 0.00012, "vol_max": 0.00040,
    },
}

# Regional distribution center mappings:
WAREHOUSE_MAPPING = {
    "Mumbai": {
        "warehouse_id": "WH-MUM-01",
        "facility_name": "Bhiwandi Western Hub",
        "region": "West",
    },
    "Delhi": {
        "warehouse_id": "WH-DEL-01",
        "facility_name": "Kundli Northern Hub",
        "region": "North",
    },
    "Kolkata": {
        "warehouse_id": "WH-KOL-01",
        "facility_name": "Dankuni Eastern Hub",
        "region": "East",
    },
    "Bangalore": {
        "warehouse_id": "WH-BLR-01",
        "facility_name": "Nelamangala Central South Hub",
        "region": "South-Central",
    },
    "Chennai": {
        "warehouse_id": "WH-MAA-01",
        "facility_name": "Sriperumbudur Coastal South Hub",
        "region": "South-Coast",
    },
}

VALID_TRANSPORT_MODES = {"Road", "Rail", "Air", "Sea"}
VALID_INSPECTION_STATUSES = {"Pass", "Pending", "Fail"}


def assign_physical_attributes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate category-specific physical dimensions (weight, volume) using
    a reproducible normal distribution bounded by plausible CPG benchmarks.
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    weights = []
    volumes = []

    for _, row in df.iterrows():
        pt = str(row["product_type"]).lower()
        cfg = PHYSICAL_BENCHMARKS.get(pt, PHYSICAL_BENCHMARKS["skincare"])

        # Sample weight (kg) and clip to category bounds
        w = rng.normal(cfg["weight_mean"], cfg["weight_std"])
        w = np.clip(w, cfg["weight_min"], cfg["weight_max"])

        # Sample volume (m³) and clip to category bounds
        v = rng.normal(cfg["vol_mean"], cfg["vol_std"])
        v = np.clip(v, cfg["vol_min"], cfg["vol_max"])

        weights.append(round(float(w), 3))
        volumes.append(round(float(v), 6))

    df["unit_weight_kg"] = weights
    df["unit_volume_m3"] = volumes
    df["density_kg_m3"] = (df["unit_weight_kg"] / df["unit_volume_m3"]).round(1)

    return df


def assign_warehouses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map origin location city deterministically (1-to-1) to a regional warehouse.
    """
    df = df.copy()
    loc_col = "location" if "location" in df.columns else "origin_location"

    df["warehouse_id"] = df[loc_col].map(lambda loc: WAREHOUSE_MAPPING[loc]["warehouse_id"])
    df["warehouse_facility"] = df[loc_col].map(lambda loc: WAREHOUSE_MAPPING[loc]["facility_name"])
    df["warehouse_region"] = df[loc_col].map(lambda loc: WAREHOUSE_MAPPING[loc]["region"])

    return df


def build_product_catalog(input_path: str, output_path: str = None, seed: int = 42) -> pd.DataFrame:
    """
    Load cleaned supply chain data, enrich with physical dimensions and
    warehouse facilities, formulate distinct eligibility flags, and validate.
    """
    raw_df = pd.read_csv(input_path)

    # 1. Enrich with physical attributes
    enriched = assign_physical_attributes(raw_df, seed=seed)

    # 2. Enrich with warehouse mapping
    enriched = assign_warehouses(enriched)

    # 3. Explicit Eligibility Rules
    # Concept 1: Road Transport Eligibility (strictly whether transportation mode is Road)
    is_road_eligible = (enriched["transportation_modes"] == "Road")

    # Concept 2: Inspection Status (preserved as source category: Pass, Pending, Fail)
    inspection_status = enriched["inspection_results"]

    # Concept 3: Consolidation Candidacy (requires Road transport, Passed inspection, and Available Stock > 0)
    is_consolidation_candidate = is_road_eligible & (inspection_status == "Pass") & (enriched["stock_levels"] > 0)

    # 4. Construct clean, realistic product catalog schema
    catalog = pd.DataFrame({
        "sku": enriched["sku"],
        "product_type": enriched["product_type"],
        "unit_price": enriched["price"].round(2),
        "stock_level": enriched["stock_levels"].astype(int),
        "order_quantity_baseline": enriched["order_quantities"].astype(int),
        "lead_time_days": enriched["lead_times"].astype(int),
        "manufacturing_cost": enriched["manufacturing_costs"].round(2),
        "shipping_cost_baseline": enriched["shipping_costs"].round(2),
        "supplier_name": enriched["supplier_name"],
        "origin_location": enriched["location"],
        "warehouse_id": enriched["warehouse_id"],
        "warehouse_facility": enriched["warehouse_facility"],
        "warehouse_region": enriched["warehouse_region"],
        "transportation_mode": enriched["transportation_modes"],
        "assigned_route": enriched["routes"],
        "inspection_status": inspection_status,
        "defect_rate_pct": enriched["defect_rates"].round(2),
        "unit_weight_kg": enriched["unit_weight_kg"],
        "unit_volume_m3": enriched["unit_volume_m3"],
        "density_kg_m3": enriched["density_kg_m3"],
        "is_road_eligible": is_road_eligible,
        "is_consolidation_candidate": is_consolidation_candidate,
    })

    # Sort deterministically by numeric SKU index (SKU0 to SKU99)
    catalog["sku_num"] = catalog["sku"].str.replace("SKU", "").astype(int)
    catalog = catalog.sort_values("sku_num").drop(columns=["sku_num"]).reset_index(drop=True)

    # Validate against all 10 domain rules
    validation_results = validate_product_catalog(catalog)
    if not validation_results["all_passed"]:
        raise ValueError(f"Product catalog validation failed: {validation_results['failures']}")

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        catalog.to_csv(output_path, index=False)
        print(f"[OK] Product catalog successfully created and saved to: {output_path} ({catalog.shape[0]} rows, {catalog.shape[1]} columns)")

    return catalog


def validate_product_catalog(df: pd.DataFrame) -> dict:
    """
    Run the complete 10-point integrity test suite on the prepared products dataset.
    """
    checks = {}
    failures = []

    # 1. Exactly 100 rows
    checks["exact_100_rows"] = len(df) == 100
    if not checks["exact_100_rows"]:
        failures.append(f"Expected 100 rows, got {len(df)}")

    # 2. Exactly 100 unique SKUs
    checks["exact_100_unique_skus"] = df["sku"].nunique() == 100
    if not checks["exact_100_unique_skus"]:
        failures.append(f"Expected 100 unique SKUs, got {df['sku'].nunique()}")

    # 3. All original SKU identifiers preserved
    expected_skus = {f"SKU{i}" for i in range(100)}
    actual_skus = set(df["sku"])
    checks["all_sku_ids_preserved"] = expected_skus == actual_skus
    if not checks["all_sku_ids_preserved"]:
        failures.append(f"SKU identifier mismatch: missing {expected_skus - actual_skus}")

    # 4. No unexpected missing values in required fields
    missing_count = int(df.isnull().sum().sum())
    checks["no_missing_values"] = missing_count == 0
    if not checks["no_missing_values"]:
        failures.append(f"Found {missing_count} unexpected missing values")

    # 5. Positive weight and volume
    weights_pos = bool((df["unit_weight_kg"] > 0).all())
    volumes_pos = bool((df["unit_volume_m3"] > 0).all())
    checks["positive_weight_and_volume"] = weights_pos and volumes_pos
    if not checks["positive_weight_and_volume"]:
        failures.append("Found non-positive weights or volumes")

    # 6. Density correctly calculated from weight and volume
    expected_density = (df["unit_weight_kg"] / df["unit_volume_m3"]).round(1)
    density_matches = bool((df["density_kg_m3"] == expected_density).all())
    checks["density_correctly_calculated"] = density_matches
    if not checks["density_correctly_calculated"]:
        failures.append("Density column does not equal weight / volume")

    # 7. Non-negative stock levels
    checks["non_negative_stock"] = bool((df["stock_level"] >= 0).all())
    if not checks["non_negative_stock"]:
        failures.append("Found negative stock levels")

    # 8. Valid warehouse mappings
    all_wh_valid = bool(df["warehouse_id"].str.startswith("WH-").all())
    cities_in_mapping = bool(df["origin_location"].isin(WAREHOUSE_MAPPING.keys()).all())
    checks["valid_warehouse_mappings"] = all_wh_valid and cities_in_mapping
    if not checks["valid_warehouse_mappings"]:
        failures.append("Invalid warehouse mapping or unrecognized origin city")

    # 9. Valid transportation and inspection categories
    trans_valid = bool(set(df["transportation_mode"]).issubset(VALID_TRANSPORT_MODES))
    insp_valid = bool(set(df["inspection_status"]).issubset(VALID_INSPECTION_STATUSES))
    checks["valid_categories"] = trans_valid and insp_valid
    if not checks["valid_categories"]:
        failures.append(f"Invalid category found: transport={set(df['transportation_mode'])}, inspection={set(df['inspection_status'])}")

    # 10. Eligibility flags consistent with documented rules
    expected_road = (df["transportation_mode"] == "Road")
    expected_cand = expected_road & (df["inspection_status"] == "Pass") & (df["stock_level"] > 0)
    road_ok = bool((df["is_road_eligible"] == expected_road).all())
    cand_ok = bool((df["is_consolidation_candidate"] == expected_cand).all())
    checks["eligibility_flags_consistent"] = road_ok and cand_ok
    if not checks["eligibility_flags_consistent"]:
        failures.append("Eligibility flags are inconsistent with documented rules")

    checks["all_passed"] = len(failures) == 0
    checks["failures"] = failures

    return checks


if __name__ == "__main__":
    input_file = os.path.join("data", "processed", "supply_chain_data_cleaned.csv")
    output_file = os.path.join("data", "processed", "products.csv")
    build_product_catalog(input_file, output_file)
