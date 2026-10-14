"""
CargoSense -- Phase 1: Dataset Inspection Script
=================================================
This script loads the two raw CSV files and prints a comprehensive
inspection report for each. It does NOT modify the data.

How to run (from the project root):
    python src/phase1_inspect.py

Requirements:
    pip install pandas
"""

import os
import sys
import pandas as pd

# Reconfigure stdout/stderr encoding to UTF-8 if available, falling back gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Paths are relative to the project root (where you run the command from).
DATA_DIR = os.path.join("data", "raw")

FILES = {
    "Supply-Chain Dataset": os.path.join(DATA_DIR, "supply_chain_data.csv"),
    "Smart-Logistics Dataset": os.path.join(DATA_DIR, "smart_logistics_dataset.csv"),
}

# Columns we specifically want to list distinct values for (if they exist):
SPECIAL_COLS = ["Asset_ID", "Transportation modes", "Shipment_Status", "Traffic_Status"]

# A column is treated as "categorical" if it has fewer unique values than this
# threshold OR if its dtype is 'object'.
CATEGORICAL_UNIQUE_THRESHOLD = 25

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def inspect_dataset(name: str, filepath: str) -> dict:
    """
    Load a CSV file into a Pandas DataFrame and print every
    inspection item requested in Phase 1.

    Parameters
    ----------
    name : str
        A human-readable label for this dataset.
    filepath : str
        Path to the CSV file.

    Returns
    -------
    dict
        A summary dictionary with key statistics for report generation.
    """

    section(f"INSPECTING: {name}")
    print(f"File: {filepath}")

    # --- Load ---------------------------------------------------------------
    # pd.read_csv() reads a CSV file into a DataFrame (a table-like structure).
    df = pd.read_csv(filepath)

    summary = {}  # collect key stats here

    # --- Shape --------------------------------------------------------------
    # .shape returns (number_of_rows, number_of_columns).
    n_rows, n_cols = df.shape
    summary["rows"] = n_rows
    summary["cols"] = n_cols
    print(f"\nRows: {n_rows:,}   Columns: {n_cols}")

    # --- Column names and data types ----------------------------------------
    # .dtypes tells us how Pandas interpreted each column (int64, float64,
    # object = text, etc.).
    subsection("Column Names and Data Types")
    for col in df.columns:
        print(f"  {col:<35s}  {str(df[col].dtype)}")

    # --- First five rows ----------------------------------------------------
    # .head(5) returns the first 5 rows. Useful for a quick sanity check.
    subsection("First 5 Rows")
    # pd.set_option temporarily widens the display so columns aren't clipped.
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(5).to_string(index=False))

    # --- Last five rows -----------------------------------------------------
    # .tail(5) returns the last 5 rows.
    subsection("Last 5 Rows")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.tail(5).to_string(index=False))

    # --- Descriptive statistics (numerical) ---------------------------------
    # .describe() computes count, mean, std, min, 25%, 50%, 75%, max for
    # every numerical column.
    subsection("Numerical Descriptive Statistics")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.2f}".format
    ):
        print(df.describe().to_string())

    # --- Missing values per column ------------------------------------------
    # .isnull().sum() counts NaN/None values in each column.
    subsection("Missing Values per Column")
    missing = df.isnull().sum()
    summary["total_missing"] = int(missing.sum())
    for col in df.columns:
        count = missing[col]
        pct = 100.0 * count / n_rows
        flag = "  [!]" if count > 0 else ""
        print(f"  {col:<35s}  {count:>6}  ({pct:5.1f}%){flag}")

    # --- Duplicate rows -----------------------------------------------------
    # .duplicated().sum() counts rows that are exact copies of an earlier row.
    n_dup = int(df.duplicated().sum())
    summary["duplicates"] = n_dup
    subsection("Duplicate Rows")
    print(f"  Exact duplicate rows: {n_dup}")

    # --- Unique values per column -------------------------------------------
    # .nunique() counts distinct values (excluding NaN).
    subsection("Unique Values per Column")
    for col in df.columns:
        print(f"  {col:<35s}  {df[col].nunique():>6}")

    # --- Categorical columns: unique values + frequency counts --------------
    subsection("Categorical Columns -- Values and Frequencies")
    cat_cols = [
        col for col in df.columns
        if df[col].dtype == "object"
        or df[col].nunique() <= CATEGORICAL_UNIQUE_THRESHOLD
    ]
    summary["categorical_columns"] = cat_cols

    for col in cat_cols:
        print(f"\n  Column: {col}  (unique={df[col].nunique()}, dtype={df[col].dtype})")
        # .value_counts() returns each unique value and its frequency,
        # sorted from most to least common.
        vc = df[col].value_counts(dropna=False)
        for val, cnt in vc.items():
            print(f"    {str(val):<40s}  {cnt:>5}  ({100*cnt/n_rows:5.1f}%)")

    # --- Special columns (Asset_ID, Transportation modes, etc.) -------------
    subsection("Distinct Values of Special Columns (if present)")
    for col in SPECIAL_COLS:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique(), key=str)
            print(f"\n  {col} ({len(vals)} distinct):")
            for v in vals:
                print(f"    - {v}")
        else:
            print(f"\n  {col}: NOT FOUND in this dataset.")

    # --- Timestamp inspection -----------------------------------------------
    subsection("Timestamp / Date Columns")
    ts_found = False
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["time", "date", "stamp"]):
            ts_found = True
            # Distinguish numeric intervals/durations from true timestamps
            if pd.api.types.is_numeric_dtype(df[col]):
                val_min = df[col].min()
                val_max = df[col].max()
                if val_max > 1e8:  # potential epoch timestamp
                    try:
                        unit = "ms" if val_max > 1e11 else "s"
                        parsed = pd.to_datetime(df[col], unit=unit, errors="coerce")
                        valid = parsed.notna().sum()
                        failed = parsed.isna().sum()
                        print(f"\n  Column: {col} (numeric epoch timestamp, unit={unit})")
                        print(f"    Successfully parsed: {valid}  |  Failed: {failed}")
                        if valid > 0:
                            print(f"    Min (earliest):  {parsed.min()}")
                            print(f"    Max (latest):    {parsed.max()}")
                    except Exception as e:
                        print(f"\n  Column: {col} (numeric epoch) - could not parse: {e}")
                else:
                    print(f"\n  Column: {col} (numeric duration/interval metric, not calendar date)")
                    print(f"    Min: {val_min}  |  Max: {val_max}  |  Mean: {df[col].mean():.2f}")
                    print(f"    Sample values: {df[col].iloc[:5].tolist()}")
            else:
                # String / object column
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    valid = parsed.notna().sum()
                    failed = parsed.isna().sum()
                    print(f"\n  Column: {col} (string date/timestamp)")
                    print(f"    Successfully parsed: {valid}  |  Failed: {failed}")
                    if valid > 0:
                        print(f"    Min (earliest):  {parsed.min()}")
                        print(f"    Max (latest):    {parsed.max()}")
                        print(f"    Sample formats:  {df[col].iloc[:3].tolist()}")
                except Exception as e:
                    print(f"\n  Column: {col} - could not parse: {e}")
    if not ts_found:
        print("  No columns with 'time', 'date', or 'stamp' in their name.")

    # --- Potentially invalid / suspicious values ----------------------------
    subsection("Potentially Suspicious Values")

    # Numerical columns: flag extreme values or unexpected signs
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        issues = []
        # Negative values where you'd expect positive
        if col_min < 0 and any(kw in col.lower() for kw in [
            "price", "cost", "revenue", "quantity", "stock", "level",
            "sold", "volume", "weight", "utilization", "forecast",
            "frequency", "waiting", "lead", "time",
        ]):
            issues.append(f"negative values present (min={col_min})")
        # Outlier check: values more than 5 std from the mean
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            low = mean - 5 * std
            high = mean + 5 * std
            n_outliers = int(((df[col] < low) | (df[col] > high)).sum())
            if n_outliers > 0:
                issues.append(f"{n_outliers} values beyond 5 std dev (range [{low:.1f}, {high:.1f}])")
        if issues:
            print(f"  {col}: {'; '.join(issues)}")

    # Latitude / Longitude sanity check
    if "Latitude" in df.columns:
        bad_lat = int(((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum())
        if bad_lat > 0:
            print(f"  Latitude: {bad_lat} values outside [-90, 90]")
        else:
            print(f"  Latitude: all values within valid geographic range [-90, 90]")
    if "Longitude" in df.columns:
        bad_lon = int(((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum())
        if bad_lon > 0:
            print(f"  Longitude: {bad_lon} values outside [-180, 180]")
        else:
            print(f"  Longitude: all values within valid geographic range [-180, 180]")

    # Object columns: check for unexpected whitespace or mixed-case variants
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        vals = df[col].dropna().unique()
        stripped = [str(v).strip() for v in vals]
        n_whitespace = sum(1 for v, s in zip(vals, stripped) if str(v) != s)
        if n_whitespace > 0:
            print(f"  {col}: {n_whitespace} values have leading/trailing whitespace")
        # Check for possible case inconsistencies
        lower_map = {}
        for v in vals:
            lower_map.setdefault(str(v).strip().lower(), []).append(v)
        for key, variants in lower_map.items():
            if len(variants) > 1:
                print(f"  {col}: possible case inconsistency - {variants}")

    # --- Ambiguous / similarly-named columns --------------------------------
    subsection("Potentially Ambiguous Column Names")
    col_list = list(df.columns)
    flagged = set()
    for i, c1 in enumerate(col_list):
        for c2 in col_list[i + 1:]:
            # Simple heuristic: share a significant word
            words1 = set(c1.lower().replace("_", " ").split())
            words2 = set(c2.lower().replace("_", " ").split())
            common = words1 & words2 - {"of", "the", "a", "an", "and", "in", "to"}
            if common:
                pair = (c1, c2)
                if pair not in flagged:
                    flagged.add(pair)
                    print(f"  {c1}  <->  {c2}   (shared keywords: {common})")
    if not flagged:
        print("  No obvious ambiguous pairs detected.")

    return summary


# ---------------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  CargoSense -- Phase 1 Dataset Inspection")
    print("=" * 70)

    summaries = {}
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"\n[!] File not found: {path}")
            print("    Make sure you run this script from the project root directory.")
            sys.exit(1)
        summaries[name] = inspect_dataset(name, path)

    # --- Final overview -----------------------------------------------------
    section("INSPECTION COMPLETE -- OVERVIEW")
    for name, s in summaries.items():
        print(f"\n  {name}")
        print(f"    Rows: {s['rows']:,}   Columns: {s['cols']}")
        print(f"    Total missing cells: {s['total_missing']:,}")
        print(f"    Duplicate rows: {s['duplicates']}")

    print("\n[OK] Inspection finished. No data was modified.\n")


if __name__ == "__main__":
    main()
