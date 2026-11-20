# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18
"""Part A: Reproducible Data Quality, Validation, Cleaning & Insights Pipeline.

This module loads raw NYC 311 service request records, inspects their schema,
cleans the fields using modular tested functions, validates data constraints,
enforces a loud quality gate, generates summary statistics with Pandas and NumPy,
produces visualizations, and loads the data into SQLite for analytical queries.
"""

from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# File paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "nyc_311.csv"
CLEAN_DATA_PATH = ROOT / "data" / "cleaned" / "nyc_311_clean.csv"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"

# Core columns to retain for downstream analysis and tabular ML
KEEP_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "borough",
    "status",
    "latitude",
    "longitude",
]


class DataQualityGateError(ValueError):
    """Raised when dataset validation pass rate falls below the required threshold."""
    pass


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV without modifying the source file on disk.

    Parameters
    ----------
    path : Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path, low_memory=False)


def inspect_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect and record schema properties of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Schema table with columns: column, dtype, missing_count, missing_pct,
        unique_count, and sample_value.
    """
    records = []
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = round((missing / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        sample_val = str(series.dropna().iloc[0]) if not series.dropna().empty else "None"
        records.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": missing,
            "missing_pct": missing_pct,
            "unique_count": unique_cnt,
            "sample_value": sample_val[:50],
        })
    return pd.DataFrame(records)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lower case with stripped whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column headers.
    """
    data = df.copy()
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    return data


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and convert numeric coordinate fields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardized column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed date and numeric types.
    """
    data = df.copy()
    for col in ["created_date", "closed_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce", utc=True)
    for col in ["latitude", "longitude"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize string categorical columns (title case, strip whitespace, nullify placeholders).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed types.

    Returns
    -------
    pd.DataFrame
        DataFrame with clean, consistent categorical values.
    """
    data = df.copy()
    cat_cols = ["complaint_type", "descriptor", "agency", "agency_name", "borough", "status"]
    placeholder_nulls = {"", "nan", "none", "null", "n/a", "unspecified"}
    for col in cat_cols:
        if col in data.columns:
            cleaned_series = data[col].astype("string").str.strip()
            # Normalize title case for readable aggregations
            cleaned_series = cleaned_series.str.title()
            # Replace placeholder string nulls with genuine pd.NA
            mask = cleaned_series.str.lower().isin(placeholder_nulls)
            cleaned_series[mask] = pd.NA
            data[col] = cleaned_series
    return data


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and drop duplicate rows based on primary identifier or exact row matches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame possibly containing duplicate submissions.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    data = df.copy()
    if "unique_key" in data.columns and data["unique_key"].notna().any():
        data = data.drop_duplicates(subset=["unique_key"]).reset_index(drop=True)
    else:
        data = data.drop_duplicates().reset_index(drop=True)
    return data


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or standardize missing values according to domain business logic.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing value treatment applied.
    """
    data = df.copy()
    if "borough" in data.columns:
        data["borough"] = data["borough"].fillna("Unknown Borough")
    if "descriptor" in data.columns:
        data["descriptor"] = data["descriptor"].fillna("Not Provided")
    if "status" in data.columns:
        data["status"] = data["status"].fillna("Unknown")
    return data


def handle_coordinate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Nullify coordinates that fall outside valid New York City geographic bounding boxes.

    NYC Bounding Box roughly:
    Latitude: [40.40, 41.00]
    Longitude: [-74.30, -73.65]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing latitude and longitude columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with out-of-bounds coordinates replaced with np.nan.
    """
    data = df.copy()
    if "latitude" in data.columns and "longitude" in data.columns:
        invalid_lat = ~data["latitude"].between(40.40, 41.00)
        invalid_lon = ~data["longitude"].between(-74.30, -73.65)
        outlier_mask = invalid_lat | invalid_lon
        data.loc[outlier_mask, "latitude"] = np.nan
        data.loc[outlier_mask, "longitude"] = np.nan
    return data


def handle_temporal_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate or drop records where closed_date precedes created_date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with created_date and closed_date.

    Returns
    -------
    pd.DataFrame
        DataFrame with inverted timestamps corrected (closed_date set to pd.NaT).
    """
    data = df.copy()
    if "created_date" in data.columns and "closed_date" in data.columns:
        invalid_temporal = data["closed_date"] < data["created_date"]
        data.loc[invalid_temporal, "closed_date"] = pd.NaT
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """High-level data cleaning orchestrator chaining all modular transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned, type-safe, deduplicated, and validated DataFrame.
    """
    data = clean_column_names(df)
    # Select available subset of relevant columns
    cols_to_keep = [col for col in KEEP_COLUMNS if col in data.columns]
    data = data[cols_to_keep].copy()

    data = standardize_data_types(data)
    data = standardize_categories(data)
    data = handle_duplicates(data)
    data = handle_coordinate_outliers(data)
    data = handle_temporal_outliers(data)
    data = handle_missing_values(data)
    return data


# --- Validation Functions ---

def validate_schema_and_types(df: pd.DataFrame) -> bool:
    """Validate that expected mandatory columns exist and have appropriate data types."""
    required = ["created_date", "complaint_type", "agency"]
    has_cols = all(col in df.columns for col in required)
    if not has_cols:
        return False
    return pd.api.types.is_datetime64_any_dtype(df["created_date"])


def validate_ranges(df: pd.DataFrame) -> dict:
    """Validate that coordinate ranges and temporal sequences are within valid domain bounds."""
    results = {}
    if "latitude" in df.columns:
        valid_lat = df["latitude"].dropna().between(40.40, 41.00).all()
        results["valid_latitude_range"] = bool(valid_lat)
    if "longitude" in df.columns:
        valid_lon = df["longitude"].dropna().between(-74.30, -73.65).all()
        results["valid_longitude_range"] = bool(valid_lon)
    if "created_date" in df.columns and "closed_date" in df.columns:
        has_dates = df["closed_date"].notna() & df["created_date"].notna()
        valid_chronology = (df.loc[has_dates, "closed_date"] >= df.loc[has_dates, "created_date"]).all()
        results["valid_temporal_order"] = bool(valid_chronology)
    return results


def validate_uniqueness(df: pd.DataFrame) -> bool:
    """Validate that records do not contain exact duplicate rows."""
    if "unique_key" in df.columns and df["unique_key"].notna().any():
        return not bool(df["unique_key"].duplicated().any())
    return not bool(df.duplicated().any())


def validate_null_constraints(df: pd.DataFrame) -> dict:
    """Check that mandatory primary keys and timestamps satisfy non-null constraints."""
    results = {}
    if "created_date" in df.columns:
        pass_rate = float(df["created_date"].notna().mean())
        results["created_date_non_null"] = pass_rate >= 0.95
    if "complaint_type" in df.columns:
        pass_rate = float(df["complaint_type"].notna().mean())
        results["complaint_type_non_null"] = pass_rate >= 0.90
    return results


def validate_data(df: pd.DataFrame) -> tuple[dict, float]:
    """Execute all validation rules and compute the composite pass rate.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    tuple[dict, float]
        A dictionary of individual check results and the overall float pass rate [0.0, 1.0].
    """
    checks = {
        "has_rows": len(df) > 0,
        "schema_and_types_valid": validate_schema_and_types(df),
        "uniqueness_valid": validate_uniqueness(df),
    }
    ranges = validate_ranges(df)
    checks.update(ranges)
    nulls = validate_null_constraints(df)
    checks.update(nulls)

    pass_rate = sum(1 for v in checks.values() if v is True) / len(checks)
    return checks, round(pass_rate, 4)


# --- Summaries, Visualizations, and SQLite Queries ---

def compute_summaries_and_plots(df: pd.DataFrame):
    """Generate Pandas/NumPy statistical summaries, 3+ figures, and run SQLite queries."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pandas summaries
    borough_counts = df["borough"].value_counts(dropna=False).reset_index()
    borough_counts.columns = ["borough", "request_count"]
    borough_counts.to_csv(REPORTS_DIR / "requests_by_borough.csv", index=False)

    top_complaints = df["complaint_type"].value_counts().head(20).reset_index()
    top_complaints.columns = ["complaint_type", "count"]
    top_complaints.to_csv(REPORTS_DIR / "top_complaints.csv", index=False)

    # Agency aggregation
    agency_summary = df.groupby("agency", dropna=False).agg(
        total_requests=("agency", "count"),
        unique_complaints=("complaint_type", "nunique"),
    ).reset_index().sort_values("total_requests", ascending=False)
    agency_summary.to_csv(REPORTS_DIR / "agency_summary.csv", index=False)

    # Pivot table: Borough vs Top 5 Complaint Types
    top_5_types = df["complaint_type"].value_counts().head(5).index.tolist()
    filtered_for_pivot = df[df["complaint_type"].isin(top_5_types)]
    pivot = pd.pivot_table(
        filtered_for_pivot,
        index="borough",
        columns="complaint_type",
        values="created_date",
        aggfunc="count",
        fill_value=0
    )
    pivot.to_csv(REPORTS_DIR / "borough_complaint_pivot.csv")

    # 2. NumPy Vectorized Operations
    # Vectorized Operation 1: Coordinate validity mask using np.where
    df["coordinates_valid"] = np.where(df["latitude"].notna() & df["longitude"].notna(), True, False)

    # Vectorized Operation 2: Resolution duration in hours computed via numpy timedelta conversion
    has_both_dates = df["closed_date"].notna() & df["created_date"].notna()
    duration_array = np.full(len(df), np.nan)
    duration_array[has_both_dates] = (
        (df.loc[has_both_dates, "closed_date"] - df.loc[has_both_dates, "created_date"])
        / np.timedelta64(1, "h")
    )
    df["resolution_hours"] = duration_array

    # Vectorized Operation 3: Vectorized day-part classification with np.select
    hours = df["created_date"].dt.hour.to_numpy()
    conditions = [
        (hours >= 5) & (hours < 12),
        (hours >= 12) & (hours < 17),
        (hours >= 17) & (hours < 22),
    ]
    choices = ["Morning", "Afternoon", "Evening"]
    df["day_part"] = np.select(conditions, choices, default="Night")

    # 3. Create at least 3 Useful Visualizations with Title, Labelled Axes, and Units
    if HAS_MATPLOTLIB:
        plt.style.use("tableau-colorblind10" if "tableau-colorblind10" in plt.style.available else "default")

        # Figure 1: Requests by Borough
        plt.figure(figsize=(9, 5))
        borough_series = df["borough"].value_counts().drop("Unknown Borough", errors="ignore")
        bars = plt.bar(borough_series.index, borough_series.values, color="#2b5c8f", edgecolor="black")
        plt.title("NYC 311 Service Requests by Borough", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("NYC Borough Name", fontsize=11, labelpad=8)
        plt.ylabel("Number of Requests (Count)", fontsize=11, labelpad=8)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + (max(borough_series.values) * 0.01),
                     f"{int(h):,}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "borough_counts.png", dpi=300)
        plt.close()

        # Figure 2: Top 15 Complaint Categories
        plt.figure(figsize=(10, 6))
        top15 = df["complaint_type"].value_counts().head(15).sort_values(ascending=True)
        top15.plot(kind="barh", color="#388e3c", edgecolor="black")
        plt.title("Top 15 NYC 311 Complaint Categories", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Number of Service Requests (Count)", fontsize=11, labelpad=8)
        plt.ylabel("Complaint Category", fontsize=11, labelpad=8)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_complaints.png", dpi=300)
        plt.close()

        # Figure 3: Hourly Intake Pattern across the Day
        plt.figure(figsize=(9, 5))
        hourly = df["created_date"].dt.hour.value_counts().sort_index()
        plt.plot(hourly.index, hourly.values, marker="o", linewidth=2.5, color="#d32f2f", markersize=6)
        plt.title("Diurnal Pattern of 311 Incident Reports by Hour", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Hour of Submission (0 - 23, UTC)", fontsize=11, labelpad=8)
        plt.ylabel("Total Inflow Volume (Requests Count)", fontsize=11, labelpad=8)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "requests_by_hour.png", dpi=300)
        plt.close()

        # Figure 4: Average Resolution Duration across Top Agencies (Supplementary Insight)
        valid_durations = df[df["resolution_hours"].between(0, 24 * 60)]
        top_agencies = df["agency"].value_counts().head(6).index
        agency_durations = valid_durations[valid_durations["agency"].isin(top_agencies)].groupby("agency")["resolution_hours"].median().sort_values()
        if not agency_durations.empty:
            plt.figure(figsize=(9, 5))
            agency_durations.plot(kind="bar", color="#7b1fa2", edgecolor="black")
            plt.title("Median Resolution Duration Across Major Agencies", fontsize=14, fontweight="bold", pad=12)
            plt.xlabel("City Agency Acronym", fontsize=11, labelpad=8)
            plt.ylabel("Median Resolution Time (Hours)", fontsize=11, labelpad=8)
            plt.xticks(rotation=0)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "agency_resolution_hours.png", dpi=300)
            plt.close()
    else:
        print("Note: Matplotlib not available; visualization generation skipped.")

    # 4. SQLite Storage and Analytical Query
    db_path = REPORTS_DIR / "nyc311.sqlite"
    with sqlite3.connect(db_path) as conn:
        # Load cleaned data table
        clean_subset = df.drop(columns=["coordinates_valid", "day_part"], errors="ignore").copy()
        clean_subset["created_date"] = clean_subset["created_date"].astype(str)
        clean_subset["closed_date"] = clean_subset["closed_date"].astype(str)
        clean_subset.to_sql("service_requests", conn, if_exists="replace", index=False)

        # Analytical SQL query: aggregation of volume, average duration, and distinct complaint types by borough
        sql_query = """
        SELECT
            borough,
            COUNT(*) AS total_requests,
            ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours,
            COUNT(DISTINCT complaint_type) AS distinct_complaints,
            ROUND(100.0 * SUM(CASE WHEN latitude IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_with_coordinates
        FROM service_requests
        WHERE borough != 'Unknown Borough'
        GROUP BY borough
        ORDER BY total_requests DESC;
        """
        sql_result = pd.read_sql_query(sql_query, conn)
        sql_result.to_csv(REPORTS_DIR / "sqlite_borough_summary.csv", index=False)
        header = "-- Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18\n"
        (SQL_DIR / "borough_agency_summary.sql").write_text(header + sql_query.strip() + "\n", encoding="utf-8")
        (SQL_DIR / "borough_counts.sql").write_text(
            header + "SELECT borough, COUNT(*) AS request_count FROM service_requests GROUP BY borough ORDER BY request_count DESC;\n",
            encoding="utf-8"
        )


def run_pipeline(threshold: float = 0.85) -> pd.DataFrame:
    """Execute end-to-end Part A pipeline: load, clean, validate, check quality gate, and summarize.

    Parameters
    ----------
    threshold : float
        Minimum validation pass rate threshold required to pass the quality gate.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated dataset.
    """
    raw_df = load_raw_data(RAW_DATA_PATH)
    rows_before = len(raw_df)
    missing_cells_before = int(raw_df.isna().sum().sum())

    # Generate and save schema document
    schema_df = inspect_schema(raw_df)
    schema_df.to_csv(REPORTS_DIR / "schema.csv", index=False)
    schema_df.to_csv(DOCS_DIR / "schema.csv", index=False)

    # Clean data
    clean_df = clean_data(raw_df)
    rows_after = len(clean_df)
    missing_cells_after = int(clean_df.isna().sum().sum())

    # Validate data
    validation_checks, pass_rate = validate_data(clean_df)

    # Persist cleaned dataset
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    # Compute summaries, plots, and SQL analytics
    compute_summaries_and_plots(clean_df)

    # Write quality metrics report
    metrics = {
        "dataset_name": "NYC 311 Service Requests",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": rows_before - rows_after,
        "missing_cells_before": missing_cells_before,
        "missing_cells_after": missing_cells_after,
        "missing_pct_before": round(missing_cells_before / (rows_before * len(raw_df.columns)) * 100, 2),
        "missing_pct_after": round(missing_cells_after / (rows_after * len(clean_df.columns)) * 100, 2),
        "validation_checks": validation_checks,
        "validation_pass_rate": pass_rate,
        "quality_gate_threshold": threshold,
        "quality_gate_passed": pass_rate >= threshold,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "quality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Challenge Bonus (+2): Loud Quality Gate
    if pass_rate < threshold:
        raise DataQualityGateError(
            f"Quality Gate Failed! Validation pass rate {pass_rate:.1%} is below required threshold {threshold:.1%}."
        )

    print(f"Part A Pipeline completed successfully. Pass rate: {pass_rate:.1%}")
    return clean_df


if __name__ == "__main__":
    run_pipeline()
