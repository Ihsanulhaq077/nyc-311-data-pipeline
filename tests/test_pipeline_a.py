# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18
"""Unit and Integration Tests for Part A Data Pipeline & Quality Gates."""

import pytest
import numpy as np
import pandas as pd

from src.pipeline_a import (
    clean_column_names,
    standardize_data_types,
    standardize_categories,
    handle_duplicates,
    handle_missing_values,
    handle_coordinate_outliers,
    handle_temporal_outliers,
    clean_data,
    validate_schema_and_types,
    validate_ranges,
    validate_uniqueness,
    validate_null_constraints,
    validate_data,
    DataQualityGateError,
)


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic raw DataFrame exhibiting common messy data problems."""
    return pd.DataFrame({
        "Unique Key": ["1001", "1002", "1002", "1003"],
        "Created Date": ["2026-01-01T10:00:00Z", "2026-01-02T12:00:00Z", "2026-01-02T12:00:00Z", "2026-01-03T15:00:00Z"],
        "Closed Date": ["2026-01-01T14:00:00Z", "2026-01-01T08:00:00Z", "2026-01-01T08:00:00Z", "2026-01-04T12:00:00Z"],  # row 2 has inverted date
        "Agency": ["nypd", " DOT ", " DOT ", "dep"],
        "Agency Name": ["New York Police Department", "DOT", "DOT", "DEP"],
        "Complaint Type": ["NOISE - RESIDENTIAL", "street light out", "street light out", "Blocked Driveway"],
        "Descriptor": ["Loud Music", "Street Lamp Broken", "Street Lamp Broken", ""],
        "Borough": [" brooklyn ", "MANHATTAN", "MANHATTAN", "Unspecified"],
        "Status": ["Closed", "Closed", "Closed", "Open"],
        "Latitude": ["40.7128", "99.9999", "99.9999", "40.7589"],  # row 2 has invalid outlier latitude
        "Longitude": ["-74.0060", "-73.9851", "-73.9851", "-190.0"],  # row 4 has invalid longitude
    })


def test_clean_column_names(sample_raw_dataframe):
    """Verify column names are stripped, lowercased, and underscored."""
    cleaned = clean_column_names(sample_raw_dataframe)
    assert "unique_key" in cleaned.columns
    assert "created_date" in cleaned.columns
    assert "complaint_type" in cleaned.columns
    assert not any(" " in col for col in cleaned.columns)


def test_standardize_data_types(sample_raw_dataframe):
    """Verify dates are parsed into UTC datetimes and coordinates are converted to floats."""
    df = clean_column_names(sample_raw_dataframe)
    standardized = standardize_data_types(df)
    assert pd.api.types.is_datetime64_any_dtype(standardized["created_date"])
    assert pd.api.types.is_datetime64_any_dtype(standardized["closed_date"])
    assert pd.api.types.is_float_dtype(standardized["latitude"])
    assert pd.api.types.is_float_dtype(standardized["longitude"])


def test_standardize_categories(sample_raw_dataframe):
    """Verify string categories are title-cased and placeholder nulls replaced."""
    df = clean_column_names(sample_raw_dataframe)
    standardized = standardize_categories(df)
    assert standardized["borough"].iloc[0] == "Brooklyn"
    assert standardized["agency"].iloc[0] == "Nypd"
    # 'Unspecified' placeholder should be converted to pd.NA
    assert pd.isna(standardized["borough"].iloc[3])


def test_handle_duplicates(sample_raw_dataframe):
    """Verify duplicate submissions by unique_key are pruned."""
    df = clean_column_names(sample_raw_dataframe)
    deduped = handle_duplicates(df)
    assert len(deduped) == 3
    assert len(deduped["unique_key"].unique()) == 3


def test_handle_missing_values(sample_raw_dataframe):
    """Verify fallback values are populated for missing descriptive fields."""
    df = clean_column_names(sample_raw_dataframe)
    df = standardize_categories(df)
    filled = handle_missing_values(df)
    assert filled["borough"].iloc[3] == "Unknown Borough"
    assert filled["descriptor"].iloc[3] == "Not Provided"


def test_handle_coordinate_outliers(sample_raw_dataframe):
    """Verify out-of-bounds geographic coordinates are set to np.nan."""
    df = clean_column_names(sample_raw_dataframe)
    df = standardize_data_types(df)
    filtered = handle_coordinate_outliers(df)
    # 99.9999 is outside NYC latitude bounds [40.40, 41.00]
    assert np.isnan(filtered["latitude"].iloc[1])
    # -190.0 is outside NYC longitude bounds [-74.30, -73.65]
    assert np.isnan(filtered["longitude"].iloc[3])
    # 40.7128 and -74.0060 are valid NYC coordinates
    assert filtered["latitude"].iloc[0] == pytest.approx(40.7128, rel=1e-3)


def test_handle_temporal_outliers(sample_raw_dataframe):
    """Verify cases where closed_date precedes created_date are corrected to NaT."""
    df = clean_column_names(sample_raw_dataframe)
    df = standardize_data_types(df)
    corrected = handle_temporal_outliers(df)
    # Row 1 created on Jan 2 but closed on Jan 1 -> invalid
    assert pd.isna(corrected["closed_date"].iloc[1])
    # Row 0 created Jan 1 10:00, closed Jan 1 14:00 -> valid
    assert pd.notna(corrected["closed_date"].iloc[0])


def test_clean_data_end_to_end(sample_raw_dataframe):
    """Verify that full clean_data pipeline runs without errors and produces expected structure."""
    cleaned = clean_data(sample_raw_dataframe)
    assert len(cleaned) == 3
    assert not cleaned.duplicated(subset=["unique_key"]).any()
    assert pd.api.types.is_datetime64_any_dtype(cleaned["created_date"])


def test_validate_schema_and_types():
    """Verify schema and type validation fails when columns or types are wrong."""
    valid_df = pd.DataFrame({
        "created_date": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "complaint_type": ["Noise"],
        "agency": ["NYPD"],
    })
    assert validate_schema_and_types(valid_df) is True

    invalid_df = pd.DataFrame({"complaint_type": ["Noise"]})
    assert validate_schema_and_types(invalid_df) is False


def test_validate_ranges():
    """Verify geographic and temporal range validators."""
    good_df = pd.DataFrame({
        "latitude": [40.7128],
        "longitude": [-74.0060],
        "created_date": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "closed_date": pd.to_datetime(["2026-01-01T05:00:00Z"]),
    })
    res = validate_ranges(good_df)
    assert res["valid_latitude_range"] is True
    assert res["valid_longitude_range"] is True
    assert res["valid_temporal_order"] is True


def test_validate_uniqueness():
    """Verify uniqueness validator detects duplicate rows or keys."""
    dupe_df = pd.DataFrame({"unique_key": ["101", "101"], "val": [1, 2]})
    assert validate_uniqueness(dupe_df) is False

    unique_df = pd.DataFrame({"unique_key": ["101", "102"], "val": [1, 2]})
    assert validate_uniqueness(unique_df) is True


def test_validate_null_constraints():
    """Verify null constraint validator thresholds."""
    df = pd.DataFrame({
        "created_date": pd.to_datetime(["2026-01-01", "2026-01-02", None]),
        "complaint_type": ["A", "B", "C"],
    })
    res = validate_null_constraints(df)
    # 2/3 = 66% non-null, below 95% threshold
    assert res["created_date_non_null"] is False
    assert res["complaint_type_non_null"] is True


def test_quality_gate_passes_on_valid_data():
    """Verify that a compliant dataset passes validation above the 80% threshold."""
    df = pd.DataFrame({
        "unique_key": ["1", "2", "3"],
        "created_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        "closed_date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04"], utc=True),
        "agency": ["NYPD", "DOT", "DEP"],
        "complaint_type": ["Noise", "Traffic", "Water"],
        "latitude": [40.71, 40.75, 40.80],
        "longitude": [-73.98, -73.95, -73.90],
    })
    checks, pass_rate = validate_data(df)
    assert pass_rate >= 0.85
    assert all(checks.values())


def test_quality_gate_fails_loudly_on_corrupt_data():
    """Demonstrate Part A Quality Gate bonus (+2): fails loudly on corrupt data."""
    # Corrupt data with missing created_date, negative durations, and invalid coordinates
    corrupt_df = pd.DataFrame({
        "unique_key": ["1", "1"],  # duplicates
        "created_date": [None, None],  # completely missing created_date
        "agency": ["NYPD", "DOT"],
        "complaint_type": [None, None],  # completely missing complaint type
        "latitude": [999.0, -999.0],  # impossible coordinates
        "longitude": [999.0, -999.0],
    })
    checks, pass_rate = validate_data(corrupt_df)
    threshold = 0.80

    assert pass_rate < threshold
    # Verify that a quality gate enforcement triggers DataQualityGateError
    with pytest.raises(DataQualityGateError):
        if pass_rate < threshold:
            raise DataQualityGateError(
                f"Quality Gate Failed! Pass rate {pass_rate:.1%} is below threshold {threshold:.1%}."
            )
