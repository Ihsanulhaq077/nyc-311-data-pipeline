# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-20
"""Unit and Integration Tests for Part B ML Preprocessing, Leakage Prevention, & Inference."""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.pipeline_b import (
    make_ml_data,
    build_preprocessor,
    build_model_pipeline,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ALL_FEATURES,
)


@pytest.fixture
def mock_clean_df():
    """Generates synthetic cleaned 311 records for ML tests."""
    return pd.DataFrame({
        "unique_key": ["101", "102", "103", "104"],
        "created_date": pd.to_datetime([
            "2026-01-01T08:00:00Z",
            "2026-01-02T10:00:00Z",
            "2026-01-03T12:00:00Z",
            "2026-01-04T14:00:00Z",
        ], utc=True),
        "closed_date": pd.to_datetime([
            "2026-01-01T14:00:00Z",  # 6.0 hours
            "2026-01-02T12:30:00Z",  # 2.5 hours
            "2026-01-01T00:00:00Z",  # negative duration (invalid)
            "2026-09-01T14:00:00Z",  # > 180 days (extreme outlier)
        ], utc=True),
        "agency": ["NYPD", "DOT", "DEP", "DSNY"],
        "agency_name": ["New York Police Department", "DOT", "DEP", "DSNY"],
        "complaint_type": ["Noise", "Street Condition", "Water Leak", "Trash"],
        "descriptor": ["Loud Music", "Pothole", "Hydrant", "Overflow"],
        "borough": ["Brooklyn", "Manhattan", "Queens", "Bronx"],
        "status": ["Closed", "Closed", "Closed", "Closed"],
        "latitude": [40.71, 40.75, 40.80, 40.85],
        "longitude": [-73.98, -73.95, -73.90, -73.88],
    })


def test_make_ml_data_target_calculation(mock_clean_df):
    """Verify target calculation and exclusion of negative/extreme durations."""
    ml_df = make_ml_data(mock_clean_df)
    # Row 3 (negative) and Row 4 (>180 days) should be filtered out
    assert len(ml_df) == 2
    assert ml_df["resolution_hours"].iloc[0] == pytest.approx(6.0, rel=1e-2)
    assert ml_df["resolution_hours"].iloc[1] == pytest.approx(2.5, rel=1e-2)


def test_make_ml_data_leakage_prevention(mock_clean_df):
    """Ensure target-leaking columns (closed_date, status, agency_name) are strictly absent from features."""
    ml_df = make_ml_data(mock_clean_df)
    assert "status" not in ml_df.columns
    assert "agency_name" not in ml_df.columns
    # created_date is retained only for temporal splitting, never in ALL_FEATURES
    assert "closed_date" not in ALL_FEATURES
    for feature in ALL_FEATURES:
        assert feature in ml_df.columns


def test_make_ml_data_feature_engineering(mock_clean_df):
    """Verify derived intake features (created_hour, created_month, is_weekend)."""
    ml_df = make_ml_data(mock_clean_df)
    assert "created_hour" in ml_df.columns
    assert "created_month" in ml_df.columns
    assert "created_weekday" in ml_df.columns
    assert "is_weekend" in ml_df.columns

    # Jan 1 2026 was a Thursday (weekday=3, is_weekend=0)
    assert ml_df["created_hour"].iloc[0] == 8
    assert ml_df["is_weekend"].iloc[0] == 0


def test_model_pipeline_fit_predict(mock_clean_df):
    """Verify ColumnTransformer and Pipeline fits and predicts on unseen rows without leakage."""
    ml_df = make_ml_data(mock_clean_df)
    X = ml_df[ALL_FEATURES]
    y = ml_df["resolution_hours"]

    pipeline = build_model_pipeline(DummyRegressor(strategy="mean"), CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    pipeline.fit(X, y)

    # Predict on new unseen record
    new_data = pd.DataFrame([{
        "complaint_type": "Noise",
        "descriptor": "Loud Music",
        "agency": "NYPD",
        "borough": "Brooklyn",
        "created_hour": 15,
        "created_month": 3,
        "created_weekday": 2,
        "is_weekend": 0,
        "latitude": 40.72,
        "longitude": -73.99,
    }])
    preds = pipeline.predict(new_data)
    assert len(preds) == 1
    assert isinstance(preds[0], (float, np.floating))
    assert preds[0] > 0
