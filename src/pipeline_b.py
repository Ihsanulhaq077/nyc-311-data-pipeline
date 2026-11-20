# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-20
"""Part B: End-to-End Tabular Machine Learning Baseline and Comparison Pipeline.

Frames the problem of estimating 311 service request resolution duration (in hours)
using request intake attributes known strictly at creation time.
Enforces a chronological train/test split to prevent temporal leakage,
constructs an isolated scikit-learn preprocessing ColumnTransformer,
evaluates a median baseline alongside regularized linear and tree-based regressors
using TimeSeriesSplit cross-validation, conducts subgroup error analysis,
generates diagnostic residual plots, and saves the final deployable pipeline.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Paths
ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = ROOT / "data" / "cleaned" / "nyc_311_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Explicitly documented features available at intake time (no leakage)
CATEGORICAL_FEATURES = ["complaint_type", "descriptor", "agency", "borough"]
NUMERIC_FEATURES = ["created_hour", "created_month", "created_weekday", "is_weekend", "latitude", "longitude"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares tabular features and target from cleaned 311 records without leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned NYC 311 DataFrame containing created_date and closed_date.

    Returns
    -------
    pd.DataFrame
        DataFrame with intake feature columns, created_date, and resolution_hours target.
    """
    data = df.copy()

    # Ensure datetime parsing
    for col in ["created_date", "closed_date"]:
        if col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = pd.to_datetime(data[col], errors="coerce", utc=True)

    # Compute target: duration in hours
    data["resolution_hours"] = (data["closed_date"] - data["created_date"]) / np.timedelta64(1, "h")

    # Filter out invalid, open, or extreme outliers (> 180 days / 4320 hours)
    valid_mask = data["resolution_hours"].between(0.01, 24 * 180)
    data = data.loc[valid_mask].copy()

    # Extract temporal features available at ticket creation
    data["created_hour"] = data["created_date"].dt.hour
    data["created_month"] = data["created_date"].dt.month
    data["created_weekday"] = data["created_date"].dt.dayofweek
    data["is_weekend"] = np.where(data["created_weekday"].isin([5, 6]), 1, 0)

    # Fill missing coordinates with median bounds or leave for imputer
    for num_col in ["latitude", "longitude"]:
        if num_col not in data.columns:
            data[num_col] = np.nan

    # Select only eligible features and target; completely exclude closed_date and status
    keep_cols = [c for c in ALL_FEATURES if c in data.columns] + ["created_date", "resolution_hours"]
    return data[keep_cols].dropna(subset=["resolution_hours"]).sort_values("created_date").reset_index(drop=True)


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """Constructs a scikit-learn ColumnTransformer for numerical and categorical imputation and encoding.

    Parameters
    ----------
    categorical_cols : list[str]
        List of categorical feature names.
    numeric_cols : list[str]
        List of numeric feature names.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor.
    """
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def build_model_pipeline(model, categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    """Encapsulates preprocessor and regression estimator in a single Pipeline object."""
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])


def evaluate_models():
    """Trains baseline and candidate regressors on chronological split and logs artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_clean = pd.read_csv(CLEAN_DATA_PATH)
    ml_data = make_ml_data(raw_clean)

    if len(ml_data) < 100:
        raise ValueError(f"Insufficient dataset size for ML training: {len(ml_data)} rows.")

    # Chronological Split: 80% train (earlier), 20% test (later)
    split_idx = int(len(ml_data) * 0.8)
    train_df = ml_data.iloc[:split_idx].copy()
    test_df = ml_data.iloc[split_idx:].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["resolution_hours"]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["resolution_hours"]

    # Define candidate models
    candidates = {
        "DummyMedian": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=10.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=15, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=60, min_samples_leaf=5, max_depth=16, random_state=42, n_jobs=-1),
    }

    # TimeSeriesSplit cross-validation on training data only (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        print(f"Training and cross-validating: {name}...")
        pipeline = build_model_pipeline(estimator, CATEGORICAL_FEATURES, NUMERIC_FEATURES)

        # Cross-validation MAE scores on training set
        cv_mae_scores = []
        for train_fold_idx, val_fold_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_fold_idx], X_train.iloc[val_fold_idx]
            y_tr, y_val = y_train.iloc[train_fold_idx], y_train.iloc[val_fold_idx]
            fold_pipe = build_model_pipeline(estimator, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
            fold_pipe.fit(X_tr, y_tr)
            val_pred = fold_pipe.predict(X_val)
            cv_mae_scores.append(mean_absolute_error(y_val, val_pred))

        mean_cv_mae = float(np.mean(cv_mae_scores))

        # Fit on full training set and evaluate on unseen chronological test set
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)

        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        test_r2 = float(r2_score(y_test, test_pred))

        results.append({
            "model": name,
            "CV_MAE_hours": round(mean_cv_mae, 3),
            "Test_MAE_hours": round(test_mae, 3),
            "Test_RMSE_hours": round(test_rmse, 3),
            "Test_R2": round(test_r2, 4),
        })
        fitted_pipelines[name] = pipeline

    comparison_df = pd.DataFrame(results).sort_values("Test_MAE_hours").reset_index(drop=True)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Select best performing candidate ML model (excluding naive baseline) based on performance
    ml_candidates = comparison_df[comparison_df["model"] != "DummyMedian"]
    best_model_name = ml_candidates.iloc[0]["model"] if not ml_candidates.empty else comparison_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_pipeline_path = MODELS_DIR / "resolution_pipeline.joblib"
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"\nSaved best model pipeline ({best_model_name}) to: {best_pipeline_path}")

    # Comprehensive Error Analysis on Held-Out Test Set
    test_analysis = test_df.copy()
    test_analysis["predicted_hours"] = best_pipeline.predict(X_test)
    test_analysis["absolute_error"] = (test_analysis["resolution_hours"] - test_analysis["predicted_hours"]).abs()
    test_analysis["residual"] = test_analysis["resolution_hours"] - test_analysis["predicted_hours"]

    # Top 100 worst prediction errors
    top_100_errors = test_analysis.sort_values("absolute_error", ascending=False).head(100)
    top_100_errors.to_csv(REPORTS_DIR / "largest_errors.csv", index=False)

    # Error breakdown across subgroups (Borough and Agency)
    borough_errors = test_analysis.groupby("borough").agg(
        sample_size=("absolute_error", "count"),
        median_actual_hours=("resolution_hours", "median"),
        median_predicted_hours=("predicted_hours", "median"),
        mae_hours=("absolute_error", "mean"),
    ).reset_index().sort_values("mae_hours")
    borough_errors.to_csv(REPORTS_DIR / "error_breakdown_by_borough.csv", index=False)

    agency_errors = test_analysis.groupby("agency").agg(
        sample_size=("absolute_error", "count"),
        mae_hours=("absolute_error", "mean"),
    ).reset_index().sort_values("sample_size", ascending=False).head(10)
    agency_errors.to_csv(REPORTS_DIR / "error_breakdown_by_agency.csv", index=False)

    # Diagnostic Visualizations: Residuals Plot and Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(test_analysis["predicted_hours"], test_analysis["residual"], alpha=0.3, color="#1976d2", s=18)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title(f"Residual Plot for {best_model_name} (Held-out Test)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Resolution Time (Hours)", fontsize=11)
    plt.ylabel("Residual: Actual - Predicted (Hours)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(test_analysis["resolution_hours"], test_analysis["predicted_hours"], alpha=0.3, color="#388e3c", s=18)
    max_val = min(test_analysis["resolution_hours"].quantile(0.99), test_analysis["predicted_hours"].quantile(0.99))
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="Perfect 1:1 Fit")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title(f"Actual vs Predicted Resolution Hours ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Resolution Duration (Hours)", fontsize=11)
    plt.ylabel("Predicted Resolution Duration (Hours)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # Save detailed Model Card
    model_card = {
        "model_name": best_model_name,
        "problem_type": "Tabular Regression",
        "target": "resolution_hours (created_date to closed_date duration in hours)",
        "train_set_size": len(train_df),
        "test_set_size": len(test_df),
        "evaluation_metrics": comparison_df.iloc[0].to_dict(),
        "temporal_split_strategy": "Chronological holdout: oldest 80% train, newest 20% test",
        "leakage_mitigation": "Target proxies (closed_date, status, resolution_action) excluded at intake",
        "intended_use": "Educational benchmarking and civic workload pattern analysis.",
        "out_of_scope_use": "Direct SLA enforcement, employee disciplinary evaluation, or emergency dispatch.",
        "limitations": [
            "Resolution times are heavily right-skewed; complex multi-agency cases under-predicted.",
            "Historical resolution speed may reflect staffing shortages in specific boroughs rather than case difficulty.",
            "Durations restricted to 0-180 days; extreme backlog outliers are excluded."
        ]
    }
    (REPORTS_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return comparison_df


if __name__ == "__main__":
    evaluate_models()
