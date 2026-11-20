# Reproducible Data Quality Pipeline, Tabular ML System & Systems Exploration

**Student:** Ihsan Ul Haq  
**Roll Number:** TRN-2026-000951  
**Submission Date:** November 20, 2026  
**Course Assignment:** Python Mastery Assignment 2 — Reproducible Data Pipelines, Tabular ML & Systems Exploration  

---

## Executive Overview
This repository delivers an end-to-end, production-grade data engineering and applied machine learning system built on real-world NYC 311 Service Requests (20,000 raw operational records). The project is organized into three distinct parts:
1. **Part A (Data Quality & Insights Pipeline):** Schema extraction, modular cleaning and validation functions, Pandas/NumPy vectorized transformations, 4 publication-quality visualizations, SQLite analytical aggregation, and a loud automated quality gate.
2. **Part B (End-to-End Tabular ML Baseline):** Civic duration prediction framed without leakage, chronological train/test holdout, scikit-learn `Pipeline` + `ColumnTransformer`, TimeSeriesSplit cross-validation, baseline vs. 3 candidate models, residual and subgroup error analysis, and a saved production pipeline.
3. **Part C (Systems Exploration):** Four comprehensive engineering investigations covering social feed query optimization (C1), CI/CD pipelines & code review practices (C2), system design theory and CAP trade-offs (C3), and a repeatable system design workflow with an SDD artifact (C4).
4. **All Three Bonus Challenges (+6 Marks):**
   - **Bonus A (+2):** Automated Data Quality Gate failing loudly on corrupt data and passing on valid data.
   - **Bonus B (+2):** Standalone batch CLI inference script (`predict.py`) with permutation feature importance analysis.
   - **Bonus C (+2):** Algorithmic social media feed ranker (`feed_ranker.py`) with multi-tier benchmark.

---

## Dataset Provenance & Proposal
- **Dataset:** New York City 311 Service Requests
- **Source:** NYC Open Data / Socrata Open Data API (`erm2-nwe9`)
- **API Endpoint:** `https://data.cityofnewyork.us/resource/erm2-nwe9.csv`
- **License:** Open Data NYC Terms of Use / Creative Commons Public Domain (Permitted for educational and research usage)
- **Access Date:** November 20, 2026
- **Raw Dimensions:** 20,000 rows × 11 selected operational columns (plus geospatial identifiers)
- **Why It Fits:** Sourced directly from live city dispatch logs; exhibits visible real-world dirty data (missing coordinates, irregular casing, placeholder nulls, inverted timestamps); provides a realistic regression target (`resolution_hours`).

---

## Repository Directory Layout

```text
assignment_project/
├── .github/
│   ├── workflows/ci.yml             # C2: GitHub Actions CI workflow (linting, tests, benchmarks)
│   ├── PULL_REQUEST_TEMPLATE.md     # C2: Structured PR template and formal code review checklist
│   └── CODEOWNERS                   # C2: Code ownership assignments
├── data/
│   ├── raw/                         # Original, pristine raw data (never mutated)
│   └── cleaned/                     # Pipeline output: sanitized, type-safe dataset
├── docs/
│   ├── schema.md                    # Data dictionary, types, allowed ranges, and missing counts
│   ├── assumptions.md               # Cleaning assumptions, leakage mitigation, error analysis & Model Card
│   └── exploration.md               # Part C write-ups (C1, C2, C3, C4) + System Design Document
├── models/
│   └── resolution_pipeline.joblib   # Serialized scikit-learn preprocessing + best regressor pipeline
├── outputs/
│   ├── figures/                     # 3+ Visualizations with labelled axes and units
│   │   ├── borough_counts.png
│   │   ├── top_complaint_types.png
│   │   ├── requests_by_hour.png
│   │   ├── residuals_plot.png
│   │   └── actual_vs_predicted.png
│   └── reports/                     # Data quality metrics, SQLite DB, model comparison, error reports
│       ├── quality_metrics.json
│       ├── model_comparison.csv
│       ├── largest_errors.csv
│       ├── nyc311.sqlite
│       └── feed_ranker_benchmark.txt
├── sql/
│   ├── borough_counts.sql           # SQLite query: request counts by borough
│   └── borough_agency_summary.sql   # SQLite query: multi-metric analytical aggregation
├── src/
│   ├── __init__.py
│   ├── download_data.py             # Reproducible data ingestion from Socrata API
│   ├── pipeline_a.py                # Part A: cleaning, validation, summaries, figures, quality gate
│   ├── pipeline_b.py                # Part B: ML preprocessing, temporal split, modeling, error analysis
│   ├── predict.py                   # Bonus B: standalone CLI inference and feature importance
│   ├── feed_experiment.py           # Part C1: SQLite query plan and timing experiment
│   └── feed_ranker.py               # Bonus C: algorithmic feed ranker with LRU cache benchmark
├── tests/
│   ├── test_pipeline_a.py           # Unit tests for all cleaning, validation functions, and quality gate
│   └── test_ml_data.py              # Unit tests for ML feature extraction, leakage prevention, pipeline
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Master setup, findings, and documentation
```

---

## Setup & Quickstart Guide

### 1. Environment Creation
Python 3.10 or 3.11 is recommended. Create and activate a clean virtual environment:

```bash
# Clone or navigate to the repository
cd assignment_project

# Create a virtual environment
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Reproduce the Full System (Step-by-Step)

Execute the following commands sequentially from the project root:

### Step 1: Download Pristine Raw Data
```bash
python -m src.download_data
```
*Saves 20,000 raw operational records to `data/raw/nyc_311.csv`. The raw file is strictly read-only and never modified.*

### Step 2: Execute Part A Data Quality & Insights Pipeline
```bash
python -m src.pipeline_a
```
- Inspects raw data and writes schema to `docs/schema.csv` and `docs/schema.md`.
- Applies modular cleaning: timestamp parsing, category normalization, coordinate outlier filtering, temporal inversion handling.
- Runs data validation checks and enforces the **Loud Quality Gate** (threshold = 85%).
- Executes NumPy vectorized operations (coordinate distance/validity, duration calculation, day-part classification).
- Generates 4 figures in `outputs/figures/`.
- Loads data into `outputs/reports/nyc311.sqlite` and executes analytical SQL queries.
- Writes comprehensive metrics to `outputs/reports/quality_metrics.json`.

### Step 3: Execute Part B Tabular Machine Learning Pipeline
```bash
python -m src.pipeline_b
```
- Extracts target `resolution_hours` without target leakage.
- Enforces strict chronological 80/20 train/test split.
- Builds an isolated `ColumnTransformer` (median imputation + scaling for numerics; constant imputation + one-hot encoding for categoricals).
- Conducts `TimeSeriesSplit` cross-validation on training data for baseline and 3 candidate models.
- Evaluates models on held-out test set, logs metrics to `outputs/reports/model_comparison.csv`.
- Saves the best trained pipeline to `models/resolution_pipeline.joblib`.
- Conducts subgroup error analysis, saves `largest_errors.csv`, and generates diagnostic residual plots.
- Exports the Model Card to `outputs/reports/model_card.json`.

### Step 4: Run the Complete Test Suite
```bash
pytest -v
```
*Runs all 15+ unit and integration tests across data cleaning, validation constraints, loud quality gate failure/pass behavior, ML feature extraction, and pipeline prediction.*

### Step 5: Run Part C Systems Exploration & Bonus Benchmarks
```bash
# Part C1: SQLite Indexing and EXPLAIN QUERY PLAN Experiment
python -m src.feed_experiment

# Bonus C: Algorithmic Feed Ranker with Multi-tier Cache Benchmark
python -m src.feed_ranker
```

### Step 6: Test Bonus B Standalone Prediction Script
```bash
# Run prediction on sample rows and observe feature importance
python -m src.predict data/cleaned/nyc_311_clean.csv --output outputs/reports/sample_predictions.csv
```

---

## Part A: Key Findings & Data Quality Metrics

### Summary Metrics (`quality_metrics.json`)
- **Total Records Ingested:** 20,000 rows
- **Total Records After Deduplication & Filtering:** Cleaned and saved to `data/cleaned/nyc_311_clean.csv`
- **Validation Pass Rate:** **100%** on cleaned data (Quality Gate Threshold: **85%**)
- **Quality Gate Status:** **PASSED** (failing test demonstrated on corrupt sample in `tests/test_pipeline_a.py`)

### Visualizations Generated (`outputs/figures/`)
1. **`borough_counts.png`:** Displays total service request volume across the five NYC boroughs. *Finding:* Brooklyn and Queens generate the highest municipal request volume, while Staten Island represents the smallest share.
2. **`top_complaint_types.png`:** Horizontal bar chart of the top 15 complaint types. *Finding:* Noise (Residential), Illegal Parking, and Blocked Driveway dominate civic intake, comprising over 40% of all submitted tickets.
3. **`requests_by_hour.png`:** Line chart illustrating the 24-hour diurnal submission curve (in UTC). *Finding:* Submissions peak during afternoon and evening hours (14:00 - 23:00 UTC) and reach minimum volume during early dawn hours.
4. **`agency_resolution_hours.png`:** Bar chart comparing median resolution duration across primary city agencies (NYPD, DOT, DEP, DOHMH). *Finding:* NYPD handles incidents fastest (median < 4 hours, primarily noise and parking), whereas specialized infrastructure agencies (DOT and DEP) exhibit longer median turnaround times.

### SQLite Analytical Query Result (`outputs/reports/sqlite_borough_summary.csv`)
The following SQL aggregation was executed on `outputs/reports/nyc311.sqlite`:
```sql
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
```
*Analytical Takeaway:* Geographic coordinate completeness exceeds 97% across all boroughs. Average resolution times are lowest in Manhattan and highest in Staten Island and the Bronx, reflecting differing travel times and depot densities.

---

## Part B: Tabular Machine Learning Baseline & Comparison

### Problem Framing
- **Objective:** Estimate civic ticket turnaround duration (`resolution_hours`) at intake.
- **Audience:** Municipal operational planners estimating queue backlog.
- **Leakage Prevention:** Strictly excluded `closed_date`, `status`, and downstream resolution notes.

### Model Benchmark Comparison (`outputs/reports/model_comparison.csv`)

| Model | Cross-Val MAE (Hours) | Test MAE (Hours) | Test RMSE (Hours) | Test $R^2$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | **18.42** | **17.85** | **42.10** | **0.284** | **Best candidate**: Captures non-linear agency/time interactions effectively. |
| **RandomForest** | 20.15 | 19.30 | 44.52 | 0.221 | Strong non-linear modeling; higher memory footprint. |
| **Ridge Regression** | 24.80 | 23.65 | 48.90 | 0.115 | Underfits high non-linearity in geographic coordinate clusters. |
| **Dummy (Median Baseline)** | 28.50 | 27.90 | 54.20 | -0.020 | Naive baseline; predicts median duration across all tickets. |

*Why MAE is Primary:* Because municipal resolution durations are heavily right-skewed with long tails, RMSE squares large errors and distorts typical performance. MAE offers an intuitive metric: our gradient boosting pipeline is off by an average of ~17.8 hours on unseen requests.

---

## Part C: Systems Exploration Summary

- **C1 (Optimized Social Feed Queries):** Detailed analysis of candidate retrieval, push vs. pull fan-out architectures, keyset pagination, and SQLite composite indexing. Our benchmark demonstrated a dramatic speedup with composite indexing and an additional **100x speedup** with an in-memory LRU cache.
- **C2 (CI/CD & Code Reviews):** Documented production GitHub Actions architecture, branch protection rules, `.github/CODEOWNERS`, and an engineering pull request template with a verification checklist.
- **C3 (System Design Theory):** Comprehensive 10-term glossary with concrete production examples, and an architectural trade-off analysis explaining why NYC 311 selects Eventual Consistency & High Availability (AP) under the CAP/PACELC theorem.
- **C4 (Design Workflow & SDD Artifact):** A 10-step repeatable system design sequence, accompanied by a complete one-page System Design Document (SDD) for the *NYC 311 Real-Time Civic Incident Intake & Dispatch Notification Pipeline*.

---

## Bonus Challenges Checklist (+6 Marks)

- [x] **Bonus A (+2): Data Quality Gate with Corrupt vs. Real Testing**
  - Implemented in `src/pipeline_a.py` (`DataQualityGateError`).
  - Automated tests in `tests/test_pipeline_a.py` (`test_quality_gate_passes_on_valid_data` and `test_quality_gate_fails_loudly_on_corrupt_data`).
  - Metrics continuously output to `outputs/reports/quality_metrics.json`.
- [x] **Bonus B (+2): Standalone Batch Inference Script (`predict.py`)**
  - Fully featured CLI: `python -m src.predict <input_csv> --output <output_csv>`.
  - Performs schema validation and missing column detection before inference.
  - Automatically derives temporal features (`created_hour`, `is_weekend`) if raw timestamps are supplied.
  - Generates domain feature importance commentary explaining why `agency` and `complaint_type` dominate civic duration predictions.
- [x] **Bonus C (+2): Algorithmic Social Feed Ranker (`feed_ranker.py`)**
  - Implemented gravity engagement score: $\text{Score} = \frac{\text{Likes} + 1}{(\text{AgeInHours} + 2)^{1.5}}$.
  - Ranks top 10 personalized feed posts for User #1.
  - Multi-tier latency benchmark comparing unindexed scan, composite index, and in-memory LRU cache saved to `outputs/reports/feed_ranker_benchmark.txt`.

---

## AI & Tool Disclosure
An AI coding assistant (Google DeepMind Antigravity) was used for scaffolding project architecture, validating syntax, and formatting markdown tables. All code, analytical justifications, error inspection patterns, and system design documents were reviewed, verified, and understood by the student.
