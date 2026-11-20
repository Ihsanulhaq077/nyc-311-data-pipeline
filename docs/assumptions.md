# Project Assumptions, Limitations, Risk Analysis & Model Card

**Project:** NYC 311 Civic Incident Duration Forecasting  
**Author:** Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: November 20, 2026  

---

## 1. Data Cleaning Decisions, Assumptions & Omissions

### What Was Cleaned and Why
1. **Timestamp Normalization & UTC Coercion:**
   - Raw records contain inconsistent ISO 8601 string timestamps. All `created_date` and `closed_date` values are explicitly parsed into UTC datetimes. This ensures accurate duration computations without clock-drift anomalies.
2. **Standardization of Categorical Attributes:**
   - Categorical fields (`complaint_type`, `agency`, `borough`, `descriptor`) suffered from irregular casing (e.g. `nypd`, `NYPD`, `Nypd`), surrounding whitespace, and synthetic placeholders (`"Unspecified"`, `"None"`, `""`). These were cleaned into Title Case, and pseudo-null placeholders were coerced to genuine missing indicators (`pd.NA`).
3. **Causality & Temporal Order Correction:**
   - In rare operational edge cases, data logging errors caused `closed_date` to precede `created_date`. These represent recording corruption; rather than dropping the entire record, `closed_date` was set to `pd.NaT`, preserving the complaint for descriptive intake summaries while excluding it from duration training.
4. **Geospatial Bounding Box Truncation:**
   - Coordinate sensors or network defaults occasionally emit `(0.0, 0.0)` or `(99.99, 99.99)`. We bounded latitude to `[40.40, 41.00]` and longitude to `[-74.30, -73.65]`. Outliers are reset to `NaN` and later imputed via median in the ML pipeline.

### What Was Chosen NOT to Fix (and Justification)
1. **Open Requests Without a `closed_date`:**
   - Approximately 10–20% of service requests are currently in progress or assigned. We chose **not** to impute closure dates for these tickets. Imputing an arbitrary closure date (such as data extraction date) would introduce severe survival-bias artifacts into our regression targets. Instead, open requests are analyzed in Part A volume summaries but naturally excluded from the supervised target `resolution_hours` in Part B.
2. **Missing Street-Level Descriptors:**
   - Roughly 2.5% of records have blank descriptor fields. We chose not to drop these rows because doing so would systematically bias against agencies that do not mandate sub-category categorization. Instead, missing descriptors are imputed with a `"Not Provided"` token in the categorical pipeline.
3. **High-Duration Right-Tail Truncation at 180 Days (Analytical Boundary):**
   - We filtered out tickets taking longer than 180 days (4,320 hours) for model training. While certain complex infrastructure cases (e.g., major capital water main reconstructions) legitimately take over six months, they follow unique administrative workflows distinct from routine municipal dispatch. Treating them as standard tabular rows causes severe gradient distortion in linear and tree estimators.

---

## 2. Machine Learning Problem Framing & Leakage Mitigation

### Target Variable Definition
- **Target:** `resolution_hours` — continuous elapsed time in decimal hours between `created_date` and `closed_date`.
- **Target Distribution:** Heavily right-skewed with a median of ~4 to 12 hours (depending on agency) and a long tail extending into hundreds of hours.

### Leakage Identification & Mitigation Strategy
In tabular machine learning for civic operations, data leakage is an acute risk:
- **Columns Excluded to Prevent Target Leakage:**
  - `closed_date`: Directly forms the target mathematical formula; including it or any derivative allows the model to calculate the answer trivially.
  - `status`: Labels such as `"Closed"`, `"Pending"`, or `"Assigned"` correlate directly with resolution duration and are updated dynamically throughout the ticket lifecycle.
  - `resolution_action_updated_date`: Emitted when dispatch teams update their logs.
  - `agency_name`: Perfectly collinear with `agency` code; excluded to avoid feature redundancy.
- **Allowed Features (Known Strictly at Intake):**
  - Incident category: `complaint_type`, `descriptor`
  - Responding department: `agency`
  - Geography: `borough`, `latitude`, `longitude`
  - Intake time context: `created_hour`, `created_month`, `created_weekday`, `is_weekend`

---

## 3. Train/Test Split Justification

We reject random K-fold splitting in favor of a **strict Chronological Holdout Split** (oldest 80% train, newest 20% test):
1. **Temporal Causality:** In real municipal deployments, models trained on historical requests must predict incoming, future requests. A random split permits future information to leak into past predictions, artificially deflating error metrics.
2. **Seasonality & Concept Drift:** Staffing patterns, weather anomalies, and mayoral policy directives change over time. A temporal holdout accurately measures how well the model generalizes across shifting operational conditions.
3. **Cross-Validation Scheme:** Within the 80% training set, we use `TimeSeriesSplit(n_splits=3)` so that hyperparameter tuning and model validation also strictly respect the arrow of time.

---

## 4. Evaluation Metric Justification

We evaluate candidate models using **Mean Absolute Error (MAE)** as our primary loss and selection metric, supported by Root Mean Squared Error (RMSE) and $R^2$:
- **Why MAE is Primary:** Resolution hours exhibit heavy tails with large positive outliers. RMSE squares the residuals, disproportionately penalizing rare multi-week delays and forcing models to systematically over-predict typical tickets. MAE provides an intuitive, robust interpretation: *"On average, the model's estimate is off by X hours."*
- **Role of RMSE:** Tracked to monitor sensitivity to large forecast misses.
- **Role of $R^2$:** Measures the proportion of variance explained relative to a naive mean baseline.

---

## 5. Error Analysis & Concrete Failure Patterns

Inspection of the largest 100 absolute prediction errors (`outputs/reports/largest_errors.csv`) and subgroup breakdowns revealed distinct systematic failure modes:
1. **The "Inter-Agency Referral" Failure Mode:**
   - Complaints initially filed under `DOT` (e.g., Street Condition) that turn out to require major underground utility repair by `DEP` or private utility providers experience administrative handoff latency. The model, unaware of downstream internal referrals, severely underestimates duration.
2. **Noise Complaint Bimodality:**
   - For `NYPD` residential noise complaints, tickets resolved during the night shift are typically handled within 1–3 hours. However, complaints filed on Sunday afternoon during peak shift turnover sometimes linger for 36+ hours before being batch-closed. The model predicts the median (~4 hours), producing substantial residuals on delayed cases.
3. **Borough Disparities:**
   - As documented in `outputs/reports/error_breakdown_by_borough.csv`, prediction error varies across boroughs. Higher density areas (e.g., Manhattan and Brooklyn) have denser staffing and faster median resolution, whereas outlying districts in Queens and Staten Island exhibit higher variance due to broader transit geography.

---

## 6. Official Model Card

### Intended Use
- **Primary Domain:** Civic operations research, educational demonstration of tabular preprocessing pipelines, and exploratory analysis of municipal service demand.
- **Intended Users:** Data science students, municipal analysts exploring general workload patterns.

### Out-of-Scope & Prohibited Uses
- **Prohibited:** Real-time emergency dispatching (911/fire/EMS).
- **Prohibited:** Disciplinary evaluations or automated KPI scoring of individual city workers or field inspectors.
- **Prohibited:** Public-facing SLAs that guarantee completion times to citizens.

### Ethical & Fairness Risks
- **Historical Bias Propagation:** Service resolution speed is historically correlated with municipal resource allocation. If lower-income census tracts historically received fewer inspection teams, an uncalibrated model may learn to predict longer resolution times for these areas, reinforcing unequal service expectations.
- **Mitigation:** The model should include disparity auditing across geographic zones and never be used to deprioritize requests based on predicted delay.
