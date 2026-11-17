# Dataset Schema Specification: NYC 311 Service Requests

**Dataset Source:** NYC Open Data / Socrata API (`erm2-nwe9`)  
**Source URL:** [https://data.cityofnewyork.us/resource/erm2-nwe9](https://data.cityofnewyork.us/resource/erm2-nwe9)  
**Permitted License:** Public Domain / Open Data NYC Terms of Use (Educational & Research Permitted)  
**Access Date:** November 2026  
**Raw Record Count:** 20,000 service request records  

---

## 1. Schema Definition & Data Dictionary

| Column Name | Raw Data Type | Cleaned Data Type | Description & Semantic Meaning | Units / Scale | Allowed / Valid Values | Missing Count (Raw) | Missing Pct (Raw) | Validation & Cleaning Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `unique_key` | `int64` / `string` | `string` | Unique identifier generated for each 311 incident submission. | Identifier | Positive integers | 0 | 0.0% | Checked for uniqueness; deduplicated on conflict. |
| `created_date` | `string` (ISO 8601) | `datetime64[ns, UTC]` | Official timestamp when the 311 incident was received by NYC dispatch. | Timestamp (UTC) | Valid calendar dates between 2010 and present | 0 | 0.0% | Parsed to UTC datetime; mandatory non-null constraint (threshold 95%). |
| `closed_date` | `string` (ISO 8601) | `datetime64[ns, UTC]` | Timestamp when the responding municipal agency resolved and closed the incident. | Timestamp (UTC) | Date >= `created_date` | Variable (~10-25% open) | ~15-20% | Parsed to UTC datetime. Records with `closed_date < created_date` are set to `pd.NaT`. |
| `agency` | `string` | `string` (Title Case) | Acronym code of the city agency responsible for resolving the complaint (e.g., NYPD, DOT, DOHMH). | Categorical Code | Valid agency codes (NYPD, DOT, DSNY, DEP, DOB, HPD, DOHMH, etc.) | 0 | 0.0% | Standardized to uppercase/titlecase; stripped whitespace. |
| `agency_name` | `string` | `string` (Title Case) | Full official legal name of the responding NYC municipal agency. | Categorical Name | City department names | 0 | 0.0% | Stripped and standardized; excluded from ML to avoid redundancy. |
| `complaint_type`| `string` | `string` (Title Case) | Primary classification category of the reported issue. | Categorical | Noise, Illegal Parking, Street Condition, Water System, Heat/Hot Water, etc. | 0 | 0.0% | Normalized to Title Case; stripped trailing spaces. |
| `descriptor` | `string` | `string` (Title Case) | Sub-category specifying the precise condition or nuance of the complaint. | Categorical Text | e.g., Loud Music/Party, Pothole, Hydrant Running, Blocked Sidewalk | ~500 | ~2.5% | Empty strings or nulls imputed with `"Not Provided"`. |
| `borough` | `string` | `string` (Title Case) | New York City administrative borough where the incident took place. | Categorical | Bronx, Brooklyn, Manhattan, Queens, Staten Island | ~100 | ~0.5% | Standardized to Title Case; 'Unspecified' and nulls mapped to `"Unknown Borough"`. |
| `status` | `string` | `string` (Title Case) | Lifecycle state of the request. | Categorical State | Closed, Open, Pending, Assigned, In Progress | 0 | 0.0% | Standardized. Excluded from predictive ML baseline to prevent leakage. |
| `latitude` | `float64` | `float64` | WGS84 geographic latitude coordinate of the reported incident location. | Decimal Degrees | [40.40, 41.00] (Valid NYC Geographic Bounding Box) | ~600 | ~3.0% | Out-of-bounds numbers (>41.00 or <40.40) set to `np.nan`. Imputed with median in ML pipeline. |
| `longitude` | `float64` | `float64` | WGS84 geographic longitude coordinate of the reported incident location. | Decimal Degrees | [-74.30, -73.65] (Valid NYC Geographic Bounding Box) | ~600 | ~3.0% | Out-of-bounds numbers (<-74.30 or >-73.65) set to `np.nan`. Imputed with median in ML pipeline. |

---

## 2. Derived Feature Schema (Engineered for ML Pipeline)

| Derived Column Name | Data Type | Formula / Origin | Semantic Meaning | Usage |
| :--- | :--- | :--- | :--- | :--- |
| `resolution_hours` | `float64` | `(closed_date - created_date) / 1 hour` | Duration between incident reporting and resolution in elapsed hours. | Target Variable in Part B ML Regression. |
| `created_hour` | `int64` | `created_date.dt.hour` | Hour of submission (0 to 23, UTC). Captures diurnal operational shifts. | Feature (Numeric) |
| `created_month` | `int64` | `created_date.dt.month` | Month of submission (1 to 12). Captures seasonal trends. | Feature (Numeric) |
| `created_weekday` | `int64` | `created_date.dt.dayofweek` | Day of the week (0 = Monday, ..., 6 = Sunday). | Feature (Numeric) |
| `is_weekend` | `int64` | `1 if created_weekday in [5, 6] else 0` | Binary indicator of weekend dispatch latency. | Feature (Binary) |
| `coordinates_valid` | `bool` | `latitude.notna() & longitude.notna()` | Indicator of high-precision geospatial geocoding. | Part A Summary & Validation |
| `day_part` | `string` | NumPy `np.select` over `created_hour` | Categorical partition: Morning, Afternoon, Evening, Night. | Part A Summary |

---

## 3. Data Integrity & Loud Validation Rules

1. **Primary Key Uniqueness:** Every service request must possess a unique `unique_key`. Deduplication removes retransmitted packets.
2. **Temporal Order Constraint:** A ticket cannot close before it was created. Any instance where `closed_date < created_date` violates causality and is set to null.
3. **Geographic Bounding Box:** NYC spans latitudes 40.40 to 41.00 and longitudes -74.30 to -73.65. Sensor glitches or default nulls (e.g. 0.0, 99.99) outside this box are filtered.
4. **Target Leakage Prohibition:** Features derived after incident intake (`closed_date`, `status`, `resolution_action_updated_date`) are strictly quarantined from model input matrices.
