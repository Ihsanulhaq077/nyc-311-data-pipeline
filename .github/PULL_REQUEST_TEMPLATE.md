## Pull Request Description
Please provide a clear, concise summary of the changes proposed in this pull request and the motivation behind them.

- **Component Affected:** [ ] Data Ingestion / Pipeline A  [ ] ML Modeling / Pipeline B  [ ] Systems Exploration / Part C  [ ] Tests & Documentation
- **Related Issue / Checkpoint:** Checkpoint #_____

### Summary of Changes
- 

### Why is this Change Necessary?
- 

---

## Verification & Test Plan
How were these changes tested? Please attach commands, sample outputs, or test results.

- [ ] Unit tests added/updated in `tests/`
- [ ] Pytest passes cleanly (`pytest -v`)
- [ ] Data quality gate evaluated against raw and corrupt samples
- [ ] No target leakage introduced in training or evaluation code
- [ ] Tested from a clean workspace / environment

```bash
# Reproduction command used:
pytest -v
```

---

## Formal Code Review Checklist (For Reviewers)
*Every reviewer must verify each of the following points before approving merge:*

### 1. Correctness & Integrity
- [ ] Does the code run without unhandled exceptions or hidden side effects?
- [ ] Are raw data files strictly read-only and preserved without mutation?
- [ ] Are numerical and date boundary conditions explicitly validated?

### 2. Machine Learning & Data Leakage
- [ ] Is the train/test split strictly chronological and performed *prior* to fitting preprocessors or models?
- [ ] Are post-creation attributes (`closed_date`, `status`, `resolution_action`) completely excluded from feature sets?
- [ ] Are evaluation metrics robust to skewness (e.g., MAE instead of sole reliance on MSE/accuracy)?

### 3. Maintainability, Architecture & Style
- [ ] Does every `.py` file start with the required student header (Name, Roll No, Date)?
- [ ] Does every function have an explanatory docstring detailing parameters, return values, and exceptions?
- [ ] Are variable and function names self-documenting and descriptive?

### 4. Documentation & Reproducibility
- [ ] Are schema assumptions and model limitations documented in `docs/`?
- [ ] Does the `README.md` provide clear, copy-pasteable execution commands?
- [ ] Are figures and summary tables properly saved to `outputs/` with labelled units?
