---
name: feature-engineer
description: >-
  Provides a robust Python script for automated feature generation.
---

### 1. `generate_features.py`

Automatically identifies column types, imputes missing values, and calculates
a row-wise mean feature.

**Usage via `run_skill_script`**:
```python
run_skill_script(
    skill_name="feature_engineer",
    script_name="generate_features.py",
    args="--train train.csv --test test.csv --target target",
)
```

**Arguments**:
- `--train`: Path to training CSV (default: `train.csv`).
- `--test`: Path to test CSV (default: `test.csv`).
- `--target`: Name of the target column (default: `target`).

**Outputs**: Creates `train_engineered.csv` and `test_engineered.csv`.

---

## Domain Knowledge Resources

### `leakage_checklist.md`

A concise guide on preventing data leakage during feature engineering. Read
it using the `load_skill_resource` tool:
```python
load_skill_resource(
    skill_name="feature_engineer",
    resource_name="leakage_checklist.md",
)
```
