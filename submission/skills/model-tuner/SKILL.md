---
name: model-tuner
description: >-
  Batch-trains a spread of gradient-boosting hyperparameter configs (LightGBM,
  XGBoost, CatBoost) with cross-validation in one call, instead of training
  and checking one model at a time.
---

### 1. `run_grid.py`

Trains 7 hyperparameter "bands" (shallow-conservative through deep-aggressive,
plus a baseline) for each requested model family, using stratified k-fold
cross-validation, across one or more random seeds. Reports a compact,
sorted results table and writes ready-to-submit test predictions for
*every* config — no further training code needs to be written.

**First call — keep it small, so you get a result back fast regardless of how large or slow the real dataset is:**
```python
run_skill_script(
    skill_name="model_tuner",
    script_name="run_grid.py",
    args="--train train.csv --test test.csv --target target --id-col id --model-types lightgbm --n-folds 3 --seeds 0 --max-seconds 150",
)
```
**Submit the top result from this call immediately** (see system prompt Phase 1) before making any wider call like this second one:
```python
run_skill_script(
    skill_name="model_tuner",
    script_name="run_grid.py",
    args="--train train_engineered.csv --test test_engineered.csv --target target --id-col id --model-types lightgbm,xgboost --n-folds 5 --seeds 0,1 --max-seconds 240",
)
```

**Arguments**:
- `--train` / `--test`: Paths to CSVs (use the feature-engineered versions if you ran the `feature-engineer` skill first, or the raw files — boosting models handle missing values natively, so imputation is optional).
- `--target`: Target column name.
- `--id-col`: ID column to carry through into the output prediction files, if `sample_submission.csv` has one.
- `--model-types`: Comma-separated subset of `lightgbm`, `xgboost`, `catboost`. Two families x 7 bands = 14 configs is a good *wider* pass once you already have a submission in — add `catboost` if the data looks categorical-heavy.
- `--n-folds`: Stratified k-fold count (default 5; use more on small datasets, fewer for a fast first pass).
- `--seeds`: Comma-separated seeds, averaged together — don't trust a single seed's ranking on small validation sets.
- `--max-seconds`: Hard wall-clock cap (default 240). The script **never blocks past this** — it skips remaining configs and returns results for whatever finished, so a slow/large dataset can't silently eat your whole session. Lower this for the fast first call.

**Outputs**:
- `grid_results.json`: every config's mean/std cross-validated AUC, sorted best-first.
- `grid_oof_predictions.csv`: out-of-fold predictions per config — use this to check *correlation* between your top candidates before ensembling (see `resources/tuning_playbook.md`).
- `grid_test_<config_id>.csv`: full-data-fit test predictions per config, already in submission format if `--id-col` was supplied.

**Golden rule**: prefer this skill over writing your own training script by hand. Hand-written training code is the most common source of harness failures — this script is already tested and deterministic. Only write custom code for something this script genuinely can't do.

---

## Domain Knowledge Resources

### `tuning_playbook.md`

Guidance on reading the results, refining around winners, and building a
robust ensemble. Read it with:
```python
load_skill_resource(
    skill_name="model_tuner",
    resource_name="tuning_playbook.md",
)
```
