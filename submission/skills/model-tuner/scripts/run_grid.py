#!/usr/bin/env python3
"""Batch hyperparameter grid search across gradient-boosting model families.

Trains a whole spread of candidate configs with stratified k-fold CV in ONE
call, instead of the agent spending one LLM turn per model. Prints a compact
results table and writes full-data-fit test predictions for every config, so
the agent can compare a whole landscape and pick/ensemble without writing any
more training code itself.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Generic hyperparameter "bands" — spread wide across shallow/medium/deep,
# conservative/aggressive, rather than fine-tuning one corner. Depth and
# num_leaves are kept consistent (num_leaves < 2**depth) to avoid the classic
# overfitting trap of setting both high independently.
BANDS = [
    {"name": "baseline", "depth": 6, "lr": 0.05, "reg": 0.0, "min_frac": 0.01, "leaf_frac": 0.5},
    {"name": "shallow_a", "depth": 3, "lr": 0.03, "reg": 2.0, "min_frac": 0.05, "leaf_frac": 0.5},
    {"name": "shallow_b", "depth": 5, "lr": 0.03, "reg": 1.0, "min_frac": 0.03, "leaf_frac": 0.5},
    {"name": "medium_a", "depth": 6, "lr": 0.05, "reg": 0.3, "min_frac": 0.02, "leaf_frac": 0.5},
    {"name": "medium_b", "depth": 8, "lr": 0.05, "reg": 0.1, "min_frac": 0.015, "leaf_frac": 0.5},
    {"name": "deep_a", "depth": 10, "lr": 0.08, "reg": 0.0, "min_frac": 0.005, "leaf_frac": 0.4},
    {"name": "deep_b", "depth": 12, "lr": 0.10, "reg": 0.0, "min_frac": 0.005, "leaf_frac": 0.3},
]

EARLY_STOPPING_PATIENCE = 80
MAX_ESTIMATORS = 2000


def build_model(model_type, band, n_rows, seed):
    depth = band["depth"]
    lr = band["lr"]
    reg = band["reg"]
    min_samples = max(2, int(n_rows * band["min_frac"]))
    num_leaves = max(4, int(band["leaf_frac"] * (2 ** depth)))

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=MAX_ESTIMATORS, max_depth=depth, num_leaves=num_leaves,
            learning_rate=lr, reg_alpha=reg, reg_lambda=reg,
            min_child_samples=min_samples, feature_fraction=0.8,
            random_state=seed, verbosity=-1,
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=MAX_ESTIMATORS, max_depth=depth, learning_rate=lr,
            reg_alpha=reg, reg_lambda=max(reg, 1.0), min_child_weight=min_samples,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            eval_metric="auc", early_stopping_rounds=EARLY_STOPPING_PATIENCE,
        )
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=MAX_ESTIMATORS, depth=min(depth, 12), learning_rate=lr,
            l2_leaf_reg=max(reg, 1.0), min_data_in_leaf=min_samples,
            random_seed=seed, verbose=False,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_with_early_stopping(model, model_type, X_tr, y_tr, X_val, y_val):
    if model_type == "lightgbm":
        import lightgbm as lgb
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False)])
    elif model_type == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  early_stopping_rounds=EARLY_STOPPING_PATIENCE)
    return model


def main():
    parser = argparse.ArgumentParser(description="Batch-train a hyperparameter grid.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--id-col", default=None, help="ID column to carry through to predictions, if any")
    parser.add_argument("--model-types", default="lightgbm,xgboost",
                         help="Comma-separated: lightgbm,xgboost,catboost")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1", help="Comma-separated seeds; results are averaged across them")
    parser.add_argument("--out-prefix", default="grid")
    parser.add_argument("--max-seconds", type=int, default=240,
                         help="Hard wall-clock cap for this call (default 240 = 4 minutes). "
                              "If exceeded mid-grid, remaining configs are skipped and results "
                              "are written for whatever finished — this call NEVER blocks past "
                              "this cap, protecting your session's time budget.")
    args = parser.parse_args()
    start_time = time.time()

    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    id_col = args.id_col
    test_ids = test_df[id_col] if id_col and id_col in test_df.columns else None

    y = train_df[args.target].values
    drop_cols = [args.target] + ([id_col] if id_col else [])
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    X_test = X_test[[c for c in X.columns if c in X_test.columns]]
    X = X[X_test.columns]

    # Auto-detect categoricals and encode — never assume column identities,
    # since this must generalize to unseen datasets from the same family.
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols:
        combined = pd.concat([X[c], X_test[c]], axis=0).astype("category")
        X[c] = pd.Categorical(X[c], categories=combined.cat.categories).codes
        X_test[c] = pd.Categorical(X_test[c], categories=combined.cat.categories).codes

    n_rows = len(X)
    results = []
    oof_store = {}
    test_pred_store = {}

    configs = [(mt, band) for mt in model_types for band in BANDS]
    print(f"Training {len(configs)} configs x {len(seeds)} seed(s) "
          f"= {len(configs) * len(seeds)} model fits, {args.n_folds}-fold CV each "
          f"(hard cap: {args.max_seconds}s)...")

    skipped = []
    for model_type, band in configs:
        elapsed = time.time() - start_time
        if elapsed > args.max_seconds:
            skipped.append(f"{model_type}_{band['name']}")
            continue

        config_id = f"{model_type}_{band['name']}"
        seed_aucs = []
        oof_pred = np.zeros(n_rows)
        test_preds_this_config = []

        for seed in seeds:
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
            seed_oof = np.zeros(n_rows)
            for tr_idx, val_idx in skf.split(X, y):
                model = build_model(model_type, band, n_rows, seed)
                model = fit_with_early_stopping(
                    model, model_type,
                    X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx],
                )
                seed_oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
            seed_aucs.append(roc_auc_score(y, seed_oof))
            oof_pred += seed_oof / len(seeds)

            final_model = build_model(model_type, band, n_rows, seed)
            final_model.fit(X, y)
            test_preds_this_config.append(final_model.predict_proba(X_test)[:, 1])

        # cv_auc_mean/std are computed across seeds (each already averaged over
        # all folds via the OOF array) — this measures seed-to-seed stability,
        # which is what the tuning playbook's "don't trust one seed" tip is
        # actually about. Use --seeds with 2+ values for a meaningful std.
        cv_mean = float(np.mean(seed_aucs))
        cv_std = float(np.std(seed_aucs))
        oof_store[config_id] = oof_pred
        test_pred_store[config_id] = np.mean(test_preds_this_config, axis=0)

        results.append({
            "config_id": config_id, "model_type": model_type, "band": band["name"],
            "cv_auc_mean": round(cv_mean, 5), "cv_auc_std": round(cv_std, 5),
        })
        print(f"  {config_id:28s} CV AUC = {cv_mean:.5f} (+/- {cv_std:.5f})")

    results.sort(key=lambda r: r["cv_auc_mean"], reverse=True)
    elapsed_total = time.time() - start_time

    # Save a compact results table for the agent to read.
    with open(f"{args.out_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if not results:
        print(f"\nNo configs finished within the {args.max_seconds}s cap — the dataset or "
              f"fold count is too slow for this budget. Retry with a smaller grid: fewer "
              f"--model-types, a lower --n-folds, or a single seed. Do not spend more time "
              f"on this before you have at least one submission in.")
        return

    # Save OOF predictions (for correlation-based ensemble diversity checks)
    # and full-data test predictions for every config, ready to submit or average.
    oof_df = pd.DataFrame(oof_store)
    oof_df.to_csv(f"{args.out_prefix}_oof_predictions.csv", index=False)

    for config_id, preds in test_pred_store.items():
        out = pd.DataFrame({id_col: test_ids} if test_ids is not None else {})
        out[args.target] = preds
        out.to_csv(f"{args.out_prefix}_test_{config_id}.csv", index=False)

    print(f"\nTop config: {results[0]['config_id']} (CV AUC {results[0]['cv_auc_mean']}) "
          f"— {len(results)}/{len(configs)} configs completed in {elapsed_total:.0f}s.")
    if skipped:
        print(f"Skipped {len(skipped)} configs to stay under the {args.max_seconds}s cap: {skipped}")
    print(f"Wrote {args.out_prefix}_results.json, {args.out_prefix}_oof_predictions.csv, "
          f"and one {args.out_prefix}_test_<config_id>.csv per config.")
    print(f"NEXT STEP: submit_predictions({args.out_prefix}_test_{results[0]['config_id']}.csv) "
          f"now, before doing anything else — you have a valid result, don't risk finishing "
          f"the session without submitting it.")


if __name__ == "__main__":
    main()
