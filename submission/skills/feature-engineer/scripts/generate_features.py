#!/usr/bin/env python3
"""Robust CLI script for automated feature generation.

Automatically identifies column types, imputes missing values, and calculates
a row-wise mean feature.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def main():
    parser = argparse.ArgumentParser(description="Generate automated ML features.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train CSV")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    args = parser.parse_args()

    print(f"Loading datasets: {args.train}, {args.test}")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    target_series = None
    if args.target in train_df.columns:
        target_series = train_df[args.target]
        train_df = train_df.drop(columns=[args.target])
    else:
        print(f"Warning: Target column '{args.target}' not found in train_df.")

    # Align columns — never assume train/test have the exact same schema.
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    print(f"Initial shape: train={train_df.shape}, test={test_df.shape}")

    # Identify column types at runtime — no hardcoded column names, so this
    # works on any dataset from the same family, not just the ones we've seen.
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Impute missing values — fit on train only, transform test with those
    #    same fitted values, to keep cross-validation estimates honest.
    if num_cols:
        print(f"Imputing missing values for {len(num_cols)} numeric columns...")
        num_imputer = SimpleImputer(strategy="median")
        train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = num_imputer.transform(test_df[num_cols])

    if cat_cols:
        print(f"Imputing missing values for {len(cat_cols)} categorical columns...")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    # 2. Aggregation feature.
    if num_cols:
        print("Calculating row-wise mean feature...")
        train_df["row_mean"] = train_df[num_cols].mean(axis=1)
        test_df["row_mean"] = test_df[num_cols].mean(axis=1)

    # Re-attach target.
    if target_series is not None:
        train_df[args.target] = target_series

    print(f"Engineered shape: train={train_df.shape}, test={test_df.shape}")
    train_df.to_csv("train_engineered.csv", index=False)
    test_df.to_csv("test_engineered.csv", index=False)
    print("Saved train_engineered.csv and test_engineered.csv successfully.")


if __name__ == "__main__":
    main()
