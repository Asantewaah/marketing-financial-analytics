"""
02_features.py — Feature Engineering
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Encodes categorical variables, engineers derived features,
handles class imbalance, and produces train/test splits
saved to data/ for use by subsequent scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank_marketing.csv"


def load_raw(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def encode_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'unknown' strings with NaN then impute with mode per column.
    This is preferable to treating 'unknown' as a valid category for
    tree-based models that can split on ordinal encodings.
    """
    df = df.copy()
    cat_cols = [c for c in df.columns
                if c != "y" and not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = df[col].replace("unknown", np.nan)
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive interpretable features that bridge marketing activity
    and financial/behavioural signals.
    """
    df = df.copy()

    # ── Contact intensity ──────────────────────────────────────────────────
    # High campaign contact count is often associated with lower conversion
    df["contacted_heavily"] = (df["campaign"] > 3).astype(int)

    # Was the client previously contacted in a prior campaign?
    df["was_previously_contacted"] = (df["pdays"] < 999).astype(int)

    # Successful prior campaign is the single strongest positive signal
    df["prior_success"] = (df["poutcome"] == "success").astype(int)

    # ── Economic environment ───────────────────────────────────────────────
    # Low employment variation → more stable economy → higher propensity to save
    df["stable_economy"] = (df["emp.var.rate"] < 0).astype(int)

    # High Euribor3m → higher interest rates → term deposits more attractive
    df["high_rates"] = (df["euribor3m"] > 3.0).astype(int)

    # Composite economic score (normalised)
    df["econ_score"] = (
        -df["emp.var.rate"]          # negative = better
        + df["cons.conf.idx"] / 10   # higher confidence = better
        - (df["euribor3m"] - 3.5)    # centred around median
    )

    # ── Client financial risk profile ──────────────────────────────────────
    df["has_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)
    df["clean_credit"] = (df["default"] == "no").astype(int)

    # ── Demographic proxies ────────────────────────────────────────────────
    df["is_young"]  = (df["age"] < 30).astype(int)
    df["is_senior"] = (df["age"] >= 60).astype(int)
    df["degree_educated"] = (df["education"] == "university.degree").astype(int)

    # ── Call quality ───────────────────────────────────────────────────────
    # Duration is known only after a call ends; useful for insight,
    # but excluded from production-score features to avoid leakage.
    # We keep it here for the interpretability section only.
    df["long_call"] = (df["duration"] > 300).astype(int)

    # ── Seasonal signal ────────────────────────────────────────────────────
    high_conv_months = ["mar", "sep", "oct", "dec"]
    df["peak_month"] = df["month"].isin(high_conv_months).astype(int)

    return df


def ordinal_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode remaining object columns (models handle trees fine)."""
    df = df.copy()
    cat_cols = [c for c in df.columns
                if c != "y" and not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def build_splits(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    """
    Return (X_train, X_test, y_train, y_test) with stratification.
    duration is excluded from X to avoid post-treatment leakage.
    """
    df = df.copy()
    df["y_bin"] = (df["y"] == "yes").astype(int)

    # Duration excluded from model features (observable only after call)
    drop_cols = ["y", "y_bin", "duration", "long_call"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["y_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Train conversion rate: {y_train.mean():.1%}")
    print(f"Test  conversion rate: {y_test.mean():.1%}")
    print(f"Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, feature_cols


def run_pipeline(save: bool = True):
    print("=== Feature Engineering Pipeline ===\n")
    df = load_raw()
    df = encode_unknowns(df)
    df = engineer_features(df)
    df = ordinal_encode_categoricals(df)
    X_train, X_test, y_train, y_test, feature_cols = build_splits(df)

    if save:
        out = ROOT / "data"
        X_train.to_csv(out / "X_train.csv", index=False)
        X_test.to_csv(out  / "X_test.csv",  index=False)
        y_train.to_csv(out / "y_train.csv", index=False)
        y_test.to_csv(out  / "y_test.csv",  index=False)
        pd.Series(feature_cols).to_csv(out / "feature_cols.csv", index=False)
        print("\n✓ Splits saved to data/")

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    run_pipeline()
