"""Data loading, cleaning, and train-test splitting utilities."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load CSV, clean column names (strip, lower, replace spaces with _),
    encode target: status -> 1 (Alive) / 0 (Dead),
    drop survival_months (data leakage).
    Returns cleaned DataFrame.
    """
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    df["status"] = df["status"].map({"Alive": 1, "Dead": 0})
    if "survival_months" in df.columns:
        df = df.drop(columns=["survival_months"])
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    One-hot encode all object/category columns with drop_first=True.
    Return X (DataFrame), y (Series).
    Note: survival_months must already be dropped before calling this.
    """
    y = df["status"]
    X = df.drop(columns=["status"])
    object_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(object_cols) > 0:
        X = pd.get_dummies(X, columns=object_cols, drop_first=True)
    return X, y


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split.
    Returns X_train, X_test, y_train, y_test.
    Always use stratify=y.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
