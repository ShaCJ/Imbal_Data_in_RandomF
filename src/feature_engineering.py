"""Feature engineering utilities for the breast cancer dataset."""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two engineered features to the DataFrame (in place):
      - node_ratio = reginol_node_positive / (regional_node_examined + 1)
      - tumor_size_age_ratio = tumor_size / (age + 1)
    Return the modified DataFrame.
    The +1 in denominators prevents division by zero.
    """
    df["node_ratio"] = df["reginol_node_positive"] / (
        df["regional_node_examined"] + 1
    )
    df["tumor_size_age_ratio"] = df["tumor_size"] / (df["age"] + 1)
    return df
