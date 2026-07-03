import pandas as pd
import numpy as np

def detect_datetime_column(df: pd.DataFrame):
    """
    Attempts to find a datetime-like column in a DataFrame.
    Returns the column name or None if not found.
    """
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "timestamp" in cl:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    return None

def select_ts_target(df: pd.DataFrame, exclude_cols=None):
    """
    Selects a numeric target column for time series forecasting.
    Prefers the first numeric column not in exclude_cols.
    """
    exclude_cols = exclude_cols or []
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[0]]
    else:
        # Try to coerce the first non-excluded column to numeric
        for c in df.columns:
            if c not in exclude_cols:
                try:
                    return pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    continue
    raise ValueError("No suitable numeric target column found for time series.")

def coerce_numeric_series(series: pd.Series):
    """
    Forces a pandas Series to numeric dtype, coercing errors to NaN.
    """
    return pd.to_numeric(series, errors="coerce")

def make_lag_features(df: pd.DataFrame, target_col: str, lags: int = 3):
    """
    Creates lag features for time series modeling.
    Returns a DataFrame with lag columns added.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        df_lagged[f"{target_col}_lag{lag}"] = df_lagged[target_col].shift(lag)

    # Drop rows with NaNs created by lagging
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    return df_lagged