import pandas as pd
import numpy as np

def detect_task_type(y: pd.Series, cfg) -> str:
    """
    Decide classification vs regression by analyzing
    whether the target values are discrete or continuous.
    """

    print(f"[DEBUG] Detecting task type for: {y.name}")
    print(f"[DEBUG] dtype={y.dtype}, unique={y.nunique()}, total={len(y)}")

    # 1. Non-numeric → classification
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        print("[DEBUG] → CLASSIFICATION (categorical dtype)")
        return "classification"

    # 2. Numeric
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        total = len(y.dropna())
        ratio = nunique / max(total, 1)
        is_integer = np.allclose(y.dropna() % 1, 0)

        print(f"[DEBUG] Numeric target → Unique={nunique}, Total={total}, Ratio={ratio:.4f}, IsInteger={is_integer}")

        # ✅ RULE:
        # If very few distinct values (discrete labels), classify
        if nunique <= cfg.cls_cardinality_threshold:
            print("[DEBUG] → CLASSIFICATION (discrete labels, low unique count)")
            return "classification"

        # If ratio of unique/total is high → regression
        if ratio > 0.05:   # at least 5% values unique → continuous
            print("[DEBUG] → REGRESSION (high diversity → continuous target)")
            return "regression"

        # If float values and not just small set of repeats → regression
        if not is_integer:
            print("[DEBUG] → REGRESSION (continuous float values)")
            return "regression"

        # Otherwise treat as classification
        print("[DEBUG] → CLASSIFICATION (repeated integer labels)")
        return "classification"

    # 3. Fallback
    print("[DEBUG] → REGRESSION (fallback rule)")
    return "regression"