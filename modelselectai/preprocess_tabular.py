import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List

def build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str], List[str]]:
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # drop fully empty columns
    X = X.loc[:, X.notna().sum() > 0]

    # detect numeric cols (including bool)
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    # everything else treated as categorical (object/category/datetime/string)
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Convert timezone-aware datetime or actual datetime to string categories to preserve info
    for c in list(cat_cols):
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str)

    transformers = []

    if num_cols:
        # StandardScaler requires dense numeric arrays; impute missing with median first
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        # OneHotEncoder is good for low-cardinality, but bad for high-cardinality.
        # We will split the categorical columns based on cardinality to optimize.
        low_card_cat_cols = [c for c in cat_cols if X[c].nunique() <= 50]
        high_card_cat_cols = [c for c in cat_cols if X[c].nunique() > 50]
        
        if low_card_cat_cols:
            low_card_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
                ]
            )
            transformers.append(("cat_low", low_card_pipeline, low_card_cat_cols))
            
        if high_card_cat_cols:
            # OrdinalEncoder is a simple and fast way to handle high-cardinality.
            high_card_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
            )
            transformers.append(("cat_high", high_card_pipeline, high_card_cat_cols))

    if not transformers:
        raise ValueError("No usable feature columns detected in dataset after preprocessing.")

    pre = ColumnTransformer(transformers=transformers, remainder='drop')

    return X, y, pre, num_cols, cat_cols
    
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, stratify: bool = False):
    """
    Wrapper around train_test_split.
    If stratify=True and y is suitable for stratification, stratify by y.
    Suitable for stratification if target is non-numeric or has low cardinality.
    """
    stratify_arg = None
    if stratify and y is not None:
        try:
            if (not pd.api.types.is_numeric_dtype(y)) or (y.nunique(dropna=True) <= 20):
                stratify_arg = y
        except Exception:
            pass
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)