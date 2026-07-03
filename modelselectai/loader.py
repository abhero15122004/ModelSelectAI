# modelselectai/loader.py
import os
import zipfile
import pandas as pd
from typing import Optional

def load_table(path: str, dtype: Optional[dict] = None) -> pd.DataFrame:
    """
    Robust table loader. Keeps string/object columns as strings (no forced numeric coercion).
    Supports: .csv, .parquet, .json, .ndjson
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=dtype, keep_default_na=True, na_values=["", "NA", "N/A", "null"])
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in [".json", ".ndjson"]:
        df = pd.read_json(path, lines=(ext == ".ndjson"))
    else:
        # fallback: try CSV
        df = pd.read_csv(path, dtype=dtype, keep_default_na=True, na_values=["", "NA", "N/A", "null"])

    # strip whitespace for object columns and preserve empty->NA
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c] == "", c] = pd.NA
        except Exception:
            pass

    return df


def is_image_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                return True
    return False


def is_dicom_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".dcm"):
                return True
    return False


def try_extract_archive(path: str, out_dir: Optional[str] = None) -> str:
    """
    If path is a zip archive, extract it to out_dir (or a sibling folder) and return extracted folder path.
    Otherwise return input path.
    """
    out_dir = out_dir or os.path.splitext(path)[0] + "_extracted"
    if zipfile.is_zipfile(path):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(out_dir)
        return out_dir
    return path