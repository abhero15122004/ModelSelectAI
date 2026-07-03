import pandas as pd
from .loader import is_image_folder, is_dicom_folder

def detect_input_type(input_path: str = None, input_folder: str = None, task_hint: str = "auto") -> str:
    if task_hint and task_hint.lower() != "auto":
        return task_hint.lower()

    if input_folder:
        if is_dicom_folder(input_folder):
            return "medical"
        if is_image_folder(input_folder):
            return "image"
        return "multimodal"

    if input_path:
        ext = input_path.lower().split(".")[-1]

        if ext in ("csv", "parquet", "json", "ndjson"):
            try:
                if ext == "csv":
                    df = pd.read_csv(input_path, nrows=500)
                elif ext == "parquet":
                    df = pd.read_parquet(input_path)
                else:
                    df = pd.read_json(input_path, lines=(ext == "ndjson"))
            except Exception:
                return "tabular"

            # Step 1: Identify datetime columns
            datetime_cols = []
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.85:
                        datetime_cols.append(c)
                except Exception:
                    continue

            if datetime_cols:
                # Step 2: Count numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

                # Step 3: Apply strict rules
                if (
                    len(numeric_cols) == 1  # Only 1 target-like numeric column
                    and df.shape[0] > 10  # Enough rows
                    and df.shape[0] > 5 * df.shape[1]  # Time series usually has many more rows than cols
                ):
                    # Step 4: Check sequential datetime
                    dt_col = datetime_cols[0]
                    dt_series = pd.to_datetime(df[dt_col], errors="coerce")
                    increasing_ratio = (dt_series.diff().dropna() >= pd.Timedelta(0)).mean()
                    if increasing_ratio > 0.9:
                        return "timeseries"

            return "tabular"

        if ext == "dcm":
            return "medical"
        if ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff"):
            return "image"

    return "tabular"