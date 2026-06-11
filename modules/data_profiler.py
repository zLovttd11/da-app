"""Profile a DataFrame: shape, dtypes, missing values, outliers."""

import pandas as pd
import numpy as np


def profile_data(df: pd.DataFrame) -> dict:
    """Return a comprehensive data profile dictionary.

    Keys:
    - shape: (rows, cols)
    - dtypes: dict of column -> dtype string
    - missing: dict of column -> (count, percent)
    - total_missing_cells: int
    - duplicate_rows: int
    - outliers: dict of numeric column -> count of IQR outliers
    - memory_usage: string
    """
    profile: dict = {}

    profile["shape"] = df.shape
    profile["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    missing = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct = round(n_miss / max(len(df), 1) * 100, 2)
        missing[col] = {"count": n_miss, "percent": pct}
    profile["missing"] = missing
    profile["total_missing_cells"] = int(df.isna().sum().sum())

    profile["duplicate_rows"] = int(df.duplicated().sum())

    outliers: dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        outliers[col] = n_out
    profile["outliers"] = outliers

    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024:
        profile["memory_usage"] = f"{mem_bytes} B"
    elif mem_bytes < 1024 ** 2:
        profile["memory_usage"] = f"{mem_bytes / 1024:.1f} KB"
    else:
        profile["memory_usage"] = f"{mem_bytes / (1024 ** 2):.1f} MB"

    return profile
