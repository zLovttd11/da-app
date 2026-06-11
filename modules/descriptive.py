"""Descriptive statistics and categorical frequency summaries."""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns.

    Returns a DataFrame with rows = columns, columns = stats.
    """
    records: list[dict] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        records.append({
            "Variable": col,
            "Count": len(series),
            "Mean": round(series.mean(), 4),
            "Median": round(series.median(), 4),
            "Std Dev": round(series.std(ddof=1), 4) if len(series) > 1 else np.nan,
            "Min": round(series.min(), 4),
            "25%": round(series.quantile(0.25), 4),
            "50%": round(series.quantile(0.50), 4),
            "75%": round(series.quantile(0.75), 4),
            "Max": round(series.max(), 4),
            "Skewness": round(float(sp_stats.skew(series)), 4),
            "Kurtosis": round(float(sp_stats.kurtosis(series)), 4),
        })
    return pd.DataFrame(records)


def categorical_summary(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Return a frequency table for each categorical column."""
    result: dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = ["Category", "Count"]
        freq["Percent"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
        result[col] = freq.head(20)
    return result
