"""Correlation analysis: Pearson/Spearman matrix and heatmap."""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helpers import setup_matplotlib_style, figure_to_png_bytes


def correlation_analysis(df: pd.DataFrame, numeric_cols: list[str],
                         method: str = "pearson") -> dict:
    """Compute correlation matrix and return results as a dict.

    Keys:
    - matrix: the correlation DataFrame
    - heatmap_png: PNG bytes of the heatmap
    - summary: text summary of strongest correlations
    - method: the method used
    """
    if len(numeric_cols) < 2:
        return {"matrix": pd.DataFrame(), "heatmap_png": b"", "summary": "Insufficient numeric columns.", "method": method}

    setup_matplotlib_style()
    corr_df = df[numeric_cols].corr(method=method)

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 0.9), max(5, len(numeric_cols) * 0.7)))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=13, fontweight="bold")
    heatmap_png = figure_to_png_bytes(fig)
    plt.close(fig)

    # Text summary of strongest correlations
    pairs = []
    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            pairs.append((corr_df.columns[i], corr_df.columns[j], corr_df.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    summary_lines = [f"Top correlations ({method}):"]
    for a, b, val in pairs[:10]:
        summary_lines.append(f"  {a} vs {b}: r = {val:.3f}")

    return {
        "matrix": corr_df,
        "heatmap_png": heatmap_png,
        "summary": "\n".join(summary_lines),
        "method": method,
    }
