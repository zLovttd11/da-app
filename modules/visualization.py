"""Generate interactive Plotly charts and static matplotlib charts for reports."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os

from utils.helpers import setup_matplotlib_style, temp_output_dir, figure_to_png_bytes


def generate_plots(df: pd.DataFrame, col_types: dict, target_col: str | None = None) -> dict[str, list]:
    """Generate interactive Plotly chart objects organized by category.

    Returns dict with keys: distributions, boxplots, categorical, scatter_matrix,
    qq_plots.
    """
    setup_matplotlib_style()
    plots: dict[str, list] = {
        "distributions": [],
        "boxplots": [],
        "categorical": [],
        "scatter_matrix": [],
        "qq_plots": [],
    }

    numeric = col_types.get("numeric", [])
    categorical = col_types.get("categorical", [])

    # Histograms for each numeric column
    for col in numeric:
        series = df[col].dropna()
        fig = px.histogram(series, nbins=30, title=f"Distribution of {col}",
                           labels={"value": col, "count": "Frequency"},
                           color_discrete_sequence=["#3366cc"])
        fig.update_layout(bargap=0.05, template="plotly_white")
        plots["distributions"].append(fig)

    # Box plots (up to 15 numeric columns in one combined view)
    if len(numeric) >= 1:
        n_show = min(len(numeric), 15)
        box_data = df[numeric[:n_show]].melt(var_name="Variable", value_name="Value")
        fig = px.box(box_data, x="Variable", y="Value", title="Box Plot Comparison",
                     color_discrete_sequence=["#3366cc"])
        fig.update_layout(template="plotly_white")
        plots["boxplots"].append(fig)

    # Bar charts for categorical columns (up to first 8)
    for col in categorical[:8]:
        counts = df[col].value_counts().nlargest(20).reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.bar(counts, x="Category", y="Count", title=f"Frequency of {col}",
                     color_discrete_sequence=["#dc3912"])
        fig.update_layout(template="plotly_white")
        plots["categorical"].append(fig)

    # Scatter matrix for numeric columns (up to 6)
    if len(numeric) >= 2:
        scatter_cols = numeric[:min(len(numeric), 6)]
        if target_col and target_col in scatter_cols:
            color_col = target_col
        else:
            color_col = None
        fig = px.scatter_matrix(df[scatter_cols], dimensions=scatter_cols,
                                title="Scatter Matrix", color=color_col,
                                opacity=0.6)
        fig.update_layout(template="plotly_white")
        plots["scatter_matrix"].append(fig)

    # QQ plots
    for col in numeric[:10]:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        from scipy import stats as sp_stats
        qq = sp_stats.probplot(series, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                                 marker=dict(size=4, color="#3366cc"), name="Data"))
        slope, intercept, _ = qq[1]
        x_vals = np.array([min(qq[0][0]), max(qq[0][0])])
        y_vals = slope * x_vals + intercept
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines",
                                 line=dict(color="#dc3912", dash="dash"), name="Normal"))
        fig.update_layout(title=f"Q-Q Plot: {col}", xaxis_title="Theoretical Quantiles",
                          yaxis_title="Sample Quantiles", template="plotly_white")
        plots["qq_plots"].append(fig)

    return plots


def generate_static_charts(df: pd.DataFrame, col_types: dict,
                           target_col: str | None = None) -> dict[str, list[bytes]]:
    """Generate matplotlib static charts as PNG bytes for embedding in Word reports.

    Returns dict keyed by chart type, each value is a list of PNG byte strings.
    """
    setup_matplotlib_style()
    charts: dict[str, list[bytes]] = {
        "histograms": [],
        "boxplot": [],
        "barcharts": [],
        "heatmap": [],
        "scatter": [],
    }
    tmp_dir = temp_output_dir()

    numeric = col_types.get("numeric", [])
    categorical = col_types.get("categorical", [])

    # Histograms
    for col in numeric:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=30, color="#3366cc", edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        charts["histograms"].append(figure_to_png_bytes(fig))
        plt.close(fig)

    # Combined box plot
    if len(numeric) >= 1:
        fig, ax = plt.subplots(figsize=(max(6, len(numeric[:15]) * 1.2), 4))
        n_show = min(len(numeric), 15)
        ax.boxplot([df[c].dropna() for c in numeric[:n_show]], labels=numeric[:n_show])
        ax.set_title("Box Plot Comparison")
        ax.tick_params(axis="x", rotation=45)
        charts["boxplot"].append(figure_to_png_bytes(fig))
        plt.close(fig)

    # Bar charts for categorical columns
    for col in categorical[:8]:
        counts = df[col].value_counts().nlargest(15)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#dc3912", alpha=0.85)
        ax.set_title(f"Frequency of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        charts["barcharts"].append(figure_to_png_bytes(fig))
        plt.close(fig)

    return charts
