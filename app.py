"""Autonomous Data Analysis & Report Generator -- Streamlit Web App."""

import sys
import os
import io
from pathlib import Path

# Ensure modules/ is on path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import time

from modules.data_loader import load_data, infer_column_types
from modules.data_profiler import profile_data
from modules.descriptive import descriptive_stats, categorical_summary
from modules.visualization import generate_plots, generate_static_charts
from modules.correlation import correlation_analysis
from modules.regression import regression_analysis
from modules.hypothesis import hypothesis_tests
from modules.tableau_exporter import export_to_hyper
from modules.report_generator import generate_report
from utils.helpers import setup_matplotlib_style

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DA App - Autonomous Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_matplotlib_style()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
DEFAULTS = {
    "df": None,
    "dataset_name": "",
    "col_types": {},
    "target_col": None,
    "step": 1,
    "profile": None,
    "desc_stats": None,
    "cat_summary": None,
    "plots": None,
    "static_charts": None,
    "corr_result": None,
    "reg_result": None,
    "hyp_result": None,
    "report_path": None,
    "hyper_path": None,
    "analysis_done": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 DA App")
st.sidebar.caption("Autonomous Data Analysis")

steps = [
    (1, "📂 Upload Data"),
    (2, "⚙️ Configure"),
    (3, "🔬 Run Analysis"),
    (4, "📄 Download Report"),
]

current_step = st.sidebar.radio(
    "Navigation",
    options=[s[0] for s in steps],
    format_func=lambda x: steps[x - 1][1],
    index=st.session_state.step - 1,
    key="nav_radio",
)
st.session_state.step = current_step

st.sidebar.divider()
st.sidebar.caption("Made for UTS Data Analysis")

# =========================================================================
# STEP 1: Upload Data
# =========================================================================
if st.session_state.step == 1:
    st.title("📂 Upload Your Data")
    st.markdown("Upload a CSV, Excel (.xlsx/.xls), or Tableau (.hyper) file to begin.")

    uploaded = st.file_uploader(
        "Choose a data file",
        type=["csv", "xlsx", "xls", "hyper"],
        help="Supported: CSV, Excel, Tableau Hyper",
    )

    if uploaded is not None:
        try:
            with st.spinner("Loading data..."):
                df, name = load_data(uploaded)
                st.session_state.df = df
                st.session_state.dataset_name = name
                st.session_state.col_types = infer_column_types(df)
                st.session_state.target_col = None

            st.success(f"✅ Loaded **{name}** — {df.shape[0]} rows × {df.shape[1]} columns")

            # Quick preview
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df.shape[0])
                st.metric("Numeric Columns", len(st.session_state.col_types.get("numeric", [])))
            with col2:
                st.metric("Columns", df.shape[1])
                st.metric("Categorical Columns", len(st.session_state.col_types.get("categorical", [])))

            if st.button("→ Next: Configure Analysis", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()

        except Exception as e:
            st.error(f"Failed to load file: {e}")

# =========================================================================
# STEP 2: Configure
# =========================================================================
elif st.session_state.step == 2:
    st.title("⚙️ Configure Analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first. Go back to Step 1.")
    else:
        df = st.session_state.df
        col_types = st.session_state.col_types

        st.subheader("Column Overview")

        # Column type table
        type_rows = []
        for col in df.columns:
            if col in col_types.get("numeric", []):
                ctype = "Numeric"
            elif col in col_types.get("categorical", []):
                ctype = "Categorical"
            elif col in col_types.get("datetime", []):
                ctype = "Datetime"
            else:
                ctype = "Text"
            n_miss = int(df[col].isna().sum())
            type_rows.append({"Column": col, "Type": ctype, "Missing": n_miss})
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

        st.subheader("Target Variable")
        st.caption("Select the dependent variable you want to model. Defaults to the last column.")

        all_cols = list(df.columns)
        default_idx = len(all_cols) - 1 if all_cols else 0

        target_choice = st.selectbox(
            "Target variable",
            options=["(None — skip regression)"] + all_cols,
            index=default_idx + 1 if default_idx < len(all_cols) else 0,
        )
        if target_choice.startswith("(None"):
            st.session_state.target_col = None
        else:
            st.session_state.target_col = target_choice

        if st.session_state.target_col:
            st.info(f"🎯 Target set to: **{st.session_state.target_col}**")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Upload", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col_b:
            if st.button("→ Next: Run Analysis", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

# =========================================================================
# STEP 3: Run Analysis
# =========================================================================
elif st.session_state.step == 3:
    st.title("🔬 Analysis Pipeline")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first. Go back to Step 1.")
    elif not st.session_state.analysis_done:
        df = st.session_state.df
        col_types = st.session_state.col_types
        target = st.session_state.target_col

        st.markdown("Click the button below to run the complete analysis pipeline.")

        if st.button("🚀 Start Full Analysis", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting analysis pipeline...")
            status = st.status("Running analysis...", expanded=True)

            total_steps = 7

            # 1. Profile
            status.update(label="Step 1/7: Profiling data...")
            st.session_state.profile = profile_data(df)
            progress.progress(1 / total_steps, text="Data profiling complete")
            time.sleep(0.3)

            # 2. Descriptive stats
            status.update(label="Step 2/7: Computing descriptive statistics...")
            numeric_cols = col_types.get("numeric", [])
            st.session_state.desc_stats = descriptive_stats(df, numeric_cols)
            st.session_state.cat_summary = categorical_summary(df, col_types.get("categorical", []))
            progress.progress(2 / total_steps, text="Descriptive stats complete")
            time.sleep(0.3)

            # 3. Static charts
            status.update(label="Step 3/7: Generating static charts for report...")
            st.session_state.static_charts = generate_static_charts(df, col_types, target)
            progress.progress(3 / total_steps, text="Static charts generated")
            time.sleep(0.3)

            # 4. Interactive plots
            status.update(label="Step 4/7: Building interactive visualizations...")
            st.session_state.plots = generate_plots(df, col_types, target)
            progress.progress(4 / total_steps, text="Visualizations ready")
            time.sleep(0.3)

            # 5. Correlation
            status.update(label="Step 5/7: Correlation analysis...")
            st.session_state.corr_result = correlation_analysis(df, numeric_cols)
            progress.progress(5 / total_steps, text="Correlation done")
            time.sleep(0.3)

            # 6. Regression
            status.update(label="Step 6/7: Regression analysis...")
            if target and target in numeric_cols:
                predictors = [c for c in numeric_cols if c != target]
                st.session_state.reg_result = regression_analysis(df, predictors, target)
            else:
                st.session_state.reg_result = {"success": False, "error": "No valid numeric target selected."}
            progress.progress(6 / total_steps, text="Regression done")
            time.sleep(0.3)

            # 7. Hypothesis tests
            status.update(label="Step 7/7: Hypothesis testing...")
            st.session_state.hyp_result = hypothesis_tests(df, col_types, target)
            progress.progress(100, text="All analyses complete! ✅")

            status.update(label="Analysis complete!", state="complete")
            st.session_state.analysis_done = True
            st.balloons()
            time.sleep(1)
            st.rerun()

    # --- Display results when analysis is done ---
    if st.session_state.analysis_done:
        st.success("✅ Analysis completed successfully!")

        tabs = st.tabs([
            "📋 Overview", "📊 Charts", "🔗 Correlation",
            "📈 Regression", "🧪 Hypothesis"
        ])

        # Tab: Overview
        with tabs[0]:
            st.subheader("Data Profile")
            if st.session_state.profile:
                prof = st.session_state.profile
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", prof["shape"][0])
                c2.metric("Columns", prof["shape"][1])
                c3.metric("Missing Cells", prof["total_missing_cells"])
                c4.metric("Duplicate Rows", prof["duplicate_rows"])

            if st.session_state.desc_stats is not None and not st.session_state.desc_stats.empty:
                st.subheader("Descriptive Statistics")
                st.dataframe(st.session_state.desc_stats, use_container_width=True)

            if st.session_state.cat_summary:
                st.subheader("Categorical Summaries")
                for col, freq_df in st.session_state.cat_summary.items():
                    st.caption(f"**{col}**")
                    st.dataframe(freq_df, use_container_width=True)

        # Tab: Charts
        with tabs[1]:
            st.subheader("Interactive Visualizations")
            if st.session_state.plots:
                plots = st.session_state.plots
                if plots.get("distributions"):
                    st.caption("Distribution Histograms")
                    for fig in plots["distributions"]:
                        st.plotly_chart(fig, use_container_width=True)
                if plots.get("boxplots"):
                    st.caption("Box Plot Comparison")
                    for fig in plots["boxplots"]:
                        st.plotly_chart(fig, use_container_width=True)
                if plots.get("categorical"):
                    st.caption("Categorical Bar Charts")
                    for fig in plots["categorical"]:
                        st.plotly_chart(fig, use_container_width=True)
                if plots.get("scatter_matrix"):
                    st.caption("Scatter Matrix")
                    for fig in plots["scatter_matrix"]:
                        st.plotly_chart(fig, use_container_width=True)
                if plots.get("qq_plots"):
                    st.caption("Q-Q Plots")
                    for fig in plots["qq_plots"]:
                        st.plotly_chart(fig, use_container_width=True)

        # Tab: Correlation
        with tabs[2]:
            st.subheader("Correlation Analysis")
            if st.session_state.corr_result and not st.session_state.corr_result.get("matrix", pd.DataFrame()).empty:
                corr = st.session_state.corr_result
                st.text(corr.get("summary", ""))
                st.dataframe(corr["matrix"].style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                             use_container_width=True)
                if corr.get("heatmap_png"):
                    st.image(corr["heatmap_png"], caption="Correlation Heatmap")

        # Tab: Regression
        with tabs[3]:
            st.subheader("Regression Analysis")
            if st.session_state.reg_result:
                reg = st.session_state.reg_result
                if reg.get("success"):
                    st.metric("R-squared", reg["r_squared"])
                    st.metric("Adjusted R-squared", reg["adj_r_squared"])
                    st.metric("F-statistic", f"{reg['f_stat']} (p={reg['f_pvalue']})")
                    st.text(reg.get("interpretation", ""))
                    if not reg.get("coefficients", pd.DataFrame()).empty:
                        st.subheader("Coefficient Estimates")
                        st.dataframe(reg["coefficients"], use_container_width=True)
                    if reg.get("residual_plot_png"):
                        st.image(reg["residual_plot_png"], caption="Residual Diagnostics")
                    if reg.get("summary_html"):
                        with st.expander("Full OLS Summary"):
                            st.components.v1.html(reg["summary_html"], height=600, scrolling=True)
                else:
                    st.warning(f"Regression not available: {reg.get('error', '')}")

        # Tab: Hypothesis
        with tabs[4]:
            st.subheader("Hypothesis Testing")
            if st.session_state.hyp_result:
                hyp = st.session_state.hyp_result
                st.info(hyp.get("summary", ""))

                if hyp.get("t_tests"):
                    st.caption("Independent t-Tests")
                    st.dataframe(pd.DataFrame(hyp["t_tests"]), use_container_width=True)
                if hyp.get("anovas"):
                    st.caption("One-way ANOVA")
                    st.dataframe(pd.DataFrame(hyp["anovas"]), use_container_width=True)
                if hyp.get("chi_squares"):
                    st.caption("Chi-Square Tests")
                    st.dataframe(pd.DataFrame(hyp["chi_squares"]), use_container_width=True)

        # Navigation
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Configure", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col_b:
            if st.button("→ Next: Download Report", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

# =========================================================================
# STEP 4: Download Report
# =========================================================================
elif st.session_state.step == 4:
    st.title("📄 Download Report")

    if not st.session_state.analysis_done:
        st.warning("⚠️ Please run the analysis first. Go back to Step 3.")
    else:
        st.success("Analysis results are ready. Generate your report below.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Word Report (.docx)")
            if st.button("📝 Generate Word Report", type="primary", use_container_width=True):
                with st.spinner("Building report..."):
                    try:
                        path = generate_report(
                            dataset_name=st.session_state.dataset_name,
                            df=st.session_state.df,
                            col_types=st.session_state.col_types,
                            profile=st.session_state.profile,
                            desc_stats=st.session_state.desc_stats,
                            cat_summary=st.session_state.cat_summary,
                            static_charts=st.session_state.static_charts,
                            corr_result=st.session_state.corr_result,
                            reg_result=st.session_state.reg_result,
                            hyp_result=st.session_state.hyp_result,
                            target_col=st.session_state.target_col,
                        )
                        st.session_state.report_path = path
                        st.success(f"Report generated!")
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")

            if st.session_state.report_path and os.path.exists(st.session_state.report_path):
                with open(st.session_state.report_path, "rb") as f:
                    _report_bytes = f.read()
                st.download_button(
                    "📥 Download Word Report",
                    data=_report_bytes,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_docx",
                )

            st.subheader("Tableau Extract (.hyper)")
            if st.button("📊 Generate Tableau Extract", type="primary", use_container_width=True):
                with st.spinner("Exporting to .hyper..."):
                    try:
                        hpath = export_to_hyper(st.session_state.df)
                        st.session_state.hyper_path = hpath
                        st.success("Tableau extract generated!")
                    except Exception as e:
                        st.error(f"Failed to generate .hyper: {e}")

            if st.session_state.hyper_path and os.path.exists(st.session_state.hyper_path):
                with open(st.session_state.hyper_path, "rb") as f:
                    _hyper_bytes = f.read()
                st.download_button(
                    "📥 Download Tableau Extract",
                    data=_hyper_bytes,
                    file_name="analysis_export.hyper",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="dl_hyper",
                )


        st.divider()

        st.subheader("Session Summary")
        st.json({
            "Dataset": st.session_state.dataset_name,
            "Shape": list(st.session_state.df.shape) if st.session_state.df is not None else None,
            "Target": st.session_state.target_col,
            "Numeric Cols": st.session_state.col_types.get("numeric", []),
            "Categorical Cols": st.session_state.col_types.get("categorical", []),
        })

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Results", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col_b:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                for k in DEFAULTS:
                    st.session_state[k] = DEFAULTS[k]
                st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.divider()
if st.session_state.df is not None:
    st.sidebar.caption(f"📁 {st.session_state.dataset_name}")
    st.sidebar.caption(f"📐 {st.session_state.df.shape[0]}×{st.session_state.df.shape[1]}")
