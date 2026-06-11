"""Autonomous Data Analysis & Report Generator -- Streamlit Web App v2."""

import sys, os, io, base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import time

from modules.data_loader import load_data, infer_column_types
from modules.data_profiler import profile_data
from modules.descriptive import descriptive_stats, categorical_summary
from modules.visualization import generate_plots, generate_static_charts
from modules.correlation import correlation_analysis
from modules.regression import regression_analysis, regression_compare_models
from modules.hypothesis import hypothesis_tests
from modules.classification import classify_compare_models
from modules.clustering import cluster_analysis
from modules.feature_engineering import feature_importance_analysis
from modules.cross_validation import split_data, kfold_cross_validation, stratified_kfold
from modules.web_fetcher import fetch_page, download_data_file, try_load_as_dataframe, set_cookie_header
from modules.auto_ml import auto_analyze
from modules.data_preprocessing import auto_preprocess
from modules.tableau_exporter import export_to_hyper
from modules.report_generator import generate_report
from utils.helpers import setup_matplotlib_style

st.set_page_config(page_title="DA App v2 - Advanced Data Analysis", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")
setup_matplotlib_style()

DEFAULTS = {
    "df": None, "dataset_name": "", "col_types": {}, "target_col": None,
    "step": 1, "analysis_mode": "auto",
    "profile": None, "desc_stats": None, "cat_summary": None,
    "plots": None, "static_charts": None,
    "corr_result": None, "reg_result": None, "reg_compare_result": None,
    "hyp_result": None, "class_result": None,
    "cluster_result": None, "fi_result": None, "split_result": None,
    "uploaded_files": [], "active_file_index": 0,
    "problem_statement": "",
    "auto_ml_result": None, "preprocess_log": None,
    "report_path": None, "hyper_path": None, "analysis_done": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.title(" DA App v2")
st.sidebar.caption("Advanced Autonomous Data Analysis")

steps = [(1, " Upload Data"), (2, " Configure"), (3, " Run Analysis"), (4, " Download Report")]
current_step = st.sidebar.radio("Navigation", options=[s[0] for s in steps],
                                format_func=lambda x: steps[x - 1][1],
                                index=st.session_state.step - 1, key="nav_radio")
if current_step != st.session_state.step:
            st.session_state.step = current_step
st.sidebar.divider()
st.sidebar.caption("Made for UTS Data Analysis")

# ================================================================
# STEP 1: Upload Data
# ================================================================
if st.session_state.step == 1:
    st.title(" Upload Your Data")

    # ---- URL Fetch Section ----
    st.subheader("Fetch from URL (Optional)")
    st.caption("Paste an assignment page URL to auto-extract problem description and data files.")
    url_col1, url_col2 = st.columns([4, 1])
    with url_col1:
        fetch_url_input = st.text_input(
            "Enter URL (e.g., Canvas, Moodle, or direct data link):",
            placeholder="https://...",
            key="fetch_url"
        )
    with url_col2:
        st.write("")
        st.write("")
        fetch_clicked = st.button("Fetch URL", use_container_width=True, key="fetch_btn")

        with st.expander("Need to access a login-protected page? Add browser cookies"):
            st.caption("1. Open the page in Chrome (logged in)  2. Press F12 > Application > Cookies  3. Copy the Cookie header value for the site")
            cookie_input = st.text_input(
                "Paste Cookie string here:",
                placeholder="session=abc123; token=xyz789; ...",
                key="cookie_string"
            )
            if cookie_input:
                set_cookie_header(cookie_input)
                st.success("Cookie set! Fetch will use your login session.")

    if fetch_clicked and fetch_url_input:
        with st.spinner("Fetching webpage content..."):
            page = fetch_page(fetch_url_input)
            if page["success"]:
                st.success("Page fetched: **{}**".format(page["title"][:100] if page["title"] else "Untitled"))
                # Set problem statement from extracted text
                extracted = page["text"][:3000]
                st.session_state.problem_statement = extracted
                st.info("Extracted {:,} characters of text. First section shown below:".format(len(extracted)))
                with st.expander("View extracted text (first 20,000 chars)"):
                    st.text(extracted[:20000])

                # Show data files found
                if page["data_files"]:
                    st.subheader("Data Files Found ({})".format(len(page["data_files"])))
                    for i, df_info in enumerate(page["data_files"][:10]):
                        col_a, col_b = st.columns([6, 2])
                        with col_a:
                            st.markdown("- `{}`".format(df_info["url"][:120]))
                        with col_b:
                            if st.button("Download & Load", key="dl_file_{}".format(i), use_container_width=True):
                                with st.spinner("Downloading..."):
                                    raw, fname, err = download_data_file(df_info["url"])
                                    if raw:
                                        df = try_load_as_dataframe(raw, fname)
                                        if df is not None:
                                            st.session_state.df = df
                                            st.session_state.dataset_name = fname
                                            st.session_state.col_types = infer_column_types(df)
                                            st.session_state.target_col = None
                                            st.success("Loaded **{}**  {} rows x {} columns!".format(fname, df.shape[0], df.shape[1]))
                                            st.dataframe(df.head(50), use_container_width=True)
                                        else:
                                            st.error("Could not parse as CSV/Excel. File may be a PDF (download via browser).")
                                    else:
                                        st.error("Download failed: {}".format(err))
                    if st.button("Auto-download & Load First Data File", use_container_width=True):
                        with st.spinner("Downloading data..."):
                            first = page["data_files"][0]
                            raw, fname, err = download_data_file(first["url"])
                            if raw:
                                df = try_load_as_dataframe(raw, fname)
                                if df is not None:
                                    st.session_state.df = df
                                    st.session_state.dataset_name = fname
                                    st.session_state.col_types = infer_column_types(df)
                                    st.session_state.target_col = None
                                    st.success("Loaded **{}** -- {} rows x {} columns!".format(fname, df.shape[0], df.shape[1]))
                                    st.dataframe(df.head(50), use_container_width=True)
                                else:
                                    st.error("Could not parse file as CSV/Excel. Try downloading manually.")
                            else:
                                st.error("Download failed: {}".format(err))
                else:
                    st.info("No data files (.csv/.xlsx) found on the page. You can still upload manually below.")
            else:
                st.error("Failed to fetch: {}".format(page.get("error", "Unknown error")))

    st.divider()
    st.subheader("Or Upload File Directly")
    st.markdown("Upload a CSV or Excel file to begin. Supports multi-variable datasets of any size.")

    st.subheader("Assignment / Problem Description")
    st.caption("Describe what this analysis is for. This shapes the analysis approach and report narrative.")
    st.session_state.problem_statement = st.text_area(
        "Enter your assignment requirements or business problem:",
        value=st.session_state.get("problem_statement", ""),
        placeholder="e.g. OzMart needs to automatically classify new product categories from vendor photos with minimal labelled data. The goal is to reduce manual labelling time from weeks to hours.",
        height=150,
        key="problem_text"
    )

    uploaded_files = st.file_uploader("Choose data files (up to 10)", type=["csv", "xlsx", "xls"],
                                       accept_multiple_files=True,
                                       help="Supported: CSV, Excel. You can upload up to 10 files at once.")

    # Store uploaded files in session state to survive rerenders
    if uploaded_files:
        # Limit to 10
        if len(uploaded_files) > 10:
            st.warning("Maximum 10 files allowed. Only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        # Load and save to session state
        loaded_files = []
        for i, uf in enumerate(uploaded_files):
            try:
                df, name = load_data(uf)
                loaded_files.append({"index": i, "name": name, "df": df, "shape": df.shape})
            except Exception as e:
                st.error("Failed to load {}: {}".format(uf.name, str(e)))

        if loaded_files:
            st.session_state.uploaded_files = loaded_files

    # Display loaded files from session state (survives rerenders)
    if st.session_state.get("uploaded_files"):
        loaded_files = st.session_state.uploaded_files
        st.subheader("Uploaded Files ({})".format(len(loaded_files)))

        file_options = ["{} ({} rows x {} cols)".format(f["name"], f["shape"][0], f["shape"][1])
                       for f in loaded_files]
        selected_idx = st.selectbox(
            "Select file to analyze:", range(len(loaded_files)),
            format_func=lambda i: file_options[i],
            key="file_selector",
            index=min(st.session_state.get("active_file_index", 0), len(loaded_files) - 1)
        )

        selected = loaded_files[selected_idx]
        st.session_state.df = selected["df"]
        st.session_state.dataset_name = selected["name"]
        st.session_state.col_types = infer_column_types(selected["df"])
        st.session_state.target_col = None
        st.session_state.active_file_index = selected_idx

        st.success("Active: **{}** -- {} rows x {} columns".format(
            selected["name"], selected["shape"][0], selected["shape"][1]))
        st.subheader("Data Preview")
        st.dataframe(selected["df"].head(100), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", selected["shape"][0])
        col2.metric("Columns", selected["shape"][1])
        col3.metric("Numeric Cols", len(st.session_state.col_types.get("numeric", [])))

        # Analysis mode selection
        st.subheader("Analysis Mode")
        st.session_state.analysis_mode = st.radio(
            "Select analysis depth:",
            options=["auto", "comprehensive"],
            format_func=lambda x: "Auto-detect" if x == "auto" else "Comprehensive",
            horizontal=True,
        )

        if st.button("Next: Configure Analysis", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

# ================================================================
# STEP 2: Configure
# ================================================================
elif st.session_state.step == 2:
    st.title(" Configure Analysis")
    if st.session_state.df is None:
        st.warning("Please upload data first.")
    else:
        df = st.session_state.df
        col_types = st.session_state.col_types

        if st.session_state.problem_statement:
            st.subheader("Analysis Context")
            st.info(st.session_state.problem_statement[:500] + ("..." if len(st.session_state.problem_statement) > 500 else ""))

            st.subheader("Column Overview")
        type_rows = []
        for col in df.columns:
            if col in col_types.get("numeric", []):
                ct = "Numeric"
            elif col in col_types.get("categorical", []):
                ct = "Categorical"
            elif col in col_types.get("datetime", []):
                ct = "Datetime"
            else:
                ct = "Text"
            type_rows.append({"Column": col, "Type": ct, "Missing": int(df[col].isna().sum())})
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

        st.subheader("Target Variable")
        all_cols = list(df.columns)
        default_idx = len(all_cols) - 1 if all_cols else 0
        target_choice = st.selectbox("Target variable (for modeling)",
                                     options=["(None -- skip predictive modeling)"] + all_cols,
                                     index=default_idx + 1 if default_idx < len(all_cols) else 0)
        if target_choice.startswith("(None"):
            st.session_state.target_col = None
        else:
            st.session_state.target_col = target_choice
        if st.session_state.target_col:
            st.info("Target set to: **{}**".format(st.session_state.target_col))

        st.subheader("Analysis Pipeline")
        st.markdown("The following analyses will run:")
        pipeline_items = [
            "Data profiling & quality check",
            "Descriptive statistics (numeric + categorical)",
            "Interactive visualizations (histograms, boxplots, scatter matrix, QQ plots)",
            "Correlation analysis with heatmap",
        ]
        if st.session_state.target_col:
            t = st.session_state.target_col
            t_nunique = df[t].dropna().nunique()
            if t_nunique <= 20:
                pipeline_items.append("**Classification analysis** -- 3-model comparison (LR, RF, GB) with ROC curves")
            pipeline_items.append("**Regression analysis** -- OLS + multi-model comparison (LR, RF, GB)")
            pipeline_items.append("**Feature importance** -- Random Forest + Mutual Information")
        pipeline_items += [
            "Hypothesis testing with effect sizes (Cohen's d, Eta-squared, Cramer's V)",
            "**Clustering analysis** -- K-means with silhouette optimization + PCA visualization",
        ]
        for item in pipeline_items:
            st.markdown("- " + item)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Back to Upload", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col_b:
            if st.button("Run Full Analysis Pipeline", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

# ================================================================
# STEP 3: Run Analysis
# ================================================================
elif st.session_state.step == 3:
    st.title(" Analysis Pipeline")
    if st.session_state.df is None:
        st.warning("Please upload data first.")
    elif not st.session_state.analysis_done:
        df = st.session_state.df
        col_types = st.session_state.col_types
        target = st.session_state.target_col
        numeric = col_types.get("numeric", [])
        mode = st.session_state.analysis_mode

        st.markdown("Click to run the complete analysis pipeline (10 steps).")

        if st.button(" Start Full Analysis", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting analysis pipeline...")
            status = st.status("Running analysis...", expanded=True)
            total_steps = 11

            # 1. Profile
            status.update(label="Step 1/10: Profiling data...")
            st.session_state.profile = profile_data(df)
            progress.progress(1 / total_steps, text="Data profiling complete")
            time.sleep(0.2)

            # 2. Descriptive
            status.update(label="Step 2/10: Computing descriptive statistics...")
            st.session_state.desc_stats = descriptive_stats(df, numeric)
            st.session_state.cat_summary = categorical_summary(df, col_types.get("categorical", []))
            progress.progress(2 / total_steps, text="Descriptive stats complete")
            time.sleep(0.2)

            # 3. Static charts
            status.update(label="Step 3/10: Generating static charts...")
            st.session_state.static_charts = generate_static_charts(df, col_types, target)
            progress.progress(3 / total_steps, text="Static charts generated")
            time.sleep(0.2)

            # 4. Interactive plots
            status.update(label="Step 4/10: Building interactive visualizations...")
            st.session_state.plots = generate_plots(df, col_types, target)
            progress.progress(4 / total_steps, text="Visualizations ready")
            time.sleep(0.2)

            # 5. Correlation
            status.update(label="Step 5/10: Correlation analysis...")
            st.session_state.corr_result = correlation_analysis(df, numeric)
            progress.progress(5 / total_steps, text="Correlation done")
            time.sleep(0.2)

            # 6. Regression
            status.update(label="Step 6/10: Regression analysis...")
            if target and target in numeric:
                predictors = [c for c in numeric if c != target]
                st.session_state.reg_result = regression_analysis(df, predictors, target)
                try:
                    st.session_state.reg_compare_result = regression_compare_models(df, predictors, target)
                except Exception:
                    st.session_state.reg_compare_result = {"success": False, "error": "Multi-model comparison failed."}
            else:
                st.session_state.reg_result = {"success": False, "error": "No valid numeric target selected."}
                st.session_state.reg_compare_result = {"success": False, "error": "No valid target."}
            progress.progress(6 / total_steps, text="Regression done")
            time.sleep(0.2)

            # 7. Classification
            status.update(label="Step 7/11: Classification analysis...")
            if target and target in df.columns:
                t_nunique = df[target].dropna().nunique()
                if t_nunique <= 20 and t_nunique >= 2:
                    try:
                        st.session_state.class_result = classify_compare_models(df, target, numeric)
                    except Exception as e:
                        st.session_state.class_result = {"success": False, "error": str(e)}
                else:
                    st.session_state.class_result = {"success": False,
                                                     "error": "Target has {} unique values -- classification requires 2-20.".format(t_nunique)}
            else:
                st.session_state.class_result = {"success": False, "error": "No target selected."}
            progress.progress(7 / 11, text="Classification done")
            time.sleep(0.2)

            # 7.5. Auto-ML Adaptive Analysis
            status.update(label="Step 7.5/11: Auto-ML adaptive analysis...");
            if target and target in df.columns:
                try:
                    st.session_state.auto_ml_result = auto_analyze(df, target, numeric)
                    st.session_state.preprocess_log, _ = auto_preprocess(df, target, numeric)
                except Exception as e:
                    st.session_state.auto_ml_result = {"success": False, "error": str(e)}
                    st.session_state.preprocess_log = []
            else:
                st.session_state.auto_ml_result = {"success": False, "error": "No target selected."}
                st.session_state.preprocess_log = []
            progress.progress(7.5 / 11, text="Auto-ML complete");
            time.sleep(0.2);

            # 8. Hypothesis
            status.update(label="Step 8/11: Hypothesis testing...")
            st.session_state.hyp_result = hypothesis_tests(df, col_types, target)
            progress.progress(8 / 11, text="Hypothesis testing done")
            time.sleep(0.2)

            # 9. Feature importance
            status.update(label="Step 9/11: Feature importance analysis...")
            if target and target in numeric:
                try:
                    st.session_state.fi_result = feature_importance_analysis(df, target, numeric)
                except Exception as e:
                    st.session_state.fi_result = {"success": False, "error": str(e)}
            else:
                st.session_state.fi_result = {"success": False, "error": "No numeric target."}
            progress.progress(9 / 11, text="Feature importance done")
            time.sleep(0.2)

            # 10. Clustering
            status.update(label="Step 10/11: Clustering analysis...")
            if len(numeric) >= 2:
                try:
                    st.session_state.cluster_result = cluster_analysis(df, numeric)
                except Exception as e:
                    st.session_state.cluster_result = {"success": False, "error": str(e)}
            else:
                st.session_state.cluster_result = {"success": False, "error": "Need at least 2 numeric columns."}
            progress.progress(10 / 11, text="Clustering done")

            progress.progress(1.0, text="All analyses complete! ");
            status.update(label="Analysis complete!", state="complete")
            st.session_state.analysis_done = True
            st.balloons()
            time.sleep(1)
            st.rerun()

    # --- Display results ---
    if st.session_state.analysis_done:
        st.success("Analysis completed successfully!")

        tabs = st.tabs([
            " Overview", " Charts", " Correlation",
            " Regression", " Classification",
            " Hypothesis", " Features", " Clustering",
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
                    st.caption("**{}**".format(col))
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
            if st.session_state.reg_result and st.session_state.reg_result.get("success"):
                reg = st.session_state.reg_result
                c1, c2, c3 = st.columns(3)
                c1.metric("R-squared", reg["r_squared"])
                c2.metric("Adj R-squared", reg["adj_r_squared"])
                c3.metric("F-stat (p)", "{:.3f} (p={:.4f})".format(reg["f_stat"], reg["f_pvalue"]))
                st.text(reg.get("interpretation", ""))
                if not reg.get("coefficients", pd.DataFrame()).empty:
                    st.subheader("Coefficients")
                    st.dataframe(reg["coefficients"], use_container_width=True)
                if reg.get("residual_plot_png"):
                    st.image(reg["residual_plot_png"], caption="Residual Diagnostics")
                if reg.get("summary_html"):
                    with st.expander("Full OLS Summary"):
                        st.components.v1.html(reg["summary_html"], height=600, scrolling=True)
            # Multi-model comparison
            if st.session_state.reg_compare_result and st.session_state.reg_compare_result.get("success"):
                st.subheader("Multi-Model Comparison")
                comp = st.session_state.reg_compare_result
                comp_df = pd.DataFrame(comp.get("metrics", []))
                if not comp_df.empty:
                    st.dataframe(comp_df, use_container_width=True)
                st.info("Best model: **{}**".format(comp.get("best_model", "N/A")))
                for k, png in comp.get("charts", {}).items():
                    st.image(png, caption=k.replace("_", " ").title())

        # Tab: Classification
        with tabs[4]:
            st.subheader("Classification Analysis")
            if st.session_state.class_result and st.session_state.class_result.get("success"):
                cr = st.session_state.class_result
                st.success("Best model: **{}** | {} classes".format(cr["best_model"], cr["n_classes"]))
                comp = pd.DataFrame(cr.get("metrics", []))
                if not comp.empty:
                    st.dataframe(comp, use_container_width=True)
                for k, b64 in cr.get("charts", {}).items():
                    try:
                        st.image(base64.b64decode(b64), caption=k.replace("_", " ").title())
                    except Exception:
                        pass
            else:
                st.info(st.session_state.class_result.get("error", "Classification not applicable for this dataset.") if st.session_state.class_result else "No result.")

        # Tab: Hypothesis
        with tabs[5]:
            st.subheader("Hypothesis Testing")
            if st.session_state.hyp_result:
                hyp = st.session_state.hyp_result
                st.info(hyp.get("summary", ""))
                if hyp.get("t_tests"):
                    st.caption("Independent t-Tests (with Cohen's d)")
                    st.dataframe(pd.DataFrame(hyp["t_tests"]), use_container_width=True)
                if hyp.get("anovas"):
                    st.caption("One-way ANOVA (with Eta-squared)")
                    st.dataframe(pd.DataFrame(hyp["anovas"]), use_container_width=True)
                if hyp.get("chi_squares"):
                    st.caption("Chi-Square Tests (with Cramer's V)")
                    st.dataframe(pd.DataFrame(hyp["chi_squares"]), use_container_width=True)

        # Tab: Feature Importance
        with tabs[6]:
            st.subheader("Feature Importance Analysis")
            if st.session_state.fi_result and st.session_state.fi_result.get("success"):
                fi = st.session_state.fi_result
                methods = fi.get("methods", {})
                if methods.get("random_forest"):
                    st.caption("Random Forest Importance")
                    st.dataframe(pd.DataFrame(methods["random_forest"]), use_container_width=True)
                if methods.get("mutual_information"):
                    st.caption("Mutual Information")
                    st.dataframe(pd.DataFrame(methods["mutual_information"]), use_container_width=True)
                if methods.get("target_correlation"):
                    st.caption("Target Correlation")
                    st.dataframe(pd.DataFrame(methods["target_correlation"]), use_container_width=True)
                for k, b64 in fi.get("charts", {}).items():
                    try:
                        st.image(base64.b64decode(b64), caption=k.replace("_", " ").title())
                    except Exception:
                        pass
            else:
                st.info("Feature importance not applicable.")

        # Tab: Clustering
        with tabs[7]:
            st.subheader("Clustering Analysis")
            if st.session_state.cluster_result and st.session_state.cluster_result.get("success"):
                cl = st.session_state.cluster_result
                c1, c2, c3 = st.columns(3)
                c1.metric("Optimal k", cl["best_k"])
                c2.metric("Silhouette", "{:.3f}".format(max(s["silhouette"] for s in cl["silhouette_scores"])))
                c3.metric("Davies-Bouldin", str(cl.get("db_score", "N/A")))
                profiles = pd.DataFrame(cl.get("profiles", []))
                if not profiles.empty:
                    st.subheader("Cluster Profiles")
                    st.dataframe(profiles, use_container_width=True)
                for k, b64 in cl.get("charts", {}).items():
                    try:
                        st.image(base64.b64decode(b64), caption=k.replace("_", " ").title())
                    except Exception:
                        pass
            else:
                st.info("Clustering not applicable -- need at least 2 numeric columns.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Back to Configure", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col_b:
            if st.button("Next: Download Report", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

# ================================================================
# STEP 4: Download Report
# ================================================================
elif st.session_state.step == 4:
    st.title(" Download Report")
    if not st.session_state.analysis_done:
        st.warning("Please run the analysis first.")
    else:
        st.success("Analysis complete. Generate your comprehensive report below.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Word Report (.docx)")
            if st.button("Generate Full Report", type="primary", use_container_width=True):
                with st.spinner("Building comprehensive report..."):
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
                            class_result=st.session_state.class_result,
                            cluster_result=st.session_state.cluster_result,
                            fi_result=st.session_state.fi_result,
                            reg_compare_result=st.session_state.reg_compare_result,
                            target_col=st.session_state.target_col,
                            analysis_mode=st.session_state.analysis_mode,
                        auto_ml_result=st.session_state.auto_ml_result,
                        preprocess_log=st.session_state.preprocess_log,
                        problem_statement=st.session_state.problem_statement,
                        )
                        st.session_state.report_path = path
                        st.success("Report generated!")
                    except Exception as e:
                        st.error("Failed to generate report: {}".format(e))

            if st.session_state.report_path and os.path.exists(st.session_state.report_path):
                with open(st.session_state.report_path, "rb") as f:
                    _rbytes = f.read()
                st.download_button(" Download Word Report", data=_rbytes,
                                   file_name=os.path.basename(st.session_state.report_path),
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   use_container_width=True, key="dl_docx_v2")

        with col2:
            st.subheader("Tableau Extract (.hyper)")
            if st.button("Generate Tableau Extract", type="primary", use_container_width=True):
                with st.spinner("Exporting..."):
                    try:
                        hpath = export_to_hyper(st.session_state.df)
                        st.session_state.hyper_path = hpath
                        st.success("Tableau extract generated!")
                    except Exception as e:
                        st.error("Failed: {}".format(e))

            if st.session_state.hyper_path and os.path.exists(st.session_state.hyper_path):
                with open(st.session_state.hyper_path, "rb") as f:
                    _hbytes = f.read()
                st.download_button(" Download Tableau Extract", data=_hbytes,
                                   file_name="analysis_export.hyper",
                                   mime="application/octet-stream",
                                   use_container_width=True, key="dl_hyper_v2")

        st.divider()
        st.subheader("Session Summary")
        st.json({"Dataset": st.session_state.dataset_name,
                 "Shape": list(st.session_state.df.shape) if st.session_state.df is not None else None,
                 "Target": st.session_state.target_col,
                 "Problem": st.session_state.problem_statement[:100] + "..." if len(st.session_state.problem_statement) > 100 else st.session_state.problem_statement,
                 "Mode": st.session_state.analysis_mode,
                 "Numeric Cols": len(st.session_state.col_types.get("numeric", [])),
                 "Categorical Cols": len(st.session_state.col_types.get("categorical", [])),
                 "Analyses Run": 11})
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Back to Results", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col_b:
            if st.button(" Start New Analysis", use_container_width=True):
                for k in DEFAULTS:
                    st.session_state[k] = DEFAULTS[k]
                st.rerun()

st.sidebar.divider()
if st.session_state.df is not None:
    st.sidebar.caption(" {}".format(st.session_state.dataset_name))
    st.sidebar.caption(" {} files loaded".format(len(st.session_state.get("uploaded_files", []))))
    st.sidebar.caption(" Active: {} x{}".format(st.session_state.dataset_name, st.session_state.df.shape[0], st.session_state.df.shape[1]))
