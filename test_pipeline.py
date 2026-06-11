"""End-to-end test of the analysis pipeline."""
import sys
sys.path.insert(0, r'D:\DA APP')
import pandas as pd
from modules.data_loader import infer_column_types
from modules.data_profiler import profile_data
from modules.descriptive import descriptive_stats, categorical_summary
from modules.visualization import generate_plots, generate_static_charts
from modules.correlation import correlation_analysis
from modules.regression import regression_analysis
from modules.hypothesis import hypothesis_tests
from modules.report_generator import generate_report
from modules.tableau_exporter import export_to_hyper

df = pd.read_csv(r'D:\DA APP\outputs\sample_data.csv')
dataset_name = 'sample_data.csv'
print(f'Loaded: {df.shape}')

col_types = infer_column_types(df)
print(f'Column types: numeric={len(col_types["numeric"])}, categorical={len(col_types["categorical"])}')

prof = profile_data(df)
print(f'Profile: shape={prof["shape"]}, missing={prof["total_missing_cells"]}')

numeric_cols = col_types["numeric"]
desc = descriptive_stats(df, numeric_cols)
print(f'Descriptive stats: {desc.shape[0]} variables')

cat = categorical_summary(df, col_types["categorical"])
print(f'Categorical summaries: {len(cat)} variables')

static = generate_static_charts(df, col_types, target_col="income")
print(f'Static charts: hist={len(static["histograms"])}, box={len(static["boxplot"])}, bar={len(static["barcharts"])}')

plots = generate_plots(df, col_types, target_col="income")
print(f'Interactive plots: dist={len(plots["distributions"])}, box={len(plots["boxplots"])}, cat={len(plots["categorical"])}, scatter={len(plots["scatter_matrix"])}, qq={len(plots["qq_plots"])}')

corr = correlation_analysis(df, numeric_cols)
print(f'Correlation: shape={corr["matrix"].shape}, heatmap={len(corr["heatmap_png"])} bytes')

predictors = [c for c in numeric_cols if c != "income"]
reg = regression_analysis(df, predictors, "income")
print(f'Regression: success={reg["success"]}, R2={reg.get("r_squared", "N/A")}')

hyp = hypothesis_tests(df, col_types)
print(f'Hypothesis: t_tests={len(hyp["t_tests"])}, anovas={len(hyp["anovas"])}, chi2={len(hyp["chi_squares"])}')

report_path = generate_report(
    dataset_name=dataset_name, df=df, col_types=col_types, profile=prof,
    desc_stats=desc, cat_summary=cat, static_charts=static,
    corr_result=corr, reg_result=reg, hyp_result=hyp, target_col="income",
)
print(f'Report generated: {report_path}')

hyper_path = export_to_hyper(df)
print(f'Hyper export: {hyper_path}')

print("\n=== ALL TESTS PASSED ===")
