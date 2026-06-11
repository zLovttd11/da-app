"""Generate a Word (.docx) course report from analysis results."""

import os
import io
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def generate_report(
    dataset_name: str,
    df: pd.DataFrame,
    col_types: dict,
    profile: dict,
    desc_stats: pd.DataFrame,
    cat_summary: dict[str, pd.DataFrame],
    static_charts: dict[str, list[bytes]],
    corr_result: dict,
    reg_result: dict,
    hyp_result: dict,
    target_col: str | None = None,
) -> str:
    """Generate the Word report and return the output file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()

    # --- Page setup ---
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)

    # --- Helper functions ---
    def add_heading(text, level=1):
        h = doc.add_heading(text, level=level)
        for run in h.runs:
            run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
        return h

    def add_paragraph(text, bold=False, size=None):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = bold
        if size:
            run.font.size = Pt(size)
        return p

    def add_image_from_bytes(png_bytes, width_inches=5.5):
        if not png_bytes:
            return
        stream = io.BytesIO(png_bytes)
        doc.add_picture(stream, width=Inches(width_inches))
        doc.add_paragraph()

    def add_dataframe_table(pdf: pd.DataFrame, col_widths=None):
        if pdf.empty:
            add_paragraph("No data available.", size=10)
            return
        table = doc.add_table(rows=len(pdf) + 1, cols=len(pdf.columns), style="Light Grid Accent 1")
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, col_name in enumerate(pdf.columns):
            cell = table.rows[0].cells[j]
            cell.text = str(col_name)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(9)
        for i, (_, row) in enumerate(pdf.iterrows()):
            for j, val in enumerate(row):
                cell = table.rows[i + 1].cells[j]
                if isinstance(val, float):
                    cell.text = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2f}"
                else:
                    cell.text = str(val)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph()

    # =====================================================================
    # COVER PAGE
    # =====================================================================
    for _ in range(6):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Data Analysis Report")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(f"Dataset: {dataset_name}")
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_para.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    tool_para = doc.add_paragraph()
    tool_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = tool_para.add_run("Autonomous Analysis by DA App")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

    doc.add_page_break()

    # =====================================================================
    # 1. DATA OVERVIEW
    # =====================================================================
    add_heading("1. Data Overview", level=1)
    rows, cols = profile.get("shape", (0, 0))
    add_paragraph(f"This report analyzes the dataset \"{dataset_name}\", which contains {rows} observations and {cols} variables.")

    add_heading("1.1 Dataset Summary", level=2)
    summary_data = [
        ("Observations", str(rows)),
        ("Variables", str(cols)),
        ("Missing Cells", str(profile.get("total_missing_cells", 0))),
        ("Duplicate Rows", str(profile.get("duplicate_rows", 0))),
        ("Memory Usage", str(profile.get("memory_usage", "N/A"))),
        ("Target Variable", target_col if target_col else "None selected"),
    ]
    table = doc.add_table(rows=len(summary_data), cols=2, style="Light Grid Accent 1")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(summary_data):
        table.rows[i].cells[0].text = k
        table.rows[i].cells[1].text = v
        for cell in table.rows[i].cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()

    add_heading("1.2 Column Types", level=2)
    type_data = [
        ("Numeric", len(col_types.get("numeric", []))),
        ("Categorical", len(col_types.get("categorical", []))),
        ("Datetime", len(col_types.get("datetime", []))),
        ("Text / Other", len(col_types.get("text", []))),
    ]
    table = doc.add_table(rows=len(type_data) + 1, cols=2, style="Light Grid Accent 1")
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.rows[0].cells[0].text = "Type"
    table.rows[0].cells[1].text = "Count"
    for i, (k, v) in enumerate(type_data):
        table.rows[i + 1].cells[0].text = k
        table.rows[i + 1].cells[1].text = str(v)
    doc.add_paragraph()

    add_heading("1.3 Missing Values", level=2)
    missing = profile.get("missing", {})
    if missing:
        miss_df = pd.DataFrame([
            {"Variable": col, "Missing Count": info["count"], "Missing %": f"{info['percent']}%"}
            for col, info in missing.items() if info["count"] > 0
        ])
        if not miss_df.empty:
            add_dataframe_table(miss_df)
        else:
            add_paragraph("No missing values detected in the dataset.", size=10)
    doc.add_page_break()

    # =====================================================================
    # 2. DESCRIPTIVE STATISTICS
    # =====================================================================
    add_heading("2. Descriptive Statistics", level=1)
    add_paragraph("The following tables summarize the central tendency, dispersion, and shape of numeric variables.")

    if not desc_stats.empty:
        add_heading("2.1 Numeric Variables", level=2)
        add_dataframe_table(desc_stats)

    if cat_summary:
        add_heading("2.2 Categorical Variables", level=2)
        for col, freq_df in cat_summary.items():
            add_paragraph(f"Frequency table for '{col}':", bold=True, size=10)
            add_dataframe_table(freq_df)

    doc.add_page_break()

    # =====================================================================
    # 3. VISUALIZATIONS
    # =====================================================================
    add_heading("3. Data Visualizations", level=1)

    if static_charts.get("histograms"):
        add_heading("3.1 Distribution Histograms", level=2)
        for i, png in enumerate(static_charts["histograms"]):
            add_image_from_bytes(png, width_inches=5.0)

    if static_charts.get("boxplot"):
        add_heading("3.2 Box Plot Comparison", level=2)
        for png in static_charts["boxplot"]:
            add_image_from_bytes(png, width_inches=5.5)

    if static_charts.get("barcharts"):
        add_heading("3.3 Categorical Bar Charts", level=2)
        for png in static_charts["barcharts"]:
            add_image_from_bytes(png, width_inches=5.0)

    doc.add_page_break()

    # =====================================================================
    # 4. CORRELATION ANALYSIS
    # =====================================================================
    add_heading("4. Correlation Analysis", level=1)
    corr_df = corr_result.get("matrix", pd.DataFrame())
    add_paragraph(f"Method: {corr_result.get('method', 'pearson').capitalize()} correlation.")
    add_paragraph(corr_result.get("summary", ""), size=10)

    if not corr_df.empty:
        add_heading("4.1 Correlation Matrix", level=2)
        display_df = corr_df.round(3).reset_index()
        display_df.rename(columns={"index": "Variable"}, inplace=True)
        add_dataframe_table(display_df)

    if corr_result.get("heatmap_png"):
        add_heading("4.2 Correlation Heatmap", level=2)
        add_image_from_bytes(corr_result["heatmap_png"], width_inches=5.5)

    add_paragraph(
        "The correlation matrix above reveals the pairwise linear relationships "
        "between numeric variables. Variables with |r| > 0.7 indicate strong "
        "correlation; 0.3 < |r| < 0.7 indicate moderate correlation; and "
        "|r| < 0.3 indicate weak or negligible correlation."
    )
    doc.add_page_break()

    # =====================================================================
    # 5. REGRESSION ANALYSIS
    # =====================================================================
    add_heading("5. Regression Analysis", level=1)
    if reg_result.get("success"):
        add_paragraph(f"Target Variable: {target_col}")
        add_paragraph(f"R-squared: {reg_result['r_squared']}, Adjusted R-squared: {reg_result['adj_r_squared']}")
        add_paragraph(f"F-statistic: {reg_result['f_stat']}, p-value: {reg_result['f_pvalue']}")

        if not reg_result.get("coefficients", pd.DataFrame()).empty:
            add_heading("5.1 Coefficient Estimates", level=2)
            add_dataframe_table(reg_result["coefficients"])

        add_heading("5.2 Interpretation", level=2)
        add_paragraph(reg_result.get("interpretation", ""), size=10)

        if reg_result.get("residual_plot_png"):
            add_heading("5.3 Residual Diagnostics", level=2)
            add_image_from_bytes(reg_result["residual_plot_png"], width_inches=5.5)
    else:
        add_paragraph(f"Regression analysis could not be performed: {reg_result.get('error', 'Unknown error')}")

    doc.add_page_break()

    # =====================================================================
    # 6. HYPOTHESIS TESTING
    # =====================================================================
    add_heading("6. Hypothesis Testing", level=1)
    add_paragraph(hyp_result.get("summary", ""))

    if hyp_result.get("t_tests"):
        add_heading("6.1 Independent t-Tests", level=2)
        add_dataframe_table(pd.DataFrame(hyp_result["t_tests"]))

    if hyp_result.get("anovas"):
        add_heading("6.2 One-way ANOVA", level=2)
        add_dataframe_table(pd.DataFrame(hyp_result["anovas"]))

    if hyp_result.get("chi_squares"):
        add_heading("6.3 Chi-Square Tests", level=2)
        add_dataframe_table(pd.DataFrame(hyp_result["chi_squares"]))

    doc.add_page_break()

    # =====================================================================
    # 7. CONCLUSIONS
    # =====================================================================
    add_heading("7. Conclusions & Recommendations", level=1)

    n_numeric = len(col_types.get("numeric", []))
    n_cat = len(col_types.get("categorical", []))

    if reg_result.get("success"):
        reg_summary = (f"The regression model for '{target_col}' achieved an R-squared of "
                       f"{reg_result['r_squared']:.3f}, indicating that the selected predictors "
                       f"{'significantly' if reg_result.get('f_pvalue', 1) < 0.05 else 'do not significantly'} "
                       f"explain variation in the target variable.")
    else:
        reg_summary = "No regression model was fitted (either no target was selected or insufficient data)."

    top_corrs = corr_result.get("summary", "").split("\n")[1:4]
    corr_summary = "Key bivariate relationships: " + "; ".join(
        [line.strip() for line in top_corrs if line.strip()]
    ) if top_corrs else "No pairwise correlations exceeded the threshold for strong association."

    sig_hyp = sum(
        len(hyp_result.get(k, [])) for k in ["t_tests", "anovas", "chi_squares"]
    )
    hyp_summary = (f"A total of {sig_hyp} hypothesis tests were conducted. "
                   f"See Section 6 for detailed results.")

    numeric_cols = col_types.get("numeric", [])
    key_vars = ", ".join(numeric_cols[:5]) if numeric_cols else "the dataset variables"

    add_paragraph(
        f"This report analyzed the \"{dataset_name}\" dataset containing {rows} observations "
        f"across {cols} variables ({n_numeric} numeric, {n_cat} categorical)."
    )
    add_paragraph(reg_summary)
    add_paragraph(corr_summary)
    add_paragraph(hyp_summary)
    add_paragraph(
        f"These findings suggest that further investigation into {key_vars} "
        f"may yield actionable insights for decision-making. Future analyses could "
        f"explore interaction effects, non-linear transformations, or additional "
        f"data sources to strengthen the conclusions drawn here."
    )

    # --- Save ---
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"report_{safe_name}_{timestamp}.docx"
    doc.save(str(out_path))
    return str(out_path)
