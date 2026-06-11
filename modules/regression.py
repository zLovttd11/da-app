"""OLS linear regression via statsmodels, with residual diagnostics."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.helpers import setup_matplotlib_style, figure_to_png_bytes


def regression_analysis(df: pd.DataFrame, predictors: list[str],
                        target: str) -> dict:
    """Run OLS linear regression and return a full results dictionary.

    Keys:
    - success: bool
    - summary_html: OLS summary as HTML string
    - coefficients: DataFrame of coefficient estimates
    - r_squared, adj_r_squared: floats
    - f_stat, f_pvalue: model significance
    - residual_plot_png: PNG bytes of residual diagnostics
    - interpretation: text interpretation
    - error: error message if success is False
    """
    if not predictors or not target:
        return {"success": False, "error": "Missing predictors or target variable."}

    setup_matplotlib_style()
    data = df[[target] + predictors].dropna()
    if len(data) < len(predictors) + 10:
        return {"success": False, "error": f"Only {len(data)} complete cases — insufficient for regression."}

    try:
        X = sm.add_constant(data[predictors].astype(float))
        y = data[target].astype(float)
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Coefficients table
    coef_df = pd.DataFrame({
        "Predictor": model.params.index,
        "Coefficient": model.params.values.round(4),
        "Std Error": model.bse.values.round(4),
        "t-Statistic": model.tvalues.values.round(4),
        "p-Value": model.pvalues.values.round(4),
    })
    coef_df["Significant (p<0.05)"] = coef_df["p-Value"].apply(lambda p: "Yes" if p < 0.05 else "No")

    # Residual diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Residuals vs fitted
    fitted = model.fittedvalues
    residuals = model.resid
    axes[0, 0].scatter(fitted, residuals, alpha=0.5, color="#3366cc", s=20)
    axes[0, 0].axhline(y=0, color="#dc3912", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # Q-Q plot
    from scipy import stats as sp_stats
    qq = sp_stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(qq[0][0], qq[0][1], alpha=0.5, color="#3366cc", s=20)
    slope, intercept, _ = qq[1]
    x_vals = np.array(qq[0][0])
    axes[0, 1].plot(x_vals, slope * x_vals + intercept, color="#dc3912", linewidth=1)
    axes[0, 1].set_title("Q-Q Plot of Residuals")
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")

    # Scale-Location
    std_resid = residuals / float(np.std(residuals)) if np.std(residuals) > 0 else residuals
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.5, color="#3366cc", s=20)
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("sqrt(|Standardized Residuals|)")

    # Residuals histogram
    axes[1, 1].hist(residuals, bins=20, color="#3366cc", edgecolor="white", alpha=0.85)
    axes[1, 1].set_title("Residuals Histogram")
    axes[1, 1].set_xlabel("Residuals")

    fig.tight_layout()
    residual_png = figure_to_png_bytes(fig)
    plt.close(fig)

    # Interpretation
    sig_predictors = [r["Predictor"] for _, r in coef_df.iterrows()
                      if r["Predictor"] != "const" and r["p-Value"] < 0.05]
    interpretation = (
        f"The model explains {model.rsquared:.2%} of the variance in {target} "
        f"(adjusted R² = {model.rsquared_adj:.2%}). "
    )
    if sig_predictors:
        interpretation += f"Significant predictors (p < 0.05): {', '.join(sig_predictors)}. "
    else:
        interpretation += "No predictors reached significance at the p < 0.05 level. "
    interpretation += (
        f"The overall model is {'significant' if model.f_pvalue < 0.05 else 'not significant'} "
        f"(F = {model.fvalue:.3f}, p = {model.f_pvalue:.4f})."
    )

    return {
        "success": True,
        "summary_html": model.summary().as_html(),
        "coefficients": coef_df,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_stat": round(float(model.fvalue), 4),
        "f_pvalue": round(float(model.f_pvalue), 4),
        "residual_plot_png": residual_png,
        "interpretation": interpretation,
        "error": None,
    }
