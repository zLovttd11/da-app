"""OLS linear regression via statsmodels, with multi-model comparison and residual diagnostics."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from utils.helpers import setup_matplotlib_style, figure_to_png_bytes


def regression_analysis(df, predictors, target):
    """Run OLS linear regression and return full results dictionary."""
    if not predictors or not target:
        return {"success": False, "error": "Missing predictors or target variable."}

    setup_matplotlib_style()
    data = df[[target] + predictors].dropna()
    if len(data) < len(predictors) + 10:
        return {"success": False, "error": "Only {} complete cases - insufficient for regression.".format(len(data))}

    try:
        X = sm.add_constant(data[predictors].astype(float))
        y = data[target].astype(float)
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return {"success": False, "error": str(e)}

    coef_df = pd.DataFrame({
        "Predictor": model.params.index,
        "Coefficient": model.params.values.round(4),
        "Std Error": model.bse.values.round(4),
        "t-Statistic": model.tvalues.values.round(4),
        "p-Value": model.pvalues.values.round(4),
    })
    coef_df["Significant (p<0.05)"] = coef_df["p-Value"].apply(lambda p: "Yes" if p < 0.05 else "No")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fitted = model.fittedvalues
    residuals = model.resid
    axes[0, 0].scatter(fitted, residuals, alpha=0.5, color="#3366cc", s=20)
    axes[0, 0].axhline(y=0, color="#dc3912", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    from scipy import stats as sp_stats
    qq = sp_stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(qq[0][0], qq[0][1], alpha=0.5, color="#3366cc", s=20)
    slope, intercept, _ = qq[1]
    x_vals = np.array(qq[0][0])
    axes[0, 1].plot(x_vals, slope * x_vals + intercept, color="#dc3912", linewidth=1)
    axes[0, 1].set_title("Q-Q Plot of Residuals")
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")
    std_resid = residuals / float(np.std(residuals)) if np.std(residuals) > 0 else residuals
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.5, color="#3366cc", s=20)
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("sqrt(|Standardized Residuals|)")
    axes[1, 1].hist(residuals, bins=20, color="#3366cc", edgecolor="white", alpha=0.85)
    axes[1, 1].set_title("Residuals Histogram")
    axes[1, 1].set_xlabel("Residuals")
    fig.tight_layout()
    residual_png = figure_to_png_bytes(fig)
    plt.close(fig)

    sig_predictors = [r["Predictor"] for _, r in coef_df.iterrows()
                      if r["Predictor"] != "const" and r["p-Value"] < 0.05]
    interpretation = "The model explains {:.2%} of the variance in {} (adjusted R-squared = {:.2%}). ".format(
        model.rsquared, target, model.rsquared_adj)
    if sig_predictors:
        interpretation += "Significant predictors (p < 0.05): {}.".format(", ".join(sig_predictors))
    else:
        interpretation += "No predictors reached significance at the p < 0.05 level. "
    interpretation += "The overall model is {} (F = {:.3f}, p = {:.4f}).".format(
        "significant" if model.f_pvalue < 0.05 else "not significant", model.fvalue, model.f_pvalue)

    return {"success": True, "summary_html": model.summary().as_html(), "coefficients": coef_df,
            "r_squared": round(model.rsquared, 4), "adj_r_squared": round(model.rsquared_adj, 4),
            "f_stat": round(float(model.fvalue), 4), "f_pvalue": round(float(model.f_pvalue), 4),
            "residual_plot_png": residual_png, "interpretation": interpretation, "error": None}


def regression_compare_models(df, predictors, target, cv_folds=5, random_state=42):
    """Compare OLS, LinearRegression, Random Forest, and Gradient Boosting."""
    if not predictors or not target:
        return {"success": False, "error": "Missing predictors or target variable."}

    setup_matplotlib_style()
    data = df[[target] + predictors].dropna()
    if len(data) < 20:
        return {"success": False, "error": "Insufficient data (need >= 20 complete cases)."}

    X = data[predictors].astype(float)
    y = data[target].astype(float)

    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    all_metrics = []
    feature_importance_combined = {}
    cv = KFold(n_splits=min(cv_folds, len(X) // 5), shuffle=True, random_state=random_state)

    for name, model in models.items():
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            metrics = {"model": name, "r2": round(float(r2_score(y, y_pred)), 4),
                       "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                       "mae": round(float(mean_absolute_error(y, y_pred)), 4),
                       "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
                       "cv_r2_std": round(float(np.std(cv_scores)), 4)}
            all_metrics.append(metrics)

            if hasattr(model, "feature_importances_"):
                for f, imp in zip(predictors, model.feature_importances_):
                    feature_importance_combined[f] = feature_importance_combined.get(f, 0) + imp
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 0:
                    for f, c in zip(predictors, np.abs(coef)):
                        feature_importance_combined[f] = feature_importance_combined.get(f, 0) + c
        except Exception as e:
            all_metrics.append({"model": name, "error": str(e)})

    comparison_df = pd.DataFrame(all_metrics)

    # Feature importance chart
    charts = {}
    if feature_importance_combined:
        fi = sorted(feature_importance_combined.items(), key=lambda x: x[1], reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.3)))
        names = [x[0][:30] for x in fi]
        values = [x[1] for x in fi]
        colors = ["#3366cc" if v > np.mean(values) else "#999999" for v in values]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Aggregate Importance")
        ax.set_title("Feature Importance (all models)")
        fig.tight_layout()
        charts["feature_importance"] = figure_to_png_bytes(fig)

    # Model comparison chart
    valid = comparison_df[comparison_df["r2"].notna()] if not comparison_df.empty else comparison_df
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(valid))
        w = 0.25
        ax.bar(x - w, valid["r2"], w, label="R-squared", color="#3366cc")
        ax.bar(x, valid["cv_r2_mean"], w, label="CV R-squared", color="#dc3912")
        ax.bar(x + w, [1 - r for r in valid["r2"]], w, label="1 - R-squared", color="#cccccc", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(valid["model"], rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Regression Model Comparison")
        ax.legend()
        fig.tight_layout()
        charts["model_comparison"] = figure_to_png_bytes(fig)

    best_model = valid.loc[valid["cv_r2_mean"].idxmax(), "model"] if (not valid.empty and "cv_r2_mean" in valid.columns) else "N/A"

    return {"success": True, "metrics": all_metrics,
            "comparison_df": comparison_df.to_dict(orient="records"),
            "best_model": best_model, "predictors": predictors, "target": target,
            "charts": charts}
