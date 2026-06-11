"""Hypothesis tests: independent t-test, one-way ANOVA, chi-square test, effect sizes."""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from itertools import combinations


def cohens_d(x, y):
    """Calculate Cohen's d effect size for independent samples."""
    n1, n2 = len(x), len(y)
    s_pooled = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled


def eta_squared(f_stat, df1, df2):
    """Calculate eta-squared from ANOVA F-statistic."""
    if f_stat is None or df2 == 0:
        return 0.0
    return (f_stat * df1) / (f_stat * df1 + df2)


def cramers_v(chi2, n, r, k):
    """Calculate Cramer's V from chi-square."""
    if chi2 is None or n == 0 or min(r, k) <= 1:
        return 0.0
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def _interpret_effect(effect_type, value):
    """Interpret effect size magnitude."""
    if effect_type == "cohens_d":
        if abs(value) < 0.2:
            return "Negligible"
        elif abs(value) < 0.5:
            return "Small"
        elif abs(value) < 0.8:
            return "Medium"
        else:
            return "Large"
    elif effect_type == "eta_squared":
        if value < 0.01:
            return "Negligible"
        elif value < 0.06:
            return "Small"
        elif value < 0.14:
            return "Medium"
        else:
            return "Large"
    elif effect_type == "cramers_v":
        if value < 0.1:
            return "Negligible"
        elif value < 0.3:
            return "Small"
        elif value < 0.5:
            return "Medium"
        else:
            return "Large"
    return "Unknown"


def _independent_ttest(df, numeric_col, group_col):
    """Independent t-test with Cohen's d for each pair of groups."""
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return []
    results = []
    for g1, g2 in combinations(groups, 2):
        sample1 = df.loc[df[group_col] == g1, numeric_col].dropna()
        sample2 = df.loc[df[group_col] == g2, numeric_col].dropna()
        if len(sample1) < 3 or len(sample2) < 3:
            continue
        try:
            t_stat, p_val = sp_stats.ttest_ind(sample1, sample2, equal_var=False)
            d = cohens_d(sample1, sample2)
            results.append({"Test": "Independent t-test", "Variable": numeric_col,
                            "Group 1": str(g1), "Group 2": str(g2),
                            "N1": len(sample1), "N2": len(sample2),
                            "Mean1": round(float(sample1.mean()), 4),
                            "Mean2": round(float(sample2.mean()), 4),
                            "t-Statistic": round(float(t_stat), 4),
                            "p-Value": round(float(p_val), 4),
                            "Cohen's d": round(float(d), 4),
                            "Effect Size": _interpret_effect("cohens_d", d),
                            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"})
        except Exception:
            continue
    return results


def _anova(df, numeric_col, group_col):
    """One-way ANOVA with eta-squared."""
    groups = df[group_col].dropna().unique()
    samples = [df.loc[df[group_col] == g, numeric_col].dropna().values for g in groups]
    samples = [s for s in samples if len(s) >= 3]
    if len(samples) < 2:
        return None
    try:
        f_stat, p_val = sp_stats.f_oneway(*samples)
        total_n = sum(len(s) for s in samples)
        df1 = len(samples) - 1
        df2 = total_n - len(samples)
        eta2 = eta_squared(f_stat, df1, df2)
        return {"Test": "One-way ANOVA", "Variable": numeric_col, "Grouping": group_col,
                "Groups": len(samples), "F-Statistic": round(float(f_stat), 4),
                "p-Value": round(float(p_val), 4),
                "Eta-squared": round(float(eta2), 4),
                "Effect Size": _interpret_effect("eta_squared", eta2),
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"}
    except Exception:
        return None


def _chi_square(df, col1, col2):
    """Chi-square test with Cramer's V."""
    try:
        ctab = pd.crosstab(df[col1], df[col2])
        if ctab.shape[0] < 2 or ctab.shape[1] < 2:
            return None
        chi2, p_val, dof, expected = sp_stats.chi2_contingency(ctab)
        n = ctab.sum().sum()
        v = cramers_v(chi2, n, ctab.shape[0], ctab.shape[1])
        return {"Test": "Chi-square", "Variable 1": col1, "Variable 2": col2,
                "Chi-squared": round(float(chi2), 4), "df": dof,
                "p-Value": round(float(p_val), 4),
                "Cramer's V": round(float(v), 4),
                "Effect Size": _interpret_effect("cramers_v", v),
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"}
    except Exception:
        return None


def hypothesis_tests(df, col_types, target_col=None):
    """Run a battery of hypothesis tests with effect sizes."""
    numeric = col_types.get("numeric", [])
    categorical = col_types.get("categorical", [])

    t_test_results = []
    anova_results = []
    chi2_results = []

    for cat_col in categorical:
        n_unique = df[cat_col].nunique()
        if n_unique < 2 or n_unique > 20:
            continue
        for num_col in numeric:
            if n_unique == 2:
                t_test_results.extend(_independent_ttest(df, num_col, cat_col))
            elif 3 <= n_unique:
                anova = _anova(df, num_col, cat_col)
                if anova:
                    anova_results.append(anova)

    if len(categorical) >= 2:
        for i in range(min(len(categorical), 4)):
            for j in range(i + 1, min(len(categorical), 4)):
                result = _chi_square(df, categorical[i], categorical[j])
                if result:
                    chi2_results.append(result)

    total_sig = (sum(1 for r in t_test_results if r.get("Significant (p<0.05)") == "Yes") +
                 sum(1 for r in anova_results if r.get("Significant (p<0.05)") == "Yes") +
                 sum(1 for r in chi2_results if r.get("Significant (p<0.05)") == "Yes"))
    total_tests = len(t_test_results) + len(anova_results) + len(chi2_results)

    # Find largest effects
    effects = []
    for r in t_test_results:
        effects.append((r["Variable"] + " vs " + r["Group 1"] + "/" + r["Group 2"], abs(r.get("Cohen's d", 0))))
    for r in anova_results:
        effects.append((r["Variable"] + " by " + r["Grouping"], r.get("Eta-squared", 0)))
    for r in chi2_results:
        effects.append((r["Variable 1"] + " x " + r["Variable 2"], r.get("Cramer's V", 0)))
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    top_effects = effects[:5]

    summary = "Ran {} hypothesis tests total. {} tests yielded statistically significant results (p < 0.05).".format(
        total_tests, total_sig)
    if top_effects:
        summary += " Largest effects: {}.".format("; ".join(
            "{} ({:.3f})".format(e[0], e[1]) for e in top_effects if abs(e[1]) > 0.01))

    return {"t_tests": t_test_results, "anovas": anova_results, "chi_squares": chi2_results,
            "summary": summary, "total_tests": total_tests, "significant_count": total_sig}
