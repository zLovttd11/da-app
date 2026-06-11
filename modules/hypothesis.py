"""Hypothesis tests: independent t-test, one-way ANOVA, chi-square test."""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from itertools import combinations


def _independent_ttest(df: pd.DataFrame, numeric_col: str, group_col: str) -> list[dict]:
    """Independent t-test for each pair of groups."""
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
            results.append({
                "Test": "Independent t-test",
                "Variable": numeric_col,
                "Group 1": str(g1), "Group 2": str(g2),
                "N1": len(sample1), "N2": len(sample2),
                "Mean1": round(float(sample1.mean()), 4),
                "Mean2": round(float(sample2.mean()), 4),
                "t-Statistic": round(float(t_stat), 4),
                "p-Value": round(float(p_val), 4),
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
            })
        except Exception:
            continue
    return results


def _anova(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict | None:
    """One-way ANOVA."""
    groups = df[group_col].dropna().unique()
    samples = [df.loc[df[group_col] == g, numeric_col].dropna().values for g in groups]
    samples = [s for s in samples if len(s) >= 3]
    if len(samples) < 2:
        return None
    try:
        f_stat, p_val = sp_stats.f_oneway(*samples)
        return {
            "Test": "One-way ANOVA",
            "Variable": numeric_col,
            "Grouping": group_col,
            "Groups": len(samples),
            "F-Statistic": round(float(f_stat), 4),
            "p-Value": round(float(p_val), 4),
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
        }
    except Exception:
        return None


def _chi_square(df: pd.DataFrame, col1: str, col2: str) -> dict | None:
    """Chi-square test of independence between two categorical variables."""
    try:
        ctab = pd.crosstab(df[col1], df[col2])
        if ctab.shape[0] < 2 or ctab.shape[1] < 2:
            return None
        chi2, p_val, dof, expected = sp_stats.chi2_contingency(ctab)
        return {
            "Test": "Chi-square",
            "Variable 1": col1, "Variable 2": col2,
            "Chi²": round(float(chi2), 4),
            "df": dof,
            "p-Value": round(float(p_val), 4),
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
        }
    except Exception:
        return None


def hypothesis_tests(df: pd.DataFrame, col_types: dict,
                     target_col: str | None = None) -> dict:
    """Run a battery of hypothesis tests and return structured results.

    Returns dict with keys:
    - t_tests: list of t-test result dicts
    - anovas: list of ANOVA result dicts
    - chi_squares: list of chi-square result dicts
    - summary: text summary
    """
    numeric = col_types.get("numeric", [])
    categorical = col_types.get("categorical", [])

    t_test_results: list[dict] = []
    anova_results: list[dict] = []
    chi2_results: list[dict] = []

    # t-tests and ANOVA: for each categorical with 2-10 groups, test each numeric
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

    # Chi-square tests between categorical pairs (up to 6 combinations)
    if len(categorical) >= 2:
        for i in range(min(len(categorical), 4)):
            for j in range(i + 1, min(len(categorical), 4)):
                result = _chi_square(df, categorical[i], categorical[j])
                if result:
                    chi2_results.append(result)

    # Summary
    total_sig = (sum(1 for r in t_test_results if r["Significant (p<0.05)"] == "Yes") +
                 sum(1 for r in anova_results if r["Significant (p<0.05)"] == "Yes") +
                 sum(1 for r in chi2_results if r["Significant (p<0.05)"] == "Yes"))
    total_tests = len(t_test_results) + len(anova_results) + len(chi2_results)
    summary = (f"Ran {total_tests} hypothesis tests total. "
               f"{total_sig} tests yielded statistically significant results (p < 0.05).")

    return {
        "t_tests": t_test_results,
        "anovas": anova_results,
        "chi_squares": chi2_results,
        "summary": summary,
    }
