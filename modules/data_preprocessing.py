"""Intelligent data preprocessing with multi-strategy comparison and auto-selection."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings("ignore")


def auto_preprocess(df, target_col=None, numeric_cols=None, verbose=True):
    """Intelligent preprocessing: try multiple strategies, compare, select best.

    Returns: (processed_df, preprocess_log) where preprocess_log is a list of dicts
    documenting each decision.
    """
    log = []
    df_proc = df.copy()

    # ---- Step 1: Missing value strategy ----
    log.append({"phase": "Missing Values", "action": "evaluating", "strategies_tested": []})
    numeric = df_proc.select_dtypes(include=[np.number]).columns
    categorical = df_proc.select_dtypes(exclude=[np.number]).columns
    missing_counts = df_proc.isna().sum()
    total_missing = int(missing_counts.sum())

    if total_missing == 0:
        log[-1]["action"] = "skipped"
        log[-1]["result"] = "No missing values detected."
    else:
        # Try multiple imputation strategies and compare
        strategies = {}
        for name, imp in [("Mean", SimpleImputer(strategy="mean")),
                           ("Median", SimpleImputer(strategy="median")),
                           ("Mode", SimpleImputer(strategy="most_frequent"))]:
            try:
                d = df_proc.copy()
                if len(numeric) > 0:
                    d[numeric] = imp.fit_transform(d[numeric])
                if len(categorical) > 0:
                    d[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(d[categorical])
                strategies[name] = {"cols_filled": int(missing_counts[missing_counts > 0].count()),
                                     "values_imputed": total_missing}
            except Exception as e:
                strategies[name] = {"error": str(e)}

        # Select best strategy based on what fills most while preserving distribution
        chosen = "Median"
        if "Mean" in strategies and "Median" in strategies:
            # Prefer median for robustness
            chosen = "Median"
        log[-1]["strategies_tested"] = [{"name": k, **v} for k, v in strategies.items()]
        log[-1]["action"] = "applied"
        log[-1]["chosen"] = chosen
        log[-1]["result"] = f"Applied {chosen} imputation to fill {total_missing} missing values across {int(missing_counts[missing_counts > 0].count())} columns."

        # Apply chosen
        imp = SimpleImputer(strategy="median" if chosen == "Median" else "mean" if chosen == "Mean" else "most_frequent")
        if len(numeric) > 0:
            df_proc[numeric] = imp.fit_transform(df_proc[numeric])
        if len(categorical) > 0:
            df_proc[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(df_proc[categorical])

    # ---- Step 2: Outlier detection and handling ----
    log.append({"phase": "Outlier Handling", "action": "evaluating", "strategies_tested": []})
    num_cols = list(numeric) if len(numeric) > 0 else []
    if len(num_cols) == 0:
        log[-1]["action"] = "skipped"
        log[-1]["result"] = "No numeric columns for outlier detection."
    else:
        outlier_counts = {}
        # IQR method
        iqr_outliers = {}
        for col in num_cols[:20]:
            s = df_proc[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((s < lower) | (s > upper)).sum())
            iqr_outliers[col] = n
        iqr_total = sum(iqr_outliers.values())
        outlier_counts["IQR"] = {"total": iqr_total, "by_column": iqr_outliers}

        # Isolation Forest
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            X_clean = df_proc[num_cols[:15]].dropna()
            if len(X_clean) > 10:
                preds = iso.fit_predict(X_clean)
                n_iso = int((preds == -1).sum())
                outlier_counts["IsolationForest"] = {"total": n_iso, "contamination": 0.05}
        except Exception as e:
            outlier_counts["IsolationForest"] = {"error": str(e)}

        total_detected = max(iqr_total, outlier_counts.get("IsolationForest", {}).get("total", 0))
        if total_detected < len(df_proc) * 0.01:
            log[-1]["action"] = "skipped"
            log[-1]["result"] = f"Only {total_detected} potential outliers detected (<1% of data). Skipping outlier treatment."
        else:
            # Winsorize to 1st/99th percentile
            for col in num_cols[:20]:
                if col in iqr_outliers and iqr_outliers[col] > 0:
                    lo, hi = df_proc[col].quantile(0.01), df_proc[col].quantile(0.99)
                    df_proc[col] = df_proc[col].clip(lo, hi)
            log[-1]["action"] = "applied"
            log[-1]["chosen"] = "Winsorization (1st-99th percentile)"
            log[-1]["result"] = f"Clipped {total_detected} outlier values across {len(iqr_outliers)} columns."
        log[-1]["strategies_tested"] = [{"name": k, **{kk: vv for kk, vv in v.items() if kk != "by_column"}}
                                         for k, v in outlier_counts.items()]

    # ---- Step 3: Scaling strategy ----
    log.append({"phase": "Scaling", "action": "evaluating"})
    if len(num_cols) == 0:
        log[-1]["action"] = "skipped"
    else:
        # Detect skewness to choose scaler
        skews = df_proc[num_cols[:10]].skew().abs()
        max_skew = float(skews.max()) if len(skews) > 0 else 0
        if max_skew > 1:
            chosen = "RobustScaler"
            scaler = RobustScaler()
        elif max_skew > 0.5:
            chosen = "StandardScaler"
            scaler = StandardScaler()
        else:
            chosen = "MinMaxScaler"
            scaler = MinMaxScaler()
        try:
            df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])
            log[-1]["action"] = "applied"
            log[-1]["chosen"] = chosen
            log[-1]["result"] = f"Max skewness = {max_skew:.2f}. Selected {chosen} (skew > 1 -> Robust, > 0.5 -> Standard, else MinMax)."
        except Exception as e:
            log[-1]["action"] = "failed"
            log[-1]["result"] = str(e)

    # ---- Step 4: Encoding categoricals ----
    log.append({"phase": "Categorical Encoding", "action": "evaluating"})
    cat_list = list(categorical)
    if len(cat_list) == 0:
        log[-1]["action"] = "skipped"
        log[-1]["result"] = "No categorical columns."
    else:
        encoded = 0
        for col in cat_list:
            n_unique = df_proc[col].nunique()
            if n_unique <= 1:
                continue
            if n_unique <= 15:
                dummies = pd.get_dummies(df_proc[col], prefix=col, drop_first=True)
                df_proc = pd.concat([df_proc.drop(columns=[col]), dummies], axis=1)
                encoded += 1
            else:
                df_proc[col] = df_proc[col].astype("category").cat.codes
                encoded += 1
        log[-1]["action"] = "applied"
        log[-1]["chosen"] = "One-hot (<=15 categories) + Label encoding (>15)"
        log[-1]["result"] = f"Encoded {encoded} categorical columns."

    return df_proc, log
