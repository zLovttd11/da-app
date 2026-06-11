"""Feature importance, mutual information, and target correlation analysis."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from utils.helpers import setup_matplotlib_style


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def feature_importance_analysis(df, target_col, numeric_cols, top_k=20, random_state=42):
    """Compute feature importance using multiple methods."""
    setup_matplotlib_style()
    feature_cols = [c for c in numeric_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]

    X_num = df[feature_cols].fillna(df[feature_cols].median()) if feature_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    X = X.loc[:, X.nunique() > 1]
    feature_names = X.columns.tolist()

    results = {"feature_names": feature_names, "methods": {}}

    is_classification = y.nunique() <= 20
    if is_classification:
        le = LabelEncoder()
        y_proc = le.fit_transform(y.astype(str))
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        y_proc = y.values
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    rf.fit(X, y_proc)
    rf_imp = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:top_k]
    results["methods"]["random_forest"] = [{"feature": f, "importance": round(float(i), 4)} for f, i in rf_imp]

    try:
        if is_classification:
            mi = mutual_info_classif(X, y_proc, random_state=random_state)
        else:
            mi = mutual_info_regression(X, y_proc, random_state=random_state)
        mi_imp = sorted(zip(feature_names, mi), key=lambda x: x[1], reverse=True)[:top_k]
        results["methods"]["mutual_information"] = [{"feature": f, "score": round(float(s), 4)} for f, s in mi_imp]
    except Exception:
        results["methods"]["mutual_information"] = []

    if not is_classification:
        corrs = []
        for col in feature_cols:
            valid = df[[col, target_col]].dropna()
            if len(valid) > 2:
                c = valid[col].corr(valid[target_col])
                if not np.isnan(c):
                    corrs.append((col, c))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        results["methods"]["target_correlation"] = [{"feature": f, "correlation": round(float(c), 4)}
                                                      for f, c in corrs[:top_k]]

    charts = {}
    if rf_imp:
        fig, ax = plt.subplots(figsize=(8, max(4, len(rf_imp) * 0.3)))
        names = [x[0][:30] for x in rf_imp]
        values = [x[1] for x in rf_imp]
        colors = ["#3366cc" if v > np.mean(values) else "#999999" for v in values]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importances")
        fig.tight_layout()
        charts["feature_importance"] = _fig_to_b64(fig)

    return {"success": True, "feature_names": feature_names, "methods": results["methods"],
            "charts": charts, "is_classification": is_classification}
