"""Classification: Logistic Regression, Random Forest, Gradient Boosting with performance metrics."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from utils.helpers import setup_matplotlib_style


def _prepare_data(df, target_col, numeric_cols):
    """Prepare features and target for classification."""
    feature_cols = [c for c in numeric_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]
    X_num = df[feature_cols].fillna(df[feature_cols].median()) if feature_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    X = X.loc[:, X.nunique() > 1]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return X_scaled, y_enc, le, X.columns.tolist()


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def classify_compare_models(df, target_col, numeric_cols, test_size=0.2, random_state=42):
    """Train multiple classifiers, compare performance, return full results."""
    setup_matplotlib_style()
    X, y, le, feature_names = _prepare_data(df, target_col, numeric_cols)
    if len(np.unique(y)) < 2:
        return {"success": False, "error": "Target must have at least 2 classes."}
    if len(np.unique(y)) > 50:
        return {"success": False, "error": "Too many classes. Consider grouping or use regression."}

    stratify_arg = y if len(np.unique(y)) <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    all_metrics = []
    roc_curves = []
    n_classes = len(np.unique(y))
    avg = "macro" if n_classes > 2 else "binary"

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {"model": name, "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                       "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                       "recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                       "f1_score": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)}
            n_splits_cv = min(5, min(np.bincount(y)))
            skf = StratifiedKFold(n_splits=max(2, n_splits_cv), shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            metrics["cv_mean"] = round(float(np.mean(cv_scores)), 4)
            metrics["cv_std"] = round(float(np.std(cv_scores)), 4)
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                importances = np.ones(len(feature_names))
            fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
            metrics["feature_importance"] = [{"feature": f, "importance": round(float(i), 4)} for f, i in fi]
            all_metrics.append(metrics)
            if n_classes == 2 and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                roc_curves.append({"model": name, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc), 4)})
        except Exception as e:
            all_metrics.append({"model": name, "error": str(e)})

    comparison_df = pd.DataFrame(all_metrics)
    valid = comparison_df[comparison_df["f1_score"].notna()] if not comparison_df.empty else comparison_df
    best_model = valid.loc[valid["f1_score"].idxmax(), "model"] if (not valid.empty and "f1_score" in valid.columns) else "N/A"

    charts = {}
    setup_matplotlib_style()

    if not valid.empty and "accuracy" in valid.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(valid))
        w = 0.25
        ax.bar(x - w, valid["accuracy"], w, label="Accuracy", color="#3366cc")
        ax.bar(x, valid["f1_score"], w, label="F1 Score", color="#dc3912")
        ax.bar(x + w, valid["cv_mean"], w, label="CV Mean", color="#109618")
        ax.set_xticks(x)
        ax.set_xticklabels(valid["model"], rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        charts["model_comparison"] = _fig_to_b64(fig)

    if roc_curves:
        fig, ax = plt.subplots(figsize=(7, 6))
        for rc in roc_curves:
            ax.plot(rc["fpr"], rc["tpr"], label="{} (AUC={})".format(rc["model"], rc["auc"]))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        fig.tight_layout()
        charts["roc_curves"] = _fig_to_b64(fig)

    return {"success": True, "metrics": all_metrics, "comparison_df": comparison_df.to_dict(orient="records"),
            "best_model": best_model, "n_classes": n_classes,
            "classes": le.classes_.tolist() if hasattr(le, "classes_") else [],
            "feature_names": feature_names, "charts": charts, "test_size": test_size}
