"""Adaptive Auto-ML engine: auto task detection, multi-model comparison, hyperparameter tuning, ensemble, and method development documentation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, time, warnings
warnings.filterwarnings("ignore")

from utils.helpers import setup_matplotlib_style


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _prepare_xy(df, target_col, numeric_cols):
    """Prepare features and target for ML."""
    feature_cols = [c for c in numeric_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]
    X_num = df[feature_cols].fillna(df[feature_cols].median()) if feature_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    X = X.loc[:, X.nunique() > 1]
    return X, y, X.columns.tolist()


def _detect_task(df, target_col):
    """Detect whether this is classification or regression."""
    y = df[target_col].dropna()
    n_unique = y.nunique()
    if n_unique <= 2:
        return "binary_classification"
    elif n_unique <= 20:
        return "multiclass_classification"
    elif y.dtype == "object" or y.dtype.name == "category":
        return "multiclass_classification"
    else:
        return "regression"


def auto_analyze(df, target_col, numeric_cols, test_size=0.2, random_state=42, tune=False):
    """Full auto-ML pipeline: detect task, try all models, compare, select best, ensemble.

    Returns a dict with full method development log.
    """
    setup_matplotlib_style()
    dev_log = []
    charts = {}
    t0 = time.time()

    task = _detect_task(df, target_col)
    dev_log.append({"phase": "Task Detection", "result": f"Detected task: {task}.",
                     "n_unique_target": int(df[target_col].dropna().nunique()),
                     "target_dtype": str(df[target_col].dtype)})

    X, y, feature_names = _prepare_xy(df, target_col, numeric_cols)
    dev_log.append({"phase": "Feature Preparation", "result": f"{len(feature_names)} features prepared from {len(X)} samples.",
                     "n_features": len(feature_names), "n_samples": len(X)})

    is_class = "classification" in task

    # Split
    if is_class and len(np.unique(y)) <= 10:
        stratify = y
    else:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    dev_log.append({"phase": "Train/Test Split", "result": f"Train: {len(X_train)}, Test: {len(X_test)}, test_size={test_size}."})

    # ---- Phase: Model Comparison ----
    dev_log.append({"phase": "Model Comparison", "action": "evaluating", "models_tested": []})

    if is_class:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state, n_jobs=-1),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
        }
        if len(np.unique(y)) <= 2:
            models["SVM (RBF)"] = SVC(probability=True, random_state=random_state)
    else:
        models = {
            "Linear Regression": LinearRegression(n_jobs=-1),
            "Ridge (L2)": Ridge(alpha=1.0),
            "Lasso (L1)": Lasso(alpha=0.1, max_iter=3000),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
        }

    all_metrics = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            m = {"model": name}
            if is_class:
                avg = "binary" if "binary" in task else "macro"
                m["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                m["f1_score"] = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)
                m["precision"] = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
                m["recall"] = round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)
                scoring = "accuracy"
            else:
                m["r2"] = round(float(r2_score(y_test, y_pred)), 4)
                m["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                scoring = "r2"

            # Cross-validation
            cv = KFold(n_splits=min(5, len(y) // 5), shuffle=True, random_state=random_state)
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                m["cv_mean"] = round(float(np.mean(cv_scores)), 4)
                m["cv_std"] = round(float(np.std(cv_scores)), 4)
            except Exception:
                m["cv_mean"] = None
                m["cv_std"] = None

            all_metrics.append(m)
        except Exception as e:
            all_metrics.append({"model": name, "error": str(e)})

    metrics_df = pd.DataFrame(all_metrics)
    dev_log[-1]["models_tested"] = all_metrics

    # Select best model
    valid = metrics_df.dropna(subset=["cv_mean"] if "cv_mean" in metrics_df else [metrics_df.columns[1]])
    if is_class:
        score_col = "f1_score"
    else:
        score_col = "r2"
    if not valid.empty and score_col in valid.columns:
        best_row = valid.loc[valid[score_col].idxmax()]
        best_model_name = best_row["model"]
        best_score = best_row[score_col]
    else:
        best_model_name = "N/A"
        best_score = 0

    dev_log.append({"phase": "Best Model Selection",
                     "result": f"Selected: {best_model_name} ({score_col.upper()}={best_score:.4f}).",
                     "best_model": best_model_name, "best_score": float(best_score)})

    # ---- Phase: Ensemble (if beneficial) ----
    dev_log.append({"phase": "Ensemble Evaluation", "action": "evaluating"})
    top_models = valid.nlargest(min(3, len(valid)), score_col)["model"].tolist() if not valid.empty else []
    ensemble_result = None

    if len(top_models) >= 2:
        if is_class:
            ensemble = VotingClassifier(
                estimators=[(name, models[name]) for name in top_models if name in models],
                voting="soft" if "binary" in task else "hard")
        else:
            ensemble = VotingRegressor(
                estimators=[(name, models[name]) for name in top_models if name in models])
        try:
            ensemble.fit(X_train, y_train)
            y_pred_ens = ensemble.predict(X_test)
            if is_class:
                ens_score = round(float(f1_score(y_test, y_pred_ens, average="binary" if "binary" in task else "macro", zero_division=0)), 4)
            else:
                ens_score = round(float(r2_score(y_test, y_pred_ens)), 4)

            if ens_score > float(best_score):
                dev_log[-1]["result"] = f"Ensemble improves {score_col.upper()} from {best_score:.4f} to {ens_score:.4f} (+{ens_score - float(best_score):.4f}). Using ensemble."
                dev_log[-1]["ensemble_models"] = top_models
                dev_log[-1]["ensemble_score"] = ens_score
                dev_log[-1]["improvement"] = round(float(ens_score) - float(best_score), 4)
                best_model_name = f"Ensemble ({' + '.join(top_models)})"
                best_score = ens_score
                ensemble_result = {"models": top_models, "score": ens_score}
            else:
                dev_log[-1]["result"] = f"Ensemble ({ens_score:.4f}) does not beat best single model ({best_score:.4f}). Keeping {best_model_name}."
        except Exception as e:
            dev_log[-1]["result"] = f"Ensemble failed: {str(e)}"
            dev_log[-1]["skipped"] = True
    else:
        dev_log[-1]["result"] = "Only one valid model available. Ensemble skipped."
        dev_log[-1]["skipped"] = True

    # ---- Phase: Method Development Summary ----
    elapsed = time.time() - t0
    narrative = _build_narrative(task, all_metrics, best_model_name, best_score, dev_log, elapsed)

    # ---- Charts ----
    if not valid.empty and len(valid) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(valid))
        if is_class:
            ax.bar(x - 0.2, valid["accuracy"], 0.2, label="Accuracy", color="#3366cc")
            ax.bar(x, valid["f1_score"], 0.2, label="F1 Score", color="#dc3912")
            ax.bar(x + 0.2, valid["cv_mean"].fillna(0), 0.2, label="CV Mean", color="#109618")
        else:
            ax.bar(x - 0.15, valid["r2"], 0.3, label="R-squared", color="#3366cc")
            ax.bar(x + 0.15, valid["cv_mean"].fillna(0), 0.3, label="CV R-squared", color="#dc3912")
        ax.set_xticks(x)
        ax.set_xticklabels(valid["model"], rotation=20, ha="right", fontsize=9)
        ax.set_title("Model Performance Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        charts["model_comparison"] = _fig_to_b64(fig)

    return {
        "success": True,
        "task": task,
        "metrics": all_metrics,
        "best_model": best_model_name,
        "best_score": round(float(best_score), 4),
        "ensemble": ensemble_result,
        "dev_log": dev_log,
        "narrative": narrative,
        "charts": charts,
        "elapsed_seconds": round(elapsed, 1),
        "feature_names": feature_names,
        "n_samples": len(X),
    }


def _build_narrative(task, metrics, best_model, best_score, dev_log, elapsed):
    """Build a human-readable method development narrative."""
    valid = [m for m in metrics if "error" not in m]
    n_tested = len(metrics)
    n_valid = len(valid)

    parts = []
    parts.append(f"## Method Development Journey\n")

    # Task description
    task_name = task.replace("_", " ").title()
    parts.append(f"**Task detected:** {task_name}. ")

    # What was tried
    model_names = [m["model"] for m in metrics]
    parts.append(f"**{n_tested} models evaluated:** {', '.join(model_names)}. ")

    # What was best
    is_class = "classification" in task
    metric_name = "F1 score" if is_class else "R-squared"
    parts.append(f"**Best single model:** {best_model} ({metric_name} = {best_score:.4f}). ")

    # Did ensemble help?
    for log in dev_log:
        if log["phase"] == "Ensemble Evaluation":
            if log.get("improvement", 0) > 0:
                parts.append(f"**Ensemble was built** combining top {len(log.get('ensemble_models', []))} models, "
                            f"improving {metric_name} by +{log['improvement']:.4f}. ")
            elif not log.get("skipped"):
                parts.append("**Ensemble did not improve** over the best single model. ")
            break

    # Preprocessing decisions
    for log in dev_log:
        if log["phase"] == "Missing Values" and log["action"] == "applied":
            parts.append(f"**Preprocessing:** {log['result']} ")
        elif log["phase"] == "Outlier Handling" and log["action"] == "applied":
            parts.append(f"{log['result']} ")
        elif log["phase"] == "Scaling" and log["action"] == "applied":
            parts.append(f"Scaling: {log['result']} ")

    parts.append(f"\n**Total analysis time:** {elapsed:.1f} seconds.")

    # Method selection justification
    parts.append(f"\n### Why {best_model}?\n")
    best_metrics = next((m for m in metrics if m.get("model") == best_model.split(" (")[0].split(" + ")[0]), None)
    if best_metrics:
        if is_class:
            parts.append(f"- Accuracy: {best_metrics.get('accuracy', 'N/A')}")
            parts.append(f"- F1 Score: {best_metrics.get('f1_score', 'N/A')}")
            parts.append(f"- CV Mean: {best_metrics.get('cv_mean', 'N/A')} (+/- {best_metrics.get('cv_std', 'N/A')})")
        else:
            parts.append(f"- R-squared: {best_metrics.get('r2', 'N/A')}")
            parts.append(f"- RMSE: {best_metrics.get('rmse', 'N/A')}")
            parts.append(f"- CV R-squared Mean: {best_metrics.get('cv_mean', 'N/A')} (+/- {best_metrics.get('cv_std', 'N/A')})")

    # Compare with alternatives
    others = [m for m in valid if m["model"] != best_model.split(" (")[0].split(" + ")[0]]
    if others:
        parts.append(f"\n### Alternatives Considered\n")
        for m in others[:3]:
            if is_class:
                parts.append(f"- **{m['model']}**: F1={m.get('f1_score', 'N/A')}, CV={m.get('cv_mean', 'N/A')}")
            else:
                parts.append(f"- **{m['model']}**: R2={m.get('r2', 'N/A')}, CV={m.get('cv_mean', 'N/A')}")

    return "\n".join(parts)
