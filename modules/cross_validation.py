"""Train/test splitting, k-fold cross-validation, stratified sampling."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def split_data(df, target_col, test_size=0.2, stratify=True, random_state=42):
    """Split data into train/test sets."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    stratify_arg = None
    if stratify:
        try:
            stratify_arg = y if y.nunique() <= 20 else pd.qcut(y, q=5, labels=False, duplicates="drop")
        except Exception:
            stratify_arg = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "test_size": test_size, "train_size": len(X_train), "test_n": len(X_test)}


def kfold_cross_validation(df, target_col, model, n_splits=5, scoring="accuracy", random_state=42):
    """Perform k-fold cross-validation."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {"scores": scores.tolist(), "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "n_splits": n_splits, "scoring": scoring}


def stratified_kfold(df, target_col, model, n_splits=5, scoring="accuracy", random_state=42):
    """Stratified k-fold preserving class distribution."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col].dropna()
    X = X.loc[y.index]
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y_enc, cv=skf, scoring=scoring)
    return {"scores": scores.tolist(), "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "n_splits": n_splits, "scoring": scoring}
