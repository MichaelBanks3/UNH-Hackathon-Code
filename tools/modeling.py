import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, classification_report, mean_absolute_error
from sklearn.inspection import permutation_importance

def _split_cols(df, target):
    # Exclude target and other outcome variables to prevent data leakage
    outcome_cols = {"response_success", "Financial_Loss_MUSD", "actual_days_to_stabilization", "bad_success"}
    cols_to_drop = {target} | outcome_cols
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, num_cols, cat_cols

def fit_classification(df: pd.DataFrame, target="response_success", random_state=42):
    df = df.dropna(subset=[target]).copy()
    X, num_cols, cat_cols = _split_cols(df, target)
    y = df[target].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    rep = classification_report(yte, (proba >= 0.5).astype(int), output_dict=True)

    # permutation importance (more robust than raw coefficients w/ one-hot)
    pi = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=random_state, scoring="roc_auc")
    importances = pd.Series(pi['importances_mean'], index=X.columns).sort_values(ascending=False)

    return pipe, {"auc": auc, "report": rep, "perm_importance": importances}

def fit_regression(df: pd.DataFrame, target: str, random_state=42):
    df = df.dropna(subset=[target]).copy()
    X, num_cols, cat_cols = _split_cols(df, target)
    y = df[target].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    model = Ridge(alpha=1.0)
    pipe = Pipeline([("pre", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=random_state)
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    mae = mean_absolute_error(yte, pred)

    pi = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=random_state, scoring="neg_mean_absolute_error")
    importances = pd.Series(pi['importances_mean'], index=X.columns).sort_values(ascending=False)

    return pipe, {"mae": mae, "perm_importance": importances}
