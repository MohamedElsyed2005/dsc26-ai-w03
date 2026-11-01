"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

# Known when the parcel is created. Excluded: parcel_id (row id), courier_notes and
# delivered_at (written after the attempt), failed (the label).
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]
CAT_COLUMNS = ["area", "promised_slot"]
NUM_COLUMNS = ["n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
COD_COLUMN = ["cod_amount"]
ADDRESS_COLUMN = "address_text"

CAT_COLUMNS = ["area", "promised_slot"]
NUM_COLUMNS = ["n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
COD_COLUMN = ["cod_amount"]
ADDRESS_COLUMN = "address_text"

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    out = df.copy()
    tmp = pd.DataFrame({
        "c": df["customer_id"].to_numpy(),
        "t": df["created_at"].to_numpy(),
        "f": df[TARGET].to_numpy().astype(float),
    })
    g = (tmp.groupby(["c", "t"], sort=True, dropna=False)
            .agg(n=("f", "size"), k=("f", "sum"))
            .reset_index())
    g["prior_n"] = g.groupby("c", dropna=False)["n"].cumsum() - g["n"]
    g["prior_k"] = g.groupby("c", dropna=False)["k"].cumsum() - g["k"]
    merged = tmp.merge(g[["c", "t", "prior_n", "prior_k"]], on=["c", "t"], how="left")
    prior_n = merged["prior_n"].to_numpy().astype(int)
    prior_k = merged["prior_k"].to_numpy().astype(float)
    rate = np.full(len(prior_n), np.nan)
    has = prior_n > 0
    rate[has] = prior_k[has] / prior_n[has]
    out["customer_prior_parcels"] = prior_n
    out["customer_prior_fail_rate"] = rate

    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")
    n = len(df)
    n_test = math.ceil(n * test_fraction)
    ordered = df[time_col].sort_values(kind="stable")
    boundary = ordered.iloc[n - n_test]
    test = df[df[time_col] >= boundary]
    train = df[df[time_col] < boundary]

    return train, test

# ---- preprocessing helpers (module level so the pipeline stays picklable) ----

def _clean_cat(X):
    X = pd.DataFrame(X).copy()
    X = X.astype(object).where(X.notna(), "missing")
    return X.astype(str)


def _clean_num(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float)


def _clean_cod(X):
    X = _clean_num(X)
    return np.log1p(X.clip(lower=0))

# Placeholder for empty/missing/whitespace-only addresses. Without it, a training set whose
# addresses are ALL empty gives TfidfVectorizer an empty vocabulary and fit() raises ValueError.
# A character no real address contains, so it never collides with genuine n-grams.
_EMPTY_ADDRESS_TOKEN = "\u2205"

def _clean_text(X):
    s = pd.DataFrame(X).iloc[:, 0] if not isinstance(X, pd.Series) else X
    cleaned = s.astype(object).where(s.notna(), "").astype(str).str.strip()
    return cleaned.where(cleaned != "", _EMPTY_ADDRESS_TOKEN).tolist()


def _address_empty_flag(X):
    s = pd.DataFrame(X).iloc[:, 0]
    empty = s.isna() | (s.astype(object).where(s.notna(), "").astype(str).str.strip() == "")
    return empty.astype(float).to_frame()

def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    cat = Pipeline([
        ("clean", FunctionTransformer(_clean_cat)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    num = Pipeline([
        ("clean", FunctionTransformer(_clean_num)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cod = Pipeline([
        ("clean", FunctionTransformer(_clean_cod)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    text = Pipeline([
        ("clean", FunctionTransformer(_clean_text)),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=1, sublinear_tf=True)),
    ])
    pre = ColumnTransformer([
        ("cat", cat, CAT_COLUMNS),
        ("num", num, NUM_COLUMNS),
        ("cod", cod, COD_COLUMN),
        ("text", text, ADDRESS_COLUMN),
        ("addr_empty", FunctionTransformer(_address_empty_flag), [ADDRESS_COLUMN]),
    ], remainder="drop")
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    # Drop columns that are constant in the training data. They carry no information, and a
    # constant column (e.g. the "address is empty" flag when every training address is empty)
    # would otherwise absorb part of the intercept and shift every prediction at serving time.
    # A no-op on normal data.
    return Pipeline([("pre", pre), ("var", VarianceThreshold()), ("clf", clf)])

def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    y = test_df[TARGET]
    p = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    m, d = evaluate(model, test_df), evaluate(dummy, test_df)
    return {"model": m, "dummy": d, "lift_pr_auc": m["pr_auc"] - d["pr_auc"]}
