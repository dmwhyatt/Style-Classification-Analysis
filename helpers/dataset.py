"""Shared dataset loading, train/test split, CV, and logistic pipeline helpers."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from helpers import output_paths as OP
from helpers.feature_selection import numeric_model_feature_columns, prepare_numeric_feature_matrix
from helpers.pearce_exclusion import filter_features_df_pearce, pearce_default_idyom_basename_set

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_SPLITS = 5


def load_features_df(
    path: str | None = None,
    *,
    require_exists: bool = True,
) -> pd.DataFrame:
    """Load the cached features CSV and drop Pearce-overlap rows."""
    csv_path = path or OP.FEATURES_CSV
    if not os.path.isfile(csv_path):
        if require_exists:
            raise RuntimeError(
                f"{csv_path} not found. Run logistic.py first to generate features."
            )
        raise FileNotFoundError(csv_path)

    print(f"Loading cached features from {csv_path} ...")
    features_df = pd.read_csv(csv_path)
    pearce = pearce_default_idyom_basename_set()
    n0 = len(features_df)
    features_df = filter_features_df_pearce(features_df, pearce)
    if len(features_df) < n0:
        print(f"Excluded {n0 - len(features_df)} row(s) overlapping pearce_default_idyom.")
    return features_df


def prepare_xy(
    features_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str], object, LabelEncoder]:
    """Return ``(X, feature_cols, y_enc, label_encoder)`` for continent classification."""
    feature_cols = numeric_model_feature_columns(features_df)
    X, feature_cols = prepare_numeric_feature_matrix(features_df, feature_cols)
    print(f"Numeric features used for modeling: {len(feature_cols)}")
    y = features_df["continent"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, feature_cols, y_enc, le


def make_train_test_split(
    X,
    y_enc,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split shared across logistic / XGBoost / comparison."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def make_cv(
    *,
    n_splits: int = N_CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_logreg_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
