"""Shared numeric feature selection for Essen classifiers in Python.

Aligns with factor_logistic.R:
  - numeric columns only
  - exclude melody_num and id/metadata-like column names
  - replace inf, fillna(0), then drop zero-variance / non-finite std columns
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

_METADATA_SUBSTRINGS = [
    "file",
    "path",
    "name",
    "title",
    "source",
    "prefix",
    "region",
    "country",
    "collection",
    "folder",
    "subdir",
    "dataset",
    "melody_num",
]


def is_excluded_feature_column(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _METADATA_SUBSTRINGS)


def numeric_model_feature_columns(dataframe: pd.DataFrame) -> List[str]:
    numeric = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if not is_excluded_feature_column(c)]


def prepare_numeric_feature_matrix(
    dataframe: pd.DataFrame,
    feature_cols: List[str] | None = None,
    *,
    drop_zero_variance: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    if feature_cols is None:
        feature_cols = numeric_model_feature_columns(dataframe)
    if not feature_cols:
        raise RuntimeError("No feature columns selected for modeling.")
    X = dataframe[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if drop_zero_variance:
        stds = X.std()
        bad = stds.index[(stds == 0) | stds.isna()].tolist()
        if bad:
            print(f"Dropping {len(bad)} zero-variance (or invalid-variance) column(s): {bad}")
            X = X.drop(columns=bad)
        feature_cols = list(X.columns)
    return X, feature_cols
