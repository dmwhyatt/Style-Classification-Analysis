"""Exclude melodies whose basename appears in melody_features ``pearce_default_idyom``."""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

PEARCE_BASENAMES_TXT = "pearce_default_idyom_basenames.txt"


def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


def pearce_default_idyom_basename_set() -> set[str]:
    """Lowercase basenames (no extension) of all files in the Pearce IDyOM default corpus."""
    import melody_features

    d = os.path.join(
        os.path.dirname(melody_features.__file__), "corpora", "pearce_default_idyom"
    )
    if not os.path.isdir(d):
        return set()
    return {
        basename_no_ext(f)
        for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f))
    }


def write_pearce_basename_sidecar(path: str = PEARCE_BASENAMES_TXT) -> None:
    """Write sorted basenames for optional use by ``factor_logistic.R``."""
    names = sorted(pearce_default_idyom_basename_set())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names))
        if names:
            f.write("\n")


def filter_paths_and_labels_pearce(
    paths: List[str], labels: List[str], pearce: set[str]
) -> Tuple[List[str], List[str]]:
    """Drop path/label pairs whose basename is in ``pearce``."""
    out_p: List[str] = []
    out_l: List[str] = []
    for p, lab in zip(paths, labels):
        if basename_no_ext(p) not in pearce:
            out_p.append(p)
            out_l.append(lab)
    return out_p, out_l


def filter_features_df_pearce(
    df: pd.DataFrame,
    pearce: set[str],
    *,
    melody_id_col: str = "melody_id",
) -> pd.DataFrame:
    """Drop rows whose ``melody_id`` basename appears in ``pearce``."""
    if melody_id_col not in df.columns or not pearce:
        return df
    basenames = df[melody_id_col].astype(str).map(basename_no_ext)
    return df.loc[~basenames.isin(pearce)].reset_index(drop=True)
