"""
Confusion matrices for factor_logistic.R).

Run after factor_logistic.R to produce confusion matrices that are aesthetically identical 
to those produced by xgbclassifer.py and logistic.py.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_ORDER_CSV = "factor_logistic_class_order.csv"
TEST_PRED_CSV = "factor_logistic_predictions_test.csv"
CV_PRED_CSV = "factor_logistic_predictions_cv.csv"


def _plot_cm(
    df: pd.DataFrame,
    classes: list[str],
    out_path: str,
) -> None:
    y_true = df["true_label"].astype(str)
    y_pred = df["predicted"].astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(classes)):
        for j in range(len(classes)):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    if not os.path.isfile(CLASS_ORDER_CSV):
        print(
            f"Missing {CLASS_ORDER_CSV}. Run factor_logistic.R first.",
            file=sys.stderr,
        )
        sys.exit(1)

    classes = pd.read_csv(CLASS_ORDER_CSV)["class_label"].astype(str).tolist()

    if os.path.isfile(TEST_PRED_CSV):
        _plot_cm(
            pd.read_csv(TEST_PRED_CSV),
            classes,
            "factor_logistic_confusion_matrix_test.pdf",
        )
    else:
        print(f"Skip test CM: missing {TEST_PRED_CSV}", file=sys.stderr)

    if os.path.isfile(CV_PRED_CSV):
        _plot_cm(
            pd.read_csv(CV_PRED_CSV),
            classes,
            "factor_logistic_confusion_matrix_cv.pdf",
        )
    else:
        print(f"Skip CV CM: missing {CV_PRED_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
