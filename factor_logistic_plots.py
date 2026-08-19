"""Confusion matrices and permutation-importance plots for the factor logistic model.

Run after ``factor_logistic.R``. Writes Figures 4 and 5.
Pass ``--preprint`` to also write a combined two-panel figure for the preprint.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from helpers import output_paths as OP
from helpers.pearce_exclusion import basename_no_ext
from helpers.plotting import (
    confusion_heatmap,
    confusion_importance_multipanel,
    signed_permutation_importance_bar,
)

CLASS_ORDER_CSV = OP.data_path("factor_logistic_class_order.csv")
TEST_PRED_CSV = OP.data_path("factor_logistic_predictions_test.csv")
CV_PRED_CSV = OP.data_path("factor_logistic_predictions_cv.csv")
SCORES_CSV = OP.data_path("factor_scores_for_logreg.csv")
TOP_LOADINGS_CSV = OP.data_path("factor_top_loadings.csv")
OUT_CSV = OP.data_path("factor_logistic_permutation_importance.csv")
OUT_PDF = OP.fig_path(OP.FIG_FACTOR_LOGREG_IMPORTANCE, "factor_logreg_permutation_importance")


def _load_factor_labels() -> dict[str, str]:
    df = pd.read_csv(TOP_LOADINGS_CSV)
    mapping = df.drop_duplicates("factor")[["factor", "factor_name"]]
    return dict(zip(mapping["factor"], mapping["factor_name"], strict=True))


def plot_confusion_matrices() -> None:
    if not os.path.isfile(CLASS_ORDER_CSV):
        print(
            f"Missing {CLASS_ORDER_CSV}. Run factor_logistic.R first.",
            file=sys.stderr,
        )
        sys.exit(1)

    classes = pd.read_csv(CLASS_ORDER_CSV)["class_label"].astype(str).tolist()

    if os.path.isfile(TEST_PRED_CSV):
        df = pd.read_csv(TEST_PRED_CSV)
        confusion_heatmap(
            df["true_label"].astype(str),
            df["predicted"].astype(str),
            classes,
            OP.fig_path(OP.FIG_FACTOR_LOGREG_CONFUSION, "factor_logreg_confusion_matrix"),
            save_png_twin=True,
        )
        print(
            f"Saved {OP.fig_path(OP.FIG_FACTOR_LOGREG_CONFUSION, 'factor_logreg_confusion_matrix')}"
        )
    else:
        print(f"Skip test CM: missing {TEST_PRED_CSV}", file=sys.stderr)

    if os.path.isfile(CV_PRED_CSV):
        df = pd.read_csv(CV_PRED_CSV)
        confusion_heatmap(
            df["true_label"].astype(str),
            df["predicted"].astype(str),
            classes,
            OP.supp_fig_path("factor_logreg_confusion_matrix_cv"),
            save_png_twin=False,
        )
        print(f"Saved {OP.supp_fig_path('factor_logreg_confusion_matrix_cv')}")
    else:
        print(f"Skip CV CM: missing {CV_PRED_CSV}", file=sys.stderr)


def plot_permutation_importance() -> None:
    for path in (SCORES_CSV, OP.FEATURES_CSV, TOP_LOADINGS_CSV, CLASS_ORDER_CSV):
        if not os.path.isfile(path):
            print(f"Missing {path}. Run factor_logistic.R first.", file=sys.stderr)
            sys.exit(1)

    scores = pd.read_csv(SCORES_CSV)
    features = pd.read_csv(OP.FEATURES_CSV, usecols=["melody_id", "continent"])
    features["melody_key"] = features["melody_id"].map(basename_no_ext)

    factor_cols = sorted(
        [c for c in scores.columns if c.startswith("F") and c[1:].isdigit()],
        key=lambda name: int(name[1:]),
    )
    if not factor_cols:
        print("No factor score columns found in factor_scores_for_logreg.csv.", file=sys.stderr)
        sys.exit(1)

    model_df = scores.merge(features[["melody_key", "continent"]], on="melody_key", how="inner")
    if model_df.empty:
        print("No rows matched factor scores to continent labels.", file=sys.stderr)
        sys.exit(1)

    X = model_df[factor_cols]
    y = model_df["continent"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    final_model = LogisticRegression(penalty="l2", C=np.inf, solver="lbfgs", max_iter=2000)
    final_model.fit(X_train, y_train)

    print("Computing permutation importance on test set (n_repeats=10)...")
    perm = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )

    factor_labels = _load_factor_labels()
    perm_out = (
        pd.DataFrame(
            {
                "feature": factor_cols,
                "coefficient": final_model.coef_[0],
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .assign(
            pretty_feature=lambda df: df["feature"].map(factor_labels).fillna(df["feature"])
        )
        .sort_values("importance_mean", ascending=False)
    )
    perm_out.to_csv(OUT_CSV, index=False)

    plot_df = perm_out.iloc[::-1].reset_index(drop=True)
    classes = pd.read_csv(CLASS_ORDER_CSV)["class_label"].astype(str).tolist()
    neg_class, pos_class = classes[0], classes[1]

    signed_permutation_importance_bar(
        pretty_names=plot_df["pretty_feature"],
        importance_mean=plot_df["importance_mean"],
        importance_std=plot_df["importance_std"],
        coefficients=plot_df["coefficient"],
        pdf_path=OUT_PDF,
        pos_class=pos_class,
        neg_class=neg_class,
        ylabel="Factor",
        # Match Figure 2's final PDF footprint (so width=\textwidth scales identically).
        figsize=(782.8575 / 72, 395.4203125 / 72),
        left=0.32,
        bbox_inches=None,
    )

    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_PDF}")
    print(
        perm_out[["feature", "coefficient", "importance_mean", "importance_std"]].to_string(
            index=False
        )
    )
    return perm_out, plot_df, neg_class, pos_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprint",
        action="store_true",
        help=(
            "Also write a two-panel preprint figure combining the confusion "
            "matrix and permutation importance. Does not replace Figures 4-5."
        ),
    )
    args = parser.parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    plot_confusion_matrices()
    _perm_out, plot_df, neg_class, pos_class = plot_permutation_importance()
    if args.preprint:
        if not os.path.isfile(TEST_PRED_CSV):
            print(f"Skip preprint panel: missing {TEST_PRED_CSV}", file=sys.stderr)
            return
        pred = pd.read_csv(TEST_PRED_CSV)
        classes = pd.read_csv(CLASS_ORDER_CSV)["class_label"].astype(str).tolist()
        preprint_path = OP.preprint_fig_path(OP.PREPRINT_FACTOR_LOGREG)
        confusion_importance_multipanel(
            y_true=pred["true_label"].astype(str),
            y_pred=pred["predicted"].astype(str),
            class_names=classes,
            pretty_names=plot_df["pretty_feature"],
            importance_mean=plot_df["importance_mean"],
            importance_std=plot_df["importance_std"],
            coefficients=plot_df["coefficient"],
            pdf_path=preprint_path,
            pos_class=pos_class,
            neg_class=neg_class,
            importance_ylabel="Factor",
        )
        print(f"Saved preprint panel: '{preprint_path}'")


if __name__ == "__main__":
    main()
