"""
Permutation-importance bar chart for the factor logistic model.

Run after factor_logistic.R. Mirrors logistic.py / xgbclassifer.py styling.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SCORES_CSV = "factor_scores_for_logreg.csv"
FEATURES_CSV = "essen_china_europe_features.csv"
TOP_LOADINGS_CSV = "factor_top_loadings.csv"
CLASS_ORDER_CSV = "factor_logistic_class_order.csv"
OUT_CSV = "factor_logistic_permutation_importance.csv"
OUT_PDF = "factor_logistic_permutation_importance_bar.pdf"


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0].lower()


def _load_factor_labels() -> dict[str, str]:
    df = pd.read_csv(TOP_LOADINGS_CSV)
    mapping = df.drop_duplicates("factor")[["factor", "factor_name"]]
    return dict(zip(mapping["factor"], mapping["factor_name"], strict=True))


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    for path in (SCORES_CSV, FEATURES_CSV, TOP_LOADINGS_CSV, CLASS_ORDER_CSV):
        if not os.path.isfile(path):
            print(f"Missing {path}. Run factor_logistic.R first.", file=sys.stderr)
            sys.exit(1)

    scores = pd.read_csv(SCORES_CSV)
    features = pd.read_csv(FEATURES_CSV, usecols=["melody_id", "continent"])
    features["melody_key"] = features["melody_id"].map(_basename_no_ext)

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
    coef_sign = np.sign(plot_df["coefficient"].to_numpy(dtype=float))
    coef_sign = np.where(coef_sign == 0, 1.0, coef_sign)
    signed_imp = plot_df["importance_mean"].to_numpy() * coef_sign
    bar_colors = np.where(plot_df["coefficient"].to_numpy() >= 0, "#2166ac", "#b2182b")

    classes = pd.read_csv(CLASS_ORDER_CSV)["class_label"].astype(str).tolist()
    neg_class, pos_class = classes[0], classes[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        plot_df["pretty_feature"],
        signed_imp,
        xerr=plot_df["importance_std"],
        color=bar_colors,
        edgecolor="white",
    )
    ax.axvline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.legend(
        handles=[
            Patch(
                facecolor="#2166ac",
                edgecolor="white",
                label=f"β > 0 (higher → {pos_class})",
            ),
            Patch(
                facecolor="#b2182b",
                edgecolor="white",
                label=f"β < 0 (higher → {neg_class})",
            ),
        ],
        loc="lower right",
        frameon=True,
        fontsize=9,
    )
    ax.set_xlabel(
        "Mean accuracy decrease\n(sign from logistic coefficient)",
        fontsize=12,
    )
    ax.set_ylabel("Factor", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    fig.subplots_adjust(left=0.36, bottom=0.14, right=0.96, top=0.90)
    plt.savefig(OUT_PDF, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close()

    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_PDF}")
    print(
        perm_out[["feature", "coefficient", "importance_mean", "importance_std"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
