#!/usr/bin/env python3
"""Build preprint-only two-panel figures (confusion matrix + permutation importance).

Reads saved prediction and importance CSVs from ``outputs/data`` and writes
``outputs/figures/preprint_*.{pdf,png}``. Does not change the numbered paper
figures (fig01–fig05).

Usage
-----
    python build_preprint_figures.py
    python run_analysis.py --preprint
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from helpers import output_paths as OP
from helpers.plotting import confusion_importance_multipanel

LOGREG_PRED = OP.data_path("logreg_predictions_test.csv")
LOGREG_PERM = OP.data_path("logistic_permutation_importance.csv")
FACTOR_PRED = OP.data_path("factor_logistic_predictions_test.csv")
FACTOR_PERM = OP.data_path("factor_logistic_permutation_importance.csv")
FACTOR_CLASSES = OP.data_path("factor_logistic_class_order.csv")
XGB_PRED = OP.data_path("xgb_predictions_test.csv")
XGB_PERM = OP.data_path("xgb_permutation_importance.csv")


def _require(*paths: str) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        joined = "\n  ".join(missing)
        print(
            "Missing input(s) for preprint panels:\n  "
            f"{joined}\n"
            "Run the corresponding analysis stage first "
            "(e.g. python logistic.py, python factor_logistic_plots.py).",
            file=sys.stderr,
        )
        sys.exit(1)


def _class_order(pred_df: pd.DataFrame, class_order_csv: str | None = None) -> list[str]:
    if class_order_csv and os.path.isfile(class_order_csv):
        return pd.read_csv(class_order_csv)["class_label"].astype(str).tolist()
    labels = pd.concat(
        [pred_df["true_label"].astype(str), pred_df["predicted"].astype(str)]
    )
    return sorted(labels.unique())


def _importance_plot_df(perm_csv: str, top_k: int | None) -> pd.DataFrame:
    df = pd.read_csv(perm_csv).sort_values("importance_mean", ascending=False)
    if top_k is not None:
        df = df.head(top_k)
    if "pretty_feature" not in df.columns:
        df = df.assign(pretty_feature=df["feature"].astype(str))
    return df.iloc[::-1].reset_index(drop=True)


def _write_panel(
    *,
    pred_csv: str,
    perm_csv: str,
    slug: str,
    top_k: int | None,
    importance_ylabel: str = "Feature",
    class_order_csv: str | None = None,
    importance_xlabel: str = "Mean accuracy decrease",
    pos_legend: str | None = None,
    neg_legend: str | None = None,
) -> str:
    pred = pd.read_csv(pred_csv)
    classes = _class_order(pred, class_order_csv)
    # LabelEncoder / logistic.py: classes_[0] is the negative class.
    neg_class, pos_class = classes[0], classes[1]
    plot_df = _importance_plot_df(perm_csv, top_k)
    out_path = OP.preprint_fig_path(slug)
    confusion_importance_multipanel(
        y_true=pred["true_label"].astype(str),
        y_pred=pred["predicted"].astype(str),
        class_names=classes,
        pretty_names=plot_df["pretty_feature"],
        importance_mean=plot_df["importance_mean"],
        importance_std=plot_df["importance_std"],
        coefficients=plot_df["coefficient"],
        pdf_path=out_path,
        pos_class=pos_class,
        neg_class=neg_class,
        importance_xlabel=importance_xlabel,
        importance_ylabel=importance_ylabel,
        pos_legend=pos_legend,
        neg_legend=neg_legend,
    )
    return out_path


def _ensure_logreg_predictions() -> None:
    """Refit the logistic model if test predictions were not saved by an older run."""
    if os.path.isfile(LOGREG_PRED):
        return
    print(
        "logreg_predictions_test.csv missing; refitting logistic regression "
        "to recover test-set predictions (permutation importance is not recomputed)..."
    )
    from helpers import dataset

    features_df = dataset.load_features_df()
    X, _feature_cols, y_enc, le = dataset.prepare_xy(features_df)
    X_train_full, X_test, y_train_full, y_test = dataset.make_train_test_split(X, y_enc)
    model = dataset.build_logreg_pipeline(random_state=42)
    model.fit(X_train_full, y_train_full)
    pd.DataFrame(
        {
            "true_label": le.inverse_transform(y_test),
            "predicted": le.inverse_transform(model.predict(X_test)),
        }
    ).to_csv(LOGREG_PRED, index=False)
    print(f"Wrote {LOGREG_PRED}")


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    OP.ensure_output_dirs()

    _require(LOGREG_PERM)
    _ensure_logreg_predictions()
    _require(LOGREG_PRED)
    logreg_path = _write_panel(
        pred_csv=LOGREG_PRED,
        perm_csv=LOGREG_PERM,
        slug=OP.PREPRINT_LOGREG,
        top_k=20,
        importance_ylabel="Feature",
    )
    print(f"Saved preprint panel: '{logreg_path}'")

    _require(FACTOR_PRED, FACTOR_PERM, FACTOR_CLASSES)
    factor_path = _write_panel(
        pred_csv=FACTOR_PRED,
        perm_csv=FACTOR_PERM,
        slug=OP.PREPRINT_FACTOR_LOGREG,
        top_k=None,
        importance_ylabel="Factor",
        class_order_csv=FACTOR_CLASSES,
    )
    print(f"Saved preprint panel: '{factor_path}'")

    if os.path.isfile(XGB_PRED) and os.path.isfile(XGB_PERM):
        pred = pd.read_csv(XGB_PRED)
        classes = _class_order(pred)
        pos_class = classes[1]
        neg_class = classes[0]
        xgb_path = _write_panel(
            pred_csv=XGB_PRED,
            perm_csv=XGB_PERM,
            slug=OP.PREPRINT_XGB,
            top_k=20,
            importance_ylabel="Feature",
            importance_xlabel="Mean accuracy decrease (sign from mean SHAP contribution)",
            pos_legend=f"SHAP > 0 (higher → {pos_class})",
            neg_legend=f"SHAP < 0 (higher → {neg_class})",
        )
        print(f"Saved preprint panel (archive): '{xgb_path}'")
    else:
        print("Skip XGBoost preprint panel (archived outputs not present).")


if __name__ == "__main__":
    main()
