"""Shared matplotlib helpers for confusion matrices and signed importance bars."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

from helpers import output_paths as OP


def prettify_feature_name(feature_name: str) -> str:
    if not feature_name:
        return feature_name
    parts = feature_name.split(".")
    category = parts[0] if parts else ""
    remainder = " ".join(parts[1:]) if len(parts) > 1 else ""

    def _clean(segment: str) -> str:
        return segment.replace("_", " ").strip().title()

    category_text = _clean(category)
    if remainder:
        return f"{category_text}: {_clean(remainder)}"
    return category_text


def confusion_heatmap(
    y_true,
    y_pred,
    class_names: Sequence[str],
    pdf_path: str,
    *,
    save_png_twin: bool = True,
) -> None:
    """Save a count + row-% annotated confusion heatmap."""
    classes = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    n = len(classes)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({pct[i, j]:.1f}%)"

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
    if save_png_twin:
        OP.save_current_matplotlib_fig(pdf_path, dpi=150)
    else:
        import os

        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
        plt.savefig(pdf_path, dpi=150)
    plt.close()


def signed_permutation_importance_bar(
    *,
    pretty_names: Sequence[str],
    importance_mean: Sequence[float],
    importance_std: Sequence[float],
    coefficients: Sequence[float],
    pdf_path: str,
    pos_class: str,
    neg_class: str,
    xlabel: str = "Mean accuracy decrease\n(sign from logistic coefficient)",
    ylabel: str = "Feature",
    pos_legend: str | None = None,
    neg_legend: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    left: float = 0.32,
) -> None:
    """Horizontal bar chart of permutation importance, signed by coefficient direction."""
    coef = np.asarray(coefficients, dtype=float)
    imp = np.asarray(importance_mean, dtype=float)
    std = np.asarray(importance_std, dtype=float)
    coef_sign = np.sign(coef)
    coef_sign = np.where(coef_sign == 0, 1.0, coef_sign)
    signed_imp = imp * coef_sign
    bar_colors = np.where(coef >= 0, "#2166ac", "#b2182b")

    if pos_legend is None:
        pos_legend = f"β > 0 (higher → {pos_class})"
    if neg_legend is None:
        neg_legend = f"β < 0 (higher → {neg_class})"

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        list(pretty_names),
        signed_imp,
        xerr=std,
        color=bar_colors,
        edgecolor="white",
    )
    ax.axvline(0, color="0.35", linewidth=0.8, zorder=0)
    ax.legend(
        handles=[
            Patch(facecolor="#2166ac", edgecolor="white", label=pos_legend),
            Patch(facecolor="#b2182b", edgecolor="white", label=neg_legend),
        ],
        loc="lower right",
        frameon=True,
        fontsize=9,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    fig.subplots_adjust(left=left, bottom=0.14, right=0.96, top=0.90)
    OP.save_current_matplotlib_fig(pdf_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close()
