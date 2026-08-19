"""Shared matplotlib helpers for confusion matrices and signed importance bars."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
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


def draw_confusion_heatmap(
    ax: Axes,
    y_true,
    y_pred,
    class_names: Sequence[str],
    *,
    cbar: bool = True,
) -> None:
    """Draw a count + row-% annotated confusion heatmap onto ``ax``."""
    classes = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    n = len(classes)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({pct[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=cbar,
        cbar_kws={"shrink": 0.8} if cbar else None,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def draw_signed_permutation_importance_bar(
    ax: Axes,
    *,
    pretty_names: Sequence[str],
    importance_mean: Sequence[float],
    importance_std: Sequence[float],
    coefficients: Sequence[float],
    pos_class: str,
    neg_class: str,
    xlabel: str = "Mean accuracy decrease\n(sign from logistic coefficient)",
    ylabel: str = "Feature",
    pos_legend: str | None = None,
    neg_legend: str | None = None,
) -> None:
    """Draw a signed permutation-importance bar chart onto ``ax``."""
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


def confusion_heatmap(
    y_true,
    y_pred,
    class_names: Sequence[str],
    pdf_path: str,
    *,
    save_png_twin: bool = True,
) -> None:
    """Save a count + row-% annotated confusion heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_confusion_heatmap(ax, y_true, y_pred, class_names)
    fig.tight_layout()
    if save_png_twin:
        OP.save_current_matplotlib_fig(pdf_path, dpi=150)
    else:
        import os

        os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
        fig.savefig(pdf_path, dpi=150)
    plt.close(fig)


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
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.12,
) -> None:
    """Horizontal bar chart of permutation importance, signed by coefficient direction."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_signed_permutation_importance_bar(
        ax,
        pretty_names=pretty_names,
        importance_mean=importance_mean,
        importance_std=importance_std,
        coefficients=coefficients,
        pos_class=pos_class,
        neg_class=neg_class,
        xlabel=xlabel,
        ylabel=ylabel,
        pos_legend=pos_legend,
        neg_legend=neg_legend,
    )
    fig.subplots_adjust(left=left, bottom=0.14, right=0.96, top=0.90)
    save_kwargs: dict = {"dpi": 150, "pad_inches": pad_inches}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    OP.save_current_matplotlib_fig(pdf_path, **save_kwargs)
    plt.close(fig)


# Shared width so logistic / factor panels match font size at ``\textwidth``.
# Height may shrink when there are fewer importance bars (factor model).
PREPRINT_FIG_WIDTH_IN = 11.0
PREPRINT_CM_SIZE_IN = 3.4
PREPRINT_IMP_PLOT_WIDTH_IN = 4.6
PREPRINT_IMP_HEIGHT_IN = 4.55
# Upper bound for long logistic feature labels.
PREPRINT_TICK_LEFT_MAX_IN = 5.9


def _axes_in_inches(fig, left: float, bottom: float, width: float, height: float) -> list[float]:
    fig_w, fig_h = fig.get_size_inches()
    return [left / fig_w, bottom / fig_h, width / fig_w, height / fig_h]


def confusion_importance_multipanel(
    *,
    y_true,
    y_pred,
    class_names: Sequence[str],
    pretty_names: Sequence[str],
    importance_mean: Sequence[float],
    importance_std: Sequence[float],
    coefficients: Sequence[float],
    pdf_path: str,
    pos_class: str,
    neg_class: str,
    importance_xlabel: str = "Mean accuracy decrease",
    importance_ylabel: str = "Feature",
    pos_legend: str | None = None,
    neg_legend: str | None = None,
    fig_width: float = PREPRINT_FIG_WIDTH_IN,
    cm_size: float = PREPRINT_CM_SIZE_IN,
) -> None:
    """Save a stacked preprint figure: (a) confusion matrix, (b) permutation importance.

    Width is fixed across models so fonts match at ``\\textwidth``. Importance
    height tracks the number of bars. The importance block (labels + bars) is
    centred under the confusion matrix.
    """
    names = list(pretty_names)
    n_bars = max(len(names), 1)
    # Tighter vertical packing when there are few importance bars (factor panel).
    compact = n_bars <= 10
    top_pad = 0.40 if compact else 0.48
    cm_xlabel_pad = 0.32 if compact else 0.42
    gap = 0.32 if compact else 0.45
    bottom_pad = 0.45 if compact else 0.55
    cm_size_eff = 2.85 if compact else cm_size
    # Compact for few factors; full height for the 20-feature logistic panel.
    imp_height = max(2.05, min(PREPRINT_IMP_HEIGHT_IN, 0.18 * n_bars + (0.95 if compact else 1.15)))
    fig_height = top_pad + cm_size_eff + cm_xlabel_pad + gap + imp_height + bottom_pad

    max_chars = max((len(str(n)) for n in names), default=12)
    tick_left = min(PREPRINT_TICK_LEFT_MAX_IN, max(1.85, 0.098 * max_chars + 0.55))
    plot_width = PREPRINT_IMP_PLOT_WIDTH_IN
    # Prefer bars centred under the CM; shift right only if tick labels need room.
    imp_left = max(tick_left, (fig_width - plot_width) / 2)

    fig = plt.figure(figsize=(fig_width, fig_height))
    cm_left = (fig_width - cm_size_eff) / 2
    cm_bottom = bottom_pad + imp_height + gap + cm_xlabel_pad
    ax_cm = fig.add_axes(_axes_in_inches(fig, cm_left, cm_bottom, cm_size_eff, cm_size_eff))
    ax_imp = fig.add_axes(
        _axes_in_inches(fig, imp_left, bottom_pad, plot_width, imp_height)
    )

    draw_confusion_heatmap(ax_cm, y_true, y_pred, class_names, cbar=False)
    ax_cm.set_aspect("equal", adjustable="box")
    draw_signed_permutation_importance_bar(
        ax_imp,
        pretty_names=names,
        importance_mean=importance_mean,
        importance_std=importance_std,
        coefficients=coefficients,
        pos_class=pos_class,
        neg_class=neg_class,
        xlabel=importance_xlabel,
        ylabel=importance_ylabel,
        pos_legend=pos_legend,
        neg_legend=neg_legend,
    )
    # Shared figure-x so (a)/(b) titles line up even when the axes are offset.
    title_x = min(cm_left, imp_left) / fig_width
    title_dy = 0.08 / fig_height
    for y_top_in, text in (
        (cm_bottom + cm_size_eff, "(a) Confusion matrix"),
        (bottom_pad + imp_height, "(b) Permutation importance"),
    ):
        fig.text(
            title_x,
            y_top_in / fig_height + title_dy,
            text,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            transform=fig.transFigure,
        )
    # No tight crop: fixed width MediaBox keeps LaTeX font scaling consistent.
    OP.save_current_matplotlib_fig(pdf_path, dpi=150, pad_inches=0.12)
    plt.close(fig)
