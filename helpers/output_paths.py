"""Canonical output locations for the analysis pipeline.

Single source of truth for where every generated artefact lives, keyed to the
figure/table numbers used in ``paper.tex``'s "List of figures" and its three
result tables. Import from here instead of hardcoding filenames so the
orchestrator (``run_analysis.py``), the report builder, and each analysis
script always agree on paths.

Layout::

    outputs/
      figures/   fig01_...pdf .. fig05_...pdf   (paper figures, PDF + PNG twin)
                  supp_...                       (CV/diagnostic plots, not in paper)
      tables/     table2_..., table3_..., table_s1_...  (.csv + .tex fragments)
      data/       intermediate CSVs/JSON consumed by later stages or the report
      report.html

``essen_china_europe_features.csv``, ``usable_china.txt``, ``usable_europa.txt``,
and the small cross-language sidecars (``pearce_default_idyom_basenames.txt``)
stay at the repo root since they're inputs/caches shared by both the Python
and R stages. ``paper.tex`` includes figures via ``outputs/figures/fig0N_*.pdf``.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")
COEFFICIENTS_DIR = os.path.join(DATA_DIR, "coefficients")

# Shared feature cache (written by logistic.py; read by later stages + R).
FEATURES_CSV = os.path.join(REPO_ROOT, "essen_china_europe_features.csv")


def ensure_output_dirs() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, DATA_DIR, COEFFICIENTS_DIR):
        os.makedirs(d, exist_ok=True)


ensure_output_dirs()


def fig_path(number: int, slug: str, ext: str = "pdf") -> str:
    """Path for a numbered paper figure, e.g. ``fig01_logreg_confusion_matrix.pdf``."""
    return os.path.join(FIGURES_DIR, f"fig{number:02d}_{slug}.{ext}")


def supp_fig_path(slug: str, ext: str = "pdf") -> str:
    """Path for a supplementary/diagnostic plot not numbered in the paper."""
    return os.path.join(FIGURES_DIR, f"supp_{slug}.{ext}")


def table_path(table_id: str, slug: str, ext: str = "tex") -> str:
    """Path for a table fragment, e.g. ``table3_source_comparison.tex``."""
    return os.path.join(TABLES_DIR, f"{table_id}_{slug}.{ext}")


def data_path(name: str) -> str:
    """Path for an intermediate CSV/JSON artefact under ``outputs/data``."""
    return os.path.join(DATA_DIR, name)


def save_current_matplotlib_fig(pdf_path: str, *, dpi: int = 150, **savefig_kwargs) -> str:
    """Save the current matplotlib figure as ``pdf_path`` plus a PNG twin.

    The PNG twin (same basename, ``.png`` extension) is what the report
    HTML report embeds inline; the PDF is what ``paper.tex`` includes. Returns
    the PNG path.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    plt.savefig(pdf_path, dpi=dpi, **savefig_kwargs)
    png_path = os.path.splitext(pdf_path)[0] + ".png"
    plt.savefig(png_path, dpi=dpi, **savefig_kwargs)
    return png_path


# Figure numbers match the manuscript figure order (XGBoost archive is unnumbered).
FIG_LOGREG_CONFUSION = 1
FIG_LOGREG_IMPORTANCE = 2
FIG_FACTOR_SCREE = 3
FIG_FACTOR_LOGREG_CONFUSION = 4
FIG_FACTOR_LOGREG_IMPORTANCE = 5
