#!/usr/bin/env python3
"""Build a standalone HTML report at outputs/report.html.

Reads only from outputs/figures, outputs/tables, and outputs/data (performs
no modelling itself) and renders a single self-contained HTML file — every
figure is embedded as a base64 data URI, so the report has no external
asset dependencies and can be opened, emailed, or archived on its own.

Called by run_analysis.py; can also be run standalone once the
figures/tables/data under outputs/ already exist.
"""

from __future__ import annotations

import argparse
import base64
import html
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from helpers import output_paths as OP  # noqa: E402

REPORT_PATH = REPO_ROOT / "outputs" / "report.html"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _missing_note(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    return (
        f'<p class="missing">Missing <code>{html.escape(str(rel))}</code> '
        f"— run the pipeline first.</p>"
    )


def _img_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fig_html(
    fig_number: int | None = None,
    slug: str | None = None,
    *,
    supp_slug: str | None = None,
    preprint_slug: str | None = None,
    caption: str | None = None,
    width: int = 640,
) -> str:
    if preprint_slug is not None:
        path = Path(OP.preprint_fig_path(preprint_slug, ext="png"))
    elif supp_slug is not None:
        path = Path(OP.supp_fig_path(supp_slug, ext="png"))
    else:
        path = Path(OP.fig_path(fig_number, slug, ext="png"))
    uri = _img_data_uri(path)
    if uri is None:
        return _missing_note(path)
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    alt = html.escape(caption or path.name)
    return f'<figure><img src="{uri}" width="{width}" alt="{alt}">{cap}</figure>'


def table_html(path: str | Path, caption: str | None = None, **read_csv_kwargs) -> str:
    path = Path(path)
    if not path.exists():
        return _missing_note(path)
    df = pd.read_csv(path, **read_csv_kwargs)
    table = df.to_html(
        classes="report-table",
        index=False,
        border=0,
        float_format=lambda x: f"{x:.4f}",
        na_rep="—",
    )
    cap = f'<p class="table-caption">{html.escape(caption)}</p>' if caption else ""
    return cap + table


def section(title: str, body: str, *, anchor: str | None = None) -> str:
    anchor_attr = f' id="{anchor}"' if anchor else ""
    return f"<section{anchor_attr}><h2>{html.escape(title)}</h2>{body}</section>"


def build_metadata_block() -> str:
    git_commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    git_dirty = bool(_git("status", "--porcelain"))
    commit = f"{git_commit}{'*' if git_dirty else ''}"
    r_raw = subprocess.run(
        ["Rscript", "--version"], capture_output=True, text=True, check=False
    )
    r_version = (r_raw.stdout or r_raw.stderr).strip().splitlines()[0] if (r_raw.stdout or r_raw.stderr) else "—"
    # Keep R line short (first line only).
    if len(r_version) > 80:
        r_version = r_version[:77] + "…"
    rows = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Commit", commit),
        ("Python", platform.python_version()),
        ("R", html.escape(r_version)),
    ]
    body_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="meta-table">{body_rows}</table>'


def build_dataset_block() -> str:
    features_csv = Path(OP.FEATURES_CSV)
    if not features_csv.exists():
        return _missing_note(features_csv)
    n_cols = len(pd.read_csv(features_csv, nrows=0).columns)
    class_counts = pd.read_csv(features_csv, usecols=["continent"])["continent"].value_counts()
    parts = [f"{int(v):,} {k}" for k, v in class_counts.items()]
    return (
        f"<p>{class_counts.sum():,} melodies "
        f"({'; '.join(parts)}); {n_cols} columns in the feature cache.</p>"
    )


def build_table_s1_block() -> str:
    path = Path(OP.table_path("table_s1", "factor_loadings_top10", "csv"))
    if not path.exists():
        return _missing_note(path)
    df = pd.read_csv(path)
    parts = []
    for factor_name, group in df.groupby("factor_name", sort=False):
        sub = group[["rank", "variable", "loading"]].reset_index(drop=True)
        parts.append(
            f"<h3>{html.escape(str(factor_name))}</h3>"
            + sub.to_html(
                classes="report-table",
                index=False,
                border=0,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    return "".join(parts)


CSS = """
:root {
  --ink: #1c1917;
  --muted: #78716c;
  --rule: #d6d3d1;
  --paper: #fafaf9;
  --accent: #0f766e;
  --body-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
    Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: var(--body-font);
  font-size: 16px;
  line-height: 1.5;
}

.wrap {
  max-width: 42rem;
  margin: 0 auto;
  padding: 3rem 1.25rem 4rem;
}

body.preprint .wrap {
  max-width: 56rem;
}

header {
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 1.75rem;
}

header h1 {
  font-weight: 600;
  font-size: 1.85rem;
  line-height: 1.25;
  letter-spacing: -0.01em;
  margin: 0 0 0.6rem;
}

header .lede {
  margin: 0;
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 36rem;
}

nav.toc {
  margin: 0 0 2.5rem;
  padding: 0;
  font-size: 0.92rem;
}

nav.toc h2 {
  font-weight: 600;
  font-size: 1.25rem;
  margin: 0 0 0.75rem;
}

nav.toc ol {
  margin: 0;
  padding-left: 1.25rem;
}

nav.toc a {
  color: #0969da;
  text-decoration: underline;
}

nav.toc a:hover {
  color: #0550ae;
}

section {
  margin: 2.25rem 0 0;
  padding-top: 1.75rem;
  border-top: 1px solid var(--rule);
}

section h2 {
  font-weight: 600;
  font-size: 1.25rem;
  margin: 0 0 0.85rem;
}

section h3 {
  font-weight: 600;
  font-size: 0.95rem;
  margin: 1.4rem 0 0.4rem;
  color: var(--muted);
}

figure {
  margin: 1rem 0 1.4rem;
  text-align: center;
}

figure img {
  max-width: 100%;
  height: auto;
  display: inline-block;
}

figcaption,
p.table-caption {
  font-size: 0.86rem;
  color: var(--muted);
  margin: 0.55rem 0 0;
  text-align: left;
}

p.table-caption { margin: 0 0 0.35rem; }

p.missing {
  color: #9a3412;
  border-left: 3px solid #ea580c;
  padding: 0.35rem 0 0.35rem 0.75rem;
  margin: 0.8rem 0;
  font-size: 0.92rem;
}

table.report-table,
table.meta-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.4rem 0 1.2rem;
  font-size: 0.88rem;
}

table.report-table th,
table.meta-table th,
table.report-table td,
table.meta-table td {
  text-align: left;
  padding: 0.4rem 0.55rem;
  border-bottom: 1px solid var(--rule);
  vertical-align: top;
}

table.report-table th,
table.meta-table th {
  font-weight: 600;
  color: var(--muted);
  background: transparent;
}

table.meta-table th { width: 7rem; color: var(--muted); font-weight: 500; }
table.meta-table td { font-variant-numeric: tabular-nums; }

code {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.86em;
}

footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 0.82rem;
}

@media (prefers-color-scheme: dark) {
  :root {
    --ink: #f5f5f4;
    --muted: #a8a29e;
    --rule: #44403c;
    --paper: #1c1917;
    --accent: #5eead4;
  }
  p.missing { color: #fdba74; border-left-color: #fb923c; }
  nav.toc a { color: #4493f8; }
  nav.toc a:hover { color: #79b8ff; }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprint",
        action="store_true",
        help=(
            "Embed combined two-panel figures (confusion matrix + permutation "
            "importance) instead of the separate paper figures."
        ),
    )
    args = parser.parse_args()
    OP.ensure_output_dirs()

    if args.preprint:
        logreg_figs = fig_html(
            preprint_slug=OP.PREPRINT_LOGREG,
            caption=(
                "Figure 1. Logistic regression: (a) confusion matrix and "
                "(b) permuted feature importance."
            ),
            width=900,
        )
        factor_figs = fig_html(
            preprint_slug=OP.PREPRINT_FACTOR_LOGREG,
            caption=(
                "Figure 3. Factor logistic regression: (a) confusion matrix and "
                "(b) permuted feature importance."
            ),
            width=900,
        )
        scree_caption = "Figure 2. Scree plot of eigenvalues for 235 factors."
    else:
        logreg_figs = fig_html(
            OP.FIG_LOGREG_CONFUSION,
            "logreg_confusion_matrix",
            caption="Figure 1. Confusion matrix for logistic regression model.",
        ) + fig_html(
            OP.FIG_LOGREG_IMPORTANCE,
            "logreg_permutation_importance",
            caption="Figure 2. Permuted feature importance for logistic regression model.",
        )
        factor_figs = fig_html(
            OP.FIG_FACTOR_LOGREG_CONFUSION,
            "factor_logreg_confusion_matrix",
            caption="Figure 4. Confusion matrix for EFA logistic regression model.",
        ) + fig_html(
            OP.FIG_FACTOR_LOGREG_IMPORTANCE,
            "factor_logreg_permutation_importance",
            caption=(
                "Figure 5. Permuted feature importance for factor logistic "
                "regression model."
            ),
        )
        scree_caption = "Figure 3. Scree plot of eigenvalues for 235 factors."

    sections = [
        ("run-info", "Run", build_metadata_block()),
        ("dataset", "Data", build_dataset_block()),
        (
            "logreg",
            "Logistic regression",
            table_html(OP.data_path("logreg_metrics.csv"))
            + logreg_figs,
        ),
        (
            "efa",
            "Factor analysis",
            fig_html(
                OP.FIG_FACTOR_SCREE,
                "factor_eigenvalues_elbow",
                caption=scree_caption,
            )
            + table_html(
                OP.table_path("table2", "efa_variance", "csv"),
                caption="Variance explained",
            ),
        ),
        (
            "factor-logreg",
            "Factor logistic model",
            table_html(OP.data_path("logistic_factor_metrics.csv"))
            + factor_figs,
        ),
        (
            "table3",
            "By feature source",
            table_html(OP.table_path("table3", "source_comparison", "csv")),
        ),
        (
            "table-s1",
            "Top loadings",
            build_table_s1_block(),
        ),
        (
            "archive",
            "Archived code",
            (
                "<p>An XGBoost classifier script (<code>xgbclassifier.py</code>) remains "
                "in the repository as an archive of earlier exploratory work. It is "
                "no longer part of the analysis pipeline or this report; run it "
                "manually only if you need those historical outputs.</p>"
            ),
        ),
    ]

    toc = "".join(
        f'<li><a href="#{anchor}">{html.escape(title)}</a></li>'
        for anchor, title, _ in sections
    )
    body = "".join(
        section(title, content, anchor=anchor) for anchor, title, content in sections
    )

    body_class = ' class="preprint"' if args.preprint else ""
    lede = (
        "China vs Europe melody classification on Essen features — "
        "logistic regression and a factor-score model."
        + (" Preprint figure layout (combined panels)." if args.preprint else "")
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Style classification — results</title>
<style>{CSS}</style>
</head>
<body{body_class}>
<div class="wrap">
<header>
<h1>Style classification</h1>
<p class="lede">{html.escape(lede)}</p>
</header>
<nav class="toc"><h2>Contents</h2><ol>{toc}</ol></nav>
{body}
<footer>Seeds fixed at 42 · same train/test split across models</footer>
</div>
</body>
</html>
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(doc, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
