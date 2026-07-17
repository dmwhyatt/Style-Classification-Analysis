#!/usr/bin/env python3
"""Single entry point for the full style-classification analysis pipeline.

Runs every stage in dependency order (feature extraction/caching happens
lazily inside the first stage), writes every figure/table into ``outputs/``
using the numbering from ``paper.tex``, and finally builds a consolidated
standalone HTML report at ``outputs/report.html``.

Usage
-----
    python run_analysis.py                  # run everything, skipping stages
                                              # whose declared outputs already exist
    python run_analysis.py --force           # re-run every stage regardless
    python run_analysis.py --only logistic,xgboost   # run a subset of stages
    python run_analysis.py --list            # list stage names and exit

Determinism
-----------
Every stage seeds its RNGs at ``42`` (Python's ``random``, scikit-learn's
``random_state``, R's ``set.seed``) and uses the same cached feature matrix
(``essen_china_europe_features.csv``) and train/test split parameters, so
re-running this script from a clean ``outputs/`` directory reproduces the
same numbers reported in the paper bit-for-bit (modulo upstream library
version drift).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from helpers import output_paths as OP  # noqa: E402


@dataclass
class Stage:
    name: str
    description: str
    cmd: list[str]
    outputs: list[Path] = field(default_factory=list)

    def outputs_exist(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _fig(number: int, slug: str) -> Path:
    return Path(OP.fig_path(number, slug))


def _tbl(table_id: str, slug: str, ext: str) -> Path:
    return Path(OP.table_path(table_id, slug, ext))


def _dat(name: str) -> Path:
    return Path(OP.data_path(name))


def build_stages() -> list[Stage]:
    rscript = shutil.which("Rscript") or "Rscript"
    return [
        Stage(
            name="logistic",
            description="Full-feature logistic regression (Figures 1-2)",
            cmd=_py("logistic.py"),
            outputs=[
                _fig(OP.FIG_LOGREG_CONFUSION, "logreg_confusion_matrix"),
                _fig(OP.FIG_LOGREG_IMPORTANCE, "logreg_permutation_importance"),
                _dat("logreg_metrics.csv"),
            ],
        ),
        Stage(
            name="xgboost",
            description="XGBoost benchmark (Figures 3-4)",
            cmd=_py("xgbclassifier.py"),
            outputs=[
                _fig(OP.FIG_XGB_CONFUSION, "xgb_confusion_matrix"),
                _fig(OP.FIG_XGB_IMPORTANCE, "xgb_permutation_importance"),
                _dat("xgb_metrics.csv"),
            ],
        ),
        Stage(
            name="efa",
            description="Exploratory factor analysis + factor logistic regression in R "
            "(Figure 5, Table 2, Table S1)",
            cmd=[rscript, "factor_logistic.R"],
            outputs=[
                _fig(OP.FIG_FACTOR_SCREE, "factor_eigenvalues_elbow"),
                _tbl("table2", "efa_variance", "csv"),
                _tbl("table_s1", "factor_loadings_top10", "csv"),
                _dat("factor_logistic_predictions_test.csv"),
            ],
        ),
        Stage(
            name="factor_plots",
            description="Factor logistic confusion matrix + permutation importance (Figures 6-7)",
            cmd=_py("factor_logistic_plots.py"),
            outputs=[
                _fig(OP.FIG_FACTOR_LOGREG_CONFUSION, "factor_logreg_confusion_matrix"),
                _fig(OP.FIG_FACTOR_LOGREG_IMPORTANCE, "factor_logreg_permutation_importance"),
            ],
        ),
        Stage(
            name="comparison",
            description="Per-source logistic benchmarks (Table 3)",
            cmd=_py("comparison.py"),
            outputs=[_tbl("table3", "source_comparison", "csv")],
        ),
    ]


def build_report(force: bool) -> tuple[bool, float]:
    report_path = Path(OP.OUTPUTS_DIR) / "report.html"
    if not force and report_path.exists():
        build_script_mtime = (REPO_ROOT / "build_report.py").stat().st_mtime
        if report_path.stat().st_mtime > build_script_mtime:
            print("  report.html is newer than build_report.py; skipping rebuild (--force to redo)")
            return True, 0.0
    start = time.time()
    result = subprocess.run(_py("build_report.py"), cwd=REPO_ROOT, check=False)
    return result.returncode == 0, time.time() - start


def run_stage(stage: Stage, force: bool) -> tuple[str, float]:
    if not force and stage.outputs_exist():
        return "skipped (outputs exist)", 0.0
    start = time.time()
    result = subprocess.run(stage.cmd, cwd=REPO_ROOT, check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage.name}' failed (exit code {result.returncode}): {' '.join(stage.cmd)}"
        )
    return "ran", elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-run every stage, ignoring cached outputs.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated stage names to run (see --list).",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip building outputs/report.html.")
    parser.add_argument("--list", action="store_true", help="List stage names and exit.")
    args = parser.parse_args()

    stages = build_stages()

    if args.list:
        for s in stages:
            print(f"{s.name:24s} {s.description}")
        return

    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {s.name for s in stages}
        if unknown:
            parser.error(f"Unknown stage name(s): {', '.join(sorted(unknown))}")
        stages = [s for s in stages if s.name in wanted]

    OP.ensure_output_dirs()

    print("=" * 80)
    print("Style Classification Analysis — orchestrated run")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Stages: {', '.join(s.name for s in stages)}")
    print("=" * 80)

    summary: list[tuple[str, str, float]] = []
    for stage in stages:
        print(f"\n--- [{stage.name}] {stage.description} ---")
        status, elapsed = run_stage(stage, args.force)
        summary.append((stage.name, status, elapsed))
        print(f"--- [{stage.name}] {status} ({elapsed:.1f}s) ---")

    if not args.no_report:
        print("\n--- [report] Building outputs/report.html ---")
        ok, elapsed = build_report(args.force)
        summary.append(("report", "ran" if ok else "FAILED", elapsed))
        if not ok:
            raise RuntimeError("Report build failed; see output above.")
        print(f"--- [report] done ({elapsed:.1f}s) ---")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, status, elapsed in summary:
        print(f"  {name:24s} {status:24s} {elapsed:6.1f}s")
    print("=" * 80)
    print(f"Report: {OP.OUTPUTS_DIR}/report.html")


if __name__ == "__main__":
    main()
