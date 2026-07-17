# Style Classification Analysis

This repository develops classifiers to predict whether melodies from the Essen corpus originate from **China** or **Europe**, using numeric features extracted [**melody-features** ](https://github.com/dmwhyatt/melody-features)as predictors. It reproduces the manuscript’s confusion-matrix analyses, conducts exploratory factor analysis and a factor-based logistic model in R, and benchmarks logistic regression for each feature extraction source (IDyOM, jSymbolic, etc.).

Run everything from the repo root unless noted otherwise.

---

## Prerequisites

| Requirement | Notes |
|---------------|--------|
| **Python 3.10+** | Check with `python3 --version`. |
| **R** | Check with `Rscript --version`. Install from [CRAN](https://cran.r-project.org/). |
| **`melody-features`** | Installed via `requirements.txt`. It ships the Essen corpus used to resolve melody paths from basename lists, and provides all of the features used in the anaylses |

---

## One-time setup

**1. Clone and enter the repo**

```bash
git clone https://github.com/dmwhyatt/Style-Classification-Analysis.git
cd Style-Classification-Analysis
```

**2. Python environment**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. R packages** (for `factor_logistic.R` only)

```bash
Rscript -e 'install.packages(c("tidyverse", "psych", "jsonlite"), repos="https://cloud.r-project.org")'
```

`ggplot2` is included in `tidyverse`.

**4. `melody-features`**

Feature extraction is run using the Python package defaults. Some inputs may be skipped (e.g. unsupported or polyphonic). This is expected behaviour.

---

## Dataset

Two files list melody basenames (no paths):

| File | Role |
|------|------|
| `usable_china.txt` | One basename per line → pool for **China**. |
| `usable_europa.txt` | One basename per line → pool for **Europe**. |

`logistic.py` uses every China basename and draws a random subset of Europe (`random.seed(42)`, at most 2200 melodies).

---

## Full pipeline

With `.venv` activated, run everything with a single command:

```bash
python run_analysis.py
```

This runs every stage below in order, skips any stage whose declared outputs already exist
(pass `--force` to re-run everything regardless), then builds a consolidated standalone HTML
report at `outputs/report.html` embedding every figure/table produced. Use `--list` to see
stage names, `--only logistic,xgboost` to run a subset, and `--no-report` to skip the report
build.

| Stage | Equivalent manual command | What it does |
|------|---------|----------------|
| Logistic Classifier | `python logistic.py` | Builds `essen_china_europe_features.csv` on first run (this can take a long time due to IDyOM runs). Same stratified train/test and CV as other stages. Writes Figure 1 & 2. |
| XGBoost Classifier | `python xgbclassifier.py` | Needs the features CSV. Same split/features as `logistic`. Writes Figures 3 & 4. |
| EFA | `Rscript factor_logistic.R` | EFA on the same numeric features (8 factors, promax, parallel analysis). Writes Figure 5, Table 2, Table S1, and the factor GLM output/predictions consumed by the next stage. |
| EFA Plotting | `python factor_logistic_plots.py` | Reads the `efa` stage's prediction CSVs and factor scores. Writes Figures 6 & 7. |
| Comparison | `python comparison.py` | Needs the features CSV from `logistic`. Builds or loads the source-to-column mapping, trains one logistic model per implementation source plus an all-features baseline. Writes Table 3 and `outputs/data/coefficients/*.csv`. |

First run of `logistic.py` can take a long time if feature extraction has not already run. After features have been extracted, later runs load the cached `essen_china_europe_features.csv` and skip re-extraction unless you delete that file.

You can still run any stage's script directly (e.g. `python logistic.py`) if you only need one piece — `run_analysis.py` just sequences the same scripts and adds caching/report-building on top.

Shared helpers live under `helpers/`: `dataset` (load / split / CV / logistic pipeline),
`plotting` (confusion heatmaps and signed importance bars), `output_paths` (canonical
artefact paths), `feature_selection`, and `pearce_exclusion`.

---

## Outputs

Every generated artefact lives under `outputs/`, named after the figure/table it corresponds to in the paper.

```
outputs/
  figures/   fig01_logreg_confusion_matrix.{pdf,png}   ... fig07_factor_logreg_permutation_importance.{pdf,png}
             supp_*.{pdf,png}                            (CV/diagnostic plots, not referenced by the paper)
  tables/    table2_efa_variance.{csv,tex}
             table3_source_comparison.{csv,tex}
             table_s1_factor_loadings_top10.{csv,tex}
  data/      intermediate CSVs/JSON (metrics, coefficients, predictions, source mappings, ...)
  report.html                                           (built by build_report.py)
```
---

## Factor network webapp

`Rscript factor_logistic.R` also writes a self-contained 3D interactive visualization of the eight-factor solution to `docs/`/. This provides an interesting way of exploring a low-dimensional version of the melodic feature space, allowing you to study the loadings on each factor and connections between them.

`python build_melody_examples.py` copies MIDI files and writes `docs/melody_examples/manifest.json` for the 3 highest and 3 lowest-scoring melodies per feature node and per factor node. Clicking a node opens an interactive [WaveRoll](https://github.com/crescent-stdio/wave-roll) piano-roll viewer (full melody, scroll/zoom, playback).

- Features are ranked by their value in `essen_china_europe_features.csv`.
- Factors are ranked by the regression factor scores in `outputs/data/factor_scores_for_logreg.csv` (produced by `factor_logistic.R`).
- Use the ‹ › controls in each panel section to browse the top or bottom three examples.

---

## Reproducibility

- **Random seeds:** `42` is fixed in the Python scripts (`train_test_split`, CV folds, Europe subsample, XGBoost, etc.) and in `factor_logistic.R` (`set.seed(42)`). Re-running `python run_analysis.py --force` from a clean `outputs/` directory reproduces the same numbers reported in the paper (modulo upstream library version drift).
- **Same train/test rows** across `logistic.py`, `comparison.py`, and `xgbclassifier.py` — Keep `test_size=0.2` and seeds unchanged.
- **Invalidate the feature cache** — Delete `essen_china_europe_features.csv` to force re-extraction (e.g. after changing `usable_*.txt` or upgrading **`melody-features`** in a way that affects columns). `run_analysis.py` doesn't track this cache's staleness itself — delete the CSV and pass `--force` when you do.
- **Stale outputs** — `run_analysis.py` skips a stage if its declared output files already exist, so if you edit a script's logic without changing its output filenames, run with `--force` (or delete `outputs/` entirely) to make sure results are regenerated.
