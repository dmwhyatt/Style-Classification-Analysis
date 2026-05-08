# Style Classification Task

This repository builds a series of classifiers that predict whether melodies from the Essen corpus originate from **China** or **Europe**, using numeric features from the Python package **`melody-features`** as predictors. It reproduces the manuscript’s confusion-matrix figures, runs exploratory factor analysis and a factor-based logistic model in **R**, and benchmarks logistic regression for each feature-extraction source (IDyOM, jSymbolic, etc.).

Run everything from the repo root unless noted otherwise.

---

## Prerequisites

| Requirement | Notes |
|---------------|--------|
| **Python 3.10+** | Check with `python3 --version`. |
| **R** | Check with `Rscript --version`. Install from [CRAN](https://cran.r-project.org/). |
| **`melody-features`** | Installed via `requirements.txt`. It ships the **Essen corpus** used to resolve melody paths from basename lists. |

---

## One-time setup

**1. Clone and enter the repo**

```bash
git clone <repository-url>
cd DMRN-20-Essen-Classifier
```

**2. Python environment**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. R packages** (for `factor_logistic.R` only)

```bash
Rscript -e 'install.packages(c("tidyverse", "psych", "plotly", "htmlwidgets", "visNetwork"), repos="https://cloud.r-project.org")'
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

With `.venv` activated:

```bash
python logistic.py
python xgbclassifer.py
Rscript factor_logistic.R
python factor_logistic_plot_confusion.py
python comparison.py
```

| Step | Command | What it does |
|------|---------|----------------|
| **1** | `python logistic.py` | Builds `essen_china_europe_features.csv` on first run (this can take a long time due to IDyOM runs). Same stratified train/test and CV as other scripts. Writes **Figure 1** (`confusion_matrix.pdf`) plus coefficient and permutation-importance artefacts. |
| **2** | `python xgbclassifer.py` | Needs the features CSV. Same split/features as **1**. Writes **Figure 2** (`xgb_confusion_matrix_test.pdf`). |
| **3** | `Rscript factor_logistic.R` | EFA on the same numeric features (9 factors, promax, parallel analysis). Writes **Figure 3** (`factor_eigenvalues_elbow.pdf`), factor GLM output, and CSVs consumed by **4**. |
| **4** | `python factor_logistic_plot_confusion.py` | Reads R’s prediction CSVs. Writes **Figure 4** (`factor_logistic_confusion_matrix_test.pdf`). |
| **5** | `python comparison.py` | Needs the features CSV from **1**. Builds or loads `source_to_csv_columns_with_novel.json`, trains one logistic model per implementation source plus an all features baseline, writes comparison CSV/TeX/PDF and `coefficients/*.csv`. |

**First run of Step 1** can take a long time. Later runs load `essen_china_europe_features.csv` and skip re-extraction unless you delete that file.

---

## Figures for `main.tex`

| Figure | Output file | Produced by |
|--------|-------------|-------------|
| 1 | `confusion_matrix.pdf` | `python logistic.py` |
| 2 | `xgb_confusion_matrix_test.pdf` | `python xgbclassifer.py` |
| 3 | `factor_eigenvalues_elbow.pdf` | `Rscript factor_logistic.R` |
| 4 | `factor_logistic_confusion_matrix_test.pdf` | `python factor_logistic_plot_confusion.py` (after **3**) |

---

## Reproducibility

- **Random seeds:** `42` is fixed in the Python scripts (`train_test_split`, CV folds, Europe subsample, XGBoost, etc.) and in `factor_logistic.R` (`set.seed(42)`).
- **Same train/test rows** across `logistic.py`, `comparison.py`, and `xgbclassifer.py` — Keep `test_size=0.2` and seeds unchanged.
- **Invalidate the feature cache** — Delete `essen_china_europe_features.csv` to force re-extraction (e.g. after changing `usable_*.txt` or upgrading **`melody-features`** in a way that affects columns).
