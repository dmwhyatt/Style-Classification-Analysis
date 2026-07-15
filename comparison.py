"""Per-source logistic benchmarks vs full feature set (same splits as logistic.py).

If ``source_to_csv_columns_with_novel.json`` is absent, builds it by matching
``melody_features`` decorator metadata to CSV column names (strict matching +
idyom deny rules; see ``_map_functions_to_csv_columns``), then trains models.
"""

import inspect
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List

import melody_features.features as features_module

from feature_selection import numeric_model_feature_columns, prepare_numeric_feature_matrix
from pearce_exclusion import filter_features_df_pearce, pearce_default_idyom_basename_set

SOURCES = [
    "idyom",
    "jsymbolic",
    "fantastic",
    "partitura",
    "simile",
    "midi_toolbox",
    "MUST",
    "novel",
]

SOURCES_FOR_MAPPING = [
    "novel",
    "idyom",
    "jsymbolic",
    "fantastic",
    "partitura",
    "simile",
    "midi_toolbox",
    "must",
]

SOURCE_MAPPING_KEYS = {
    "MUST": "must",
}

COEFFICIENTS_DIR = "coefficients"
FEATURES_CSV = "essen_china_europe_features.csv"
SOURCE_MAPPING_JSON = "source_to_csv_columns_with_novel.json"

# CSV columns that must not appear under "idyom" when the mapping is auto-built from
# melody_features names (short decorator names like "ioi" used to match ioi_contour_*).
IDYOM_AUTOGEN_DENY_EXACT = frozenset(
    {
        "tonality.proportion_inscale",
    }
)
IDYOM_AUTOGEN_DENY_PREFIXES = ("inter_onset_interval.ioi_contour",)

# Prefix extension ``func + "_" + ...`` only if the implementing name is long enough;
# otherwise only exact ``feature_part == func`` matches (avoids "ioi" -> ioi_contour_*).
_MIN_FUNC_NAME_LEN_FOR_PREFIX_MATCH = 6


def _introspect_source_to_function_names() -> Dict[str, List[str]]:
    source_to_function_names: Dict[str, List[str]] = {
        s: [] for s in SOURCES_FOR_MAPPING
    }
    for name, obj in inspect.getmembers(features_module):
        if not (
            inspect.isfunction(obj)
            or inspect.isclass(obj)
            or (hasattr(obj, "__call__") and hasattr(obj, "__name__"))
        ):
            continue
        if hasattr(obj, "_feature_sources"):
            for src in obj._feature_sources:
                if src in source_to_function_names:
                    source_to_function_names[src].append(name)
        elif hasattr(obj, "_feature_source"):
            src = obj._feature_source
            if src in source_to_function_names:
                source_to_function_names[src].append(name)
    return source_to_function_names


def _feature_matches_function(func_name: str, feature_part: str) -> bool:
    """Whether ``category.<feature_part>`` is plausibly produced by ``func_name``."""
    if func_name == feature_part:
        return True
    if len(func_name) >= _MIN_FUNC_NAME_LEN_FOR_PREFIX_MATCH and feature_part.startswith(
        func_name + "_"
    ):
        return True
    return False


def _best_matching_function_name(
    function_names: List[str], feature_part: str
) -> str | None:
    """Prefer the longest implementing name so ``ioi`` does not steal ``ioi_contour_*``."""
    best: str | None = None
    best_len = -1
    for fn in function_names:
        if _feature_matches_function(fn, feature_part) and len(fn) > best_len:
            best = fn
            best_len = len(fn)
    return best


def _map_functions_to_csv_columns(
    source_to_function_names: Dict[str, List[str]],
    csv_columns: List[str],
) -> Dict[str, List[str]]:
    """Map implementing function names to ``category.feature`` CSV columns."""
    source_to_csv_columns: Dict[str, List[str]] = {s: [] for s in SOURCES_FOR_MAPPING}
    for source, function_names in source_to_function_names.items():
        for csv_col in csv_columns:
            feature_part = csv_col.split(".", 1)[1] if "." in csv_col else csv_col
            if _best_matching_function_name(function_names, feature_part) is not None:
                source_to_csv_columns[source].append(csv_col)
    return source_to_csv_columns


def _source_mapping_key(source: str) -> str:
    """Return the melody_features source key for a benchmark display source."""
    return SOURCE_MAPPING_KEYS.get(source, source)


def _supplement_missing_source_mappings(
    mapping: Dict[str, List[str]],
    csv_columns: List[str],
) -> Dict[str, List[str]]:
    """Add mappings for sources absent from an older hand-maintained JSON."""
    missing_sources = [s for s in SOURCES_FOR_MAPPING if s not in mapping]
    if not missing_sources:
        return mapping

    fn_map = _introspect_source_to_function_names()
    missing_fn_map = {s: fn_map.get(s, []) for s in missing_sources}
    supplemental = _map_functions_to_csv_columns(missing_fn_map, csv_columns)
    for source in missing_sources:
        mapping[source] = supplemental.get(source, [])
        print(
            f"Added missing source mapping for '{source}' "
            f"({len(mapping[source])} columns)."
        )
    return mapping


def _strip_idyom_autogen_false_positives(columns: List[str]) -> tuple[List[str], List[str]]:
    """Remove columns that should never be tagged ``idyom`` by the automapper."""
    kept: List[str] = []
    dropped: List[str] = []
    for c in columns:
        if c in IDYOM_AUTOGEN_DENY_EXACT or any(
            c.startswith(p) for p in IDYOM_AUTOGEN_DENY_PREFIXES
        ):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def _idyom_mapping_offenders(columns: List[str]) -> List[str]:
    """Columns in a hand-maintained JSON that violate idyom automap hygiene rules."""
    _, dropped = _strip_idyom_autogen_false_positives(columns)
    return dropped


def write_feature_source_mapping_csvs(source_to_features: Dict[str, List[str]]) -> None:
    """Emit human-readable mapping tables"""
    feature_to_sources: Dict[str, List[str]] = {}
    for source, feats in source_to_features.items():
        for feature in feats:
            feature_to_sources.setdefault(feature, []).append(source)

    records = []
    for feature, srcs in feature_to_sources.items():
        if "." in feature:
            category, feature_name = feature.split(".", 1)
        else:
            category, feature_name = "other", feature
        records.append(
            {
                "feature": feature,
                "feature_name": feature_name,
                "category": category,
                "sources": ", ".join(sorted(srcs)),
                "num_sources": len(srcs),
            }
        )
    df = pd.DataFrame(records).sort_values(["category", "feature"])
    df.to_csv("feature_source_mapping_complete.csv", index=False)

    expanded_records = []
    for source, feats in source_to_features.items():
        for feature in feats:
            if "." in feature:
                category, feature_name = feature.split(".", 1)
            else:
                category, feature_name = "other", feature
            expanded_records.append(
                {
                    "feature": feature,
                    "feature_name": feature_name,
                    "category": category,
                    "source": source,
                }
            )
    expanded_df = pd.DataFrame(expanded_records).sort_values(
        ["source", "category", "feature"]
    )
    expanded_df.to_csv("feature_source_mapping_expanded.csv", index=False)

    pivot = expanded_df.groupby(["source", "category"]).size().reset_index(name="count")
    pivot_table = pivot.pivot(
        index="category", columns="source", values="count"
    ).fillna(0).astype(int)
    pivot_table.to_csv("feature_source_category_summary_complete.csv")

    print(
        "Wrote feature_source_mapping_complete.csv, "
        "feature_source_mapping_expanded.csv, "
        "feature_source_category_summary_complete.csv"
    )


def load_or_build_source_mapping(
    features_csv: str, json_path: str
) -> tuple[Dict[str, List[str]], pd.DataFrame]:
    """Load JSON if present; otherwise introspect melody_features and build mapping + CSVs."""
    features_df = pd.read_csv(features_csv)
    csv_columns = [
        c
        for c in features_df.columns
        if c not in ("melody_num", "melody_id", "continent")
    ]

    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            mapping = json.load(f)
        mapping = _supplement_missing_source_mappings(mapping, csv_columns)
        offenders = _idyom_mapping_offenders(mapping.get("idyom", []))
        if offenders:
            print(
                "WARNING: idyom source lists columns flagged as automap false positives "
                f"(see comparison.py IDYOM_AUTOGEN_DENY_*): {offenders}"
            )
        return mapping, features_df

    print(
        f"\n{'=' * 80}\n"
        f"No {json_path} found — building from melody_features decorators "
        f"and columns in {features_csv}\n"
        f"{'=' * 80}"
    )
    fn_map = _introspect_source_to_function_names()
    mapping = _map_functions_to_csv_columns(fn_map, csv_columns)
    if "idyom" in mapping:
        kept, dropped = _strip_idyom_autogen_false_positives(mapping["idyom"])
        mapping["idyom"] = kept
        if dropped:
            print(
                "idyom automap hygiene: removed columns that must not be tagged idyom "
                f"when rebuilding JSON: {dropped}"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote {json_path}")

    all_mapped = set()
    for cols in mapping.values():
        all_mapped.update(cols)
    print(
        f"Mapping summary: {len(csv_columns)} CSV columns, "
        f"{len(all_mapped)} uniquely mapped, "
        f"{len(csv_columns) - len(all_mapped)} unmapped"
    )
    for src in SOURCES_FOR_MAPPING:
        print(f"  {src:15s}: {len(mapping.get(src, [])):3d} columns")

    write_feature_source_mapping_csvs(mapping)
    return mapping, features_df


if not os.path.isfile(FEATURES_CSV):
    raise RuntimeError(
        f"{FEATURES_CSV} not found. Run logistic.py first to generate features."
    )

print(f"Loading cached features from {FEATURES_CSV} ...")
SOURCE_FEATURES, features_df = load_or_build_source_mapping(FEATURES_CSV, SOURCE_MAPPING_JSON)
_pearce = pearce_default_idyom_basename_set()
_n0 = len(features_df)
features_df = filter_features_df_pearce(features_df, _pearce)
if len(features_df) < _n0:
    print(f"Excluded {_n0 - len(features_df)} row(s) overlapping pearce_default_idyom.")

print(f"Dataset size: {len(features_df)}")
print(features_df["continent"].value_counts())

# Prepare data: same feature columns as logistic.py / factor_logistic.R / xgbclassifer
all_feature_cols = numeric_model_feature_columns(features_df)
X_all, all_feature_cols = prepare_numeric_feature_matrix(features_df, all_feature_cols)
print(f"Numeric features used for modeling: {len(all_feature_cols)}")

y = features_df["continent"].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Same split as logistic.py (random_state=42, test_size=0.2)
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train set size: {len(X_train_all)}, Test set size: {len(X_test_all)}")

# 5-fold stratified CV on training set only (same setup as logistic.py)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_logreg_pipeline(random_state: int = 42) -> Pipeline:
    """Same pipeline as logistic.py"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def cross_val_accuracy(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    columns: List[str],
) -> tuple[float, float]:
    """Mean and population SD of accuracy across CV folds."""
    if not columns:
        return float("nan"), float("nan")
    fold_accs: List[float] = []
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx][columns]
        X_va = X_train.iloc[valid_idx][columns]
        y_tr, y_va = y_train[train_idx], y_train[valid_idx]
        clf = build_logreg_pipeline(random_state=42)
        clf.fit(X_tr, y_tr)
        fold_accs.append(accuracy_score(y_va, clf.predict(X_va)))
    return float(np.mean(fold_accs)), float(np.std(fold_accs, ddof=0))


results: List[Dict] = []
trained_models: Dict[str, Pipeline] = {}

print("\n" + "="*80)
print("Training separate models for each feature source")
print("="*80)

for source in SOURCES:
    print(f"\n{'='*80}")
    print(f"Source: {source.upper()}")
    print(f"{'='*80}")
    
    try:
        mapping_key = _source_mapping_key(source)
        if mapping_key == "idyom":
            # Decorator JSON mapping omits idyom.* STM/LTM information-content columns
            base = [f for f in SOURCE_FEATURES.get("idyom", []) if f in all_feature_cols]
            stm_ltm = [
                c
                for c in all_feature_cols
                if c.lower().startswith("idyom.")
                and ("_stm_" in c.lower() or "_ltm_" in c.lower())
            ]
            valid_features = sorted(set(base) | set(stm_ltm))
        else:
            source_feature_cols = SOURCE_FEATURES.get(mapping_key, [])
            valid_features = [f for f in source_feature_cols if f in all_feature_cols]

        if not valid_features:
            print(f"No features found for source '{source}', skipping...")
            results.append({
                "source": source,
                "num_features": 0,
                "cv_accuracy_mean": np.nan,
                "cv_accuracy_std": np.nan,
                "train_accuracy": np.nan,
                "test_accuracy": np.nan,
            })
            continue
        
        print(f"Number of features: {len(valid_features)}")
        print(f"Sample features: {valid_features[:3]}")
        
        cv_mean, cv_std = cross_val_accuracy(X_train_all, y_train, valid_features)
        print(
            f"CV accuracy (train, 5-fold): {cv_mean:.4f} ± {cv_std:.4f}"
        )

        X_train = X_train_all[valid_features]
        X_test = X_test_all[valid_features]

        clf = build_logreg_pipeline(random_state=42)
        clf.fit(X_train, y_train)

        trained_models[source] = clf
        trained_models[f"{source}_features"] = valid_features

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")

        y_test_labels = le.inverse_transform(y_test)
        y_test_pred_labels = le.inverse_transform(y_test_pred)
        print("\nTest set classification report:")
        print(classification_report(
            y_test_labels, y_test_pred_labels, 
            target_names=list(le.classes_), 
            digits=4
        ))

        results.append({
            "source": source,
            "num_features": len(valid_features),
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

    except Exception as e:
        print(f"Error processing source '{source}': {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "source": source,
            "num_features": 0,
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
            "train_accuracy": np.nan,
            "test_accuracy": np.nan,
        })

print(f"\n{'='*80}")
print(f"BASELINE: ALL FEATURES")
print(f"{'='*80}")
print(f"Number of features: {len(all_feature_cols)}")

cv_mean_all, cv_std_all = cross_val_accuracy(X_train_all, y_train, all_feature_cols)
print(f"CV accuracy (train, 5-fold): {cv_mean_all:.4f} ± {cv_std_all:.4f}")

clf_all = build_logreg_pipeline(random_state=42)
clf_all.fit(X_train_all, y_train)

trained_models["ALL"] = clf_all
trained_models["ALL_features"] = all_feature_cols

y_train_pred_all = clf_all.predict(X_train_all)
y_test_pred_all = clf_all.predict(X_test_all)

train_acc_all = accuracy_score(y_train, y_train_pred_all)
test_acc_all = accuracy_score(y_test, y_test_pred_all)

print(f"Train accuracy: {train_acc_all:.4f}")
print(f"Test accuracy:  {test_acc_all:.4f}")

y_test_labels_all = le.inverse_transform(y_test)
y_test_pred_labels_all = le.inverse_transform(y_test_pred_all)
print("\nTest set classification report:")
print(classification_report(
    y_test_labels_all, y_test_pred_labels_all,
    target_names=list(le.classes_),
    digits=4
))

results.append({
    "source": "ALL",
    "num_features": len(all_feature_cols),
    "cv_accuracy_mean": cv_mean_all,
    "cv_accuracy_std": cv_std_all,
    "train_accuracy": train_acc_all,
    "test_accuracy": test_acc_all,
})

results_df = pd.DataFrame(results).sort_values("cv_accuracy_mean", ascending=False, na_position="last")
print("\n" + "="*80)
print("SUMMARY: Model Performance by Feature Source")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

results_df.to_csv("source_comparison_results.csv", index=False)
print("\nSaved results to 'source_comparison_results.csv'")


SOURCE_DISPLAY_NAMES = {
    "ALL": r"\textit{\textbf{melody-features}}",
    "jsymbolic": "jSymbolic",
    "fantastic": "FANTASTIC",
    "MUST": "MUST",
    "novel": "Novel",
    "midi_toolbox": "MIDI Toolbox",
    "partitura": "Partitura",
    "idyom": "IDyOM",
    "simile": "SIMILE",
}


def _latex_fmt(x: float, decimals: int = 4) -> str:
    if pd.isna(x):
        return "---"
    return f"{float(x):.{decimals}f}"


def _latex_maybe_bold(text: str, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold else text


def _best_mask(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """True where ``series`` equals the best finite value (ties allowed)."""
    vals = pd.to_numeric(series, errors="coerce")
    finite = vals.dropna()
    if finite.empty:
        return pd.Series(False, index=series.index)
    best = finite.max() if higher_is_better else finite.min()
    return vals.eq(best)


def save_results_latex_table(df: pd.DataFrame, path: str) -> None:
    """Pretty table of feature sources with best-in-column values bolded."""
    bold_n = _best_mask(df["num_features"], higher_is_better=True)
    bold_cv = _best_mask(df["cv_accuracy_mean"], higher_is_better=True)
    bold_sd = _best_mask(df["cv_accuracy_std"], higher_is_better=False)
    bold_tr = _best_mask(df["train_accuracy"], higher_is_better=True)
    bold_te = _best_mask(df["test_accuracy"], higher_is_better=True)

    lines = [
        r"\newpage",
        r"\begin{table}[h]",
        r"  \centering",
        r"  \label{tab:source-comparison}",
        r"  \begin{tabular}{@{}lccccc@{}}",
        r"    Source & No. features & CV acc.\ & CV SD & Train acc.\ & Test acc.\ \\",
        r"    \midrule",
    ]
    for i, row in df.iterrows():
        src_key = str(row["source"])
        src = SOURCE_DISPLAY_NAMES.get(src_key, src_key.replace("_", r"\_"))
        nf = int(row["num_features"]) if pd.notna(row["num_features"]) else 0
        lines.append(
            "    "
            + " & ".join(
                [
                    src,
                    _latex_maybe_bold(str(nf), bool(bold_n.loc[i])),
                    _latex_maybe_bold(
                        _latex_fmt(row["cv_accuracy_mean"]), bool(bold_cv.loc[i])
                    ),
                    _latex_maybe_bold(
                        _latex_fmt(row["cv_accuracy_std"]), bool(bold_sd.loc[i])
                    ),
                    _latex_maybe_bold(
                        _latex_fmt(row["train_accuracy"]), bool(bold_tr.loc[i])
                    ),
                    _latex_maybe_bold(
                        _latex_fmt(row["test_accuracy"]), bool(bold_te.loc[i])
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"  \end{tabular}",
            r"  \caption{Logistic regression by feature extraction source.}\end{table}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


save_results_latex_table(results_df, "source_comparison_results.tex")
print("Saved LaTeX table to 'source_comparison_results.tex'")

# Extract and save coefficients for each model
print("\n" + "="*80)
print("Extracting and saving model coefficients")
print("="*80)

os.makedirs(COEFFICIENTS_DIR, exist_ok=True)

class_names = list(le.classes_)

for source in SOURCES + ["ALL"]:
    if source not in trained_models:
        continue
    
    clf = trained_models[source]
    feature_names = trained_models[f"{source}_features"]
    
    logreg = clf.named_steps['logreg']
    coefficients = logreg.coef_
    intercepts = logreg.intercept_

    # sklearn: one coef row for binary OVR; one row per class for true multinomial.
    n_coef_rows = coefficients.shape[0]
    n_classes = len(class_names)
    
    if n_coef_rows == 1 and n_classes == 2:
        coef_data = {"feature": feature_names}
        coef_data[f"coef_{class_names[1]}"] = coefficients[0]
        
        coef_df = pd.DataFrame(coef_data)
        
        intercept_row = {"feature": "intercept"}
        intercept_row[f"coef_{class_names[1]}"] = intercepts[0]
        intercept_df = pd.DataFrame([intercept_row])
        
        coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)
    elif n_coef_rows == n_classes:
        coef_data = {"feature": feature_names}
        
        for i, class_name in enumerate(class_names):
            coef_data[f"coef_{class_name}"] = coefficients[i]
        
        coef_df = pd.DataFrame(coef_data)
        
        intercept_row = {"feature": "intercept"}
        for i, class_name in enumerate(class_names):
            intercept_row[f"coef_{class_name}"] = intercepts[i]
        intercept_df = pd.DataFrame([intercept_row])
        
        coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)
    else:
        coef_data = {"feature": feature_names}
        
        for i in range(n_coef_rows):
            class_label = class_names[i] if i < n_classes else f"class_{i}"
            coef_data[f"coef_{class_label}"] = coefficients[i]
        
        coef_df = pd.DataFrame(coef_data)
        
        intercept_row = {"feature": "intercept"}
        for i in range(len(intercepts)):
            class_label = class_names[i] if i < n_classes else f"class_{i}"
            intercept_row[f"coef_{class_label}"] = intercepts[i]
        intercept_df = pd.DataFrame([intercept_row])
        
        coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)
    
    output_file = os.path.join(COEFFICIENTS_DIR, f"coefficients_{source}.csv")
    coef_df.to_csv(output_file, index=False)
    print(f"Saved coefficients for '{source}' to '{output_file}'")
    print(f"  - {len(feature_names)} features, {len(class_names)} classes")

print("="*80)
