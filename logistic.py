import os
import random
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from melody_features import get_all_features
from melody_features.corpus import get_corpus_files
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, get_scorer

from helpers import dataset
from helpers import output_paths as OP
from helpers.pearce_exclusion import (
    basename_no_ext,
    filter_features_df_pearce,
    filter_paths_and_labels_pearce,
    pearce_default_idyom_basename_set,
    write_pearce_basename_sidecar,
)
from helpers.plotting import (
    confusion_heatmap,
    confusion_importance_multipanel,
    prettify_feature_name,
    signed_permutation_importance_bar,
)

GROUP_DISPLAY_NAMES = {
    "pitch": "Pitch",
    "rhythm": "Rhythm",
    "pitch_and_rhythm": "Pitch&Rhythm",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-feature logistic regression (Figures 1-2)."
    )
    parser.add_argument(
        "--preprint",
        action="store_true",
        help=(
            "Also write a two-panel preprint figure combining the confusion "
            "matrix and permutation importance. Does not replace Figures 1-2."
        ),
    )
    return parser.parse_args() if __name__ == "__main__" else argparse.Namespace(preprint=False)


CLI_ARGS = _parse_args()


def _assign_feature_groups(coef_df: pd.DataFrame) -> None:
    """Assign Pitch / Rhythm / Pitch&Rhythm taxonomy labels on ``coef_df``."""
    feat = coef_df["feature"].astype(str)
    pretty = coef_df["pretty_feature"].astype(str)
    groups = pd.Series("pitch", index=coef_df.index)
    groups.loc[
        feat.str.contains(r"timing|inter_onset_interval|metre", case=False, na=False)
    ] = "rhythm"
    groups.loc[
        feat.str.contains(r"corpus|lexical_diversity", case=False, na=False)
    ] = "pitch_and_rhythm"
    groups.loc[pretty.str.contains("Complebm Pitch", case=False, na=False)] = "pitch"
    groups.loc[pretty.str.contains("Complebm Rhythm", case=False, na=False)] = "rhythm"
    groups.loc[pretty.str.contains("Complebm Optimal", case=False, na=False)] = (
        "pitch_and_rhythm"
    )
    coef_df["feature_group"] = groups


def _append_taxonomy_summaries_to_coef_df(coef_df: pd.DataFrame) -> None:
    """Scale coefficients, assign Pitch/Rhythm groups, print top-|β| features."""
    c = coef_df["coefficient"].astype(float)
    std = float(c.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        coef_df["coefficient_scaled"] = np.nan
    else:
        coef_df["coefficient_scaled"] = (c - c.mean()) / std

    _assign_feature_groups(coef_df)

    abs_scaled = coef_df["coefficient_scaled"].abs()
    top3 = coef_df.assign(_abs_scaled=abs_scaled).nlargest(3, "_abs_scaled").drop(
        columns=["_abs_scaled"]
    )
    print("\nTop 3 features by |scaled coefficient|:")
    print(top3.to_string(index=False))


def _group_permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    feature_to_group: pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Permutation importance for feature groups (shared row shuffle within group)."""
    scorer = get_scorer(scoring)
    baseline = float(scorer(model, X, y))
    rng = np.random.RandomState(random_state)

    rows = []
    for group_key in ("pitch", "rhythm", "pitch_and_rhythm"):
        cols = feature_to_group.index[feature_to_group == group_key].tolist()
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = np.empty(n_repeats, dtype=float)
        for i in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm.loc[:, cols] = X.iloc[perm_idx][cols].to_numpy()
            drops[i] = baseline - float(scorer(model, X_perm, y))
        rows.append(
            {
                "feature_group": GROUP_DISPLAY_NAMES[group_key],
                "n_features": len(cols),
                "importance_mean": float(drops.mean()),
                "importance_std": float(drops.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)

USABLE_CHINA_TXT = "usable_china.txt"
USABLE_EUROPA_TXT = "usable_europa.txt"


def _read_usable_basenames_txt(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def _selected_paths_and_labels_from_usable_sets(
    usable_china: set[str],
    usable_europa: set[str],
    basename_to_path: Dict[str, str],
) -> tuple[List[str], List[str]]:
    random.seed(42)
    europe_target = min(2200, len(usable_europa))
    sampled_europa = set(random.sample(sorted(usable_europa), europe_target))

    selected_files: List[str] = []
    labels: List[str] = []
    for name in sorted(usable_china):
        p = basename_to_path.get(name)
        if p:
            selected_files.append(p)
            labels.append("China")
    for name in sorted(sampled_europa):
        p = basename_to_path.get(name)
        if p:
            selected_files.append(p)
            labels.append("Europe")
    return selected_files, labels


essen_paths = get_corpus_files("essen")
basename_to_path = {basename_no_ext(str(p)): str(p) for p in essen_paths}

for required in (USABLE_CHINA_TXT, USABLE_EUROPA_TXT):
    if not os.path.isfile(required):
        raise RuntimeError(
            f"Missing required file {required!r}. Place {USABLE_CHINA_TXT} and {USABLE_EUROPA_TXT} "
            "in the project root (one melody basename per line per region; both are tracked in git). "
            "See README.md."
        )

print(f"Loading {USABLE_CHINA_TXT} and {USABLE_EUROPA_TXT} …")
usable_china = _read_usable_basenames_txt(USABLE_CHINA_TXT)
usable_europa = _read_usable_basenames_txt(USABLE_EUROPA_TXT)

pearce_basenames = pearce_default_idyom_basename_set()
write_pearce_basename_sidecar()
if pearce_basenames:
    n_c = sum(1 for n in usable_china if n in pearce_basenames)
    n_e = sum(1 for n in usable_europa if n in pearce_basenames)
    if n_c or n_e:
        print(
            f"Excluding {len(pearce_basenames)} pearce_default_idyom basenames from selection "
            f"({n_c} in China list, {n_e} in Europe list match)."
        )
    usable_china = {n for n in usable_china if n not in pearce_basenames}
    usable_europa = {n for n in usable_europa if n not in pearce_basenames}

selected_files, labels = _selected_paths_and_labels_from_usable_sets(
    usable_china, usable_europa, basename_to_path
)
selected_files, labels = filter_paths_and_labels_pearce(
    selected_files, labels, pearce_basenames
)

print(f"Selected Essen files — China: {labels.count('China')}, Europe: {labels.count('Europe')}")

if len(selected_files) == 0:
    raise RuntimeError(
        f"No corpus paths resolved from {USABLE_CHINA_TXT} / {USABLE_EUROPA_TXT}. "
        "Basenames must exist in the melody-features Essen corpus (check package install/version)."
    )

FEATURES_CSV = OP.FEATURES_CSV

if os.path.isfile(FEATURES_CSV):
    print(f"Loading cached features from {FEATURES_CSV} ...")
    features_df = pd.read_csv(FEATURES_CSV)
    n_before = len(features_df)
    features_df = filter_features_df_pearce(features_df, pearce_basenames)
    if len(features_df) < n_before:
        print(
            f"Dropped {n_before - len(features_df)} cached row(s) overlapping "
            "pearce_default_idyom; rewriting CSV."
        )
        features_df.to_csv(FEATURES_CSV, index=False)
else:
    print(f"Extracting features (total files: {len(selected_files)})...")
    features_df = get_all_features(selected_files, skip_idyom=False)

    name_to_label = {}
    for pth, lab in zip(selected_files, labels):
        name_to_label[basename_no_ext(pth)] = lab

    path_like_columns = [
        "melody_id",
        "file_path", "path", "midi_path", "source_path", "file", "filename", "name"
    ]
    found_col = None
    for col in path_like_columns:
        if col in features_df.columns:
            found_col = col
            break

    if found_col is None:
        if len(features_df) != len(selected_files):
            raise RuntimeError(
                "Feature rows do not match inputs and no path-like column found for alignment."
            )
        aligned = pd.Series([name_to_label.get(basename_no_ext(p), None) for p in selected_files])
        features_df = features_df.copy()
        features_df["continent"] = pd.Categorical(list(aligned), categories=["China", "Europe"])
    else:
        basenames = features_df[found_col].astype(str).map(basename_no_ext)
        aligned = basenames.map(name_to_label)
        mask = aligned.notna()
        if not mask.all():
            features_df = features_df.loc[mask].reset_index(drop=True)
            aligned = aligned.loc[mask].reset_index(drop=True)
        features_df = features_df.copy()
        features_df["continent"] = pd.Categorical(list(aligned), categories=["China", "Europe"])

    features_df = filter_features_df_pearce(features_df, pearce_basenames)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved cached features to {FEATURES_CSV}")

print(features_df["continent"].value_counts())

# prepare data: use consistent feature columns across factor_logistic.R / comparison.py / xgbclassifier.py
X, feature_cols, y_enc, le = dataset.prepare_xy(features_df)

X_train_full, X_test, y_train_full, y_test = dataset.make_train_test_split(X, y_enc)
skf = dataset.make_cv()

fold_metrics: List[Dict[str, float]] = []
all_true_labels: List[str] = []
all_pred_labels: List[str] = []

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
    X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
    y_train, y_valid = y_train_full[train_idx], y_train_full[valid_idx]

    clf = dataset.build_logreg_pipeline(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    fold_acc = accuracy_score(y_valid, y_pred)
    print(f"Fold {fold_idx} accuracy: {fold_acc:.4f}")
    fold_metrics.append({"fold": fold_idx, "accuracy": fold_acc})

    all_true_labels.extend(le.inverse_transform(y_valid))
    all_pred_labels.extend(le.inverse_transform(y_pred))


metrics_df = pd.DataFrame(fold_metrics)
print("Cross-validation accuracy by fold:")
print(metrics_df)
print(
    f"Mean accuracy: {metrics_df['accuracy'].mean():.4f} "
    f"± {metrics_df['accuracy'].std(ddof=0):.4f}"
)

overall_acc = accuracy_score(all_true_labels, all_pred_labels)
print(f"Overall cross-validated accuracy: {overall_acc:.4f}")
print(
    classification_report(
        all_true_labels, all_pred_labels, target_names=list(le.classes_), digits=4
    )
)
cv_cm_path = OP.supp_fig_path("logreg_confusion_matrix_cv", ext="png")
confusion_heatmap(
    all_true_labels,
    all_pred_labels,
    list(le.classes_),
    cv_cm_path,
    save_png_twin=False,
)

print(f"Saved plots: '{cv_cm_path}'")

print("\nTraining final model on full training set...")
final_model = dataset.build_logreg_pipeline(random_state=42)
final_model.fit(X_train_full, y_train_full)

y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTest set accuracy: {test_acc:.4f}")
y_test_labels = le.inverse_transform(y_test)
y_test_pred_labels = le.inverse_transform(y_test_pred)
pred_csv_path = OP.data_path("logreg_predictions_test.csv")
pd.DataFrame(
    {"true_label": y_test_labels, "predicted": y_test_pred_labels}
).to_csv(pred_csv_path, index=False)
print(f"Saved test set predictions: '{pred_csv_path}'")
print("Test set classification report:")
print(
    classification_report(
        y_test_labels, y_test_pred_labels, target_names=list(le.classes_), digits=4
    )
)

fig01_path = OP.fig_path(OP.FIG_LOGREG_CONFUSION, "logreg_confusion_matrix")
confusion_heatmap(
    y_test_labels,
    y_test_pred_labels,
    list(le.classes_),
    fig01_path,
    save_png_twin=True,
)
print(f"Saved test set confusion matrix (Figure 1): '{fig01_path}'")

logreg_metrics_path = OP.data_path("logreg_metrics.csv")
pd.DataFrame(
    {
        "metric": [
            "cv_accuracy_mean",
            "cv_accuracy_sd",
            "cv_accuracy_overall",
            "test_accuracy",
            "n_features",
        ],
        "value": [
            metrics_df["accuracy"].mean(),
            metrics_df["accuracy"].std(ddof=0),
            overall_acc,
            test_acc,
            len(feature_cols),
        ],
    }
).to_csv(logreg_metrics_path, index=False)
print(f"Saved summary metrics: '{logreg_metrics_path}'")

logreg = final_model.named_steps["logreg"]
scaler = final_model.named_steps.get("scaler")
coef = logreg.coef_
intercept = logreg.intercept_.copy()

if scaler is not None:
    scale = getattr(scaler, "scale_", None)
    mean = getattr(scaler, "mean_", None)
    if scale is not None:
        coef = coef / scale
    if mean is not None:
        intercept = intercept - np.dot(coef, mean)

if coef.ndim == 2:
    if coef.shape[0] == 1:
        coef_df = (
            pd.DataFrame({"feature": feature_cols, "coefficient": coef[0]})
            .assign(
                abs_coefficient=lambda df: df["coefficient"].abs(),
                pretty_feature=lambda df: df["feature"].map(prettify_feature_name),
            )
            .sort_values("abs_coefficient", ascending=False)
        )
        coef_df.insert(2, "intercept", intercept[0])
        _append_taxonomy_summaries_to_coef_df(coef_df)
        coef_csv_path = OP.data_path("logistic_coefficients.csv")
        coef_df.to_csv(coef_csv_path, index=False)
        print(f"Saved logistic coefficients: '{coef_csv_path}'")

        print("\nComputing permutation importance on test set (n_repeats=10)...")
        perm = permutation_importance(
            final_model, X_test, y_test,
            n_repeats=10, random_state=42, scoring="accuracy", n_jobs=-1,
        )
        perm_df = (
            pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            })
            .merge(
                coef_df[["feature", "coefficient"]],
                on="feature",
                how="left",
            )
            .assign(pretty_feature=lambda df: df["feature"].map(prettify_feature_name))
            .sort_values("importance_mean", ascending=False)
        )
        perm_out = perm_df[
            ["feature", "coefficient", "importance_mean", "importance_std", "pretty_feature"]
        ]
        perm_csv_path = OP.data_path("logistic_permutation_importance.csv")
        perm_out.to_csv(perm_csv_path, index=False)

        topk = perm_out.head(20).iloc[::-1].reset_index(drop=True)
        pos_class, neg_class = le.classes_[1], le.classes_[0]
        fig02_path = OP.fig_path(OP.FIG_LOGREG_IMPORTANCE, "logreg_permutation_importance")
        signed_permutation_importance_bar(
            pretty_names=topk["pretty_feature"],
            importance_mean=topk["importance_mean"],
            importance_std=topk["importance_std"],
            coefficients=topk["coefficient"],
            pdf_path=fig02_path,
            pos_class=pos_class,
            neg_class=neg_class,
        )
        print(
            f"Saved permutation importance (Figure 2): '{perm_csv_path}', '{fig02_path}'"
        )
        if CLI_ARGS.preprint:
            preprint_path = OP.preprint_fig_path(OP.PREPRINT_LOGREG)
            confusion_importance_multipanel(
                y_true=y_test_labels,
                y_pred=y_test_pred_labels,
                class_names=list(le.classes_),
                pretty_names=topk["pretty_feature"],
                importance_mean=topk["importance_mean"],
                importance_std=topk["importance_std"],
                coefficients=topk["coefficient"],
                pdf_path=preprint_path,
                pos_class=pos_class,
                neg_class=neg_class,
            )
            print(f"Saved preprint panel: '{preprint_path}'")
        print("\nTop 20 features by permutation importance:")
        print(
            perm_out.head(20)[
                ["feature", "coefficient", "importance_mean", "importance_std"]
            ].to_string(index=False)
        )

        print(
            "\nComputing group permutation importance on test set "
            "(Pitch / Rhythm / Pitch&Rhythm, n_repeats=10)..."
        )
        feature_to_group = coef_df.set_index("feature")["feature_group"]
        group_perm_df = _group_permutation_importance(
            final_model,
            X_test,
            y_test,
            feature_to_group,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )
        group_perm_csv_path = OP.data_path("logistic_group_permutation_importance.csv")
        group_perm_df.to_csv(group_perm_csv_path, index=False)
        print(f"Saved group permutation importance: '{group_perm_csv_path}'")
        print("\nFeature group permutation importance:")
        print(group_perm_df.to_string(index=False))
    else:
        classes = le.inverse_transform(np.arange(coef.shape[0]))
        multi_df = pd.DataFrame(coef.T, index=feature_cols, columns=classes)
        multi_df.loc["intercept"] = intercept
        multi_coef_path = OP.data_path("logistic_coefficients.csv")
        multi_df.to_csv(multi_coef_path)
        print(f"Saved multi-class logistic coefficients: '{multi_coef_path}'")
