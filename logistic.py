import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from melody_features import get_all_features
from melody_features.corpus import get_corpus_files
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature_selection import numeric_model_feature_columns, prepare_numeric_feature_matrix


def _append_taxonomy_summaries_to_coef_df(coef_df: pd.DataFrame) -> None:
    """Scale coefficients, assign Pitch/Rhythm groups, print summaries."""
    c = coef_df["coefficient"].astype(float)
    std = float(c.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        coef_df["coefficient_scaled"] = np.nan
    else:
        coef_df["coefficient_scaled"] = (c - c.mean()) / std

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

    abs_scaled = coef_df["coefficient_scaled"].abs()
    group_importance = (
        coef_df.assign(abs_beta=abs_scaled)
        .groupby("feature_group", as_index=False)["abs_beta"]
        .sum()
        .rename(columns={"abs_beta": "sum_abs_beta"})
    )
    denom = group_importance["sum_abs_beta"].sum()
    group_importance["prop_importance"] = (
        group_importance["sum_abs_beta"] / denom if denom else 0.0
    )
    group_importance = group_importance.sort_values(
        "prop_importance", ascending=False
    )

    print("\nFeature group importance (|scaled coefficient| mass):")
    print(group_importance.to_string(index=False))

    top3 = coef_df.assign(_abs_scaled=abs_scaled).nlargest(3, "_abs_scaled").drop(
        columns=["_abs_scaled"]
    )
    print("\nTop 3 features by |scaled coefficient|:")
    print(top3.to_string(index=False))


def _save_confusion_heatmap(
    y_true,
    y_pred,
    class_names: List[str],
    *,
    title: str,
    out_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({pct[i, j]:.1f}%)"
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


USABLE_CHINA_TXT = "usable_china.txt"
USABLE_EUROPA_TXT = "usable_europa.txt"


def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


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
selected_files, labels = _selected_paths_and_labels_from_usable_sets(
    usable_china, usable_europa, basename_to_path
)

print(f"Selected Essen files — China: {labels.count('China')}, Europe: {labels.count('Europe')}")

if len(selected_files) == 0:
    raise RuntimeError(
        f"No corpus paths resolved from {USABLE_CHINA_TXT} / {USABLE_EUROPA_TXT}. "
        "Basenames must exist in the melody-features Essen corpus (check package install/version)."
    )

FEATURES_CSV = "essen_china_europe_features.csv"

if os.path.isfile(FEATURES_CSV):
    print(f"Loading cached features from {FEATURES_CSV} ...")
    features_df = pd.read_csv(FEATURES_CSV)
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

    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved cached features to {FEATURES_CSV}")

print(features_df["continent"].value_counts())

# prepare data: use consistent feature columns across factor_logistic.R / comparison.py / xgbclassifer.py
feature_cols = numeric_model_feature_columns(features_df)
X, feature_cols = prepare_numeric_feature_matrix(features_df, feature_cols)
print(f"Numeric features used for modeling: {len(feature_cols)}")
y = features_df["continent"].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train set size: {len(X_train_full)}, Test set size: {len(X_test)}")

# run 5-fold cross-validation on training set only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics: List[Dict[str, float]] = []
all_true_labels: List[str] = []
all_pred_labels: List[str] = []


def build_logreg_pipeline(random_state: int = 42) -> Pipeline:
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

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
    X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
    y_train, y_valid = y_train_full[train_idx], y_train_full[valid_idx]

    clf = build_logreg_pipeline(random_state=42)

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
_save_confusion_heatmap(
    all_true_labels,
    all_pred_labels,
    list(le.classes_),
    title="Confusion Matrix (Cross-Validation)",
    out_path="confusion_matrix_cv.png",
)

print("Saved plots: 'confusion_matrix_cv.png'")

print("\nTraining final model on full training set...")
final_model = build_logreg_pipeline(random_state=42)
final_model.fit(X_train_full, y_train_full)

# Evaluate final model on held-out test set
y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTest set accuracy: {test_acc:.4f}")
y_test_labels = le.inverse_transform(y_test)
y_test_pred_labels = le.inverse_transform(y_test_pred)
print("Test set classification report:")
print(
    classification_report(
        y_test_labels, y_test_pred_labels, target_names=list(le.classes_), digits=4
    )
)

_save_confusion_heatmap(
    y_test_labels,
    y_test_pred_labels,
    list(le.classes_),
    title="Confusion Matrix (Test Set)",
    out_path="confusion_matrix.pdf",
)
print("Saved test set confusion matrix: 'confusion_matrix.pdf'")

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
        def _prettify_feature_name(feature_name: str) -> str:
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

        coef_df = (
            pd.DataFrame({"feature": feature_cols, "coefficient": coef[0]})
            .assign(
                abs_coefficient=lambda df: df["coefficient"].abs(),
                pretty_feature=lambda df: df["feature"].map(_prettify_feature_name),
            )
            .sort_values("abs_coefficient", ascending=False)
        )
        coef_df.insert(2, "intercept", intercept[0])
        _append_taxonomy_summaries_to_coef_df(coef_df)
        coef_df.to_csv("logistic_coefficients.csv", index=False)
        print("Saved logistic coefficients: 'logistic_coefficients.csv'")

        # Permutation importance on held-out test set
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
            .assign(pretty_feature=lambda df: df["feature"].map(_prettify_feature_name))
            .sort_values("importance_mean", ascending=False)
        )
        perm_out = perm_df[
            ["feature", "coefficient", "importance_mean", "importance_std", "pretty_feature"]
        ]
        perm_out.to_csv("logistic_permutation_importance.csv", index=False)

        topk = perm_out.head(20).iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(topk["pretty_feature"], topk["importance_mean"],
                xerr=topk["importance_std"], color="#4c72b0", edgecolor="white")
        ax.set_xlabel("Mean accuracy decrease", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        plt.tight_layout(pad=0.5)
        plt.savefig("logistic_permutation_importance_bar.pdf", dpi=150)
        plt.close()
        print("Saved permutation importance: 'logistic_permutation_importance.csv', 'logistic_permutation_importance_bar.pdf'")
        print("\nTop 20 features by permutation importance:")
        print(
            perm_out.head(20)[
                ["feature", "coefficient", "importance_mean", "importance_std"]
            ].to_string(index=False)
        )
    else:
        classes = le.inverse_transform(np.arange(coef.shape[0]))
        multi_df = pd.DataFrame(coef.T, index=feature_cols, columns=classes)
        multi_df.loc["intercept"] = intercept
        multi_df.to_csv("logistic_coefficients.csv")
        print("Saved multi-class logistic coefficients: 'logistic_coefficients.csv'")
