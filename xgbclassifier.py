"""Archived XGBoost China/Europe classifier.

Kept for reference only: not part of ``run_analysis.py`` or
``outputs/report.html`` anymore. Run this script manually if you want the old
outputs (written as ``supp_xgb_*`` figures, not numbered paper figs).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from helpers import dataset
from helpers import output_paths as OP
from helpers.plotting import confusion_heatmap, prettify_feature_name, signed_permutation_importance_bar

features_df = dataset.load_features_df()
print(f"Dataset size: {len(features_df)}")
print(features_df["continent"].value_counts())

X, feature_cols, y_enc, le = dataset.prepare_xy(features_df)
X_train_full, X_test, y_train_full, y_test = dataset.make_train_test_split(X, y_enc)


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        verbosity=0,
    )


skf = dataset.make_cv()

fold_metrics: List[Dict[str, float]] = []
all_true_labels: List[str] = []
all_pred_labels: List[str] = []

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
    X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
    y_train, y_valid = y_train_full[train_idx], y_train_full[valid_idx]

    clf = build_xgb()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    fold_acc = accuracy_score(y_valid, y_pred)
    print(f"Fold {fold_idx} accuracy: {fold_acc:.4f}")
    fold_metrics.append({"fold": fold_idx, "accuracy": fold_acc})

    all_true_labels.extend(le.inverse_transform(y_valid))
    all_pred_labels.extend(le.inverse_transform(y_pred))

metrics_df = pd.DataFrame(fold_metrics)
print("\nCross-validation accuracy by fold:")
print(metrics_df.to_string(index=False))
print(
    f"Mean CV accuracy: {metrics_df['accuracy'].mean():.4f} "
    f"± {metrics_df['accuracy'].std(ddof=0):.4f}"
)

overall_acc = accuracy_score(all_true_labels, all_pred_labels)
print(f"Overall cross-validated accuracy: {overall_acc:.4f}")
print(classification_report(all_true_labels, all_pred_labels, target_names=list(le.classes_), digits=4))

cv_cm_path = OP.supp_fig_path("xgb_confusion_matrix_cv", ext="png")
confusion_heatmap(
    all_true_labels,
    all_pred_labels,
    list(le.classes_),
    cv_cm_path,
    save_png_twin=False,
)

print("\nTraining final model on full training set...")
final_model = build_xgb()
final_model.fit(X_train_full, y_train_full)

y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTest set accuracy: {test_acc:.4f}")
y_test_labels = le.inverse_transform(y_test)
y_test_pred_labels = le.inverse_transform(y_test_pred)
print("Test set classification report:")
print(classification_report(y_test_labels, y_test_pred_labels, target_names=list(le.classes_), digits=4))

fig_cm_path = OP.supp_fig_path("xgb_confusion_matrix")
confusion_heatmap(
    y_test_labels,
    y_test_pred_labels,
    list(le.classes_),
    fig_cm_path,
    save_png_twin=True,
)

print(f"Saved confusion matrices: '{cv_cm_path}', '{fig_cm_path}' (archive)")

xgb_metrics_path = OP.data_path("xgb_metrics.csv")
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
).to_csv(xgb_metrics_path, index=False)
print(f"Saved summary metrics: '{xgb_metrics_path}'")

print("\nComputing permutation importance on test set (n_repeats=10)...")
perm = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring="accuracy", n_jobs=-1,
)
_positive = le.classes_[1]
booster = final_model.get_booster()
dm_test = xgb.DMatrix(X_test, feature_names=feature_cols)
contribs = booster.predict(dm_test, pred_contribs=True)
mean_shap = contribs[:, :-1].mean(axis=0)
shap_df = pd.DataFrame({"feature": feature_cols, "coefficient": mean_shap})

perm_out = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })
    .merge(shap_df, on="feature", how="left")
    .sort_values("importance_mean", ascending=False)
)
perm_out = (
    perm_out.assign(
        pretty_feature=lambda df: df["feature"].map(prettify_feature_name)
    )[
        ["feature", "coefficient", "importance_mean", "importance_std", "pretty_feature"]
    ]
)
perm_csv_path = OP.data_path("xgb_permutation_importance.csv")
perm_out.to_csv(perm_csv_path, index=False)

topk = perm_out.head(20).iloc[::-1].reset_index(drop=True)
pos_class, neg_class = le.classes_[1], le.classes_[0]
fig_imp_path = OP.supp_fig_path("xgb_permutation_importance")
signed_permutation_importance_bar(
    pretty_names=topk["pretty_feature"],
    importance_mean=topk["importance_mean"],
    importance_std=topk["importance_std"],
    coefficients=topk["coefficient"],
    pdf_path=fig_imp_path,
    pos_class=pos_class,
    neg_class=neg_class,
    xlabel="Mean accuracy decrease\n(sign from mean SHAP contribution)",
    pos_legend=f"SHAP > 0 (higher → {pos_class})",
    neg_legend=f"SHAP < 0 (higher → {neg_class})",
)

print(f"Saved permutation importance (archive): '{perm_csv_path}', '{fig_imp_path}'")
print(
    f"Note: coefficient = mean SHAP contribution on test set; "
    f"positive values push toward encoded class 1 ({_positive}).\n"
)
print("\nTop 20 features by permutation importance:")
print(
    perm_out.head(20)[
        ["feature", "coefficient", "importance_mean", "importance_std"]
    ].to_string(index=False)
)
