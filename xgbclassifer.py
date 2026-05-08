import os
import numpy as np
import pandas as pd
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import xgboost as xgb

from feature_selection import numeric_model_feature_columns, prepare_numeric_feature_matrix

FEATURES_CSV = "essen_china_europe_features.csv"

if not os.path.isfile(FEATURES_CSV):
    raise RuntimeError(
        f"{FEATURES_CSV} not found. Run logistic.py first to generate features."
    )

print(f"Loading cached features from {FEATURES_CSV} ...")
features_df = pd.read_csv(FEATURES_CSV)

print(f"Dataset size: {len(features_df)}")
print(features_df["continent"].value_counts())

# Same feature columns as logistic.py / factor_logistic.R / comparison
feature_cols = numeric_model_feature_columns(features_df)
X, feature_cols = prepare_numeric_feature_matrix(features_df, feature_cols)
print(f"Numeric features used for modeling: {len(feature_cols)}")

y = features_df["continent"].astype(str)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Same split as logistic.py
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train set size: {len(X_train_full)}, Test set size: {len(X_test)}")


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


# 5-fold stratified CV on training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

# CV confusion matrix
cm_cv = confusion_matrix(all_true_labels, all_pred_labels, labels=list(le.classes_))
cm_cv_pct = cm_cv.astype(float) / cm_cv.sum(axis=1)[:, np.newaxis] * 100
annot_cv = np.empty_like(cm_cv, dtype=object)
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        annot_cv[i, j] = f"{cm_cv[i, j]}\n({cm_cv_pct[i, j]:.1f}%)"

plt.figure(figsize=(5, 4))
sns.heatmap(cm_cv, annot=annot_cv, fmt="", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix (Cross-Validation, XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix_cv.png", dpi=150)
plt.close()

# Final model on full training set
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

# Test set confusion matrix
cm_test = confusion_matrix(y_test_labels, y_test_pred_labels, labels=list(le.classes_))
cm_test_pct = cm_test.astype(float) / cm_test.sum(axis=1)[:, np.newaxis] * 100
annot_test = np.empty_like(cm_test, dtype=object)
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        annot_test[i, j] = f"{cm_test[i, j]}\n({cm_test_pct[i, j]:.1f}%)"

plt.figure(figsize=(5, 4))
sns.heatmap(cm_test, annot=annot_test, fmt="", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix_test.pdf", dpi=150)
plt.close()

print("Saved confusion matrices: 'xgb_confusion_matrix_cv.png', 'xgb_confusion_matrix_test.pdf'")

# Permutation importance on held-out test set
print("\nComputing permutation importance on test set (n_repeats=10)...")
perm = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring="accuracy", n_jobs=-1,
)
# signed mean SHAP-style contributions on test set (direction toward positive encoded class)
# positive class is europe here, so positive values push the classifier toward europe
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
perm_out = perm_out[["feature", "coefficient", "importance_mean", "importance_std"]]
perm_out.to_csv("xgb_permutation_importance.csv", index=False)

topk = perm_out.head(20).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(topk["feature"], topk["importance_mean"],
        xerr=topk["importance_std"], color="#4c72b0", edgecolor="white")
ax.set_xlabel("Mean accuracy decrease", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
plt.tight_layout()
plt.savefig("xgb_permutation_importance_bar.pdf", dpi=150)
plt.close()

print("Saved permutation importance: 'xgb_permutation_importance.csv', 'xgb_permutation_importance_bar.pdf'")
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
