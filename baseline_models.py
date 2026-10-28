"""
Baseline classifiers for FSCV neurotransmitter identification.
  1. Random Forest on hand-crafted features
  2. SVM + PCA on hand-crafted features
  3. Random Forest on raw CV (flattened)
  4. SVM + PCA on raw CV (flattened)

Establishes performance baselines before deep learning.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add project to path for imports
sys.path.insert(0, BASE_DIR)
from feature_extraction import extract_all

# ============================================================
# Load data
# ============================================================
print("Loading data...")
raw_train = np.load(os.path.join(BASE_DIR, "data", "fscv_train.npz"), allow_pickle=True)
raw_test = np.load(os.path.join(BASE_DIR, "data", "fscv_test.npz"), allow_pickle=True)

X_train_raw = raw_train["X"]
y_train = raw_train["y"]
X_test_raw = raw_test["X"]
y_test = raw_test["y"]
voltage = raw_train["voltage"]
analyte_names = list(raw_train["analyte_names"])

print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
print(f"  Classes: {analyte_names}")

# ============================================================
# Extract features
# ============================================================
print("\nExtracting hand-crafted features...")
F_train, feature_names = extract_all(X_train_raw, voltage)
F_test, _ = extract_all(X_test_raw, voltage)
print(f"  {len(feature_names)} features per CV")

# ============================================================
# Model 1: Random Forest on features
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: Random Forest (hand-crafted features)")
print("=" * 60)

rf_feat = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_feat.fit(F_train, y_train)
rf_feat_pred = rf_feat.predict(F_test)
rf_feat_acc = accuracy_score(y_test, rf_feat_pred)
rf_feat_f1 = f1_score(y_test, rf_feat_pred, average="macro")

print(f"  Accuracy: {rf_feat_acc:.4f}")
print(f"  F1 Macro: {rf_feat_f1:.4f}")
print("\n" + classification_report(y_test, rf_feat_pred, target_names=analyte_names))

# Feature importance
importances = rf_feat.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("Top features:")
for i in range(min(8, len(feature_names))):
    idx = sorted_idx[i]
    print(f"  {feature_names[idx]:<20s} importance={importances[idx]:.4f}")

# ============================================================
# Model 2: SVM + PCA on features
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: SVM + PCA (hand-crafted features)")
print("=" * 60)

svm_feat = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))
])
svm_feat.fit(F_train, y_train)
svm_feat_pred = svm_feat.predict(F_test)
svm_feat_acc = accuracy_score(y_test, svm_feat_pred)
svm_feat_f1 = f1_score(y_test, svm_feat_pred, average="macro")

print(f"  Accuracy: {svm_feat_acc:.4f}")
print(f"  F1 Macro: {svm_feat_f1:.4f}")
print("\n" + classification_report(y_test, svm_feat_pred, target_names=analyte_names))

# PCA variance
pca = svm_feat.named_steps["pca"]
print(f"PCA: {pca.n_components_} components explain {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

# ============================================================
# Model 3: Random Forest on raw CV (flattened)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Random Forest (raw CV, 1000 features)")
print("=" * 60)

rf_raw = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_raw.fit(X_train_raw, y_train)
rf_raw_pred = rf_raw.predict(X_test_raw)
rf_raw_acc = accuracy_score(y_test, rf_raw_pred)
rf_raw_f1 = f1_score(y_test, rf_raw_pred, average="macro")

print(f"  Accuracy: {rf_raw_acc:.4f}")
print(f"  F1 Macro: {rf_raw_f1:.4f}")
print("\n" + classification_report(y_test, rf_raw_pred, target_names=analyte_names))

# ============================================================
# Model 4: SVM + PCA on raw CV
# ============================================================
print("\n" + "=" * 60)
print("MODEL 4: SVM + PCA (raw CV, 1000 -> 50 components)")
print("=" * 60)

svm_raw = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))
])
svm_raw.fit(X_train_raw, y_train)
svm_raw_pred = svm_raw.predict(X_test_raw)
svm_raw_acc = accuracy_score(y_test, svm_raw_pred)
svm_raw_f1 = f1_score(y_test, svm_raw_pred, average="macro")

print(f"  Accuracy: {svm_raw_acc:.4f}")
print(f"  F1 Macro: {svm_raw_f1:.4f}")
print("\n" + classification_report(y_test, svm_raw_pred, target_names=analyte_names))

pca_raw = svm_raw.named_steps["pca"]
print(f"PCA: {pca_raw.n_components_} components explain {pca_raw.explained_variance_ratio_.sum()*100:.1f}% variance")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)

results = [
    ("RF (features)", rf_feat_acc, rf_feat_f1),
    ("SVM+PCA (features)", svm_feat_acc, svm_feat_f1),
    ("RF (raw CV)", rf_raw_acc, rf_raw_f1),
    ("SVM+PCA (raw CV)", svm_raw_acc, svm_raw_f1),
]

print(f"\n{'Model':<25s} {'Accuracy':>10s} {'F1 Macro':>10s}")
print("-" * 47)
for name, acc, f1 in results:
    print(f"{name:<25s} {acc:>10.4f} {f1:>10.4f}")

# ============================================================
# Confusion matrix figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
models = [
    ("RF (features)", rf_feat_pred),
    ("SVM+PCA (features)", svm_feat_pred),
    ("RF (raw CV)", rf_raw_pred),
    ("SVM+PCA (raw CV)", svm_raw_pred),
]

for ax, (name, pred) in zip(axes.flat, models):
    cm = confusion_matrix(y_test, pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(analyte_names)))
    ax.set_yticks(range(len(analyte_names)))
    ax.set_xticklabels(analyte_names)
    ax.set_yticklabels(analyte_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    acc = accuracy_score(y_test, pred)
    ax.set_title(f"{name}\nAccuracy: {acc:.1%}", fontweight="bold")

    for i in range(len(analyte_names)):
        for j in range(len(analyte_names)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                    ha="center", va="center", fontsize=9, color=color)

plt.suptitle("Baseline Model Confusion Matrices", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "baseline_confusion.png"), dpi=150)
plt.close()
print(f"\nSaved figures/baseline_confusion.png")

# Feature importance figure
fig, ax = plt.subplots(figsize=(10, 6))
top_n = min(14, len(feature_names))
top_idx = sorted_idx[:top_n]
bars = ax.barh(range(top_n), importances[top_idx], color="#3498db", edgecolor="black", linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in top_idx])
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved figures/feature_importance.png")

# Save results
np.savez(os.path.join(RESULTS_DIR, "baseline_results.npz"),
         models=["RF_feat", "SVM_feat", "RF_raw", "SVM_raw"],
         accuracies=[rf_feat_acc, svm_feat_acc, rf_raw_acc, svm_raw_acc],
         f1_scores=[rf_feat_f1, svm_feat_f1, rf_raw_f1, svm_raw_f1],
         feature_names=feature_names,
         feature_importances=importances)
print("Saved results/baseline_results.npz")
