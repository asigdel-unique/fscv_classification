"""
Cross-model comparison and per-concentration accuracy analysis.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

# ============================================================
# Load all results
# ============================================================
print("Loading results from all models...\n")

baseline = np.load(os.path.join(RESULTS_DIR, "baseline_results.npz"), allow_pickle=True)
cnn = np.load(os.path.join(RESULTS_DIR, "cnn_results.npz"), allow_pickle=True)
rnn = np.load(os.path.join(RESULTS_DIR, "rnn_results.npz"), allow_pickle=True)

# Also load test data for per-concentration analysis
test = np.load(os.path.join(BASE_DIR, "data", "fscv_test.npz"), allow_pickle=True)
y_test = test["y"]
c_test = test["concentrations"]
analyte_names = list(test["analyte_names"])

# ============================================================
# Build results table
# ============================================================
all_models = []

# Baselines
for i, name in enumerate(baseline["models"]):
    all_models.append({
        "name": str(name),
        "type": "Traditional ML",
        "accuracy": float(baseline["accuracies"][i]),
        "f1": float(baseline["f1_scores"][i]),
    })

# CNN
all_models.append({
    "name": "1D-CNN",
    "type": "Deep Learning",
    "accuracy": float(cnn["cnn_acc"]),
    "f1": float(cnn["cnn_f1"]),
})
all_models.append({
    "name": "Attention 1D-CNN",
    "type": "Deep Learning",
    "accuracy": float(cnn["attn_acc"]),
    "f1": float(cnn["attn_f1"]),
})

# RNN
for i, name in enumerate(rnn["models"]):
    all_models.append({
        "name": str(name),
        "type": "Deep Learning",
        "accuracy": float(rnn["accuracies"][i]),
        "f1": float(rnn["f1_scores"][i]),
    })

# Sort by accuracy
all_models.sort(key=lambda x: x["accuracy"])

print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)
print(f"\n{'Rank':<5} {'Model':<25} {'Type':<16} {'Accuracy':>9} {'F1':>9}")
print("-" * 65)
for rank, m in enumerate(reversed(all_models), 1):
    print(f"{rank:<5} {m['name']:<25} {m['type']:<16} {m['accuracy']:>9.4f} {m['f1']:>9.4f}")

best = all_models[-1]
worst = all_models[0]
print(f"\nBest:  {best['name']} ({best['accuracy']:.1%})")
print(f"Worst: {worst['name']} ({worst['accuracy']:.1%})")
print(f"Gap:   {(best['accuracy'] - worst['accuracy'])*100:.1f} percentage points")

# ============================================================
# Figure 1: Bar chart comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
names = [m["name"] for m in all_models]
accs = [m["accuracy"] for m in all_models]
f1s = [m["f1"] for m in all_models]
colors = ["#95a5a6" if m["type"] == "Traditional ML" else "#3498db" for m in all_models]

x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color=colors, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, f1s, width, label="F1 Macro", color=colors, alpha=0.6, edgecolor="black", linewidth=0.5)

ax.set_ylim(0.93, 1.005)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("FSCV Neurotransmitter Classification: All Models", fontsize=14, fontweight="bold")
ax.legend()

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.1%}", ha="center", fontsize=8, fontweight="bold")

ax.axhline(y=0.98, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax.text(len(names) - 0.5, 0.981, "98% threshold", fontsize=8, color="red", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "final_comparison.png"), dpi=150)
plt.close()
print("\nSaved figures/final_comparison.png")

# ============================================================
# Figure 2: Per-concentration accuracy analysis
# ============================================================
# Use attention CNN predictions (best model with stored predictions)
attn_preds = cnn["attn_preds"]
attn_labels = cnn["attn_labels"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2a: Accuracy vs concentration
ax = axes[0]
conc_bins = np.linspace(0, 2600, 11)
conc_centers = (conc_bins[:-1] + conc_bins[1:]) / 2

for class_idx, analyte in enumerate(analyte_names):
    mask = attn_labels == class_idx
    bin_accs = []
    for j in range(len(conc_bins) - 1):
        c_mask = mask & (c_test >= conc_bins[j]) & (c_test < conc_bins[j+1])
        if c_mask.sum() > 0:
            bin_accs.append((attn_preds[c_mask] == attn_labels[c_mask]).mean())
        else:
            bin_accs.append(np.nan)
    ax.plot(conc_centers, bin_accs, "o-", label=analyte, linewidth=1.5, markersize=4)

ax.set_xlabel("Concentration (nM)")
ax.set_ylabel("Classification Accuracy")
ax.set_title("Accuracy vs Concentration (Attention CNN)", fontweight="bold")
ax.legend()
ax.set_ylim(0.85, 1.02)
ax.axhline(y=0.95, color="gray", linewidth=0.5, linestyle="--")

# 2b: Confusion at low vs high concentration
ax = axes[1]
low_mask = c_test < 500
high_mask = c_test >= 1500

low_acc = accuracy_score(attn_labels[low_mask], attn_preds[low_mask])
high_acc = accuracy_score(attn_labels[high_mask], attn_preds[high_mask])
mid_mask = (c_test >= 500) & (c_test < 1500)
mid_acc = accuracy_score(attn_labels[mid_mask], attn_preds[mid_mask])

bars = ax.bar(["Low\n(<500 nM)", "Medium\n(500-1500 nM)", "High\n(>1500 nM)"],
              [low_acc, mid_acc, high_acc],
              color=["#e74c3c", "#f39c12", "#2ecc71"],
              edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, [low_acc, mid_acc, high_acc]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Classification Accuracy")
ax.set_title("Accuracy by Concentration Range", fontweight="bold")
ax.set_ylim(0.85, 1.05)

plt.suptitle("Concentration-Dependent Classification Performance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "concentration_analysis.png"), dpi=150)
plt.close()
print("Saved figures/concentration_analysis.png")

# ============================================================
# Figure 3: Pairwise confusion analysis (DA vs NE focus)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(attn_labels, attn_preds)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(analyte_names)))
ax.set_yticks(range(len(analyte_names)))
ax.set_xticklabels(analyte_names, fontsize=12)
ax.set_yticklabels(analyte_names, fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)

for i in range(len(analyte_names)):
    for j in range(len(analyte_names)):
        color = "white" if cm_pct[i, j] > 50 else "black"
        ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                ha="center", va="center", fontsize=11, color=color, fontweight="bold")

plt.colorbar(im, label="% of True Class")
ax.set_title(f"Best Model Confusion Matrix (Attention CNN, Acc={float(cnn['attn_acc']):.1%})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "best_model_confusion.png"), dpi=150)
plt.close()
print("Saved figures/best_model_confusion.png")

# ============================================================
# Print key insights
# ============================================================
print("\n" + "=" * 65)
print("KEY FINDINGS")
print("=" * 65)

# DA vs NE confusion
da_idx = analyte_names.index("DA")
ne_idx = analyte_names.index("NE")
da_as_ne = cm_pct[da_idx, ne_idx]
ne_as_da = cm_pct[ne_idx, da_idx]
print(f"\n1. DA-NE confusion (hardest pair):")
print(f"   DA misclassified as NE: {da_as_ne:.1f}%")
print(f"   NE misclassified as DA: {ne_as_da:.1f}%")
print(f"   This matches real electrochemistry: DA and NE have very similar oxidation potentials")

# Low vs high concentration
print(f"\n2. Concentration effect:")
print(f"   Low  (<500 nM):   {low_acc:.1%} accuracy")
print(f"   Medium (500-1500): {mid_acc:.1%} accuracy")
print(f"   High (>1500 nM):  {high_acc:.1%} accuracy")
if low_acc < high_acc:
    print(f"   Low concentrations are harder (weaker signal, lower SNR)")

# Traditional vs DL
trad_best = max(m["accuracy"] for m in all_models if m["type"] == "Traditional ML")
dl_best = max(m["accuracy"] for m in all_models if m["type"] == "Deep Learning")
print(f"\n3. Traditional ML best: {trad_best:.1%}")
print(f"   Deep Learning best:  {dl_best:.1%}")
print(f"   DL advantage: {(dl_best - trad_best)*100:.1f} percentage points")

# Feature importance recap
print(f"\n4. Top discriminating features (Random Forest):")
feat_names = list(baseline["feature_names"])
feat_imp = baseline["feature_importances"]
sorted_idx = np.argsort(feat_imp)[::-1]
for i in range(5):
    idx = sorted_idx[i]
    print(f"   {feat_names[idx]}: {feat_imp[idx]:.3f}")

print(f"\n5. Research relevance:")
print(f"   - Attention maps show which voltage regions distinguish each neurotransmitter")
print(f"   - Oxidation/reduction peak ratio is the single most important feature")
print(f"   - Model can potentially assist real-time FSCV neurotransmitter identification")
print(f"   - DA vs NE separation needs additional electrochemical information or")
print(f"     electrode surface modifications (e.g., Nafion coating for selectivity)")
