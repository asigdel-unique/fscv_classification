"""
Noise robustness analysis: test all models under increasing noise levels.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

sys.path.insert(0, BASE_DIR)
from feature_extraction import extract_all

# ============================================================
# Load data
# ============================================================
print("Loading data...")
train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
test = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)

X_train, y_train = train["X"], train["y"]
X_test_clean, y_test = test["X"], test["y"]
voltage = train["voltage"]
analyte_names = list(train["analyte_names"])
n_classes = len(analyte_names)

# Normalization stats
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

# Signal power for SNR reference
signal_std = X_test_clean.std()
print(f"  Clean signal std: {signal_std:.2f} nA")

# ============================================================
# Model definitions (match trained architectures)
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

class AttentionCNN1D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.attention = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1), nn.Tanh(), nn.Conv1d(64, 1, kernel_size=1))
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))
    def forward(self, x):
        f = self.conv(x)
        a = torch.softmax(self.attention(f), dim=-1)
        return self.classifier((f * a).sum(dim=-1))

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=2, n_classes=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])

# ============================================================
# Train baseline models fresh (needed for noisy test evaluation)
# ============================================================
print("\nTraining baseline models...")
F_train, feature_names = extract_all(X_train, voltage)

rf_feat = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_feat.fit(F_train, y_train)

svm_raw = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50)),
                     ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))])
svm_raw.fit(X_train, y_train)
print("  Baselines ready.")

# Load DL models
print("Loading deep learning models...")
cnn = CNN1D(n_classes)
cnn.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "cnn1d_best.pt"), weights_only=True))
cnn.eval()

attn_cnn = AttentionCNN1D(n_classes)
attn_cnn.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "attn_cnn1d_best.pt"), weights_only=True))
attn_cnn.eval()

bigru = BiGRUClassifier(input_dim=2, n_classes=n_classes)
bigru.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "bigru_best.pt"), weights_only=True))
bigru.eval()
print("  DL models loaded.")

# ============================================================
# Noise levels to test
# ============================================================
noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
np.random.seed(42)

models_to_test = {
    "RF (features)": "traditional",
    "SVM+PCA (raw)": "traditional",
    "1D-CNN": "cnn",
    "Attn 1D-CNN": "attn_cnn",
    "BiGRU": "bigru",
}

results = {name: {"acc": [], "f1": []} for name in models_to_test}

print(f"\nTesting {len(noise_multipliers)} noise levels...")
print(f"{'Noise':>6s}", end="")
for name in models_to_test:
    print(f"  {name:>15s}", end="")
print()
print("-" * (6 + 17 * len(models_to_test)))

for noise_mult in noise_multipliers:
    noise = np.random.normal(0, signal_std * noise_mult, X_test_clean.shape).astype(np.float32)
    X_noisy = X_test_clean + noise

    accs = {}

    # RF on features
    F_noisy, _ = extract_all(X_noisy, voltage)
    rf_pred = rf_feat.predict(F_noisy)
    accs["RF (features)"] = accuracy_score(y_test, rf_pred)
    results["RF (features)"]["acc"].append(accs["RF (features)"])
    results["RF (features)"]["f1"].append(f1_score(y_test, rf_pred, average="macro"))

    # SVM on raw
    svm_pred = svm_raw.predict(X_noisy)
    accs["SVM+PCA (raw)"] = accuracy_score(y_test, svm_pred)
    results["SVM+PCA (raw)"]["acc"].append(accs["SVM+PCA (raw)"])
    results["SVM+PCA (raw)"]["f1"].append(f1_score(y_test, svm_pred, average="macro"))

    # Normalize for DL
    X_norm = (X_noisy - mean) / std
    X_t = torch.FloatTensor(X_norm).unsqueeze(1)

    # 1D-CNN
    with torch.no_grad():
        cnn_pred = cnn(X_t).argmax(1).numpy()
    accs["1D-CNN"] = accuracy_score(y_test, cnn_pred)
    results["1D-CNN"]["acc"].append(accs["1D-CNN"])
    results["1D-CNN"]["f1"].append(f1_score(y_test, cnn_pred, average="macro"))

    # Attention CNN
    with torch.no_grad():
        attn_pred = attn_cnn(X_t).argmax(1).numpy()
    accs["Attn 1D-CNN"] = accuracy_score(y_test, attn_pred)
    results["Attn 1D-CNN"]["acc"].append(accs["Attn 1D-CNN"])
    results["Attn 1D-CNN"]["f1"].append(f1_score(y_test, attn_pred, average="macro"))

    # BiGRU (needs 2-channel downsampled input)
    DS = 5
    X_ds = X_norm.reshape(len(X_norm), -1, DS).mean(axis=-1)
    v_ds = voltage.reshape(-1, DS).mean(axis=-1)
    v_feat = np.tile(v_ds, (len(X_ds), 1))
    X_2ch = np.stack([X_ds, v_feat], axis=-1)
    X_gru_t = torch.FloatTensor(X_2ch)
    with torch.no_grad():
        gru_pred = bigru(X_gru_t).argmax(1).numpy()
    accs["BiGRU"] = accuracy_score(y_test, gru_pred)
    results["BiGRU"]["acc"].append(accs["BiGRU"])
    results["BiGRU"]["f1"].append(f1_score(y_test, gru_pred, average="macro"))

    print(f"{noise_mult:>5.1f}x", end="")
    for name in models_to_test:
        print(f"  {accs[name]:>14.1%}", end="")
    print()

# ============================================================
# Figure: Accuracy vs noise level
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {"RF (features)": "#95a5a6", "SVM+PCA (raw)": "#7f8c8d",
          "1D-CNN": "#3498db", "Attn 1D-CNN": "#e74c3c", "BiGRU": "#2ecc71"}
markers = {"RF (features)": "s", "SVM+PCA (raw)": "D",
           "1D-CNN": "o", "Attn 1D-CNN": "^", "BiGRU": "v"}

for name in models_to_test:
    ax1.plot(noise_multipliers, results[name]["acc"], f"-{markers[name]}",
             color=colors[name], linewidth=1.8, markersize=6, label=name)
    ax2.plot(noise_multipliers, results[name]["f1"], f"-{markers[name]}",
             color=colors[name], linewidth=1.8, markersize=6, label=name)

ax1.set_xlabel("Noise Level (× signal std)")
ax1.set_ylabel("Accuracy")
ax1.set_title("Classification Accuracy vs Noise", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, 1.05)
ax1.axhline(0.25, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.text(9, 0.27, "chance", fontsize=8, color="gray")

ax2.set_xlabel("Noise Level (× signal std)")
ax2.set_ylabel("F1 Macro")
ax2.set_title("F1 Score vs Noise", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 1.05)

plt.suptitle("Noise Robustness Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "noise_robustness.png"), dpi=150)
plt.close()
print("\nSaved figures/noise_robustness.png")

# ============================================================
# Figure: Example noisy CVs
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(16, 6))

example_idx = np.where(y_test == 0)[0][10]  # a DA sample
example_cv = X_test_clean[example_idx]

for col, noise_mult in enumerate([0.0, 1.0, 2.0, 5.0, 10.0]):
    noise = np.random.normal(0, signal_std * noise_mult, example_cv.shape)
    noisy_cv = example_cv + noise

    axes[0, col].plot(voltage, noisy_cv, color="#e74c3c", linewidth=0.6)
    axes[0, col].set_title(f"{noise_mult:.0f}× noise", fontweight="bold", fontsize=10)
    axes[0, col].set_ylim(example_cv.min() - 15, example_cv.max() + 15)
    axes[0, col].axhline(0, color="gray", linewidth=0.3)
    if col == 0:
        axes[0, col].set_ylabel("DA sample\nCurrent (nA)")

    # Show what model sees after normalization
    norm_cv = (noisy_cv - mean) / std
    axes[1, col].plot(voltage, norm_cv, color="#3498db", linewidth=0.6)
    axes[1, col].axhline(0, color="gray", linewidth=0.3)
    if col == 0:
        axes[1, col].set_ylabel("After z-score\nnormalization")
    axes[1, col].set_xlabel("V")

plt.suptitle("Effect of Noise on a Dopamine Voltammogram", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "noise_examples.png"), dpi=150)
plt.close()
print("Saved figures/noise_examples.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("NOISE ROBUSTNESS SUMMARY")
print("=" * 60)

for name in models_to_test:
    accs = results[name]["acc"]
    # Find noise level where accuracy drops below 90%
    thresh_90 = None
    for i, a in enumerate(accs):
        if a < 0.90:
            thresh_90 = noise_multipliers[i]
            break
    drop = accs[0] - accs[-1]
    print(f"\n  {name}:")
    print(f"    Clean: {accs[0]:.1%}  |  10x noise: {accs[-1]:.1%}  |  Drop: {drop*100:.1f}pp")
    if thresh_90:
        print(f"    Falls below 90% at {thresh_90:.1f}x noise")
    else:
        print(f"    Stays above 90% through all noise levels")

np.savez(os.path.join(RESULTS_DIR, "noise_robustness.npz"),
         noise_levels=noise_multipliers,
         model_names=list(models_to_test.keys()),
         accuracies={n: results[n]["acc"] for n in models_to_test},
         f1_scores={n: results[n]["f1"] for n in models_to_test})
print("\nSaved results/noise_robustness.npz")
