"""
1D-CNN and Attention 1D-CNN for FSCV neurotransmitter classification.
Attention variant produces voltage-region importance maps.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import os, gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print("Loading data...")
train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
test = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)

X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]
voltage = train["voltage"]
analyte_names = list(train["analyte_names"])

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape for Conv1D: (batch, channels=1, length=1000)
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
y_test_t = torch.LongTensor(y_test)

# Val split
val_n = int(len(y_train) * 0.15)
idx = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(42))
val_idx, train_idx = idx[:val_n], idx[val_n:]

train_ds = TensorDataset(X_train_t[train_idx], y_train_t[train_idx])
val_ds = TensorDataset(X_train_t[val_idx], y_train_t[val_idx])
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

n_classes = len(analyte_names)
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"  Input shape: {X_train_t.shape[1:]} -> {n_classes} classes")


# ============================================================
# Model architectures
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AttentionCNN1D(nn.Module):
    """1D-CNN with attention over voltage positions for interpretability."""
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # After 2x MaxPool(4): 1000 -> 250 -> 62, then conv keeps 62
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attention=False):
        features = self.conv(x)                        # (B, 128, 62)
        attn_scores = self.attention(features)         # (B, 1, 62)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, 62)
        context = (features * attn_weights).sum(dim=-1)    # (B, 128)
        logits = self.classifier(context)
        if return_attention:
            return logits, attn_weights.squeeze(1)     # (B, 62)
        return logits


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, val_loader, name, n_epochs=80, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch = 0
    wait = 0
    save_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")

    for epoch in range(n_epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)

        train_acc = correct / total
        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or wait == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f}"
                  f"{' *best*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"  Early stop at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def evaluate(model, loader, is_attention=False):
    model.eval()
    all_preds, all_labels = [], []
    all_attn = [] if is_attention else None

    with torch.no_grad():
        for xb, yb in loader:
            if is_attention:
                logits, attn = model(xb, return_attention=True)
                all_attn.append(attn.numpy())
            else:
                logits = model(xb)
            all_preds.append(logits.argmax(1).numpy())
            all_labels.append(yb.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    attn = np.concatenate(all_attn) if all_attn else None
    return preds, labels, attn


# ============================================================
# Train 1D-CNN
# ============================================================
print("\n" + "=" * 60)
print("MODEL: 1D-CNN")
print("=" * 60)

cnn = CNN1D(n_classes)
n_params = sum(p.numel() for p in cnn.parameters())
print(f"  Parameters: {n_params:,}")
cnn = train_model(cnn, train_loader, val_loader, "cnn1d")
cnn_preds, cnn_labels, _ = evaluate(cnn, test_loader)
cnn_acc = accuracy_score(cnn_labels, cnn_preds)
cnn_f1 = f1_score(cnn_labels, cnn_preds, average="macro")
print(f"\n  Test Accuracy: {cnn_acc:.4f}")
print(f"  Test F1 Macro: {cnn_f1:.4f}")
print("\n" + classification_report(cnn_labels, cnn_preds, target_names=analyte_names))

del cnn; gc.collect()

# ============================================================
# Train Attention-CNN
# ============================================================
print("\n" + "=" * 60)
print("MODEL: Attention 1D-CNN")
print("=" * 60)

attn_cnn = AttentionCNN1D(n_classes)
n_params = sum(p.numel() for p in attn_cnn.parameters())
print(f"  Parameters: {n_params:,}")
attn_cnn = train_model(attn_cnn, train_loader, val_loader, "attn_cnn1d")
attn_preds, attn_labels, attn_weights = evaluate(attn_cnn, test_loader, is_attention=True)
attn_acc = accuracy_score(attn_labels, attn_preds)
attn_f1 = f1_score(attn_labels, attn_preds, average="macro")
print(f"\n  Test Accuracy: {attn_acc:.4f}")
print(f"  Test F1 Macro: {attn_f1:.4f}")
print("\n" + classification_report(attn_labels, attn_preds, target_names=analyte_names))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("CNN RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25s} {'Accuracy':>10s} {'F1 Macro':>10s}")
print("-" * 47)
print(f"{'1D-CNN':<25s} {cnn_acc:>10.4f} {cnn_f1:>10.4f}")
print(f"{'Attention 1D-CNN':<25s} {attn_acc:>10.4f} {attn_f1:>10.4f}")

# ============================================================
# Attention visualization
# ============================================================
print("\nGenerating attention visualizations...")

# Map attention from feature space (62 points) back to voltage space (1000 points)
# After 2x MaxPool(4): 1000 -> 250 -> 62
# Each attention point covers ~16 voltage points
attn_len = attn_weights.shape[1]
v_centers = np.linspace(voltage[0], voltage[len(voltage)//2 - 1], attn_len)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"DA": "#e74c3c", "5HT": "#3498db", "NE": "#2ecc71", "pH": "#9b59b6"}

for idx, (ax, analyte) in enumerate(zip(axes.flat, analyte_names)):
    mask = (attn_labels == idx) & (attn_preds == idx)
    if mask.sum() == 0:
        continue
    mean_attn = attn_weights[mask].mean(axis=0)

    ax.bar(v_centers, mean_attn, width=(v_centers[1] - v_centers[0]) * 0.9,
           color=colors[analyte], alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"{analyte} - Voltage Region Importance", fontweight="bold",
                 color=colors[analyte])

    # Mark key electrochemical potentials
    for v_mark, label in [(0.6, "DA ox"), (0.65, "5HT/NE ox"),
                          (-0.2, "DA red"), (-0.1, "5HT red"), (-0.25, "NE red"),
                          (0.35, "5HT 2nd")]:
        if v_centers[0] <= v_mark <= v_centers[-1]:
            ax.axvline(v_mark, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

plt.suptitle("Attention Maps: Which Voltage Regions Distinguish Each Neurotransmitter",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "attention_maps.png"), dpi=150)
plt.close()
print("Saved figures/attention_maps.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, preds, labels) in zip(axes, [("1D-CNN", cnn_preds, cnn_labels),
                                              ("Attention 1D-CNN", attn_preds, attn_labels)]):
    cm = confusion_matrix(labels, preds)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(analyte_names)
    ax.set_yticklabels(analyte_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = accuracy_score(labels, preds)
    ax.set_title(f"{name}\nAccuracy: {acc:.1%}", fontweight="bold")
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%", ha="center", va="center",
                    fontsize=10, color=color)

plt.suptitle("CNN Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "cnn_confusion.png"), dpi=150)
plt.close()
print("Saved figures/cnn_confusion.png")

# Save results
np.savez(os.path.join(RESULTS_DIR, "cnn_results.npz"),
         cnn_acc=cnn_acc, cnn_f1=cnn_f1,
         attn_acc=attn_acc, attn_f1=attn_f1,
         attn_weights=attn_weights, attn_labels=attn_labels, attn_preds=attn_preds,
         analyte_names=analyte_names)
print("Saved results/cnn_results.npz")
