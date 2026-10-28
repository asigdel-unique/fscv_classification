"""
Recurrent models (LSTM, GRU, Bidirectional GRU) for FSCV classification.
Treats the CV as a sequence along the voltage sweep direction.
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
# Load and prepare data
# ============================================================
print("Loading data...")
train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
test = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)

X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]
voltage = train["voltage"]
analyte_names = list(train["analyte_names"])
n_classes = len(analyte_names)

# Normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape for RNN: (batch, seq_len, features)
# Downsample 1000 -> 200 points (5x) to reduce sequence length for RNN efficiency
# Each point covers 5 original voltage steps = ~8.5mV resolution (still very fine)
DS = 5
X_tr_ds = X_train.reshape(len(X_train), -1, DS).mean(axis=-1)
X_te_ds = X_test.reshape(len(X_test), -1, DS).mean(axis=-1)
seq_len = X_tr_ds.shape[1]  # 200

# Add voltage as second feature channel
v_ds = voltage.reshape(-1, DS).mean(axis=-1)
v_feat = np.tile(v_ds, (len(X_tr_ds), 1))
X_tr_2ch = np.stack([X_tr_ds, v_feat], axis=-1)  # (N, 200, 2)
v_feat_te = np.tile(v_ds, (len(X_te_ds), 1))
X_te_2ch = np.stack([X_te_ds, v_feat_te], axis=-1)

X_train_t = torch.FloatTensor(X_tr_2ch)
X_test_t = torch.FloatTensor(X_te_2ch)
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

n_features = X_train_t.shape[2]
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"  Sequence: {seq_len} steps x {n_features} features")


# ============================================================
# Models
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=2, n_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=2, n_classes=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=2, n_classes=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


# ============================================================
# Training loop
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


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            all_preds.append(logits.argmax(1).numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# Train all three models
# ============================================================
all_results = []

for ModelClass, name, label in [
    (LSTMClassifier, "lstm", "LSTM"),
    (GRUClassifier, "gru_cls", "GRU"),
    (BiGRUClassifier, "bigru", "Bidirectional GRU"),
]:
    print("\n" + "=" * 60)
    print(f"MODEL: {label}")
    print("=" * 60)

    model = ModelClass(input_dim=n_features, n_classes=n_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    model = train_model(model, train_loader, val_loader, name)
    preds, labels = evaluate(model, test_loader)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 Macro: {f1:.4f}")
    print("\n" + classification_report(labels, preds, target_names=analyte_names))

    all_results.append((label, acc, f1, preds, labels, n_params))
    del model; gc.collect()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("RECURRENT MODEL SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25s} {'Params':>8s} {'Accuracy':>10s} {'F1 Macro':>10s}")
print("-" * 55)
for name, acc, f1, _, _, params in all_results:
    print(f"{name:<25s} {params:>8,} {acc:>10.4f} {f1:>10.4f}")

# ============================================================
# Confusion matrices
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, acc, f1, preds, labels, _) in zip(axes, all_results):
    cm = confusion_matrix(labels, preds)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(analyte_names)
    ax.set_yticklabels(analyte_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}\nAcc: {acc:.1%}", fontweight="bold")
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%", ha="center", va="center",
                    fontsize=9, color=color)

plt.suptitle("Recurrent Model Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "rnn_confusion.png"), dpi=150)
plt.close()
print(f"\nSaved figures/rnn_confusion.png")

# Save
np.savez(os.path.join(RESULTS_DIR, "rnn_results.npz"),
         models=[r[0] for r in all_results],
         accuracies=[r[1] for r in all_results],
         f1_scores=[r[2] for r in all_results],
         params=[r[5] for r in all_results],
         analyte_names=analyte_names)
print("Saved results/rnn_results.npz")
