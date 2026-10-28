"""
Real FSCV data analysis — monkey T, session 09272024, site c8ds (Zenodo 14825992).
PCA-extracted dopamine, pH, movement, and oxidation current.

Runs DA release detection (binary) and concentration regression from iox.
"""
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "real_data")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============================================================
# Load real FSCV data
# ============================================================
print("Loading real monkey FSCV data (Zenodo 14825992)...")
mat = sio.loadmat(os.path.join(DATA_DIR, "monkey_fscv_dopamine.mat"))
data = mat["c8ds_fscv"][0]

n_trials = len(data)
n_timepoints = 601  # 10 Hz for ~60 seconds

# Extract all trials into arrays
da_traces = np.zeros((n_trials, n_timepoints))
ph_traces = np.zeros((n_trials, n_timepoints))
mov_traces = np.zeros((n_trials, n_timepoints))
iox_traces = np.zeros((n_trials, n_timepoints))

for i in range(n_trials):
    da_traces[i] = data[i]["da"].flatten()[:n_timepoints]
    ph_traces[i] = data[i]["ph"].flatten()[:n_timepoints]
    mov_traces[i] = data[i]["m"].flatten()[:n_timepoints]
    iox_traces[i] = data[i]["iox"].flatten()[:n_timepoints]

# Replace NaN with interpolation
for arr in [da_traces, ph_traces, mov_traces, iox_traces]:
    for i in range(n_trials):
        nans = np.isnan(arr[i])
        if nans.any():
            arr[i][nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], arr[i][~nans])

print(f"  Trials: {n_trials}")
print(f"  Time points per trial: {n_timepoints} (10 Hz, ~60s)")
print(f"  DA range: [{da_traces.min():.1f}, {da_traces.max():.1f}] nM")
print(f"  pH range: [{ph_traces.min():.3f}, {ph_traces.max():.3f}]")

# ============================================================
# Figure 1: Exploratory visualization
# ============================================================
print("\nGenerating exploratory figures...")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
t = np.arange(n_timepoints) / 10.0  # Convert to seconds

# Show 5 example trials
example_trials = [0, 100, 300, 500, 800]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

for trial_idx, color in zip(example_trials, colors):
    axes[0].plot(t, da_traces[trial_idx], color=color, linewidth=0.8, alpha=0.7,
                 label=f"Trial {trial_idx}")
    axes[1].plot(t, ph_traces[trial_idx], color=color, linewidth=0.8, alpha=0.7)
    axes[2].plot(t, mov_traces[trial_idx], color=color, linewidth=0.8, alpha=0.7)
    axes[3].plot(t, iox_traces[trial_idx], color=color, linewidth=0.8, alpha=0.7)

axes[0].set_ylabel("DA (nM)")
axes[0].set_title("Dopamine Concentration (PCA-extracted)", fontweight="bold")
axes[0].legend(fontsize=8)
axes[1].set_ylabel("pH shift")
axes[1].set_title("pH", fontweight="bold")
axes[2].set_ylabel("Movement")
axes[2].set_title("Movement Signal", fontweight="bold")
axes[3].set_ylabel("I_ox (nA)")
axes[3].set_xlabel("Time (s)")
axes[3].set_title("Oxidation Current at 0.6V (raw FSCV signal)", fontweight="bold")

plt.suptitle("Real FSCV Data: Monkey T, Session 09272024", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "real_data_exploration.png"), dpi=150)
plt.close()
print("Saved figures/real_data_exploration.png")

# ============================================================
# Task 1: DA release detection (binary classification)
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: Dopamine Release Detection")
print("=" * 60)

# Define DA release: peak DA > threshold in trial
peak_da = da_traces.max(axis=1)
threshold = np.percentile(peak_da, 60)
labels = (peak_da > threshold).astype(int)

print(f"  Threshold: {threshold:.2f} nM (60th percentile)")
print(f"  Release: {labels.sum()} trials, No release: {(1-labels).sum()} trials")

# Features: statistics from the iox trace per trial
def extract_trial_features(iox_trace):
    return np.array([
        iox_trace.mean(),
        iox_trace.std(),
        iox_trace.max(),
        iox_trace.min(),
        iox_trace.max() - iox_trace.min(),
        np.argmax(iox_trace) / len(iox_trace),  # time of peak
        iox_trace[:len(iox_trace)//2].mean(),    # first half mean
        iox_trace[len(iox_trace)//2:].mean(),    # second half mean
        np.abs(np.diff(iox_trace)).mean(),        # mean absolute derivative
        np.sum(iox_trace > 0) / len(iox_trace),  # fraction positive
    ])

X_features = np.array([extract_trial_features(iox_traces[i]) for i in range(n_trials)])

X_tr, X_te, y_tr, y_te = train_test_split(X_features, labels, test_size=0.2,
                                            random_state=42, stratify=labels)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr)
rf_pred = rf.predict(X_te_s)
rf_acc = accuracy_score(y_te, rf_pred)
rf_f1 = f1_score(y_te, rf_pred)
print(f"\n  Random Forest: Accuracy={rf_acc:.4f}, F1={rf_f1:.4f}")

# SVM
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_tr_s, y_tr)
svm_pred = svm.predict(X_te_s)
svm_acc = accuracy_score(y_te, svm_pred)
svm_f1 = f1_score(y_te, svm_pred)
print(f"  SVM:           Accuracy={svm_acc:.4f}, F1={svm_f1:.4f}")

print("\n" + classification_report(y_te, rf_pred, target_names=["No Release", "DA Release"]))

# ============================================================
# Task 2: DA concentration regression from iox
# ============================================================
print("=" * 60)
print("TASK 2: DA Concentration Regression")
print("=" * 60)

# Use sliding windows: predict DA at each timepoint from iox context
window = 30  # 3 seconds of context at 10 Hz

X_windows = []
y_da = []

for i in range(n_trials):
    for t_idx in range(window, n_timepoints):
        X_windows.append(iox_traces[i, t_idx-window:t_idx])
        y_da.append(da_traces[i, t_idx])

X_windows = np.array(X_windows, dtype=np.float32)
y_da = np.array(y_da, dtype=np.float32)

print(f"  Windows: {len(y_da)} (window={window} = {window/10:.1f}s)")
print(f"  DA target range: [{y_da.min():.1f}, {y_da.max():.1f}] nM")

# Subsample for tractability (full dataset is huge)
np.random.seed(42)
sub_idx = np.random.choice(len(y_da), size=min(50000, len(y_da)), replace=False)
X_sub = X_windows[sub_idx]
y_sub = y_da[sub_idx]

n_train = int(len(y_sub) * 0.8)
X_tr_w, y_tr_w = X_sub[:n_train], y_sub[:n_train]
X_te_w, y_te_w = X_sub[n_train:], y_sub[n_train:]

# Normalize
w_scaler = StandardScaler()
X_tr_w_s = w_scaler.fit_transform(X_tr_w)
X_te_w_s = w_scaler.transform(X_te_w)

# Random Forest regression
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_w_s, y_tr_w)
rf_pred_da = rf_reg.predict(X_te_w_s)
rf_r2 = r2_score(y_te_w, rf_pred_da)
rf_rmse = np.sqrt(mean_squared_error(y_te_w, rf_pred_da))
print(f"\n  RF Regression:  R2={rf_r2:.4f}, RMSE={rf_rmse:.2f} nM")

# Simple 1D-CNN regression
print("  Training 1D-CNN regressor...")

class CNN1DRegressor(nn.Module):
    def __init__(self, window_size=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

X_tr_t = torch.FloatTensor(X_tr_w_s).unsqueeze(1)
y_tr_t = torch.FloatTensor(y_tr_w)
X_te_t = torch.FloatTensor(X_te_w_s).unsqueeze(1)
y_te_t = torch.FloatTensor(y_te_w)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

cnn_reg = CNN1DRegressor()
optimizer = torch.optim.Adam(cnn_reg.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_loss = float("inf")
for epoch in range(30):
    cnn_reg.train()
    for xb, yb in train_loader:
        pred = cnn_reg(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cnn_reg.eval()
    with torch.no_grad():
        val_pred = cnn_reg(X_te_t)
        val_loss = criterion(val_pred, y_te_t).item()
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.clone() for k, v in cnn_reg.state_dict().items()}
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}: val_loss={val_loss:.4f}")

cnn_reg.load_state_dict(best_state)
cnn_reg.eval()
with torch.no_grad():
    cnn_pred_da = cnn_reg(X_te_t).numpy()
cnn_r2 = r2_score(y_te_w, cnn_pred_da)
cnn_rmse = np.sqrt(mean_squared_error(y_te_w, cnn_pred_da))
print(f"  CNN Regression: R2={cnn_r2:.4f}, RMSE={cnn_rmse:.2f} nM")

# ============================================================
# Task 2 Figures
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: predicted vs actual
ax = axes[0]
ax.scatter(y_te_w[:2000], rf_pred_da[:2000], alpha=0.3, s=5, color="#3498db", label="RF")
ax.scatter(y_te_w[:2000], cnn_pred_da[:2000], alpha=0.3, s=5, color="#e74c3c", label="CNN")
lims = [y_te_w.min(), y_te_w.max()]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("Actual DA (nM)")
ax.set_ylabel("Predicted DA (nM)")
ax.set_title(f"DA Concentration Prediction\nRF R2={rf_r2:.3f}, CNN R2={cnn_r2:.3f}", fontweight="bold")
ax.legend()

# Time series: one trial prediction
ax = axes[1]
trial_idx = 300
trial_iox = iox_traces[trial_idx]
trial_da = da_traces[trial_idx]

# Predict DA for this trial
trial_windows = []
for t_idx in range(window, n_timepoints):
    trial_windows.append(trial_iox[t_idx-window:t_idx])
trial_windows = np.array(trial_windows, dtype=np.float32)
trial_windows_s = w_scaler.transform(trial_windows)
trial_pred = rf_reg.predict(trial_windows_s)

t_pred = np.arange(window, n_timepoints) / 10.0
t_full = np.arange(n_timepoints) / 10.0

ax.plot(t_full, trial_da, "k-", linewidth=1.5, label="Actual DA")
ax.plot(t_pred, trial_pred, "r-", linewidth=1, alpha=0.8, label=f"RF Predicted (R2={rf_r2:.3f})")
ax.set_xlabel("Time (s)")
ax.set_ylabel("DA (nM)")
ax.set_title(f"Trial {trial_idx}: DA Prediction from Oxidation Current", fontweight="bold")
ax.legend()

plt.suptitle("Real FSCV Data: Dopamine Concentration Regression", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "real_data_regression.png"), dpi=150)
plt.close()
print("\nSaved figures/real_data_regression.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("REAL DATA ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nDataset: Monkey T, {n_trials} trials, 60s each at 10 Hz")
print(f"Source: Zenodo record 14825992 (CC BY 4.0)")

print(f"\nTask 1 - DA Release Detection (binary):")
print(f"  Random Forest: Accuracy={rf_acc:.1%}, F1={rf_f1:.3f}")
print(f"  SVM:           Accuracy={svm_acc:.1%}, F1={svm_f1:.3f}")

print(f"\nTask 2 - DA Concentration Regression:")
print(f"  RF Regressor:  R2={rf_r2:.4f}, RMSE={rf_rmse:.2f} nM")
print(f"  1D-CNN:        R2={cnn_r2:.4f}, RMSE={cnn_rmse:.2f} nM")

print(f"\nNote: predictions use only the oxidation current at 0.6V,")
print(f"not full voltammograms. Full CV data would improve both tasks.")

# Save
np.savez(os.path.join(RESULTS_DIR, "real_data_results.npz"),
         rf_acc=rf_acc, rf_f1=rf_f1, svm_acc=svm_acc, svm_f1=svm_f1,
         rf_r2=rf_r2, rf_rmse=rf_rmse, cnn_r2=cnn_r2, cnn_rmse=cnn_rmse)
print("\nSaved results/real_data_results.npz")
