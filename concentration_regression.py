"""
Concentration regression: predict analyte concentration (nM) from CV shape.
Tests RF, SVR, and 1D-CNN regressors per-analyte and combined.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

X_train, y_train, c_train = train["X"], train["y"], train["concentrations"]
X_test, y_test, c_test = test["X"], test["y"], test["concentrations"]
voltage = train["voltage"]
analyte_names = list(train["analyte_names"])

print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
print(f"  Concentration range: {c_train.min():.0f} - {c_train.max():.0f} nM")

# Extract features
F_train, feat_names = extract_all(X_train, voltage)
F_test, _ = extract_all(X_test, voltage)

# Normalize raw CVs
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0
X_tr_norm = (X_train - mean) / std
X_te_norm = (X_test - mean) / std

# Normalize features
f_scaler = StandardScaler()
F_tr_s = f_scaler.fit_transform(F_train)
F_te_s = f_scaler.transform(F_test)

# ============================================================
# 1D-CNN regressor
# ============================================================
class CNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(-1)

# ============================================================
# Task 1: Combined regression (all analytes, predict concentration)
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: Combined Concentration Regression (all analytes)")
print("=" * 60)

# RF on features
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(F_tr_s, c_train)
rf_pred = rf.predict(F_te_s)
rf_r2 = r2_score(c_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(c_test, rf_pred))
rf_mae = mean_absolute_error(c_test, rf_pred)
print(f"  RF (features):  R2={rf_r2:.4f}  RMSE={rf_rmse:.1f} nM  MAE={rf_mae:.1f} nM")

# SVR on raw CV (subsample for speed)
svr = SVR(kernel="rbf", C=100, gamma="scale")
svr.fit(X_tr_norm, c_train)
svr_pred = svr.predict(X_te_norm)
svr_r2 = r2_score(c_test, svr_pred)
svr_rmse = np.sqrt(mean_squared_error(c_test, svr_pred))
svr_mae = mean_absolute_error(c_test, svr_pred)
print(f"  SVR (raw CV):   R2={svr_r2:.4f}  RMSE={svr_rmse:.1f} nM  MAE={svr_mae:.1f} nM")

# CNN regressor
print("  Training 1D-CNN regressor...")
X_tr_t = torch.FloatTensor(X_tr_norm).unsqueeze(1)
c_tr_t = torch.FloatTensor(c_train)
X_te_t = torch.FloatTensor(X_te_norm).unsqueeze(1)

train_ds = TensorDataset(X_tr_t, c_tr_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

cnn_reg = CNN1DRegressor()
optimizer = torch.optim.Adam(cnn_reg.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
criterion = nn.MSELoss()

best_loss = float("inf")
for epoch in range(60):
    cnn_reg.train()
    for xb, yb in train_loader:
        pred = cnn_reg(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    cnn_reg.eval()
    with torch.no_grad():
        val_pred = cnn_reg(X_te_t)
        val_loss = criterion(val_pred, torch.FloatTensor(c_test)).item()
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.clone() for k, v in cnn_reg.state_dict().items()}
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}: val_loss={val_loss:.1f}")

cnn_reg.load_state_dict(best_state)
cnn_reg.eval()
with torch.no_grad():
    cnn_pred = cnn_reg(X_te_t).numpy()
cnn_r2 = r2_score(c_test, cnn_pred)
cnn_rmse = np.sqrt(mean_squared_error(c_test, cnn_pred))
cnn_mae = mean_absolute_error(c_test, cnn_pred)
print(f"  CNN Regressor:  R2={cnn_r2:.4f}  RMSE={cnn_rmse:.1f} nM  MAE={cnn_mae:.1f} nM")

# ============================================================
# Task 2: Per-analyte regression
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: Per-Analyte Concentration Regression (RF)")
print("=" * 60)

per_analyte_results = {}
for cls_idx, name in enumerate(analyte_names):
    tr_mask = y_train == cls_idx
    te_mask = y_test == cls_idx

    rf_a = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_a.fit(F_tr_s[tr_mask], c_train[tr_mask])
    pred_a = rf_a.predict(F_te_s[te_mask])
    r2_a = r2_score(c_test[te_mask], pred_a)
    rmse_a = np.sqrt(mean_squared_error(c_test[te_mask], pred_a))
    mae_a = mean_absolute_error(c_test[te_mask], pred_a)

    per_analyte_results[name] = {"r2": r2_a, "rmse": rmse_a, "mae": mae_a,
                                  "true": c_test[te_mask], "pred": pred_a}
    print(f"  {name:>4s}: R2={r2_a:.4f}  RMSE={rmse_a:.1f} nM  MAE={mae_a:.1f} nM")

# ============================================================
# Figure 1: Predicted vs actual scatter (combined + per-analyte)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors_m = {"RF": "#95a5a6", "SVR": "#7f8c8d", "CNN": "#e74c3c"}

ax = axes[0]
ax.scatter(c_test, rf_pred, alpha=0.2, s=8, color="#95a5a6", label=f"RF (R²={rf_r2:.3f})")
ax.scatter(c_test, cnn_pred, alpha=0.2, s=8, color="#e74c3c", label=f"CNN (R²={cnn_r2:.3f})")
ax.plot([0, 2600], [0, 2600], "k--", linewidth=1)
ax.set_xlabel("Actual Concentration (nM)")
ax.set_ylabel("Predicted Concentration (nM)")
ax.set_title("Combined (all analytes)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 2600)
ax.set_ylim(0, 2600)

colors_a = {"DA": "#e74c3c", "5HT": "#3498db", "NE": "#2ecc71", "pH": "#f39c12"}

ax = axes[1]
for name in analyte_names:
    r = per_analyte_results[name]
    ax.scatter(r["true"], r["pred"], alpha=0.3, s=10, color=colors_a[name],
               label=f"{name} (R²={r['r2']:.3f})")
ax.plot([0, 2600], [0, 2600], "k--", linewidth=1)
ax.set_xlabel("Actual Concentration (nM)")
ax.set_ylabel("Predicted Concentration (nM)")
ax.set_title("Per-Analyte RF Regression", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 2600)
ax.set_ylim(0, 2600)

# Residual distribution
ax = axes[2]
for name in analyte_names:
    r = per_analyte_results[name]
    residuals = r["pred"] - r["true"]
    ax.hist(residuals, bins=30, alpha=0.5, color=colors_a[name], label=name, edgecolor="black", linewidth=0.3)
ax.set_xlabel("Prediction Error (nM)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution", fontweight="bold")
ax.legend(fontsize=9)
ax.axvline(0, color="black", linewidth=0.5)

plt.suptitle("Concentration Regression Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "concentration_regression.png"), dpi=150)
plt.close()
print("\nSaved figures/concentration_regression.png")

# ============================================================
# Figure 2: Error vs concentration (does model struggle at low conc?)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

conc_bins = np.linspace(0, 2600, 11)
conc_centers = (conc_bins[:-1] + conc_bins[1:]) / 2

for name in analyte_names:
    r = per_analyte_results[name]
    bin_maes = []
    for j in range(len(conc_bins) - 1):
        mask = (r["true"] >= conc_bins[j]) & (r["true"] < conc_bins[j+1])
        if mask.sum() > 0:
            bin_maes.append(np.abs(r["pred"][mask] - r["true"][mask]).mean())
        else:
            bin_maes.append(np.nan)
    ax.plot(conc_centers, bin_maes, "o-", color=colors_a[name], linewidth=1.5,
            markersize=5, label=name)

ax.set_xlabel("True Concentration (nM)")
ax.set_ylabel("Mean Absolute Error (nM)")
ax.set_title("Prediction Error vs Concentration", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "regression_error_vs_conc.png"), dpi=150)
plt.close()
print("Saved figures/regression_error_vs_conc.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("CONCENTRATION REGRESSION SUMMARY")
print("=" * 60)

print(f"\n  Combined (all analytes):")
print(f"    {'Model':<18s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s}")
print(f"    {'-'*48}")
print(f"    {'RF (features)':<18s} {rf_r2:>8.4f} {rf_rmse:>9.1f} {rf_mae:>9.1f}")
print(f"    {'SVR (raw CV)':<18s} {svr_r2:>8.4f} {svr_rmse:>9.1f} {svr_mae:>9.1f}")
print(f"    {'1D-CNN':<18s} {cnn_r2:>8.4f} {cnn_rmse:>9.1f} {cnn_mae:>9.1f}")

print(f"\n  Per-analyte (RF):")
for name in analyte_names:
    r = per_analyte_results[name]
    print(f"    {name:>4s}: R²={r['r2']:.4f}  RMSE={r['rmse']:.1f} nM  MAE={r['mae']:.1f} nM")

np.savez(os.path.join(RESULTS_DIR, "concentration_regression.npz"),
         rf_r2=rf_r2, rf_rmse=rf_rmse, svr_r2=svr_r2, svr_rmse=svr_rmse,
         cnn_r2=cnn_r2, cnn_rmse=cnn_rmse,
         per_analyte={n: per_analyte_results[n]["r2"] for n in analyte_names})
print("\nSaved results/concentration_regression.npz")
