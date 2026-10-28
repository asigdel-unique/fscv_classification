"""
FSCV preprocessing: smoothing, normalization, train/val/test split.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data():
    train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
    test = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)
    return (train["X"], train["y"], train["concentrations"], train["voltage"],
            test["X"], test["y"], test["concentrations"],
            list(train["analyte_names"]))


def smooth_cv(X, window=5):
    """Apply uniform smoothing along the voltage axis of each CV."""
    return np.array([uniform_filter1d(cv, size=window) for cv in X])


def normalize_zscore(X_train, X_test):
    """Z-score normalize using training set statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def normalize_per_sample(X):
    """Normalize each CV independently to [-1, 1]."""
    maxabs = np.abs(X).max(axis=1, keepdims=True)
    maxabs[maxabs == 0] = 1.0
    return X / maxabs


def split_validation(X, y, concs, val_fraction=0.15):
    """Split training data into train and validation sets."""
    n = len(y)
    idx = np.random.permutation(n)
    val_n = int(n * val_fraction)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return (X[train_idx], y[train_idx], concs[train_idx],
            X[val_idx], y[val_idx], concs[val_idx])


# ============================================================
# Run preprocessing
# ============================================================
print("Loading raw FSCV data...")
X_train, y_train, c_train, voltage, X_test, y_test, c_test, analyte_names = load_raw_data()
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Classes: {analyte_names}")

# Step 1: Light smoothing to reduce noise
print("\nSmoothing (window=5)...")
X_train_sm = smooth_cv(X_train, window=5)
X_test_sm = smooth_cv(X_test, window=5)

# Step 2: Z-score normalization
print("Z-score normalization...")
X_train_norm, X_test_norm, global_mean, global_std = normalize_zscore(X_train_sm, X_test_sm)

# Step 3: Also create per-sample normalized version (useful for some models)
X_train_persample = normalize_per_sample(X_train_sm)
X_test_persample = normalize_per_sample(X_test_sm)

# Step 4: Train/validation split
print("Splitting validation set (15%)...")
np.random.seed(42)
(X_tr, y_tr, c_tr,
 X_val, y_val, c_val) = split_validation(X_train_norm, y_train, c_train)

(X_tr_ps, _, _,
 X_val_ps, _, _) = split_validation(X_train_persample, y_train, c_train)

print(f"\n  Train:      {X_tr.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test:       {X_test_norm.shape[0]} samples")

# Save processed data
np.savez(os.path.join(PROCESSED_DIR, "zscore.npz"),
         X_train=X_tr, y_train=y_tr, c_train=c_tr,
         X_val=X_val, y_val=y_val, c_val=c_val,
         X_test=X_test_norm, y_test=y_test, c_test=c_test,
         voltage=voltage, analyte_names=analyte_names,
         global_mean=global_mean, global_std=global_std)

np.savez(os.path.join(PROCESSED_DIR, "persample.npz"),
         X_train=X_tr_ps, y_train=y_tr, c_train=c_tr,
         X_val=X_val_ps, y_val=y_val, c_val=c_val,
         X_test=X_test_persample, y_test=y_test, c_test=c_test,
         voltage=voltage, analyte_names=analyte_names)

print("\nSaved processed_data/zscore.npz")
print("Saved processed_data/persample.npz")

# Stats
print("\n--- Dataset Statistics ---")
print(f"{'Analyte':<8} {'Train':>6} {'Val':>6} {'Test':>6} {'Conc Range (nM)':>20}")
print("-" * 50)
for i, name in enumerate(analyte_names):
    tr_n = np.sum(y_tr == i)
    va_n = np.sum(y_val == i)
    te_n = np.sum(y_test == i)
    c_all = np.concatenate([c_tr[y_tr == i], c_val[y_val == i]])
    print(f"{name:<8} {tr_n:>6} {va_n:>6} {te_n:>6} {c_all.min():>8.0f} - {c_all.max():>7.0f}")
