"""
Electrochemical feature extraction from cyclic voltammograms.
14 features: peak positions/currents, widths at half-max, anodic/cathodic
charges, ox/red ratio, secondary peaks, hysteresis area.
"""
import numpy as np
from scipy.signal import find_peaks
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")


def extract_features(cv, voltage):
    """Extract electrochemical features from a single CV."""
    n = len(cv)
    half = n // 2
    v_up = voltage[:half]
    v_down = voltage[half:]
    i_up = cv[:half]     # anodic (oxidation) scan
    i_down = cv[half:]   # cathodic (reduction) scan

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    features = {}

    # 1-2. Oxidation peak
    ox_idx = np.argmax(i_up)
    features["ox_peak_v"] = v_up[ox_idx]
    features["ox_peak_i"] = i_up[ox_idx]

    # 3-4. Reduction peak (most negative current)
    red_idx = np.argmin(i_down)
    features["red_peak_v"] = v_down[red_idx]
    features["red_peak_i"] = i_down[red_idx]

    # 5. Ox/Red ratio
    red_mag = abs(features["red_peak_i"]) if abs(features["red_peak_i"]) > 0.01 else 0.01
    features["ox_red_ratio"] = features["ox_peak_i"] / red_mag

    # 6. Oxidation peak width at half max
    half_max_ox = features["ox_peak_i"] / 2
    above_half = np.where(i_up > half_max_ox)[0]
    if len(above_half) >= 2:
        features["ox_width"] = v_up[above_half[-1]] - v_up[above_half[0]]
    else:
        features["ox_width"] = 0.0

    # 7. Reduction peak width at half max
    half_max_red = features["red_peak_i"] / 2
    below_half = np.where(i_down < half_max_red)[0]
    if len(below_half) >= 2:
        features["red_width"] = abs(v_down[below_half[-1]] - v_down[below_half[0]])
    else:
        features["red_width"] = 0.0

    # 8-9. Charge (area under curve)
    features["anodic_charge"] = _trapz(np.maximum(i_up, 0), v_up)
    features["cathodic_charge"] = abs(_trapz(np.minimum(i_down, 0), v_down))

    # 10. Charge ratio
    cat_q = features["cathodic_charge"] if features["cathodic_charge"] > 0.001 else 0.001
    features["charge_ratio"] = features["anodic_charge"] / cat_q

    # 11. Secondary oxidation peak (0.2-0.45V region, characteristic of 5-HT)
    sec_mask = (v_up >= 0.2) & (v_up <= 0.45)
    if sec_mask.any():
        features["secondary_ox_i"] = np.max(i_up[sec_mask])
    else:
        features["secondary_ox_i"] = 0.0

    # 12. Rising edge voltage (voltage where current reaches 50% of peak on rising edge)
    half_peak = features["ox_peak_i"] * 0.5
    rising = np.where(i_up[:ox_idx] >= half_peak)[0]
    features["rising_edge_v"] = v_up[rising[0]] if len(rising) > 0 else v_up[ox_idx]

    # 13. Mean current in high voltage range (0.8-1.2V)
    high_v_mask = (v_up >= 0.8) & (v_up <= 1.2)
    features["high_v_mean"] = np.mean(i_up[high_v_mask]) if high_v_mask.any() else 0.0

    # 14. Hysteresis area
    features["hysteresis"] = _trapz(i_up, v_up) - _trapz(i_down, v_down)

    return features


def extract_all(X, voltage):
    """Extract features for the full dataset."""
    feature_names = None
    all_features = []

    for cv in X:
        feat = extract_features(cv, voltage)
        if feature_names is None:
            feature_names = list(feat.keys())
        all_features.append([feat[k] for k in feature_names])

    return np.array(all_features, dtype=np.float32), feature_names


# ============================================================
# Run feature extraction
# ============================================================
if __name__ == "__main__":
    print("Loading raw data...")
    train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
    test = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)

    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]
    voltage = train["voltage"]
    analyte_names = list(train["analyte_names"])

    print(f"Extracting features from {len(X_train)} train + {len(X_test)} test CVs...")
    F_train, feature_names = extract_all(X_train, voltage)
    F_test, _ = extract_all(X_test, voltage)

    print(f"\n{len(feature_names)} features extracted:")
    for i, name in enumerate(feature_names):
        tr_mean = F_train[:, i].mean()
        tr_std = F_train[:, i].std()
        print(f"  {i+1:2d}. {name:<20s}  mean={tr_mean:>8.3f}  std={tr_std:>8.3f}")

    np.savez(os.path.join(PROCESSED_DIR, "features.npz"),
             F_train=F_train, y_train=y_train,
             F_test=F_test, y_test=y_test,
             feature_names=feature_names,
             analyte_names=analyte_names)
    print(f"\nSaved processed_data/features.npz")
