"""
Synthetic FSCV data generator.
Models DA, 5-HT, NE, and pH voltammograms using Gaussian peaks at published
oxidation/reduction potentials with electrode variation and instrument noise.
Waveform: -0.4V to +1.3V to -0.4V at 400 V/s.
"""
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

np.random.seed(42)

# FSCV voltage waveform: -0.4 -> +1.3 -> -0.4 (triangle)
N_POINTS = 1000
v_up = np.linspace(-0.4, 1.3, N_POINTS // 2)
v_down = np.linspace(1.3, -0.4, N_POINTS // 2)
voltage = np.concatenate([v_up, v_down])


def gaussian_peak(v, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((v - center) / width) ** 2)


def generate_cv(analyte, concentration_nm, electrode_variation=None):
    """
    Generate a single background-subtracted cyclic voltammogram.

    Parameters:
        analyte: 'DA', '5HT', 'NE', or 'pH'
        concentration_nm: concentration in nanomolar (0-2500)
        electrode_variation: random seed for electrode-to-electrode variation
    """
    if electrode_variation is not None:
        rng = np.random.RandomState(electrode_variation)
    else:
        rng = np.random.RandomState()

    scale = concentration_nm / 1000.0

    # Electrode-to-electrode variation in sensitivity and peak position
    sensitivity = 1.0 + rng.normal(0, 0.15)
    pos_shift = rng.normal(0, 0.02)
    width_var = 1.0 + rng.normal(0, 0.1)

    current = np.zeros(N_POINTS)

    if analyte == "DA":
        # Oxidation peak (anodic scan, first half)
        ox_center = 0.60 + pos_shift
        ox_width = 0.08 * width_var
        current[:N_POINTS//2] += gaussian_peak(v_up, ox_center, ox_width, scale * sensitivity * 8.0)
        # Small secondary oxidation
        current[:N_POINTS//2] += gaussian_peak(v_up, 0.20 + pos_shift, 0.06 * width_var, scale * sensitivity * 1.0)
        # Reduction peak (cathodic scan, second half)
        red_center = -0.20 + pos_shift
        red_width = 0.10 * width_var
        current[N_POINTS//2:] -= gaussian_peak(v_down, red_center, red_width, scale * sensitivity * 4.5)

    elif analyte == "5HT":
        # Primary oxidation (slightly higher and broader than DA)
        ox_center = 0.65 + pos_shift
        ox_width = 0.10 * width_var
        current[:N_POINTS//2] += gaussian_peak(v_up, ox_center, ox_width, scale * sensitivity * 7.0)
        # Secondary oxidation peak (characteristic of serotonin)
        current[:N_POINTS//2] += gaussian_peak(v_up, 0.35 + pos_shift, 0.07 * width_var, scale * sensitivity * 3.5)
        # Reduction peak (smaller ratio than DA)
        red_center = -0.10 + pos_shift
        red_width = 0.12 * width_var
        current[N_POINTS//2:] -= gaussian_peak(v_down, red_center, red_width, scale * sensitivity * 2.5)
        # Serotonin fouls electrode over time - adds slight baseline drift
        current += rng.normal(0, 0.1) * scale * np.linspace(0, 1, N_POINTS)

    elif analyte == "NE":
        # Oxidation peak (very similar to DA but slightly different width/ratio)
        ox_center = 0.65 + pos_shift
        ox_width = 0.09 * width_var
        current[:N_POINTS//2] += gaussian_peak(v_up, ox_center, ox_width, scale * sensitivity * 7.5)
        # Reduction peak (larger reduction:oxidation ratio than DA)
        red_center = -0.25 + pos_shift
        red_width = 0.11 * width_var
        current[N_POINTS//2:] -= gaussian_peak(v_down, red_center, red_width, scale * sensitivity * 5.5)
        # NE has a broader oxidation shoulder
        current[:N_POINTS//2] += gaussian_peak(v_up, 0.45 + pos_shift, 0.12 * width_var, scale * sensitivity * 2.0)

    elif analyte == "pH":
        # pH shift is a broad, symmetric change across the CV
        current[:N_POINTS//2] += scale * sensitivity * 2.0 * np.sin(np.pi * (v_up + 0.4) / 1.7)
        current[N_POINTS//2:] -= scale * sensitivity * 2.0 * np.sin(np.pi * (v_down + 0.4) / 1.7)

    # Instrument noise
    noise_level = 0.3 + 0.1 * scale
    current += rng.normal(0, noise_level, N_POINTS)

    # Capacitive background (residual after subtraction)
    cap_residual = rng.normal(0, 0.05) * np.ones(N_POINTS)
    cap_residual[:N_POINTS//2] += rng.normal(0, 0.1)
    cap_residual[N_POINTS//2:] -= rng.normal(0, 0.1)
    current += cap_residual

    return current


def generate_dataset(n_per_class=2000, n_concentrations=30, n_electrodes=10):
    """Generate a full dataset with multiple concentrations and electrode variations."""
    analytes = ["DA", "5HT", "NE", "pH"]
    concentrations = np.linspace(50, 2500, n_concentrations)

    all_cvs = []
    all_labels = []
    all_concentrations = []

    for label_idx, analyte in enumerate(analytes):
        print(f"  Generating {analyte}... ", end="", flush=True)
        count = 0
        for _ in range(n_per_class):
            conc = np.random.choice(concentrations) + np.random.normal(0, 20)
            conc = max(10, conc)
            electrode = np.random.randint(0, n_electrodes * 100)
            cv = generate_cv(analyte, conc, electrode_variation=electrode)
            all_cvs.append(cv)
            all_labels.append(label_idx)
            all_concentrations.append(conc)
            count += 1
        print(f"{count} samples")

    X = np.array(all_cvs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    concs = np.array(all_concentrations, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y, concs = X[idx], y[idx], concs[idx]

    return X, y, concs, analytes


# ============================================================
# Generate data
# ============================================================
print("Generating synthetic FSCV dataset...")
print(f"Voltage waveform: {voltage[0]:.1f}V -> {voltage[N_POINTS//2-1]:.1f}V -> {voltage[-1]:.1f}V")
print(f"Points per CV: {N_POINTS}\n")

X, y, concs, analyte_names = generate_dataset(n_per_class=2000)

# Train/test split (80/20)
n_total = len(y)
n_train = int(n_total * 0.8)
X_train, y_train, c_train = X[:n_train], y[:n_train], concs[:n_train]
X_test, y_test, c_test = X[n_train:], y[n_train:], concs[n_train:]

print(f"\nTotal:  {n_total} samples ({len(analyte_names)} classes)")
print(f"Train:  {len(y_train)}")
print(f"Test:   {len(y_test)}")
print(f"\nClass distribution (train):")
for i, name in enumerate(analyte_names):
    print(f"  {name}: {np.sum(y_train == i)}")

# Save
np.savez(os.path.join(DATA_DIR, "fscv_train.npz"),
         X=X_train, y=y_train, concentrations=c_train,
         voltage=voltage, analyte_names=analyte_names)
np.savez(os.path.join(DATA_DIR, "fscv_test.npz"),
         X=X_test, y=y_test, concentrations=c_test,
         voltage=voltage, analyte_names=analyte_names)
print(f"\nSaved data/fscv_train.npz and data/fscv_test.npz")

# ============================================================
# Visualization: Example CVs for each analyte
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_per = {"DA": "#e74c3c", "5HT": "#3498db", "NE": "#2ecc71", "pH": "#9b59b6"}

for idx, (ax, analyte) in enumerate(zip(axes.flat, analyte_names)):
    # Plot 5 examples at different concentrations
    example_concs = [250, 500, 1000, 1500, 2500]
    for conc in example_concs:
        cv = generate_cv(analyte, conc, electrode_variation=42)
        v_full = np.concatenate([v_up, v_down])
        ax.plot(v_full, cv, linewidth=0.8, alpha=0.8, label=f"{conc} nM")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (nA)")
    ax.set_title(f"{analyte}", fontsize=14, fontweight="bold", color=colors_per[analyte])
    ax.legend(fontsize=7)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")

plt.suptitle("Synthetic FSCV Cyclic Voltammograms by Analyte", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "example_voltammograms.png"), dpi=150)
plt.close()
print("Saved figures/example_voltammograms.png")

# ============================================================
# Visualization: Overlay all 4 analytes at same concentration
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
conc = 1000
for analyte in analyte_names:
    cv = generate_cv(analyte, conc, electrode_variation=42)
    v_full = np.concatenate([v_up, v_down])
    ax.plot(v_full, cv, linewidth=1.5, label=analyte, color=colors_per[analyte])

ax.set_xlabel("Voltage (V)", fontsize=12)
ax.set_ylabel("Current (nA)", fontsize=12)
ax.set_title(f"All Analytes at {conc} nM - Note Similar Shapes for DA vs NE vs 5-HT",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "analyte_overlay.png"), dpi=150)
plt.close()
print("Saved figures/analyte_overlay.png")

# ============================================================
# Color plot (time x voltage heatmap) example
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
half = N_POINTS // 2
for idx, (ax, analyte) in enumerate(zip(axes, analyte_names)):
    n_scans = 50
    color_plot = np.zeros((half, n_scans))
    for t in range(n_scans):
        if t < 20:
            c = 0
        elif t < 30:
            c = (t - 20) * 250
        else:
            c = max(0, 2500 - (t - 30) * 125)
        cv = generate_cv(analyte, c, electrode_variation=42 + t)
        color_plot[:, t] = cv[:half]  # forward (oxidation) scan only

    im = ax.imshow(color_plot, aspect="auto", cmap="RdBu_r",
                   extent=[0, n_scans, v_up[-1], v_up[0]],
                   vmin=-15, vmax=15, origin="lower")
    ax.set_xlabel("Scan Number")
    if idx == 0:
        ax.set_ylabel("Applied Potential (V)")
    ax.set_title(analyte, fontweight="bold", color=colors_per[analyte])

plt.suptitle("FSCV Color Plots (Oxidation Scan)", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=axes, label="Current (nA)", shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "color_plots.png"), dpi=150)
plt.close()
print("Saved figures/color_plots.png")
