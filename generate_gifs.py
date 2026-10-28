"""
Animated GIFs: CV sweep (raw vs smoothed), classification confidence, real DA dynamics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import uniform_filter1d
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
GIF_DIR = os.path.join(BASE_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)

SMOOTH_WINDOW = 15

train = np.load(os.path.join(DATA_DIR, "fscv_train.npz"), allow_pickle=True)
X_train = train["X"]
y_train = train["y"]
voltage = train["voltage"]
analyte_names = list(train["analyte_names"])

colors_analyte = {"DA": "#e74c3c", "5HT": "#3498db", "NE": "#2ecc71", "pH": "#f39c12"}

# ============================================================
# GIF 1: Raw vs Smoothed CV sweep — side by side
# ============================================================
print("GIF 1: Raw vs Smoothed CV sweep...")

fig, axes = plt.subplots(4, 2, figsize=(12, 10))

lines_raw = []
lines_smooth = []
dots_raw = []
dots_smooth = []
traces_raw = []
traces_smooth = []

for idx, name in enumerate(analyte_names):
    mask = y_train == idx
    raw = X_train[mask][5]
    smooth = uniform_filter1d(raw, size=SMOOTH_WINDOW)
    traces_raw.append(raw)
    traces_smooth.append(smooth)

    for col, trace, label in [(0, raw, "Raw"), (1, smooth, "Smoothed")]:
        ax = axes[idx, col]
        ax.set_xlim(voltage.min() - 0.05, voltage.max() + 0.05)
        y_margin = (raw.max() - raw.min()) * 0.15
        ax.set_ylim(raw.min() - y_margin, raw.max() + y_margin)
        if idx == 3:
            ax.set_xlabel("Voltage (V)", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{name}", fontsize=11, fontweight="bold",
                          color=colors_analyte[name])
        if idx == 0:
            ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0.6, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.12)
        ax.tick_params(labelsize=7)

    line_r, = axes[idx, 0].plot([], [], color=colors_analyte[name], linewidth=1.2, alpha=0.85)
    dot_r, = axes[idx, 0].plot([], [], "o", color=colors_analyte[name], markersize=4)
    line_s, = axes[idx, 1].plot([], [], color=colors_analyte[name], linewidth=2.0)
    dot_s, = axes[idx, 1].plot([], [], "o", color=colors_analyte[name], markersize=4)
    lines_raw.append(line_r)
    lines_smooth.append(line_s)
    dots_raw.append(dot_r)
    dots_smooth.append(dot_s)

fig.suptitle("FSCV Cyclic Voltammograms: Raw vs Smoothed", fontsize=14, fontweight="bold")
plt.tight_layout()

n_points = len(voltage)
step = 8
frames = list(range(0, n_points, step)) + [n_points]

def update_sweep(frame_idx):
    n = frames[frame_idx]
    out = []
    for i in range(4):
        lines_raw[i].set_data(voltage[:n], traces_raw[i][:n])
        lines_smooth[i].set_data(voltage[:n], traces_smooth[i][:n])
        if n > 0:
            dots_raw[i].set_data([voltage[n-1]], [traces_raw[i][n-1]])
            dots_smooth[i].set_data([voltage[n-1]], [traces_smooth[i][n-1]])
        else:
            dots_raw[i].set_data([], [])
            dots_smooth[i].set_data([], [])
        out += [lines_raw[i], lines_smooth[i], dots_raw[i], dots_smooth[i]]
    return out

anim = animation.FuncAnimation(fig, update_sweep, frames=len(frames),
                                interval=30, blit=True)
anim.save(os.path.join(GIF_DIR, "cv_sweep_animation.gif"),
          writer="pillow", fps=30, dpi=100)
plt.close()
print("  Saved gifs/cv_sweep_animation.gif")

# ============================================================
# GIF 2: Classification confidence (smoothed CV shown)
# ============================================================
print("GIF 2: Classification confidence animation...")

import torch
import torch.nn as nn

cnn_results = np.load(os.path.join(BASE_DIR, "results", "cnn_results.npz"), allow_pickle=True)
test_data = np.load(os.path.join(DATA_DIR, "fscv_test.npz"), allow_pickle=True)
X_test = test_data["X"]
y_test = test_data["y"]

class AttentionCNN1D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1), nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_attention=False):
        features = self.conv(x)
        attn_scores = self.attention(features)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (features * attn_weights).sum(dim=-1)
        logits = self.classifier(context)
        if return_attention:
            return logits, attn_weights.squeeze(1)
        return logits

model = AttentionCNN1D()
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "results", "attn_cnn1d_best.pt"),
                                  weights_only=True))
model.eval()

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

sample_indices = []
for cls in range(4):
    cls_mask = np.where(y_test == cls)[0]
    sample_indices.extend(cls_mask[:5].tolist())
np.random.seed(42)
np.random.shuffle(sample_indices)
sample_indices = sample_indices[:16]

all_probs = []
all_true = []
all_cvs_raw = []
all_cvs_smooth = []
for idx in sample_indices:
    x_raw = X_test[idx]
    x_smooth = uniform_filter1d(x_raw, size=SMOOTH_WINDOW)
    x_norm = (x_raw - mean) / std
    x_t = torch.FloatTensor(x_norm).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    all_probs.append(probs)
    all_true.append(y_test[idx])
    all_cvs_raw.append(x_raw)
    all_cvs_smooth.append(x_smooth)

fig, (ax_cv, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5),
                                      gridspec_kw={"width_ratios": [1.3, 1]})

line_raw, = ax_cv.plot([], [], color="#bbbbbb", linewidth=0.8, alpha=0.5, label="Raw")
line_smooth, = ax_cv.plot([], [], "k-", linewidth=1.8, label="Smoothed")
ax_cv.set_xlim(voltage.min(), voltage.max())
cv_min = min(cv.min() for cv in all_cvs_raw)
cv_max = max(cv.max() for cv in all_cvs_raw)
cv_margin = (cv_max - cv_min) * 0.1
ax_cv.set_ylim(cv_min - cv_margin, cv_max + cv_margin)
ax_cv.set_xlabel("Voltage (V)")
ax_cv.set_ylabel("Current (nA)")
ax_cv.axhline(0, color="gray", linewidth=0.3)
ax_cv.grid(True, alpha=0.15)
ax_cv.legend(fontsize=8, loc="upper left")
title_cv = ax_cv.set_title("", fontsize=12, fontweight="bold")

bar_colors_base = [colors_analyte[n] for n in analyte_names]
bars = ax_bar.barh(analyte_names, [0]*4, color=bar_colors_base, edgecolor="black", linewidth=0.5)
ax_bar.set_xlim(0, 1.05)
ax_bar.set_xlabel("Prediction Confidence")
title_bar = ax_bar.set_title("", fontsize=12, fontweight="bold")

prob_texts = []
for i, bar in enumerate(bars):
    txt = ax_bar.text(0.01, bar.get_y() + bar.get_height()/2, "",
                      va="center", fontsize=10, fontweight="bold")
    prob_texts.append(txt)

fig.suptitle("Attention CNN Classification", fontsize=13, fontweight="bold")
plt.tight_layout()

total_frames = len(sample_indices) * 25

def update_classify(frame):
    sample_idx = frame // 25
    sub_frame = frame % 25
    if sample_idx >= len(sample_indices):
        sample_idx = len(sample_indices) - 1
        sub_frame = 24

    true_cls = all_true[sample_idx]
    probs = all_probs[sample_idx]
    true_name = analyte_names[true_cls]

    line_raw.set_data(voltage, all_cvs_raw[sample_idx])
    line_smooth.set_data(voltage, all_cvs_smooth[sample_idx])
    line_smooth.set_color(colors_analyte[true_name])
    title_cv.set_text(f"Sample: {true_name}")
    title_cv.set_color(colors_analyte[true_name])

    progress = min(sub_frame / 15.0, 1.0)
    pred_cls = np.argmax(probs)
    correct = pred_cls == true_cls
    status = "CORRECT" if correct else "WRONG"
    status_color = "#2ecc71" if correct else "#e74c3c"
    title_bar.set_text(f"Prediction: {analyte_names[pred_cls]} ({status})")
    title_bar.set_color(status_color)

    for i, bar in enumerate(bars):
        w = probs[i] * progress
        bar.set_width(w)
        if progress > 0.5:
            prob_texts[i].set_text(f"{probs[i]:.1%}")
            prob_texts[i].set_x(w + 0.02)
        else:
            prob_texts[i].set_text("")

    return [line_raw, line_smooth, title_cv, title_bar] + list(bars) + prob_texts

anim2 = animation.FuncAnimation(fig, update_classify, frames=total_frames,
                                 interval=80, blit=False)
anim2.save(os.path.join(GIF_DIR, "classification_confidence.gif"),
           writer="pillow", fps=15, dpi=100)
plt.close()
print("  Saved gifs/classification_confidence.gif")

# ============================================================
# GIF 3: Real data — dopamine dynamics over trials
# ============================================================
print("GIF 3: Real data dopamine dynamics...")

import scipy.io as sio
mat = sio.loadmat(os.path.join(BASE_DIR, "real_data", "monkey_fscv_dopamine.mat"))
data = mat["c8ds_fscv"][0]
n_trials = len(data)
n_tp = 601

da_traces = np.zeros((n_trials, n_tp))
iox_traces = np.zeros((n_trials, n_tp))
for i in range(n_trials):
    da_traces[i] = data[i]["da"].flatten()[:n_tp]
    iox_traces[i] = data[i]["iox"].flatten()[:n_tp]

for arr in [da_traces, iox_traces]:
    for i in range(n_trials):
        nans = np.isnan(arr[i])
        if nans.any() and (~nans).any():
            arr[i][nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], arr[i][~nans])

t_sec = np.arange(n_tp) / 10.0

fig, (ax_da, ax_iox) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

line_da, = ax_da.plot([], [], "#e74c3c", linewidth=1.5)
fill_da = None
ax_da.set_xlim(0, 60)
ax_da.set_ylim(da_traces.min() * 1.1, da_traces.max() * 1.1)
ax_da.set_ylabel("DA (nM)", fontsize=11)
ax_da.axhline(0, color="gray", linewidth=0.3)
ax_da.grid(True, alpha=0.15)
title_da = ax_da.set_title("", fontsize=12, fontweight="bold")

line_iox, = ax_iox.plot([], [], "#3498db", linewidth=1.5)
ax_iox.set_xlim(0, 60)
ax_iox.set_ylim(iox_traces.min() * 1.1, iox_traces.max() * 1.1)
ax_iox.set_ylabel("I_ox at 0.6V (nA)", fontsize=11)
ax_iox.set_xlabel("Time (s)", fontsize=11)
ax_iox.axhline(0, color="gray", linewidth=0.3)
ax_iox.grid(True, alpha=0.15)

fig.suptitle("Real FSCV: Monkey Dopamine Dynamics", fontsize=14, fontweight="bold")
plt.tight_layout()

trial_picks = list(range(0, 400, 8))[:50]

def update_real(frame):
    global fill_da
    trial = trial_picks[frame]
    da = da_traces[trial]
    iox = iox_traces[trial]

    line_da.set_data(t_sec, da)
    line_iox.set_data(t_sec, iox)

    if fill_da is not None:
        fill_da.remove()
    fill_da = ax_da.fill_between(t_sec, 0, da, where=(da > 0),
                                  color="#e74c3c", alpha=0.2)

    peak_da = da.max()
    status = f"Peak DA: {peak_da:.1f} nM"
    if peak_da > 15:
        status += "  (RELEASE)"
        title_da.set_color("#e74c3c")
    else:
        title_da.set_color("#333333")

    title_da.set_text(f"Trial {trial} — {status}")
    return [line_da, line_iox, title_da]

anim3 = animation.FuncAnimation(fig, update_real, frames=len(trial_picks),
                                 interval=400, blit=False)
anim3.save(os.path.join(GIF_DIR, "dopamine_dynamics.gif"),
           writer="pillow", fps=5, dpi=100)
plt.close()
print("  Saved gifs/dopamine_dynamics.gif")

# ============================================================
# Static figure: Raw vs Smoothed side by side (for README)
# ============================================================
print("\nStatic figure: Raw vs Smoothed comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for idx, name in enumerate(analyte_names):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    mask = y_train == idx
    raw = X_train[mask][5]
    smooth = uniform_filter1d(raw, size=SMOOTH_WINDOW)

    ax.plot(voltage, raw, color=colors_analyte[name], linewidth=0.6, alpha=0.4, label="Raw")
    ax.plot(voltage, smooth, color=colors_analyte[name], linewidth=2.0, label="Smoothed")
    ax.set_xlabel("Voltage (V)", fontsize=10)
    ax.set_ylabel("Current (nA)", fontsize=10)
    ax.set_title(name, fontsize=13, fontweight="bold", color=colors_analyte[name])
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0.6, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.12)

fig.suptitle("Raw vs Smoothed Cyclic Voltammograms", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "raw_vs_smoothed.png"), dpi=150)
plt.close()
print("  Saved figures/raw_vs_smoothed.png")

# ============================================================
print(f"\nAll GIFs saved to gifs/ directory:")
for f in sorted(os.listdir(GIF_DIR)):
    size_mb = os.path.getsize(os.path.join(GIF_DIR, f)) / 1024 / 1024
    print(f"  {f} ({size_mb:.1f} MB)")
