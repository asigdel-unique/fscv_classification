"""
FSCV artifact removal using ML.
Simulates the FSCV crosstalk problem: voltage sweep artifacts corrupt neural
spike recordings during simultaneous echem+ephys. Trains a U-Net to predict
and subtract the artifact, using sweep phase as an auxiliary input channel.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

torch.set_num_threads(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FS = 30000       # ephys sampling rate (30 kHz)
FSCV_RATE = 10   # FSCV scans per second
SWEEP_SAMPLES = FS // FSCV_RATE  # 3000 samples per cycle
WINDOW = SWEEP_SAMPLES

# ============================================================
# Sweep phase channel (known from the instrument)
# ============================================================
# Triangle wave position within each FSCV cycle, normalized to [-1, 1]
half = SWEEP_SAMPLES // 2
phase_one_cycle = np.concatenate([
    np.linspace(-1, 1, half),   # forward sweep
    np.linspace(1, -1, half),   # reverse sweep
]).astype(np.float32)

# ============================================================
# Generate multiple independent recordings
# ============================================================
def generate_recording(duration_s, seed):
    """Generate one simulated simultaneous echem+ephys recording."""
    rng = np.random.RandomState(seed)
    n_samples = FS * duration_s
    t = np.arange(n_samples) / FS

    # Clean neural signal
    clean = np.zeros(n_samples, dtype=np.float32)

    # LFP with random phase offsets
    phi1, phi2, phi3 = rng.uniform(0, 2*np.pi, 3)
    clean += (50 * np.sin(2*np.pi*7*t + phi1)
            + 30 * np.sin(2*np.pi*12*t + phi2)
            + 15 * np.sin(2*np.pi*22*t + phi3)).astype(np.float32)

    # Spikes at random times with variable amplitude
    spike_template = np.zeros(45, dtype=np.float32)
    spike_template[5:10] = np.array([0.3, 0.8, 1.0, 0.6, 0.1])
    spike_template[10:20] = np.linspace(-0.5, -0.1, 10)
    spike_template[20:35] = np.linspace(-0.1, 0.0, 15)

    n_spikes = int(duration_s * rng.uniform(30, 80))
    spike_times = np.sort(rng.choice(n_samples - 100, size=n_spikes, replace=False))
    for st in spike_times:
        amp = rng.uniform(120, 250)
        end = min(st + len(spike_template), n_samples)
        clean[st:end] += spike_template[:end-st] * amp

    clean += rng.normal(0, 15, n_samples).astype(np.float32)

    # FSCV artifact
    dv = np.zeros(SWEEP_SAMPLES, dtype=np.float32)
    dv[:half] = 1.0
    dv[half:] = -1.0

    cap_artifact = 800 * dv
    ring = np.zeros(SWEEP_SAMPLES, dtype=np.float32)
    rlen = 200
    rt = np.arange(rlen) / FS
    ring[half:half+rlen] += 400 * np.sin(2*np.pi*1000*rt) * np.exp(-rt*500)
    ring[:rlen] += 200 * np.sin(2*np.pi*800*rt) * np.exp(-rt*400)
    base = cap_artifact + ring

    art = np.zeros(n_samples, dtype=np.float32)
    n_cyc = n_samples // SWEEP_SAMPLES
    drift = rng.uniform(0.95, 1.05)
    for i in range(n_cyc):
        s = i * SWEEP_SAMPLES
        scale = drift + 0.05*np.sin(2*np.pi*0.2*i/FSCV_RATE) + 0.02*rng.randn()
        art[s:s+SWEEP_SAMPLES] = base*scale + rng.normal(0, 20, SWEEP_SAMPLES).astype(np.float32)

    recorded = clean + art

    # Build sweep phase for full recording (tile the single-cycle pattern)
    phase = np.tile(phase_one_cycle, n_cyc)[:n_samples]

    return recorded, clean, art, phase

print("Generating multiple independent recordings...")
all_rec, all_cln, all_art, all_phase = [], [], [], []

n_recordings = 8
duration_each = 10  # seconds each

for r in range(n_recordings):
    rec, cln, art, ph = generate_recording(duration_each, seed=42 + r)
    all_rec.append(rec)
    all_cln.append(cln)
    all_art.append(art)
    all_phase.append(ph)
    print(f"  Recording {r+1}: {len(rec)//FS}s, {len(rec)//SWEEP_SAMPLES} cycles, "
          f"SNR={10*np.log10(np.var(cln)/np.var(art)):.1f} dB")

# ============================================================
# Window extraction (cycle-aligned)
# ============================================================
print("\nExtracting cycle-aligned windows...")
windows_rec, windows_art, windows_cln, windows_phase = [], [], [], []

for rec, cln, art, ph in zip(all_rec, all_cln, all_art, all_phase):
    n_cyc = len(rec) // SWEEP_SAMPLES
    for i in range(n_cyc):
        s = i * SWEEP_SAMPLES
        windows_rec.append(rec[s:s+WINDOW])
        windows_cln.append(cln[s:s+WINDOW])
        windows_art.append(art[s:s+WINDOW])
        windows_phase.append(ph[s:s+WINDOW])

X_rec = np.array(windows_rec, dtype=np.float32)
Y_art = np.array(windows_art, dtype=np.float32)
Y_cln = np.array(windows_cln, dtype=np.float32)
X_phase = np.array(windows_phase, dtype=np.float32)

# Normalize recorded signal
rec_std = X_rec.std()
X_rec_n = X_rec / rec_std
Y_art_n = Y_art / rec_std  # target: predict artifact (same scale)

# 2-channel input: [normalized recorded, sweep phase]
X_2ch = np.stack([X_rec_n, X_phase], axis=1)  # (N, 2, 3000)

n_total = len(X_2ch)
n_train = int(n_total * 0.8)
idx = np.random.RandomState(42).permutation(n_total)

X_tr = torch.FloatTensor(X_2ch[idx[:n_train]])
Y_tr = torch.FloatTensor(Y_art_n[idx[:n_train]]).unsqueeze(1)
X_te = torch.FloatTensor(X_2ch[idx[n_train:]])
Y_te = torch.FloatTensor(Y_art_n[idx[n_train:]]).unsqueeze(1)

# Also keep clean signal targets for evaluation
Y_cln_te = Y_cln[idx[n_train:]]
X_rec_te = X_rec[idx[n_train:]]

snr_before = 10 * np.log10(np.var(Y_cln) / np.var(Y_art))
print(f"  Total windows: {n_total} (train: {n_train}, test: {n_total-n_train})")
print(f"  Overall SNR before removal: {snr_before:.1f} dB")

# ============================================================
# Template subtraction baseline
# ============================================================
train_arts = Y_art[idx[:n_train]]
avg_template = train_arts.mean(axis=0)

# ============================================================
# U-Net (2-channel input, predicts artifact)
# ============================================================
class ArtifactPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 input channels: recorded signal + sweep phase
        self.enc1 = nn.Sequential(nn.Conv1d(2, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU())
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU())

        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv1d(256, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU())
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv1d(128, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU())
        self.up1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv1d(64, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU())

        self.output = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        orig_len = x.shape[-1]
        pad_len = (8 - orig_len % 8) % 8
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len))

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3[:,:,:e3.shape[2]], e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2[:,:,:e2.shape[2]], e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1[:,:,:e1.shape[2]], e1], dim=1))

        return self.output(d1)[:,:,:orig_len]

# ============================================================
# Train
# ============================================================
print("\nTraining U-Net artifact predictor...")
train_ds = TensorDataset(X_tr, Y_tr)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = ArtifactPredictor()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

best_loss = float("inf")
best_state = None
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_te)
        val_loss = criterion(val_pred, Y_te).item()
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: train={epoch_loss/n_train:.6f} val={val_loss:.6f}")

model.load_state_dict(best_state)
model.eval()

# ============================================================
# Evaluate
# ============================================================
print("\nEvaluating...")
with torch.no_grad():
    pred_artifact = model(X_te).numpy().squeeze(1) * rec_std

# Clean = recorded - predicted_artifact
ml_clean = X_rec_te - pred_artifact

# Template baseline
tmpl_clean = X_rec_te - avg_template

# Metrics on recovered clean signal
r2_ml = r2_score(Y_cln_te.flatten(), ml_clean.flatten())
rmse_ml = np.sqrt(np.mean((Y_cln_te - ml_clean) ** 2))

r2_template = r2_score(Y_cln_te.flatten(), tmpl_clean.flatten())
rmse_template = np.sqrt(np.mean((Y_cln_te - tmpl_clean) ** 2))

snr_after_ml = 10 * np.log10(np.var(Y_cln_te) / np.var(Y_cln_te - ml_clean))
snr_after_tmpl = 10 * np.log10(np.var(Y_cln_te) / np.var(Y_cln_te - tmpl_clean))

print(f"  Template subtraction: R2={r2_template:.4f}  RMSE={rmse_template:.1f} uV  SNR={snr_after_tmpl:.1f} dB")
print(f"  U-Net (predict+sub):  R2={r2_ml:.4f}  RMSE={rmse_ml:.1f} uV  SNR={snr_after_ml:.1f} dB")
print(f"  (Before removal: SNR={snr_before:.1f} dB)")

# ============================================================
# Figure 1: Before/after artifact removal (4-panel)
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

ex = 8  # example window
t_ms = np.arange(WINDOW) / FS * 1000

axes[0].plot(t_ms, X_rec_te[ex], color="#e74c3c", linewidth=0.5)
axes[0].set_ylabel("Amplitude (uV)")
axes[0].set_title("Recorded Signal (neural + FSCV artifact)", fontweight="bold")

axes[1].plot(t_ms, Y_cln_te[ex], "k-", linewidth=0.5)
axes[1].set_ylabel("Amplitude (uV)")
axes[1].set_title("True Neural Signal (ground truth)", fontweight="bold")

axes[2].plot(t_ms, tmpl_clean[ex], color="#f39c12", linewidth=0.5)
axes[2].plot(t_ms, Y_cln_te[ex], "k--", linewidth=0.3, alpha=0.4)
axes[2].set_ylabel("Amplitude (uV)")
axes[2].set_title(f"Template Subtraction (R2={r2_template:.3f})", fontweight="bold")

axes[3].plot(t_ms, ml_clean[ex], color="#3498db", linewidth=0.5)
axes[3].plot(t_ms, Y_cln_te[ex], "k--", linewidth=0.3, alpha=0.4)
axes[3].set_ylabel("Amplitude (uV)")
axes[3].set_xlabel("Time (ms)")
axes[3].set_title(f"U-Net Artifact Removal (R2={r2_ml:.3f})", fontweight="bold")

plt.suptitle("FSCV Artifact Removal from Neural Recordings", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "artifact_removal.png"), dpi=150)
plt.close()
print("\nSaved figures/artifact_removal.png")

# ============================================================
# Figure 2: Zoomed spike recovery
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

sr = slice(500, 1200)
t_z = np.arange(700) / FS * 1000

axes[0].plot(t_z, X_rec_te[ex][sr], color="#e74c3c", linewidth=0.8, label="Recorded")
axes[0].plot(t_z, Y_cln_te[ex][sr], "k--", linewidth=1.0, alpha=0.7, label="True neural")
axes[0].set_ylabel("uV")
axes[0].set_title("Recorded (artifact + neural)", fontweight="bold")
axes[0].legend(fontsize=8)

axes[1].plot(t_z, Y_cln_te[ex][sr], "k-", linewidth=1.0, label="True")
axes[1].plot(t_z, tmpl_clean[ex][sr], color="#f39c12", linewidth=0.8, alpha=0.8, label="Template")
axes[1].set_ylabel("uV")
axes[1].set_title("Template Subtraction", fontweight="bold")
axes[1].legend(fontsize=8)

axes[2].plot(t_z, Y_cln_te[ex][sr], "k-", linewidth=1.0, label="True")
axes[2].plot(t_z, ml_clean[ex][sr], color="#3498db", linewidth=0.8, alpha=0.8, label="U-Net")
axes[2].set_ylabel("uV")
axes[2].set_xlabel("Time (ms)")
axes[2].set_title("U-Net Artifact Removal", fontweight="bold")
axes[2].legend(fontsize=8)

plt.suptitle("Spike Recovery After Artifact Removal", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "spike_recovery.png"), dpi=150)
plt.close()
print("Saved figures/spike_recovery.png")

# Save
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "artifact_remover_best.pt"))
np.savez(os.path.join(RESULTS_DIR, "artifact_removal_results.npz"),
         r2_ml=r2_ml, rmse_ml=rmse_ml, r2_template=r2_template, rmse_template=rmse_template,
         snr_before=snr_before, snr_after_ml=snr_after_ml, snr_after_tmpl=snr_after_tmpl)
print("Saved results/artifact_removal_results.npz")

print("\n" + "=" * 60)
print("ARTIFACT REMOVAL SUMMARY")
print("=" * 60)
print(f"  Problem: FSCV sweep artifacts mask neural spikes")
print(f"  Signal: 30 kHz ephys, {FSCV_RATE} Hz FSCV")
print(f"  Data: {n_recordings} independent recordings x {duration_each}s = {n_total} windows")
print(f"  Model: 1D U-Net ({n_params:,} params), 2-channel input")
print(f"  Strategy: predict artifact + subtract (not direct clean prediction)")
print(f"\n  {'Method':<22s} {'R2':>8s} {'RMSE (uV)':>10s} {'SNR (dB)':>10s}")
print(f"  {'-'*52}")
print(f"  {'No removal':<22s} {'--':>8s} {'--':>10s} {snr_before:>9.1f}")
print(f"  {'Template subtraction':<22s} {r2_template:>8.4f} {rmse_template:>10.1f} {snr_after_tmpl:>9.1f}")
print(f"  {'U-Net (predict+sub)':<22s} {r2_ml:>8.4f} {rmse_ml:>10.1f} {snr_after_ml:>9.1f}")
