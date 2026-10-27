import sys, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/sharon/vision_fusion")

# ── parse training log ────────────────────────────────────────
log   = open("results/logs/full_train.log").read()
train_epochs, train_losses = [], []
val_epochs,   val_losses   = [], []

for line in log.split("\n"):
    m = re.search(r"Epoch\s+(\d+)/80.*loss=([\d.]+)", line)
    if m and "VAL" not in line:
        train_epochs.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
    m = re.search(r"VAL.*loss=([\d.]+)", line)
    if m:
        val_epochs.append(train_epochs[-1] if train_epochs else 0)
        val_losses.append(float(m.group(1)))

# ── figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel 1: Training curve
ax1 = fig.add_subplot(gs[0])
ax1.plot(train_epochs, train_losses, "b-",  lw=2, label="Train loss")
ax1.plot(val_epochs,   val_losses,   "r--", lw=2, label="Val loss",
         marker="o", markersize=4)
ax1.axvline(x=10, color="gray", ls=":", lw=1.2, label="Encoders unfreeze")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_title("(a) Training Convergence", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 80)

# Panel 2: Per-class recall bar chart
ax2 = fig.add_subplot(gs[1])
classes  = ["Car", "Two-\nwheeler", "Pedestrian", "Truck"]
recalls  = [100.0, 51.9, 0.0, 0.0]
gt_counts= [133, 54, 3, 1]
colors   = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
bars = ax2.bar(classes, recalls, color=colors, edgecolor="white", lw=0.5)
for bar, cnt in zip(bars, gt_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"n={cnt}", ha="center", va="bottom", fontsize=8, color="gray")
ax2.set_ylim(0, 115)
ax2.set_ylabel("Recall (%)", fontsize=11)
ax2.set_title("(b) Per-class Detection Recall\n(Test set, IoU proxy)",
              fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

# Panel 3: Loss reduction summary
ax3 = fig.add_subplot(gs[2])
stages = ["Epoch 1\n(random)", "Epoch 10\n(unfreeze)",
          "Epoch 40\n(mid)", "Epoch 80\n(final)"]
losses = [2.5748, 1.8, 1.3299, 1.0620]
ax3.plot(range(len(stages)), losses, "bo-", lw=2, markersize=8)
for i, (s, l) in enumerate(zip(stages, losses)):
    ax3.annotate(f"{l:.3f}", (i, l), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9)
ax3.set_xticks(range(len(stages)))
ax3.set_xticklabels(stages, fontsize=9)
ax3.set_ylabel("Focal Loss", fontsize=11)
ax3.set_title("(c) Loss at Key Training\nMilestones",
              fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.8, 2.8)

plt.suptitle("LiquidFusionNet — Detection Results on TIAND Dataset (Hyderabad, India)",
             fontsize=13, fontweight="bold", y=1.02)

out = Path("results/figures/training_results.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {out}")

# also save individual panels for LaTeX
fig2, ax = plt.subplots(figsize=(6,4))
ax.plot(train_epochs, train_losses, "b-",  lw=2, label="Train")
ax.plot(val_epochs,   val_losses,   "r--", lw=2, label="Val",
        marker="o", markersize=4)
ax.axvline(x=10, color="gray", ls=":", lw=1.2, label="Encoder unfreeze")
ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
ax.set_title("LiquidFusionNet Training Convergence")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/training_curve.png", dpi=150, bbox_inches="tight")
print("Training curve saved -> results/figures/training_curve.png")
