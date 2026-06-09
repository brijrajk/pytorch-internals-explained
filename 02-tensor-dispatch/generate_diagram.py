"""
generate_diagram.py — dispatch flow diagram for LinkedIn post 02.

Run from 02-tensor-dispatch/:
    python generate_diagram.py
Outputs: dispatch_diagram.png
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works in WSL2 / headless environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0f172a"
CARD    = "#1e293b"
TEXT    = "#f1f5f9"
MUTED   = "#94a3b8"
CPU     = "#3b82f6"   # blue
CUDA    = "#10b981"   # green
META    = "#f59e0b"   # amber
XPU     = "#475569"   # slate — not registered
DISP    = "#8b5cf6"   # purple — dispatcher
ARROW   = "#475569"

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


# ── Helpers ───────────────────────────────────────────────────────────────────

def card(x, y, w, h, border, label, sublabel=None, highlight=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor="#162032" if highlight else CARD,
        edgecolor=border,
        linewidth=3 if highlight else 1.5,
        zorder=3,
    ))
    ty = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w / 2, ty, label,
            ha="center", va="center",
            color=border if highlight else TEXT,
            fontsize=11, fontweight="bold", fontfamily="monospace", zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center",
                color=MUTED, fontsize=8.5, fontfamily="monospace", zorder=4)


def arrow(x1, y1, x2, y2, color=ARROW, lw=1.5, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=5)


def label(x, y, text, color=MUTED, size=9, weight="normal", align="left"):
    ax.text(x, y, text, color=color, fontsize=size,
            fontfamily="monospace", fontweight=weight,
            ha=align, va="center", zorder=4)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 10.55, "How PyTorch Picks the Right Kernel",
        ha="center", color=TEXT, fontsize=22, fontweight="bold",
        fontfamily="sans-serif")
ax.text(9, 10.15, "Post 02 — PyTorch Internals Explained  ·  tensor dispatch",
        ha="center", color=MUTED, fontsize=11, fontfamily="sans-serif")

# ── Python call ───────────────────────────────────────────────────────────────
card(4.5, 8.85, 9, 0.95, MUTED,
     "Python:  dispatch.scale(tensor, 2.0)",
     "tensor.key = DispatchKey.CUDA")

# ── pybind11 boundary label + arrow ──────────────────────────────────────────
arrow(9, 8.85, 9, 8.2, color=MUTED)
label(9.2, 8.52, "pybind11  ·  Python ↔ C++", color=MUTED, size=8.5)

# ── Dispatcher ────────────────────────────────────────────────────────────────
card(3.5, 7.1, 11, 1.0, DISP,
     "Dispatcher",
     "reads tensor.key  →  O(1) table lookup  →  calls fn",
     highlight=True)

# ── Fan-out arrows (dispatcher → kernels) ────────────────────────────────────
kernel_centers = [2.05, 5.85, 9.65, 13.45]
kernel_colors  = [CPU, CUDA, META, XPU]
rads           = [-0.18, -0.06, 0.06, 0.18]

for cx, kc, rad in zip(kernel_centers, kernel_colors, rads):
    arrow(9, 7.1, cx, 6.05,
          color=kc if kc != XPU else ARROW,
          lw=2.2 if kc == CUDA else 1.5,
          rad=rad)

# ── Kernel cards ──────────────────────────────────────────────────────────────
kernels = [
    (0.3,  5.0, 3.5, CPU,  "CPU kernel",  "scale_cpu()"),
    (4.1,  5.0, 3.5, CUDA, "CUDA kernel", "scale_cuda()"),
    (7.9,  5.0, 3.5, META, "Meta kernel", "scale_meta()"),
    (11.7, 5.0, 3.5, XPU,  "XPU",         "(not registered)"),
]
for x, y, w, c, t, s in kernels:
    card(x, y, w, 1.0, c, t, s, highlight=(c == CUDA))

# ── "selected" badge on CUDA ──────────────────────────────────────────────────
ax.text(5.85, 4.85, "▲  selected",
        ha="center", color=CUDA, fontsize=8.5,
        fontfamily="monospace", fontweight="bold", zorder=4)

# ── XPU error note ────────────────────────────────────────────────────────────
ax.text(13.45, 4.82, "✕",
        ha="center", color=XPU, fontsize=10,
        fontfamily="monospace", zorder=4)

# ── Hardware / simulated output block ────────────────────────────────────────
bx, by = 0.4, 3.6
ax.add_patch(FancyBboxPatch((bx, by - 0.85), 17.2, 1.55,
    boxstyle="round,pad=0.1", facecolor=CARD, edgecolor=ARROW,
    linewidth=1, zorder=2))

label(bx + 0.2, by + 0.52, "With GPU hardware:", color=CUDA, size=9, weight="bold")
label(bx + 0.2, by + 0.22,
      "  [CUDA kernel] *** RUNNING ON HARDWARE GPU ***", color=TEXT, size=8.5)
label(bx + 0.2, by - 0.05,
      "  [CUDA kernel] device : NVIDIA GeForce RTX 4090  |  VRAM: 24564 MB  |  compute: 8.9",
      color=MUTED, size=8.5)

label(9.5, by + 0.52, "Without GPU:", color=XPU, size=9, weight="bold")
label(9.5, by + 0.22,
      "  [CUDA kernel] *** SIMULATED (no GPU hardware detected) ***",
      color=TEXT, size=8.5)
label(9.5, by - 0.05,
      "  [CUDA kernel] computing on CPU memory as stand-in",
      color=MUTED, size=8.5)

# ── Key insight box ───────────────────────────────────────────────────────────
ix, iy = 0.4, 1.45
ax.add_patch(FancyBboxPatch((ix, iy - 0.75), 17.2, 1.3,
    boxstyle="round,pad=0.1", facecolor="#1a1040", edgecolor=DISP,
    linewidth=1.2, zorder=2))

label(ix + 0.3, iy + 0.35,
      "DispatchKey drives everything.  Zero if/else.  Just an O(1) hash map lookup.",
      color=TEXT, size=10, weight="bold")

items = [
    (CPU,  "CPU  → hardware CPU math"),
    (CUDA, "CUDA → GPU (hardware or simulated)"),
    (META, "Meta → shape only, zero compute"),
    (XPU,  "XPU  → RuntimeError (no kernel)"),
]
cols = [0.6, 5.0, 9.4, 13.6]
for (c, t), cx in zip(items, cols):
    ax.plot(cx, iy - 0.32, "o", color=c, markersize=6, zorder=4)
    label(cx + 0.25, iy - 0.32, t, color=c, size=9.0)

# ── Footer ────────────────────────────────────────────────────────────────────
ax.text(9, 0.35,
        "#PyTorch   #CUDA   #CPlusPlus   #MachineLearning   #MLEngineering   #DeepLearning",
        ha="center", color=MUTED, fontsize=9, fontfamily="sans-serif")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig("dispatch_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved: dispatch_diagram.png")
plt.close()
