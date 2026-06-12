"""
generate_diagram.py — autograd graph diagram for LinkedIn post 03.

Run from 03-autograd-internals/:
    python generate_diagram.py
Outputs: autograd_diagram.png
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — works in WSL2 / headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette — light, professional ─────────────────────────────────────────────
BG = "#ffffff"
PANEL_F = "#ecfdf5"  # soft green tint — forward panel
PANEL_B = "#fef2f2"  # soft red tint — backward panel
PANEL_I = "#f5f3ff"  # soft violet tint — insights panel
CARD = "#ffffff"
CARD_HL = "#faf9ff"
TEXT = "#0f172a"
MUTED = "#64748b"
LEAF = "#2563eb"  # blue — leaf tensors
NODE = "#7c3aed"  # violet — graph nodes
OUT = "#d97706"  # amber — output tensor
FWD = "#059669"  # green — forward
BWD = "#dc2626"  # red — backward
ACC = "#0e7490"  # teal — gradient accumulation

W, H = 18, 14
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def card(x, y, w, h, border, label_, sublabel=None, highlight=False, sub2=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=CARD_HL if highlight else CARD,
            edgecolor=border,
            linewidth=2.6 if highlight else 1.6,
            zorder=3,
        )
    )
    lines = [label_] + ([sublabel] if sublabel else []) + ([sub2] if sub2 else [])
    n = len(lines)
    for i, line in enumerate(lines):
        ty = y + h * (n - i - 0.5) / n
        ax.text(
            x + w / 2,
            ty,
            line,
            ha="center",
            va="center",
            color=border if i == 0 else MUTED,
            fontsize=11.5 if i == 0 else 8.8,
            fontweight="bold" if i == 0 else "normal",
            fontfamily="monospace",
            zorder=4,
        )


def arrow(x1, y1, x2, y2, color, lw=2.4, rad=0.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.22,head_length=0.45",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=5,
    )


def label(x, y, text, color=MUTED, size=9, weight="normal", align="center", bg=None):
    t = ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontfamily="monospace",
        fontweight=weight,
        ha=align,
        va="center",
        zorder=6,
    )
    if bg:
        t.set_bbox(dict(facecolor=bg, edgecolor="none", pad=2.5))


# ════════════════════════════════════════════════════════════════════════════
# Title
# ════════════════════════════════════════════════════════════════════════════
ax.text(
    W / 2,
    13.5,
    "What loss.backward() Actually Does",
    ha="center",
    color=TEXT,
    fontsize=24,
    fontweight="bold",
    fontfamily="sans-serif",
)
ax.text(
    W / 2,
    13.0,
    "PyTorch Internals Explained · Post 03 · example:  y = a·b + a",
    ha="center",
    color=MUTED,
    fontsize=12,
    fontfamily="sans-serif",
)

# ════════════════════════════════════════════════════════════════════════════
# Shared grid — IDENTICAL node positions in both panels (mirror layout).
# Columns:  leaves | Mul | Add | y
# ════════════════════════════════════════════════════════════════════════════
LX, LW = 1.0, 2.6  # leaf column
MX, MW = 6.2, 3.4  # MulBackward column
AX_, AW = 11.4, 3.2  # AddBackward column
YX, YW = 16.2, 1.3  # output column


# ════════════════════════════════════════════════════════════════════════════
# PANEL 1 — FORWARD
# ════════════════════════════════════════════════════════════════════════════
ax.add_patch(
    FancyBboxPatch(
        (0.35, 9.2),
        17.3,
        3.5,
        boxstyle="round,pad=0.12",
        facecolor=PANEL_F,
        edgecolor=FWD,
        linewidth=1.6,
        zorder=1,
    )
)
label(0.75, 12.35, "1 · FORWARD PASS", color=FWD, size=13, weight="bold", align="left")
label(
    0.75,
    11.95,
    "compute + record the graph",
    color=MUTED,
    size=9.5,
    align="left",
)

F_MID = 10.55  # vertical center of the op row
card(LX, 10.75, LW, 0.95, LEAF, "a = [2, 3]", "leaf · requires_grad")
card(LX, 9.55, LW, 0.95, LEAF, "b = [4, 5]", "leaf · requires_grad")
card(MX, 9.9, MW, 1.3, NODE, "MulBackward", "c = a·b = [8, 15]", sub2="saves a.data, b.data", highlight=True)
card(AX_, 9.9, AW, 1.3, NODE, "AddBackward", "y = c + a", highlight=True)
card(YX, 10.0, YW, 1.1, OUT, "y", "[10, 18]")

# straight, short forward arrows
arrow(LX + LW, 11.22, MX, F_MID + 0.25, FWD)  # a → Mul
arrow(LX + LW, 10.02, MX, F_MID - 0.25, FWD)  # b → Mul
arrow(MX + MW, F_MID, AX_, F_MID, FWD)  # Mul → Add
label((MX + MW + AX_) / 2, F_MID + 0.3, "c", color=FWD, size=9.5, weight="bold")
arrow(AX_ + AW, F_MID, YX, F_MID, FWD)  # Add → y

# the ONE routed arc: a feeds Add directly (skip connection), above the row
arrow(LX + 1.3, 11.7, AX_ + 1.6, 11.2, FWD, rad=-0.13)
label(
    9.0,
    12.25,
    "a is used twice → its gradients will SUM in backward",
    color=FWD,
    size=9.5,
    weight="bold",
    bg=PANEL_F,
)

# ════════════════════════════════════════════════════════════════════════════
# PANEL 2 — BACKWARD (exact mirror, arrows reversed, every edge labelled)
# ════════════════════════════════════════════════════════════════════════════
ax.add_patch(
    FancyBboxPatch(
        (0.35, 4.7),
        17.3,
        3.9,
        boxstyle="round,pad=0.12",
        facecolor=PANEL_B,
        edgecolor=BWD,
        linewidth=1.6,
        zorder=1,
    )
)
label(0.75, 8.25, "2 · BACKWARD PASS", color=BWD, size=13, weight="bold", align="left")
label(
    0.75,
    7.85,
    "same graph, walked in reverse",
    color=MUTED,
    size=9.5,
    align="left",
)

B_MID = 6.25  # vertical center of the op row (mirrors forward)
card(LX, 6.45, LW, 0.95, ACC, "a.grad = [5, 6]", "= [4,5] + [1,1]")
card(LX, 5.25, LW, 0.95, ACC, "b.grad = [2, 3]", "= g·a")
card(MX, 5.6, MW, 1.3, NODE, "MulBackward", "dL/da = g·b   dL/db = g·a", sub2="uses saved a.data, b.data", highlight=True)
card(AX_, 5.6, AW, 1.3, NODE, "AddBackward", "passes g through", highlight=True)
card(YX, 5.7, YW, 1.1, OUT, "y", "dL/dy = [1, 1]")

# straight, short backward arrows — mirror of forward, each labelled
arrow(YX, B_MID, AX_ + AW, B_MID, BWD)  # y → Add
label((YX + AX_ + AW) / 2, B_MID + 0.3, "seed", color=BWD, size=9, weight="bold")
arrow(AX_, B_MID, MX + MW, B_MID, BWD)  # Add → Mul
label((AX_ + MX + MW) / 2, B_MID + 0.3, "[1, 1]", color=BWD, size=9, weight="bold")
arrow(MX, B_MID + 0.25, LX + LW, 6.92, BWD)  # Mul → a.grad
label(4.9, 7.15, "g·b = [4,5]", color=ACC, size=9, weight="bold", bg=PANEL_B)
arrow(MX, B_MID - 0.25, LX + LW, 5.72, BWD)  # Mul → b.grad
label(4.9, 5.45, "g·a = [2,3]", color=ACC, size=9, weight="bold", bg=PANEL_B)

# the ONE routed arc: Add sends g straight back to a (skip connection)
arrow(AX_ + 1.6, 6.95, LX + 1.3, 7.4, BWD, rad=0.13)
label(
    9.8,
    7.95,
    "g = [1,1] — 2nd contribution to a.grad: ACCUMULATES (+=), never overwrites",
    color=BWD,
    size=9.5,
    weight="bold",
    bg=PANEL_B,
)

label(
    17.25,
    4.95,
    "leaf updates = AccumulateGrad — real PyTorch uses this exact name",
    color=ACC,
    size=9,
    align="right",
)

# ════════════════════════════════════════════════════════════════════════════
# INSIGHTS STRIP
# ════════════════════════════════════════════════════════════════════════════
ax.add_patch(
    FancyBboxPatch(
        (0.35, 2.0),
        17.3,
        2.3,
        boxstyle="round,pad=0.12",
        facecolor=PANEL_I,
        edgecolor=NODE,
        linewidth=1.6,
        zorder=1,
    )
)
label(0.75, 3.95, "3 · WHY THIS MATTERS", color=NODE, size=12, weight="bold", align="left")
label(
    0.75,
    3.42,
    "▸ grads accumulate (+=)            →  this is exactly why optimizer.zero_grad() exists",
    color=TEXT,
    size=10,
    align="left",
)
label(
    0.75,
    2.95,
    "▸ inputs saved for the chain rule  →  this is why training eats more memory than inference",
    color=TEXT,
    size=10,
    align="left",
)
label(
    0.75,
    2.48,
    "▸ requires_grad=False → NO graph   →  that's all torch.no_grad() does: stop recording",
    color=TEXT,
    size=10,
    align="left",
)

# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP STRIP
# ════════════════════════════════════════════════════════════════════════════
steps = [
    ("loss = model(x)", FWD, "forward: compute + record"),
    ("loss.backward()", BWD, "reverse walk + chain rule"),
    ("optimizer.step()", OUT, "uses the .grad fields"),
    ("optimizer.zero_grad()", ACC, "because grads accumulate"),
]
sx, sw, gap = 0.55, 4.0, 0.35
for i, (code, color, sub) in enumerate(steps):
    x = sx + i * (sw + gap)
    card(x, 0.7, sw, 0.95, color, code, sub)
    if i < 3:
        arrow(x + sw + 0.02, 1.17, x + sw + gap - 0.02, 1.17, MUTED, lw=1.8)

ax.text(
    W / 2,
    0.3,
    "every PyTorch training loop runs this machinery",
    ha="center",
    color=MUTED,
    fontsize=10,
    fontfamily="sans-serif",
    style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(
    "autograd_diagram.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
    edgecolor="none",
)
print("Saved: autograd_diagram.png")
plt.close()
