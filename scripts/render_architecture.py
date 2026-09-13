"""Render the Standard MoE versus SplitMoE architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(axis, x, y, width, height, label, color="#edf1ff"):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02", facecolor=color,
        edgecolor="#283044", linewidth=1.3,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#283044"))


fig, axes = plt.subplots(2, 1, figsize=(9.2, 3.8))
for axis in axes:
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 2.2)
    axis.axis("off")

axis = axes[0]
axis.text(0, 1.95, "Standard MoE", fontsize=12, fontweight="bold")
box(axis, 0.2, 0.55, 1.0, 0.65, "$x$")
box(axis, 2.0, 0.55, 1.4, 0.65, "Router")
box(axis, 4.2, 0.38, 2.3, 1.0, "$K$ selected\nfull experts", "#fdebdc")
box(axis, 7.5, 0.55, 1.4, 0.65, "$F(x)$")
arrow(axis, (1.2, 0.88), (2.0, 0.88)); arrow(axis, (3.4, 0.88), (4.2, 0.88)); arrow(axis, (6.5, 0.88), (7.5, 0.88))

axis = axes[1]
axis.text(0, 1.95, "SplitMoE", fontsize=12, fontweight="bold")
box(axis, 0.2, 0.55, 1.0, 0.65, "$x$")
box(axis, 2.0, 1.18, 2.0, 0.65, "Shared partial FFN", "#dff3e8")
box(axis, 2.0, 0.12, 1.4, 0.65, "Router")
box(axis, 4.2, 0.02, 2.3, 0.85, "$K$ selected\nprivate FFNs", "#fdebdc")
box(axis, 7.1, 0.55, 0.8, 0.65, "+")
box(axis, 8.6, 0.55, 1.2, 0.65, "$F(x)$")
arrow(axis, (1.2, 0.88), (2.0, 1.5)); arrow(axis, (1.2, 0.88), (2.0, 0.45))
arrow(axis, (3.4, 0.45), (4.2, 0.45)); arrow(axis, (4.0, 1.5), (7.1, 1.02))
arrow(axis, (6.5, 0.45), (7.1, 0.72)); arrow(axis, (7.9, 0.88), (8.6, 0.88))

fig.tight_layout()
output = Path("results/architecture.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)
print(output)
