"""Render the Standard, full-shared-expert, and SplitMoE comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


INK = "#253047"
NEUTRAL = "#EEF2FF"
ROUTED = "#FDE9D8"
SHARED = "#DDF3E5"
MERGE = "#FFF4CC"


def box(axis, x, y, width, height, label, color=NEUTRAL, fontsize=9.4):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025",
        facecolor=color,
        edgecolor=INK,
        linewidth=1.25,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.05,
    )


def arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.25,
            color=INK,
            shrinkA=0,
            shrinkB=0,
        )
    )


def row_title(axis, y, title, detail, detail_x):
    title_y = y + 1.24
    axis.text(0.05, title_y, title, fontsize=11.5, fontweight="bold", va="center")
    axis.text(detail_x, title_y, detail, fontsize=8.7, color="#536078", va="center")


def input_and_output(axis, y):
    box(axis, 0.25, y, 0.85, 0.58, "$x$")
    box(axis, 9.62, y, 0.98, 0.58, "$F(x)$")


fig, axis = plt.subplots(figsize=(11.2, 6.4))
axis.set_xlim(0, 10.9)
axis.set_ylim(-0.45, 6.65)
axis.axis("off")

# Standard sparse MoE.
y = 5.12
row_title(axis, y, "Standard MoE", "complete routed experts; no always-active expert", 1.78)
input_and_output(axis, y)
box(axis, 1.85, y, 1.25, 0.58, "Router")
box(axis, 4.02, y - 0.12, 2.22, 0.82, "$K$ selected\nfull experts", ROUTED)
box(axis, 7.32, y, 1.22, 0.58, "Weighted\nsum", MERGE, fontsize=8.8)
arrow(axis, (1.10, y + 0.29), (1.85, y + 0.29))
arrow(axis, (3.10, y + 0.29), (4.02, y + 0.29))
arrow(axis, (6.24, y + 0.29), (7.32, y + 0.29))
arrow(axis, (8.54, y + 0.29), (9.62, y + 0.29))

# Established full shared-expert pattern.
y = 3.10
row_title(
    axis,
    y,
    "Full shared-expert MoE",
    "our matched baseline, inspired by DeepSeekMoE shared-expert isolation",
    2.58,
)
input_and_output(axis, y)
box(axis, 1.75, y + 0.48, 1.72, 0.58, "Full shared FFN", SHARED)
box(axis, 1.85, y - 0.38, 1.25, 0.58, "Router")
box(axis, 4.02, y - 0.49, 2.22, 0.80, "$K$ selected\nfull-width routed experts", ROUTED, fontsize=8.8)
box(axis, 7.23, y, 1.40, 0.58, "Add shared\n+ routed", MERGE, fontsize=8.8)
arrow(axis, (1.10, y + 0.29), (1.75, y + 0.77))
arrow(axis, (1.10, y + 0.29), (1.85, y - 0.09))
arrow(axis, (3.10, y - 0.09), (4.02, y - 0.09))
arrow(axis, (3.47, y + 0.77), (7.23, y + 0.42))
arrow(axis, (6.24, y - 0.09), (7.23, y + 0.15))
arrow(axis, (8.63, y + 0.29), (9.62, y + 0.29))

# SplitMoE.
y = 1.08
row_title(axis, y, "SplitMoE", "active FFN width divided into shared and private parts", 1.18)
input_and_output(axis, y)
box(axis, 1.75, y + 0.48, 1.72, 0.58, "Partial-width\nshared FFN", SHARED, fontsize=8.8)
box(axis, 1.85, y - 0.38, 1.25, 0.58, "Router")
box(axis, 4.02, y - 0.49, 2.22, 0.80, "$K$ selected partial-width\nprivate FFNs", ROUTED, fontsize=8.8)
box(axis, 7.23, y, 1.40, 0.58, "Add shared\n+ routed", MERGE, fontsize=8.8)
arrow(axis, (1.10, y + 0.29), (1.75, y + 0.77))
arrow(axis, (1.10, y + 0.29), (1.85, y - 0.09))
arrow(axis, (3.10, y - 0.09), (4.02, y - 0.09))
arrow(axis, (3.47, y + 0.77), (7.23, y + 0.42))
arrow(axis, (6.24, y - 0.09), (7.23, y + 0.15))
arrow(axis, (8.63, y + 0.29), (9.62, y + 0.29))

legend_handles = [
    Patch(facecolor=SHARED, edgecolor=INK, label="Always-active shared capacity"),
    Patch(facecolor=ROUTED, edgecolor=INK, label="Router-selected capacity"),
    Patch(facecolor=MERGE, edgecolor=INK, label="Add/weighted combination"),
    Line2D([], [], color="none", label=r"Split merge: $\alpha[S(x)+\sum_i p_iP_i(x)]$"),
]
axis.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.075),
    ncol=4,
    frameon=False,
    fontsize=8.4,
    handlelength=1.8,
    columnspacing=1.6,
)

fig.tight_layout(pad=0.6)
output = Path("results/architecture.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(output)
