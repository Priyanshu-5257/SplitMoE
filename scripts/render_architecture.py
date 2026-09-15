"""Render an expert-bank comparison of Standard MoE, DeepSeekMoE, and SplitMoE."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch


INK = "#1F2937"
ROUTED = "#DCEAF7"
SHARED = "#CDE7BE"
SELECTED = "#F59E0B"
NEUTRAL = "#F8FAFC"


def box(axis, x, y, width, height, label, color=ROUTED, fontsize=8.5, edge=INK, linewidth=1.1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012",
        facecolor=color,
        edgecolor=edge,
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.0,
    )
    return patch


def arrow(axis, start, end, color=INK, linewidth=1.0, connectionstyle="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def selected_outline(axis, x, y, width, height):
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.012, y - 0.018),
            width + 0.024,
            height + 0.036,
            boxstyle="round,pad=0.012",
            fill=False,
            edgecolor=SELECTED,
            linewidth=1.6,
            linestyle=(0, (3, 2)),
        )
    )


def annotation(axis, x, y, label, fontsize=7.8):
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#526079",
        zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.2},
    )


def base_panel(axis, title, subtitle, merge_symbol):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.5, 0.055, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
    axis.text(0.5, 0.012, subtitle, ha="center", va="center", fontsize=7.8, color="#526079")
    box(axis, 0.39, 0.11, 0.22, 0.075, "Input hidden $x$", NEUTRAL, fontsize=8.0)
    box(axis, 0.405, 0.255, 0.19, 0.075, "Router", "#FFF1C9", fontsize=8.0)
    box(axis, 0.38, 0.855, 0.24, 0.075, "Output hidden $F(x)$", NEUTRAL, fontsize=8.0)
    merge = Circle((0.5, 0.775), 0.024, facecolor="white", edgecolor=INK, linewidth=1.1)
    axis.add_patch(merge)
    axis.text(0.5, 0.775, merge_symbol, ha="center", va="center", fontsize=10)
    arrow(axis, (0.5, 0.185), (0.5, 0.255))
    arrow(axis, (0.5, 0.799), (0.5, 0.855))


def connect_selected(axis, center_x, expert_bottom, expert_top, curve):
    arrow(
        axis,
        (0.5, 0.33),
        (center_x, expert_bottom),
        color="#475569",
        connectionstyle=f"arc3,rad={curve}",
    )
    arrow(
        axis,
        (center_x, expert_top),
        (0.5, 0.751),
        connectionstyle=f"arc3,rad={-curve / 2}",
    )


fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3))
fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.19, wspace=0.10)

# (a) Conventional MoE: N complete experts, K selected.
axis = axes[0]
base_panel(axis, "(a) Standard MoE", "$N$ complete experts; activate $K$", "$\\Sigma$")
expert_y, expert_h, expert_w = 0.47, 0.105, 0.17
xs = [0.08, 0.29, 0.50, 0.71]
labels = ["$E_1$", "$E_2$", "$\\cdots$", "$E_N$"]
for x, label in zip(xs, labels, strict=True):
    box(axis, x, expert_y, expert_w, expert_h, label, fontsize=9.0)
annotation(axis, 0.5, 0.62, "full width $D$ each", fontsize=8.0)
for index, curve in ((0, 0.18), (3, -0.18)):
    selected_outline(axis, xs[index], expert_y, expert_w, expert_h)
    connect_selected(axis, xs[index] + expert_w / 2, expert_y, expert_y + expert_h, curve)
annotation(axis, 0.5, 0.695, "router-weighted routed outputs")

# (b) DeepSeekMoE: fine-grained segmentation plus isolated shared experts.
axis = axes[1]
base_panel(
    axis,
    "(b) DeepSeekMoE",
    "fine-grained segmentation + shared-expert isolation",
    "$\\oplus$",
)
small_y, small_h, small_w = 0.47, 0.105, 0.115
xs = [0.025, 0.165, 0.305, 0.445, 0.585, 0.725, 0.865]
labels = ["$K_s$\nshared", "$R_1$", "$R_2$", "$R_3$", "$\\cdots$", "$R_{mN-1}$", "$R_{mN}$"]
for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
    color = SHARED if index == 0 else ROUTED
    box(axis, x, small_y, small_w, small_h, label, color, fontsize=7.7)
annotation(axis, 0.5, 0.62, "fine-grained width $D/m$ per expert", fontsize=8.0)
selected_outline(axis, xs[0], small_y, small_w, small_h)
arrow(axis, (0.5, 0.185), (xs[0] + small_w / 2, small_y), connectionstyle="arc3,rad=-0.28")
arrow(axis, (xs[0] + small_w / 2, small_y + small_h), (0.5, 0.751), connectionstyle="arc3,rad=0.20")
for index, curve in ((1, 0.16), (3, 0.04), (6, -0.16)):
    selected_outline(axis, xs[index], small_y, small_w, small_h)
    connect_selected(axis, xs[index] + small_w / 2, small_y, small_y + small_h, curve)
annotation(axis, 0.5, 0.695, "$K_s$ shared + $(mK-K_s)$ routed")

# (c) SplitMoE: an explicit shared/private width decomposition.
axis = axes[2]
base_panel(axis, "(c) SplitMoE", "one explicit shared/private capacity split", "$\\oplus$")
split_y, split_h = 0.455, 0.135
shared_x, shared_w = 0.035, 0.18
box(axis, shared_x, split_y, shared_w, split_h, "$S$\nshared", SHARED, fontsize=8.3)
private_xs = [0.285, 0.47, 0.655, 0.84]
private_w = 0.135
private_labels = ["$P_1$", "$P_2$", "$\\cdots$", "$P_N$"]
for x, label in zip(private_xs, private_labels, strict=True):
    box(axis, x, split_y, private_w, split_h, label, ROUTED, fontsize=8.8)
annotation(axis, shared_x + shared_w / 2, 0.625, "width $sD$")
annotation(axis, 0.63, 0.625, "private width $(1-s)D$ each")
selected_outline(axis, shared_x, split_y, shared_w, split_h)
arrow(axis, (0.5, 0.185), (shared_x + shared_w / 2, split_y), connectionstyle="arc3,rad=-0.28")
arrow(axis, (shared_x + shared_w / 2, split_y + split_h), (0.5, 0.751), connectionstyle="arc3,rad=0.20")
for index, curve in ((0, 0.12), (3, -0.12)):
    selected_outline(axis, private_xs[index], split_y, private_w, split_h)
    connect_selected(axis, private_xs[index] + private_w / 2, split_y, split_y + split_h, curve)
annotation(axis, 0.5, 0.695, "$S(x)+\\sum_i p_iP_i(x)$", fontsize=8.2)

# Panel separators and legend.
for x in (0.347, 0.678):
    fig.add_artist(
        Line2D(
            [x, x],
            [0.19, 0.91],
            transform=fig.transFigure,
            color="#94A3B8",
            linewidth=1.0,
            linestyle=(0, (5, 4)),
        )
    )

legend_handles = [
    Patch(facecolor=ROUTED, edgecolor=INK, label="Routed expert"),
    Patch(facecolor=SHARED, edgecolor=INK, label="Always-active shared expert"),
    FancyBboxPatch(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor=SELECTED,
        linewidth=1.6,
        linestyle=(0, (3, 2)),
        label="Active path",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=3,
    frameon=False,
    fontsize=8.5,
    handlelength=2.0,
    columnspacing=1.8,
)
fig.text(
    0.5,
    0.018,
    "$\\Sigma$ = router-weighted routed sum"
    "     |     "
    "$\\oplus$ = shared output + router-weighted routed sum",
    ha="center",
    va="center",
    fontsize=8.5,
    color=INK,
)

output = Path("results/architecture.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(output)
