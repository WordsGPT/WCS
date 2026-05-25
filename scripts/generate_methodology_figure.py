#!/usr/bin/env python
"""
Generate a fresh, publication-quality methodology figure for the WCS paper.
Divided into 4 vertically stacked stages, designed to fit in half a page or less.
Uses a clean, robust layout system with headers positioned above stage content
to prevent horizontal overlaps and clipping.
"""

from __future__ import annotations
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]

# Typography & Colors matching academic/journal standards
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight"
})

# Aesthetic Palette
BLACK = "#0f172a"
MUTED = "#64748b"
LIGHT_GREY = "#cbd5e1"
GRID_COLOR = "#f8fafc"
BORDER_COLOR = "#e2e8f0"

BLUE_LINE = "#2563eb"
BLUE_FILL = "#eff6ff"
BLUE_BORDER = "#bfdbfe"

GREEN_LINE = "#16a34a"
GREEN_FILL = "#f0fdf4"
GREEN_BORDER = "#bbf7d0"

RED_LINE = "#dc2626"
RED_FILL = "#fef2f2"
RED_BORDER = "#fecaca"

# Target words and metadata
TARGET_WORDS = [
    {"word": "offenders", "rank": 10104, "count": 4976449, "prefix": "... Dessave to hold an inquiry and punish the", "suffix": ""},
    {"word": "tyranny", "rank": 19906, "count": 1616527, "prefix": "... perverted into instruments of", "suffix": " and oppression"},
    {"word": "dubious", "rank": 23071, "count": 1252536, "prefix": "... outlook for his success seemed as", "suffix": " and uncertain"},
    {"word": "circulate", "rank": 26691, "count": 997906, "prefix": "... write to, when he began to", "suffix": " his letters of inquiry"},
    {"word": "precipitated", "rank": 39823, "count": 526227, "prefix": "... several months was suddenly", "suffix": " by the crisis"}
]

def load_norvig_frequency(path: Path, max_rank: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    ranks, counts = [], []
    if not path.exists():
        ranks = np.arange(1, max_rank + 1)
        counts = 1e9 / ranks
        return ranks, counts
    with path.open(encoding="utf-8") as f:
        for r, line in enumerate(f, start=1):
            if r > max_rank:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                count = int(parts[-1])
                ranks.append(r)
                counts.append(count)
            except ValueError:
                continue
    return np.asarray(ranks), np.asarray(counts)

def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.spines[["left", "bottom"]].set_linewidth(0.4)
    ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, width=0.4, pad=1.5)
    ax.grid(True, color=GRID_COLOR, linewidth=0.3, zorder=0)

def draw_stage_header(fig: plt.Figure, spec, number: int, title: str, subtitle: str) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Circle badge
    ax.text(0.01, 0.5, f"{number}", ha="center", va="center", fontsize=7.5, weight="bold", color="white",
            bbox={"boxstyle": "circle,pad=0.22", "facecolor": BLACK, "edgecolor": "none"})
    # Title
    ax.text(0.035, 0.5, title, ha="left", va="center", fontsize=8.0, weight="bold", color=BLACK)
    # Subtitle (Aligned vertically across all rows)
    ax.text(0.22, 0.5, subtitle, ha="left", va="center", fontsize=6.2, color=MUTED)

def draw_stage1(fig: plt.Figure, spec, ranks: np.ndarray, counts: np.ndarray) -> None:
    """1. Lexical Selection: Highlight rank band 10k-40k on Norvig list and show sampled words."""
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)

    # Left: log-log Zipf plot
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(ranks, counts, color=BLACK, linewidth=0.8, zorder=2)
    # Highlight 10k-40k band
    ax.axvspan(10_000, 40_000, facecolor=BLUE_FILL, edgecolor=BLUE_BORDER, linewidth=0.4, zorder=1)
    
    # Plot target words as dots
    for word in TARGET_WORDS:
        ax.scatter([word["rank"]], [word["count"]], s=6, facecolor=BLUE_LINE, edgecolor="none", zorder=3)
    
    ax.set_xlim(1_000, 100_000)
    ax.set_ylim(5e4, 5e7)
    ax.xaxis.set_major_locator(FixedLocator([10_000, 40_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.yaxis.set_major_locator(FixedLocator([100_000, 1_000_000, 10_000_000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}k"))
    
    ax.set_xlabel("Frequency Rank", fontsize=5.0, labelpad=2, color=MUTED)
    ax.set_ylabel("Count", fontsize=5.0, labelpad=2, color=MUTED)
    clean_axis(ax)
    
    # Label the band inside the highlight region
    ax.text(20_000, 2e6, "10k-40k\nband", ha="center", va="center", fontsize=5.0, color=BLUE_LINE, weight="bold")

    # Right: Word chips with ranks
    word_ax = fig.add_subplot(gs[0, 1])
    word_ax.axis("off")
    word_ax.set_xlim(0, 1)
    word_ax.set_ylim(0, 1)
    
    word_ax.text(0.01, 0.82, "Target Word Selection (Middle-Long Tail):", fontsize=5.5, weight="bold", color=BLACK)
    for i, word in enumerate(TARGET_WORDS):
        x = 0.01 + i * 0.198
        # Chip background box
        rect = FancyBboxPatch((x, 0.18), 0.178, 0.48, boxstyle="round,pad=0.002,rounding_size=0.008",
                              facecolor=BLUE_FILL, edgecolor=BLUE_BORDER, linewidth=0.4)
        word_ax.add_patch(rect)
        # Word text
        word_ax.text(x + 0.089, 0.48, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=BLUE_LINE)
        # Rank text
        word_ax.text(x + 0.089, 0.30, f"rank {word['rank']:,}", ha="center", va="center", fontsize=4.5, color=MUTED)

def draw_stage2(fig: plt.Figure, spec) -> None:
    """2. Context Pairing: Show C -> w pairing with column alignment."""
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.text(0.01, 0.94, r"Prefix / Context $C$ $\rightarrow$ Target Word $w$ (PG-19 Source):", fontsize=5.5, weight="bold", color=BLACK)
    
    y_coords = np.linspace(0.78, 0.12, 5)
    for i, word in enumerate(TARGET_WORDS):
        y = y_coords[i]
        
        # Column 1: Context prefix (Right-aligned at x=0.64)
        ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.5, family="serif", color=MUTED)
        
        # Column 2: Arrow indicator (Centered at x=0.675)
        ax.text(0.675, y, r"$\rightarrow$", ha="center", va="center", fontsize=5.5, color=MUTED)
        
        # Column 3: Highlighted Target Word Pill (Left-aligned at x=0.71)
        rect = FancyBboxPatch((0.705, y - 0.07), 0.11, 0.14, boxstyle="round,pad=0.002,rounding_size=0.005",
                              facecolor=BLUE_FILL, edgecolor=BLUE_BORDER, linewidth=0.4)
        ax.add_patch(rect)
        ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=BLUE_LINE, family="serif")
        
        # Column 4: Suffix if present (Left-aligned at x=0.835)
        if word["suffix"]:
            ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.5, family="serif", color=MUTED)

def draw_stage3(fig: plt.Figure, spec) -> None:
    """3. Sampler Audit: Explain thresholding mechanics with clean probability distributions."""
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
    
    # Generate synthetic token probabilities
    np.random.seed(42)
    x = np.arange(1, 13)
    # Exponential decay
    y = np.exp(-x * 0.25)
    y /= y.sum()
    
    # Target word w is at rank 9
    w_idx = 8
    w_x = 9
    w_y = y[w_idx]

    # 1. Top-k Sampler (k=6)
    k_ax = fig.add_subplot(gs[0, 0])
    k_val = 6
    colors_k = [GREEN_FILL if r <= k_val else RED_FILL for r in x]
    edges_k = [GREEN_LINE if r <= k_val else RED_LINE for r in x]
    k_ax.bar(x, y, color=colors_k, edgecolor=edges_k, linewidth=0.4, width=0.7, zorder=2)
    k_ax.axvline(k_val + 0.5, color=MUTED, linestyle="--", linewidth=0.6, zorder=3)
    k_ax.text(k_val + 0.5, y[0]*0.9, f" k={k_val}", fontsize=4.8, color=MUTED, ha="left")
    
    k_ax.scatter([w_x], [w_y], s=12, color=RED_LINE, zorder=4)
    k_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED_LINE)
    k_ax.text(w_x, w_y + 0.038, "Pruned", ha="center", fontsize=4.5, color=RED_LINE, weight="bold")
    k_ax.set_title("top-k", fontsize=6.5, weight="bold", pad=3, color=BLACK)
    k_ax.set_ylabel("prob.", fontsize=5.0, labelpad=3, color=MUTED)
    clean_axis(k_ax)
    
    # 2. Top-p Sampler (p=0.90)
    p_ax = fig.add_subplot(gs[0, 1])
    p_val = 0.90
    cum_y = np.cumsum(y)
    cutoff = np.where(cum_y >= p_val)[0][0] + 1
    colors_p = [GREEN_FILL if r <= cutoff else RED_FILL for r in x]
    edges_p = [GREEN_LINE if r <= cutoff else RED_LINE for r in x]
    p_ax.bar(x, y, color=colors_p, edgecolor=edges_p, linewidth=0.4, width=0.7, zorder=2)
    p_ax.axvline(cutoff + 0.5, color=MUTED, linestyle="--", linewidth=0.6, zorder=3)
    p_ax.text(cutoff + 0.5, y[0]*0.9, f" p={p_val}", fontsize=4.8, color=MUTED, ha="left")
    p_ax.plot(x, cum_y * y[0]*0.8, color=BLUE_LINE, linewidth=0.6, marker="o", markersize=1.0, zorder=4)
    
    p_ax.scatter([w_x], [w_y], s=12, color=GREEN_LINE, zorder=4)
    p_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=GREEN_LINE)
    p_ax.text(w_x, w_y + 0.038, "Kept", ha="center", fontsize=4.5, color=GREEN_LINE, weight="bold")
    p_ax.set_title("top-p (nucleus)", fontsize=6.5, weight="bold", pad=3, color=BLACK)
    clean_axis(p_ax)
    p_ax.set_yticklabels([])

    # 3. Min-p Sampler (m=0.15)
    m_ax = fig.add_subplot(gs[0, 2])
    m_val = 0.15
    threshold = m_val * y[0]
    colors_m = [GREEN_FILL if val >= threshold else RED_FILL for val in y]
    edges_m = [GREEN_LINE if val >= threshold else RED_LINE for val in y]
    m_ax.bar(x, y, color=colors_m, edgecolor=edges_m, linewidth=0.4, width=0.7, zorder=2)
    m_ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=0.6, zorder=3)
    m_ax.text(12, threshold + 0.008, r"$m \cdot p_{max}$", fontsize=4.8, color=MUTED, ha="right")
    
    m_ax.scatter([w_x], [w_y], s=12, color=RED_LINE, zorder=4)
    m_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED_LINE)
    m_ax.text(w_x, w_y + 0.038, "Pruned", ha="center", fontsize=4.5, color=RED_LINE, weight="bold")
    m_ax.set_title("min-p", fontsize=6.5, weight="bold", pad=3, color=BLACK)
    clean_axis(m_ax)
    m_ax.set_yticklabels([])

    for ax in (k_ax, p_ax, m_ax):
        ax.set_xlim(0.3, 12.7)
        ax.set_ylim(0, y[0] * 1.35)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel("Token Rank", fontsize=5.0, labelpad=2, color=MUTED)

def draw_stage4(fig: plt.Figure, spec) -> None:
    """4. WCS Measurement: Conceptual Overview of aggregation formula and matrix."""
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
    
    # Left: Math Aggregation Overview
    math_ax = fig.add_subplot(gs[0, 0])
    math_ax.axis("off")
    math_ax.set_xlim(0, 1)
    math_ax.set_ylim(0, 1)
    
    math_ax.text(0.01, 0.85, "1. Next-Token Reachability:", fontsize=5.5, weight="bold", color=BLACK)
    math_ax.text(0.08, 0.70, r"$\mathcal{R}_{\theta}(w, c) \in \{0, 1\}$", fontsize=7.5, color=BLACK)
    math_ax.text(0.08, 0.56, r"(1 if target $w$ survives filter $\theta$ in context $c$, else 0)", fontsize=4.8, color=MUTED)
    
    math_ax.text(0.01, 0.38, "2. Corpus-Level Average Score:", fontsize=5.5, weight="bold", color=BLACK)
    math_ax.text(0.08, 0.20, r"$\text{WCS}_{\theta} = \frac{1}{|\mathcal{D}|} \sum_{(w,c)\in\mathcal{D}} \mathcal{R}_{\theta}(w,c)$", fontsize=7.5, color=BLACK)

    # Right: Coverage Matrix
    mat_ax = fig.add_subplot(gs[0, 1])
    mat_ax.axis("off")
    mat_ax.set_xlim(0, 1)
    mat_ax.set_ylim(0, 1)
    
    mat_ax.text(0.01, 0.85, "Conceptual Coverage Matrix:", fontsize=5.5, weight="bold", color=BLACK)
    
    # 5 targets x 5 contexts
    matrix_data = [
        [1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ]
    
    x_coords = np.linspace(0.35, 0.75, 5)
    y_coords = np.linspace(0.64, 0.16, 5)
    
    # Header column labels
    for c in range(5):
        mat_ax.text(x_coords[c], 0.76, f"$c_{c+1}$", ha="center", fontsize=4.8, color=MUTED)
    mat_ax.text(0.89, 0.76, r"$\text{WCS}_w$", ha="center", fontsize=4.8, color=MUTED, weight="bold")
        
    for r, word in enumerate(TARGET_WORDS):
        y = y_coords[r]
        mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0, color=BLACK)
        successes = 0
        for c in range(5):
            val = matrix_data[r][c]
            face, edge, sym, sym_y = (GREEN_FILL, GREEN_BORDER, r"$\checkmark$", y) if val == 1 else (RED_FILL, RED_BORDER, r"$\times$", y)
            rect = FancyBboxPatch((x_coords[c] - 0.03, y - 0.05), 0.06, 0.10, boxstyle="round,pad=0.001,rounding_size=0.002",
                                  facecolor=face, edgecolor=edge, linewidth=0.4)
            mat_ax.add_patch(rect)
            mat_ax.text(x_coords[c], sym_y, sym, ha="center", va="center", fontsize=5.0, color=edge, weight="bold")
            if val == 1:
                successes += 1
        w_score = f"{int((successes/5)*100)}%"
        mat_ax.text(0.89, y, w_score, ha="center", va="center", fontsize=5.0, color=BLACK, weight="bold")

    # Bottom average row
    mat_ax.plot([0.30, 0.96], [0.08, 0.08], color=LIGHT_GREY, linewidth=0.4)
    mat_ax.text(0.01, 0.02, "Average WCS Score", ha="left", va="center", fontsize=5.0, color=BLACK, weight="bold")
    mat_ax.text(0.89, 0.02, "52%", ha="center", va="center", fontsize=5.0, color=BLUE_LINE, weight="bold")

def add_separators(fig: plt.Figure, grid) -> None:
    # Add horizontal lines separating the stages centered between rows
    for i in range(3):
        pos_above = grid[i, 0].get_position(fig)
        pos_below = grid[i+1, 0].get_position(fig)
        y = (pos_above.y0 + pos_below.y1) / 2.0
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=BORDER_COLOR, linewidth=0.4))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, default=ROOT / "data/raw/norvig_count_1w.txt")
    parser.add_argument("--output-png", type=Path, default=ROOT / "figs/wcs_methodology_paper.png")
    parser.add_argument("--output-pdf", type=Path, default=ROOT / "figs/wcs_methodology_paper.pdf")
    parser.add_argument("--output-svg", type=Path, default=ROOT / "figs/wcs_methodology_paper.svg")
    args = parser.parse_args()

    # Load Norvig counts
    ranks, counts = load_norvig_frequency(args.frequency)

    # 7.15 inches wide (fits half-page standard double column or full-width single column), 6.0 inches height
    fig = plt.figure(figsize=(7.15, 6.0), dpi=350)
    fig.patch.set_facecolor("white")
    
    # 4 rows of gridspec representing the 4 stages
    grid = GridSpec(4, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.04, hspace=0.25)

    for i in range(4):
        # Split each stage row into header (18%) and content (82%)
        stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.18, 0.82], hspace=0.08)
        
        # Draw Stage Header
        if i == 0:
            draw_stage_header(fig, stage_gs[0, 0], 1, "Lexical Selection", "— Identify middle-long tail target lexicon")
            draw_stage1(fig, stage_gs[1, 0], ranks, counts)
        elif i == 1:
            draw_stage_header(fig, stage_gs[0, 0], 2, "Context Pairing", "— Map target words back into human passages")
            draw_stage2(fig, stage_gs[1, 0])
        elif i == 2:
            draw_stage_header(fig, stage_gs[0, 0], 3, "Sampler Audit", "— Verify if target word survives generation filters")
            draw_stage3(fig, stage_gs[1, 0])
        elif i == 3:
            draw_stage_header(fig, stage_gs[0, 0], 4, "WCS Measurement", "— Aggregate target word reachability to compute WCS")
            draw_stage4(fig, stage_gs[1, 0])

    # Add separators
    add_separators(fig, grid)

    # Save outputs
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.output_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.output_svg, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Successfully generated figure: {args.output_png}")
    print(f"Successfully generated figure: {args.output_pdf}")
    print(f"Successfully generated figure: {args.output_svg}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
