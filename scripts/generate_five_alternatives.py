#!/usr/bin/env python
"""
Generate 5 visually distinct alternative graphic designs for the WCS paper methodology figure:
1. Dark Mode Tech Dashboard (wcs_alt_dark_mode)
2. Swiss Minimalist / Helvetica Style (wcs_alt_minimalist)
3. Hand-Drawn / Comic Style (wcs_alt_xkcd)
4. Stepped Pipeline / Flowchart Layout (wcs_alt_pipeline)
5. Warm Retro Serif / Editorial Style (wcs_alt_retro_editorial)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Circle, BoxStyle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]

# Shared word data
TARGET_WORDS = [
    {"word": "offenders", "rank": 10104, "count": 4976449, "prefix": "... Dessave to hold an inquiry and punish the", "suffix": ""},
    {"word": "tyranny", "rank": 19906, "count": 1616527, "prefix": "... perverted into instruments of", "suffix": " and oppression"},
    {"word": "dubious", "rank": 23071, "count": 1252536, "prefix": "... outlook for his success seemed as", "suffix": " and uncertain"},
    {"word": "circulate", "rank": 26691, "count": 997906, "prefix": "... write to, when he began to", "suffix": " his letters of inquiry"},
    {"word": "precipitated", "rank": 39823, "count": 526227, "prefix": "... several months was suddenly", "suffix": " by the crisis"}
]

# Generate synthetic token probabilities for Stage 3
np.random.seed(42)
x_tokens = np.arange(1, 13)
y_tokens = np.exp(-x_tokens * 0.25)
y_tokens /= y_tokens.sum()
w_x = 9
w_y = y_tokens[8]

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

def draw_header_helper(fig: plt.Figure, spec, number: int, title: str, subtitle: str, 
                       bg_color: str, text_color: str, muted_color: str) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Circle badge
    ax.text(0.01, 0.5, f"{number}", ha="center", va="center", fontsize=7.5, weight="bold", color=bg_color,
            bbox={"boxstyle": "circle,pad=0.22", "facecolor": text_color, "edgecolor": "none"})
    # Title
    ax.text(0.035, 0.5, title, ha="left", va="center", fontsize=8.0, weight="bold", color=text_color)
    # Subtitle
    ax.text(0.22, 0.5, subtitle, ha="left", va="center", fontsize=6.2, color=muted_color)

# ==========================================
# VARIANT 1: Sleek Dark Mode / Tech Dashboard
# ==========================================
def build_dark_mode(ranks, counts, out_png) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "savefig.bbox": "tight"
    })
    
    BG = "#0f172a"
    PANEL_BG = "#1e293b"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    BORDER = "#334155"
    LINE = "#38bdf8"
    LINE_FILL = "#075985"
    GREEN = "#4ade80"
    GREEN_F = "#14532d"
    RED = "#f43f5e"
    RED_F = "#4c0519"

    fig = plt.figure(figsize=(7.15, 6.0), dpi=300)
    fig.patch.set_facecolor(BG)
    grid = GridSpec(4, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.04, hspace=0.25)

    for i in range(4):
        stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.18, 0.82], hspace=0.08)
        
        # Header
        titles = [
            ("Lexical Selection", "— Identify middle-long tail target lexicon"),
            ("Context Pairing", "— Map target words back into human passages"),
            ("Sampler Audit", "— Verify if target word survives generation filters"),
            ("WCS Measurement", "— Aggregate target word reachability to compute WCS")
        ]
        draw_header_helper(fig, stage_gs[0, 0], i+1, titles[i][0], titles[i][1], BG, TEXT, MUTED)

        # Content
        spec = stage_gs[1, 0]
        if i == 0:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)
            # Zipf
            ax = fig.add_subplot(gs[0, 0])
            ax.loglog(ranks, counts, color=TEXT, linewidth=0.8, zorder=2)
            ax.axvspan(10_000, 40_000, facecolor=LINE_FILL, edgecolor=LINE, linewidth=0.4, alpha=0.3, zorder=1)
            for word in TARGET_WORDS:
                ax.scatter([word["rank"]], [word["count"]], s=6, facecolor=LINE, edgecolor="none", zorder=3)
            ax.set_xlim(1_000, 100_000)
            ax.set_ylim(5e4, 5e7)
            ax.xaxis.set_major_locator(FixedLocator([10_000, 40_000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(FixedLocator([100_000, 1_000_000, 10_000_000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}k"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            ax.grid(True, color=BORDER, linewidth=0.3, zorder=0)
            ax.text(20_000, 2e6, "10k-40k", ha="center", va="center", fontsize=5.0, color=LINE, weight="bold")
            
            # Words
            word_ax = fig.add_subplot(gs[0, 1])
            word_ax.axis("off")
            word_ax.set_xlim(0, 1)
            word_ax.set_ylim(0, 1)
            word_ax.text(0.01, 0.82, "Target Word Selection:", fontsize=5.5, weight="bold", color=TEXT)
            for j, word in enumerate(TARGET_WORDS):
                x = 0.01 + j * 0.198
                rect = FancyBboxPatch((x, 0.18), 0.178, 0.48, boxstyle="round,pad=0.002,rounding_size=0.008",
                                      facecolor=PANEL_BG, edgecolor=BORDER, linewidth=0.4)
                word_ax.add_patch(rect)
                word_ax.text(x + 0.089, 0.48, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=LINE)
                word_ax.text(x + 0.089, 0.30, f"rank {word['rank']:,}", ha="center", va="center", fontsize=4.5, color=MUTED)

        elif i == 1:
            ax = fig.add_subplot(spec)
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            y_coords = np.linspace(0.78, 0.12, 5)
            for j, word in enumerate(TARGET_WORDS):
                y = y_coords[j]
                ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.5, color=MUTED)
                ax.text(0.675, y, r"$\rightarrow$", ha="center", va="center", fontsize=5.5, color=MUTED)
                rect = FancyBboxPatch((0.705, y - 0.07), 0.11, 0.14, boxstyle="round,pad=0.002,rounding_size=0.005",
                                      facecolor=PANEL_BG, edgecolor=BORDER, linewidth=0.4)
                ax.add_patch(rect)
                ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=LINE)
                if word["suffix"]:
                    ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.5, color=MUTED)

        elif i == 2:
            gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
            # top-k
            k_ax = fig.add_subplot(gs[0, 0])
            colors_k = [GREEN if r <= 6 else RED_F for r in x_tokens]
            edges_k = [GREEN if r <= 6 else RED for r in x_tokens]
            k_ax.bar(x_tokens, y_tokens, color=colors_k, edgecolor=edges_k, linewidth=0.4, width=0.7, zorder=2)
            k_ax.axvline(6.5, color=MUTED, linestyle="--", linewidth=0.6)
            k_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            k_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED)
            k_ax.set_title("top-k", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            k_ax.set_ylabel("prob.", fontsize=5.0, labelpad=3, color=MUTED)
            k_ax.spines[["top", "right"]].set_visible(False)
            k_ax.spines[["left", "bottom"]].set_color(BORDER)
            k_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            k_ax.set_xlim(0.3, 12.7)
            k_ax.set_ylim(0, y_tokens[0] * 1.35)
            k_ax.set_xticks([1, 4, 8, 12])
            k_ax.grid(True, color=BORDER, linewidth=0.3, zorder=0)

            # top-p
            p_ax = fig.add_subplot(gs[0, 1])
            colors_p = [GREEN if r <= 9 else RED_F for r in x_tokens]
            edges_p = [GREEN if r <= 9 else RED for r in x_tokens]
            p_ax.bar(x_tokens, y_tokens, color=colors_p, edgecolor=edges_p, linewidth=0.4, width=0.7, zorder=2)
            p_ax.axvline(9.5, color=MUTED, linestyle="--", linewidth=0.6)
            p_ax.scatter([w_x], [w_y], s=12, color=GREEN, zorder=4)
            p_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=GREEN)
            p_ax.set_title("top-p", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            p_ax.spines[["top", "right"]].set_visible(False)
            p_ax.spines[["left", "bottom"]].set_color(BORDER)
            p_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            p_ax.set_xlim(0.3, 12.7)
            p_ax.set_ylim(0, y_tokens[0] * 1.35)
            p_ax.set_xticks([1, 4, 8, 12])
            p_ax.set_yticklabels([])
            p_ax.grid(True, color=BORDER, linewidth=0.3, zorder=0)

            # min-p
            m_ax = fig.add_subplot(gs[0, 2])
            threshold = 0.15 * y_tokens[0]
            colors_m = [GREEN if val >= threshold else RED_F for val in y_tokens]
            edges_m = [GREEN if val >= threshold else RED for val in y_tokens]
            m_ax.bar(x_tokens, y_tokens, color=colors_m, edgecolor=edges_m, linewidth=0.4, width=0.7, zorder=2)
            m_ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=0.6)
            m_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            m_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED)
            m_ax.set_title("min-p", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            m_ax.spines[["top", "right"]].set_visible(False)
            m_ax.spines[["left", "bottom"]].set_color(BORDER)
            m_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            m_ax.set_xlim(0.3, 12.7)
            m_ax.set_ylim(0, y_tokens[0] * 1.35)
            m_ax.set_xticks([1, 4, 8, 12])
            m_ax.set_yticklabels([])
            m_ax.grid(True, color=BORDER, linewidth=0.3, zorder=0)

        elif i == 3:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
            # Math
            math_ax = fig.add_subplot(gs[0, 0])
            math_ax.axis("off")
            math_ax.set_xlim(0, 1)
            math_ax.set_ylim(0, 1)
            math_ax.text(0.01, 0.85, "1. Next-Token Reachability:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.70, r"$\mathcal{R}_{\theta}(w, c) \in \{0, 1\}$", fontsize=7.5, color=TEXT)
            math_ax.text(0.01, 0.38, "2. Corpus-Level Average Score:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.20, r"$\text{WCS}_{\theta} = \frac{1}{|\mathcal{D}|} \sum \mathcal{R}_{\theta}(w,c)$", fontsize=7.5, color=TEXT)
            
            # Matrix
            mat_ax = fig.add_subplot(gs[0, 1])
            mat_ax.axis("off")
            mat_ax.set_xlim(0, 1)
            mat_ax.set_ylim(0, 1)
            mat_ax.text(0.01, 0.85, "Conceptual Coverage Matrix:", fontsize=5.5, weight="bold", color=TEXT)
            matrix_data = [
                [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0]
            ]
            x_coords = np.linspace(0.35, 0.75, 5)
            y_coords = np.linspace(0.64, 0.16, 5)
            for c in range(5):
                mat_ax.text(x_coords[c], 0.76, f"$c_{c+1}$", ha="center", fontsize=4.8, color=MUTED)
            mat_ax.text(0.89, 0.76, r"$\text{WCS}_w$", ha="center", fontsize=4.8, color=MUTED, weight="bold")
            for r, word in enumerate(TARGET_WORDS):
                y = y_coords[r]
                mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0, color=TEXT)
                successes = 0
                for c in range(5):
                    val = matrix_data[r][c]
                    face, edge, sym = (GREEN_F, GREEN, r"$\checkmark$") if val == 1 else (RED_F, RED, r"$\times$")
                    rect = FancyBboxPatch((x_coords[c] - 0.03, y - 0.05), 0.06, 0.10, boxstyle="round,pad=0.001",
                                          facecolor=face, edgecolor=edge, linewidth=0.4)
                    mat_ax.add_patch(rect)
                    mat_ax.text(x_coords[c], y, sym, ha="center", va="center", fontsize=5.0, color=edge)
                    if val == 1: successes += 1
                mat_ax.text(0.89, y, f"{int((successes/5)*100)}%", ha="center", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.plot([0.30, 0.96], [0.08, 0.08], color=BORDER, linewidth=0.4)
            mat_ax.text(0.01, 0.02, "Average WCS Score", ha="left", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.text(0.89, 0.02, "52%", ha="center", va="center", fontsize=5.0, color=LINE, weight="bold")

    # Separators
    for j in range(3):
        pos_above = grid[j, 0].get_position(fig)
        pos_below = grid[j+1, 0].get_position(fig)
        y = (pos_above.y0 + pos_below.y1) / 2.0
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=BORDER, linewidth=0.4))

    fig.savefig(out_png, facecolor=BG, edgecolor="none", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

# ==========================================
# VARIANT 2: Swiss Minimalist / Helvetica Style
# ==========================================
def build_minimalist(ranks, counts, out_png) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight"
    })
    
    TEXT = "#09090b"
    MUTED = "#71717a"
    BORDER = "#e4e4e7"
    ACCENT = "#2563eb"
    GREEN = "#16a34a"
    RED = "#dc2626"

    fig = plt.figure(figsize=(7.15, 6.0), dpi=300)
    grid = GridSpec(4, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.04, hspace=0.25)

    for i in range(4):
        stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.18, 0.82], hspace=0.08)
        
        # Header
        titles = [
            ("Lexical Selection", "— Filter vocabulary ranks 10k–40k"),
            ("Context Pairing", "— Align targets with source prefixes"),
            ("Sampler Audit", "— Process tokens through decoder filters"),
            ("WCS Measurement", "— Compute average corpus-level reachability")
        ]
        draw_header_helper(fig, stage_gs[0, 0], i+1, titles[i][0], titles[i][1], "white", TEXT, MUTED)

        spec = stage_gs[1, 0]
        if i == 0:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)
            ax = fig.add_subplot(gs[0, 0])
            ax.loglog(ranks, counts, color=TEXT, linewidth=0.7, zorder=2)
            ax.axvspan(10_000, 40_000, facecolor="#f4f4f5", edgecolor=BORDER, linewidth=0.4, zorder=1)
            ax.set_xlim(1_000, 100_000)
            ax.set_ylim(5e4, 5e7)
            ax.xaxis.set_major_locator(FixedLocator([10_000, 40_000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
            ax.yaxis.set_major_locator(FixedLocator([100_000, 1_000_000, 10_000_000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}k"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            ax.grid(True, color="#fafafa", linewidth=0.3, zorder=0)
            
            # Words
            word_ax = fig.add_subplot(gs[0, 1])
            word_ax.axis("off")
            word_ax.set_xlim(0, 1)
            word_ax.set_ylim(0, 1)
            for j, word in enumerate(TARGET_WORDS):
                x = 0.01 + j * 0.198
                rect = plt.Rectangle((x, 0.18), 0.178, 0.48, fill=False, edgecolor=BORDER, linewidth=0.5)
                word_ax.add_patch(rect)
                word_ax.text(x + 0.089, 0.48, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=ACCENT)
                word_ax.text(x + 0.089, 0.30, f"rank {word['rank']:,}", ha="center", va="center", fontsize=4.5, color=MUTED)

        elif i == 1:
            ax = fig.add_subplot(spec)
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            y_coords = np.linspace(0.78, 0.12, 5)
            for j, word in enumerate(TARGET_WORDS):
                y = y_coords[j]
                ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.5, color=MUTED)
                ax.text(0.675, y, "|", ha="center", va="center", fontsize=5.5, color=BORDER)
                rect = plt.Rectangle((0.705, y - 0.07), 0.11, 0.14, fill=False, edgecolor=BORDER, linewidth=0.5)
                ax.add_patch(rect)
                ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=ACCENT)
                if word["suffix"]:
                    ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.5, color=MUTED)

        elif i == 2:
            gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
            # top-k
            k_ax = fig.add_subplot(gs[0, 0])
            colors_k = [ACCENT if r <= 6 else "#f4f4f5" for r in x_tokens]
            k_ax.bar(x_tokens, y_tokens, color=colors_k, edgecolor=MUTED, linewidth=0.3, width=0.7, zorder=2)
            k_ax.axvline(6.5, color=TEXT, linestyle="-", linewidth=0.5)
            k_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            k_ax.set_title("top-k (k=6)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            k_ax.spines[["top", "right"]].set_visible(False)
            k_ax.spines[["left", "bottom"]].set_color(MUTED)
            k_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            k_ax.set_xlim(0.3, 12.7)
            k_ax.set_ylim(0, y_tokens[0] * 1.35)
            k_ax.set_xticks([1, 4, 8, 12])
            k_ax.grid(True, color="#fafafa", linewidth=0.3, zorder=0)

            # top-p
            p_ax = fig.add_subplot(gs[0, 1])
            colors_p = [ACCENT if r <= 9 else "#f4f4f5" for r in x_tokens]
            p_ax.bar(x_tokens, y_tokens, color=colors_p, edgecolor=MUTED, linewidth=0.3, width=0.7, zorder=2)
            p_ax.axvline(9.5, color=TEXT, linestyle="-", linewidth=0.5)
            p_ax.scatter([w_x], [w_y], s=12, color=GREEN, zorder=4)
            p_ax.set_title("top-p (p=0.9)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            p_ax.spines[["top", "right"]].set_visible(False)
            p_ax.spines[["left", "bottom"]].set_color(MUTED)
            p_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            p_ax.set_xlim(0.3, 12.7)
            p_ax.set_ylim(0, y_tokens[0] * 1.35)
            p_ax.set_xticks([1, 4, 8, 12])
            p_ax.set_yticklabels([])
            p_ax.grid(True, color="#fafafa", linewidth=0.3, zorder=0)

            # min-p
            m_ax = fig.add_subplot(gs[0, 2])
            threshold = 0.15 * y_tokens[0]
            colors_m = [ACCENT if val >= threshold else "#f4f4f5" for val in y_tokens]
            m_ax.bar(x_tokens, y_tokens, color=colors_m, edgecolor=MUTED, linewidth=0.3, width=0.7, zorder=2)
            m_ax.axhline(threshold, color=TEXT, linestyle="-", linewidth=0.5)
            m_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            m_ax.set_title("min-p (m=0.15)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            m_ax.spines[["top", "right"]].set_visible(False)
            m_ax.spines[["left", "bottom"]].set_color(MUTED)
            m_ax.tick_params(colors=MUTED, labelsize=5.0, length=1.5, pad=1.5)
            m_ax.set_xlim(0.3, 12.7)
            m_ax.set_ylim(0, y_tokens[0] * 1.35)
            m_ax.set_xticks([1, 4, 8, 12])
            m_ax.set_yticklabels([])
            m_ax.grid(True, color="#fafafa", linewidth=0.3, zorder=0)

        elif i == 3:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
            math_ax = fig.add_subplot(gs[0, 0])
            math_ax.axis("off")
            math_ax.set_xlim(0, 1)
            math_ax.set_ylim(0, 1)
            math_ax.text(0.01, 0.85, "1. Reachability Formula:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.70, r"$R_\theta(w, c) \in \{0, 1\}$", fontsize=7.5, color=TEXT)
            math_ax.text(0.01, 0.38, "2. Corpus Score Accumulation:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.20, r"$\text{WCS}_\theta = \frac{1}{|\mathcal{D}|}\sum \text{reach}(w,c)$", fontsize=7.5, color=TEXT)
            
            mat_ax = fig.add_subplot(gs[0, 1])
            mat_ax.axis("off")
            mat_ax.set_xlim(0, 1)
            mat_ax.set_ylim(0, 1)
            matrix_data = [
                [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0]
            ]
            x_coords = np.linspace(0.35, 0.75, 5)
            y_coords = np.linspace(0.64, 0.16, 5)
            for c in range(5):
                mat_ax.text(x_coords[c], 0.76, f"$c_{c+1}$", ha="center", fontsize=4.8, color=MUTED)
            mat_ax.text(0.89, 0.76, "WCS", ha="center", fontsize=4.8, color=MUTED, weight="bold")
            for r, word in enumerate(TARGET_WORDS):
                y = y_coords[r]
                mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0, color=TEXT)
                successes = 0
                for c in range(5):
                    val = matrix_data[r][c]
                    face = "#e4e4e7" if val == 1 else "white"
                    rect = plt.Rectangle((x_coords[c] - 0.03, y - 0.05), 0.06, 0.10, facecolor=face, edgecolor=MUTED, linewidth=0.4)
                    mat_ax.add_patch(rect)
                    mat_ax.text(x_coords[c], y, "1" if val == 1 else "0", ha="center", va="center", fontsize=5.0, color=TEXT)
                    if val == 1: successes += 1
                mat_ax.text(0.89, y, f"{int((successes/5)*100)}%", ha="center", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.plot([0.30, 0.96], [0.08, 0.08], color=MUTED, linewidth=0.4)
            mat_ax.text(0.01, 0.02, "Global WCS Score", ha="left", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.text(0.89, 0.02, "52%", ha="center", va="center", fontsize=5.0, color=ACCENT, weight="bold")

    for j in range(3):
        pos_above = grid[j, 0].get_position(fig)
        pos_below = grid[j+1, 0].get_position(fig)
        y = (pos_above.y0 + pos_below.y1) / 2.0
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=BORDER, linewidth=0.4))

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

# ==========================================
# VARIANT 3: Hand-Drawn / Comic Style (XKCD)
# ==========================================
def build_xkcd(ranks, counts, out_png) -> None:
    # Run inside plt.xkcd() context manager to ensure local styling
    with plt.xkcd(scale=0.8, length=80, randomness=1.5):
        fig = plt.figure(figsize=(7.15, 6.2), dpi=250)
        grid = GridSpec(4, 1, figure=fig, left=0.04, right=0.96, top=0.97, bottom=0.04, hspace=0.3)

        titles = [
            ("Lexical Selection", "Select target words from Norvig frequency ranks"),
            ("Context Pairing", "Pair selected targets with PG-19 sentence templates"),
            ("Sampler Audit", "Filter token probabilities and test reachability"),
            ("WCS Measurement", "Score and aggregate context reachability")
        ]

        for i in range(4):
            stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.16, 0.84], hspace=0.1)
            
            # Header
            header_ax = fig.add_subplot(stage_gs[0, 0])
            header_ax.axis("off")
            header_ax.text(0.01, 0.5, f"{i+1}. {titles[i][0]} {titles[i][1]}", fontsize=7.5, weight="bold", color="black")

            spec = stage_gs[1, 0]
            if i == 0:
                gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)
                ax = fig.add_subplot(gs[0, 0])
                ax.plot(ranks[::100], counts[::100], color="black", linewidth=0.8) # sample to speed up sketchy rendering
                ax.axvspan(10_000, 40_000, color="#dbeafe", alpha=0.5)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(1_000, 100_000)
                ax.set_ylim(5e4, 5e7)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlabel("rank", fontsize=5.0)
                ax.set_ylabel("count", fontsize=5.0)
                ax.tick_params(labelsize=4.5)
                
                word_ax = fig.add_subplot(gs[0, 1])
                word_ax.axis("off")
                word_ax.set_xlim(0, 1)
                word_ax.set_ylim(0, 1)
                for j, word in enumerate(TARGET_WORDS):
                    x = 0.01 + j * 0.198
                    # Sketchy box
                    rect = plt.Rectangle((x, 0.18), 0.178, 0.48, fill=True, facecolor="#f3f4f6", edgecolor="black", linewidth=0.8)
                    word_ax.add_patch(rect)
                    word_ax.text(x + 0.089, 0.48, word["word"], ha="center", va="center", fontsize=5.0, weight="bold")
                    word_ax.text(x + 0.089, 0.30, f"rank {word['rank']}", ha="center", va="center", fontsize=4.0)

            elif i == 1:
                ax = fig.add_subplot(spec)
                ax.axis("off")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                y_coords = np.linspace(0.78, 0.12, 5)
                for j, word in enumerate(TARGET_WORDS):
                    y = y_coords[j]
                    ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.0)
                    ax.text(0.675, y, "->", ha="center", va="center", fontsize=5.0)
                    rect = plt.Rectangle((0.705, y - 0.07), 0.11, 0.14, fill=True, facecolor="#eff6ff", edgecolor="black", linewidth=0.8)
                    ax.add_patch(rect)
                    ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.0, weight="bold")
                    if word["suffix"]:
                        ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.0)

            elif i == 2:
                gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
                # top-k
                k_ax = fig.add_subplot(gs[0, 0])
                k_ax.bar(x_tokens, y_tokens, color="white", edgecolor="black", linewidth=0.8, width=0.7)
                k_ax.axvline(6.5, color="black", linestyle="--")
                k_ax.text(9, y_tokens[8]+0.02, "w", color="red", fontsize=6.0, weight="bold")
                k_ax.set_title("top-k", fontsize=6.5)
                k_ax.spines[["top", "right"]].set_visible(False)
                k_ax.set_xlim(0.3, 12.7)
                k_ax.set_ylim(0, y_tokens[0] * 1.35)
                k_ax.tick_params(labelsize=4.5)

                # top-p
                p_ax = fig.add_subplot(gs[0, 1])
                p_ax.bar(x_tokens, y_tokens, color="white", edgecolor="black", linewidth=0.8, width=0.7)
                p_ax.axvline(9.5, color="black", linestyle="--")
                p_ax.text(9, y_tokens[8]+0.02, "w", color="green", fontsize=6.0, weight="bold")
                p_ax.set_title("top-p", fontsize=6.5)
                p_ax.spines[["top", "right"]].set_visible(False)
                p_ax.set_xlim(0.3, 12.7)
                p_ax.set_ylim(0, y_tokens[0] * 1.35)
                p_ax.set_yticklabels([])
                p_ax.tick_params(labelsize=4.5)

                # min-p
                m_ax = fig.add_subplot(gs[0, 2])
                threshold = 0.15 * y_tokens[0]
                m_ax.bar(x_tokens, y_tokens, color="white", edgecolor="black", linewidth=0.8, width=0.7)
                m_ax.axhline(threshold, color="black", linestyle="--")
                m_ax.text(9, y_tokens[8]+0.02, "w", color="red", fontsize=6.0, weight="bold")
                m_ax.set_title("min-p", fontsize=6.5)
                m_ax.spines[["top", "right"]].set_visible(False)
                m_ax.set_xlim(0.3, 12.7)
                m_ax.set_ylim(0, y_tokens[0] * 1.35)
                m_ax.set_yticklabels([])
                m_ax.tick_params(labelsize=4.5)

            elif i == 3:
                gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
                math_ax = fig.add_subplot(gs[0, 0])
                math_ax.axis("off")
                math_ax.set_xlim(0, 1)
                math_ax.set_ylim(0, 1)
                math_ax.text(0.01, 0.80, "Reachability: R(w,c) = 1 or 0", fontsize=5.5, weight="bold")
                math_ax.text(0.01, 0.35, "Aggregate: Average of R(w,c)", fontsize=5.5, weight="bold")
                
                mat_ax = fig.add_subplot(gs[0, 1])
                mat_ax.axis("off")
                mat_ax.set_xlim(0, 1)
                mat_ax.set_ylim(0, 1)
                matrix_data = [
                    [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0]
                ]
                x_coords = np.linspace(0.35, 0.75, 5)
                y_coords = np.linspace(0.64, 0.16, 5)
                for c in range(5):
                    mat_ax.text(x_coords[c], 0.76, f"c{c+1}", ha="center", fontsize=4.8)
                mat_ax.text(0.89, 0.76, "WCS", ha="center", fontsize=4.8)
                for r, word in enumerate(TARGET_WORDS):
                    y = y_coords[r]
                    mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0)
                    for c in range(5):
                        val = matrix_data[r][c]
                        face = "#dcfce7" if val == 1 else "#fee2e2"
                        rect = plt.Rectangle((x_coords[c] - 0.03, y - 0.05), 0.06, 0.10, facecolor=face, edgecolor="black", linewidth=0.8)
                        mat_ax.add_patch(rect)
                        mat_ax.text(x_coords[c], y, "Y" if val == 1 else "N", ha="center", va="center", fontsize=5.0)
                    mat_ax.text(0.89, y, f"{int((sum(matrix_data[r])/5)*100)}%", ha="center", va="center", fontsize=5.0)

        for j in range(3):
            pos_above = grid[j, 0].get_position(fig)
            pos_below = grid[j+1, 0].get_position(fig)
            y = (pos_above.y0 + pos_below.y1) / 2.0
            fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color="black", linewidth=0.5))

        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

# ==========================================
# VARIANT 4: Stepped Pipeline / Flowchart Layout
# ==========================================
def build_pipeline(ranks, counts, out_png) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "savefig.bbox": "tight"
    })
    
    TEXT = "#0f172a"
    MUTED = "#64748b"
    BORDER = "#e2e8f0"
    BLUE = "#3b82f6"
    BLUE_F = "#eff6ff"
    GREEN = "#22c55e"
    RED = "#ef4444"

    fig = plt.figure(figsize=(7.15, 6.0), dpi=300)
    grid = GridSpec(4, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.04, hspace=0.25)

    for i in range(4):
        stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.18, 0.82], hspace=0.08)
        
        # Header
        titles = [
            ("Stage 1: Lexical Funnel", "— Filtering words from large frequency databases"),
            ("Stage 2: Context Merger", "— Pairing target lexicon with human corpus text tracks"),
            ("Stage 3: Filter Gate", "— Deciding token survival via standard decoding thresholds"),
            ("Stage 4: Metric Assembly", "— Summing reachability outputs to compile WCS score")
        ]
        draw_header_helper(fig, stage_gs[0, 0], i+1, titles[i][0], titles[i][1], "white", TEXT, MUTED)

        spec = stage_gs[1, 0]
        if i == 0:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(ranks, counts, color=TEXT, linewidth=0.8, zorder=2)
            ax.fill_between(ranks, counts, where=(ranks >= 10_000) & (ranks <= 40_000), color=BLUE, alpha=0.15, zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1_000, 100_000)
            ax.set_ylim(5e4, 5e7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=5.0)
            ax.set_xlabel("Rank", fontsize=5.0)
            ax.set_ylabel("Freq", fontsize=5.0)
            ax.text(20_000, 2e6, "Funnel Band", ha="center", va="center", fontsize=5.0, color=BLUE, weight="bold")
            
            # Words
            word_ax = fig.add_subplot(gs[0, 1])
            word_ax.axis("off")
            word_ax.set_xlim(0, 1)
            word_ax.set_ylim(0, 1)
            # Draw funnel pipeline lines
            word_ax.plot([0, 0.05], [0.5, 0.5], color=BLUE, linewidth=1.0, linestyle="--")
            for j, word in enumerate(TARGET_WORDS):
                x = 0.08 + j * 0.185
                rect = FancyBboxPatch((x, 0.22), 0.165, 0.44, boxstyle="round,pad=0.002",
                                      facecolor=BLUE_F, edgecolor=BLUE, linewidth=0.5)
                word_ax.add_patch(rect)
                word_ax.text(x + 0.082, 0.52, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=BLUE)
                word_ax.text(x + 0.082, 0.36, f"r={word['rank']}", ha="center", va="center", fontsize=4.5, color=MUTED)

        elif i == 1:
            ax = fig.add_subplot(spec)
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            y_coords = np.linspace(0.78, 0.12, 5)
            for j, word in enumerate(TARGET_WORDS):
                y = y_coords[j]
                # Database icon placeholder or flow line
                ax.text(0.12, y, "PG-19 Source", ha="right", va="center", fontsize=5.0, color=MUTED, weight="bold")
                ax.plot([0.14, 0.19], [y, y], color=MUTED, linewidth=0.5)
                ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.5, color=TEXT)
                ax.text(0.675, y, "+", ha="center", va="center", fontsize=6.0, color=BLUE, weight="bold")
                rect = FancyBboxPatch((0.705, y - 0.07), 0.11, 0.14, boxstyle="round,pad=0.002",
                                      facecolor=BLUE_F, edgecolor=BLUE, linewidth=0.5)
                ax.add_patch(rect)
                ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=BLUE)
                if word["suffix"]:
                    ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.5, color=MUTED)

        elif i == 2:
            gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
            for k, (name, val_str) in enumerate([("top-k", "k=6"), ("top-p", "p=0.9"), ("min-p", "m=0.15")]):
                sax = fig.add_subplot(gs[0, k])
                colors = [GREEN if r <= 6 else "#f1f5f9" for r in x_tokens] if k == 0 else (
                         [GREEN if r <= 9 else "#f1f5f9" for r in x_tokens] if k == 1 else
                         [GREEN if r <= 8 else "#f1f5f9" for r in x_tokens])
                sax.bar(x_tokens, y_tokens, color=colors, edgecolor=BORDER, linewidth=0.3, width=0.7)
                sax.set_title(f"{name} ({val_str})", fontsize=6.5, weight="bold", pad=3, color=TEXT)
                sax.spines[["top", "right"]].set_visible(False)
                sax.spines[["left", "bottom"]].set_color(BORDER)
                sax.tick_params(colors=MUTED, labelsize=5.0, length=1.5)
                sax.set_xlim(0.3, 12.7)
                sax.set_ylim(0, y_tokens[0] * 1.3)
                sax.set_xticks([1, 6, 12])
                if k > 0: sax.set_yticklabels([])

        elif i == 3:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
            math_ax = fig.add_subplot(gs[0, 0])
            math_ax.axis("off")
            math_ax.set_xlim(0, 1)
            math_ax.set_ylim(0, 1)
            math_ax.text(0.01, 0.85, "Flow Logic:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.70, r"Input: Context C $\rightarrow$ Target w", fontsize=5.2, color=MUTED)
            math_ax.text(0.08, 0.55, r"Audit: Reachability $R \in \{0, 1\}$", fontsize=5.2, color=MUTED)
            math_ax.text(0.08, 0.35, r"Output: Average Score $WCS$", fontsize=5.2, color=MUTED)
            
            mat_ax = fig.add_subplot(gs[0, 1])
            mat_ax.axis("off")
            mat_ax.set_xlim(0, 1)
            mat_ax.set_ylim(0, 1)
            matrix_data = [
                [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0]
            ]
            x_coords = np.linspace(0.35, 0.75, 5)
            y_coords = np.linspace(0.64, 0.16, 5)
            for c in range(5):
                mat_ax.text(x_coords[c], 0.76, f"c{c+1}", ha="center", fontsize=4.8, color=MUTED)
            mat_ax.text(0.89, 0.76, "Score", ha="center", fontsize=4.8, color=MUTED, weight="bold")
            for r, word in enumerate(TARGET_WORDS):
                y = y_coords[r]
                mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0, color=TEXT)
                successes = 0
                for c in range(5):
                    val = matrix_data[r][c]
                    face = GREEN if val == 1 else RED
                    rect = plt.Rectangle((x_coords[c] - 0.02, y - 0.04), 0.04, 0.08, facecolor=face, edgecolor="none")
                    mat_ax.add_patch(rect)
                    if val == 1: successes += 1
                mat_ax.text(0.89, y, f"{int((successes/5)*100)}%", ha="center", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.plot([0.30, 0.96], [0.08, 0.08], color=BORDER, linewidth=0.4)
            mat_ax.text(0.01, 0.02, "System Average Score", ha="left", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.text(0.89, 0.02, "52%", ha="center", va="center", fontsize=5.0, color=BLUE, weight="bold")

    # Add Pipeline connector arrows on the left margin
    for j in range(3):
        pos_above = grid[j, 0].get_position(fig)
        pos_below = grid[j+1, 0].get_position(fig)
        # Draw small vertical connecting pipeline arrow
        y_mid = (pos_above.y0 + pos_below.y1) / 2.0
        fig.add_artist(plt.Line2D([0.025, 0.025], [pos_above.y0 - 0.015, pos_below.y1 + 0.015],
                                   transform=fig.transFigure, color=BLUE, linewidth=1.5, zorder=5))
        fig.add_artist(plt.Line2D([0.02, 0.98], [y_mid, y_mid], transform=fig.transFigure, color=BORDER, linewidth=0.4))

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

# ==========================================
# VARIANT 5: Warm Retro Serif / Editorial Style (Tufte-inspired)
# ==========================================
def build_retro_editorial(ranks, counts, out_png) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "figure.facecolor": "#fafaf9", # Stone-50 warm alabaster
        "axes.facecolor": "#fafaf9",
        "savefig.bbox": "tight"
    })
    
    BG = "#fafaf9"
    TEXT = "#1c1917" # Stone-900 espresso
    MUTED = "#78716c" # Stone-500
    BORDER = "#e7e5e4" # Stone-200
    ACCENT = "#4f46e5" # Indigo-600
    GREEN = "#166534" # Green-800 olive
    RED = "#991b1b" # Red-800 rust

    fig = plt.figure(figsize=(7.15, 6.0), dpi=300)
    fig.patch.set_facecolor(BG)
    grid = GridSpec(4, 1, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.04, hspace=0.25)

    for i in range(4):
        stage_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[i, 0], height_ratios=[0.18, 0.82], hspace=0.08)
        
        # Header
        titles = [
            ("I. Lexical Selection", "--- Identification of middle-long tail vocabulary ranks"),
            ("II. Context Pairing", "--- Alignment of target lexicon inside natural source prose"),
            ("III. Sampler Audit", "--- Analysis of token probability pruning boundaries"),
            ("IV. Score Measurement", "--- Compilation of reachability observations into WCS metric")
        ]
        draw_header_helper(fig, stage_gs[0, 0], i+1, titles[i][0], titles[i][1], BG, TEXT, MUTED)

        spec = stage_gs[1, 0]
        if i == 0:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.2, 3.8], wspace=0.25)
            ax = fig.add_subplot(gs[0, 0])
            ax.loglog(ranks, counts, color=TEXT, linewidth=0.6, zorder=2)
            ax.axvspan(10_000, 40_000, facecolor="#f5f5f4", edgecolor=BORDER, linewidth=0.4, zorder=1)
            ax.set_xlim(1_000, 100_000)
            ax.set_ylim(5e4, 5e7)
            ax.xaxis.set_major_locator(FixedLocator([10_000, 40_000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
            ax.yaxis.set_major_locator(FixedLocator([100_000, 1_000_000, 10_000_000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}k"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(TEXT)
            ax.spines[["left", "bottom"]].set_linewidth(0.4)
            ax.tick_params(colors=TEXT, labelsize=5.0, length=1.5, pad=1.5)
            ax.grid(True, color="#f5f5f4", linewidth=0.3, zorder=0)
            ax.text(20_000, 2e6, "[10k-40k band]", ha="center", va="center", fontsize=5.0, color=TEXT, style="italic")
            
            # Words
            word_ax = fig.add_subplot(gs[0, 1])
            word_ax.axis("off")
            word_ax.set_xlim(0, 1)
            word_ax.set_ylim(0, 1)
            word_ax.text(0.01, 0.82, "Target Lexicon & Frequency Ranks:", fontsize=5.5, weight="bold", color=TEXT)
            for j, word in enumerate(TARGET_WORDS):
                x = 0.01 + j * 0.198
                word_ax.text(x + 0.089, 0.48, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=ACCENT)
                word_ax.text(x + 0.089, 0.30, f"rank {word['rank']:,}", ha="center", va="center", fontsize=4.5, color=MUTED, style="italic")

        elif i == 1:
            ax = fig.add_subplot(spec)
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            y_coords = np.linspace(0.78, 0.12, 5)
            for j, word in enumerate(TARGET_WORDS):
                y = y_coords[j]
                ax.text(0.64, y, word["prefix"], ha="right", va="center", fontsize=5.5, color=MUTED, style="italic")
                ax.text(0.675, y, "[w]", ha="center", va="center", fontsize=5.0, color=MUTED)
                ax.text(0.76, y, word["word"], ha="center", va="center", fontsize=5.4, weight="bold", color=ACCENT)
                if word["suffix"]:
                    ax.text(0.835, y, word["suffix"], ha="left", va="center", fontsize=5.5, color=MUTED, style="italic")

        elif i == 2:
            gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25)
            # top-k
            k_ax = fig.add_subplot(gs[0, 0])
            k_ax.bar(x_tokens, y_tokens, color="white", edgecolor=TEXT, linewidth=0.4, width=0.7, zorder=2)
            k_ax.axvline(6.5, color=TEXT, linestyle=":", linewidth=0.6)
            k_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            k_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED)
            k_ax.set_title("Top-k (k=6)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            k_ax.spines[["top", "right"]].set_visible(False)
            k_ax.spines[["left", "bottom"]].set_color(TEXT)
            k_ax.spines[["left", "bottom"]].set_linewidth(0.4)
            k_ax.tick_params(colors=TEXT, labelsize=5.0, length=1.5, pad=1.5)
            k_ax.set_xlim(0.3, 12.7)
            k_ax.set_ylim(0, y_tokens[0] * 1.35)
            k_ax.set_xticks([1, 6, 12])
            k_ax.set_ylabel("probability", fontsize=5.0, labelpad=3, color=MUTED)

            # top-p
            p_ax = fig.add_subplot(gs[0, 1])
            p_ax.bar(x_tokens, y_tokens, color="white", edgecolor=TEXT, linewidth=0.4, width=0.7, zorder=2)
            p_ax.axvline(9.5, color=TEXT, linestyle=":", linewidth=0.6)
            p_ax.scatter([w_x], [w_y], s=12, color=GREEN, zorder=4)
            p_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=GREEN)
            p_ax.set_title("Top-p (p=0.9)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            p_ax.spines[["top", "right"]].set_visible(False)
            p_ax.spines[["left", "bottom"]].set_color(TEXT)
            p_ax.spines[["left", "bottom"]].set_linewidth(0.4)
            p_ax.tick_params(colors=TEXT, labelsize=5.0, length=1.5, pad=1.5)
            p_ax.set_xlim(0.3, 12.7)
            p_ax.set_ylim(0, y_tokens[0] * 1.35)
            p_ax.set_xticks([1, 9, 12])
            p_ax.set_yticklabels([])

            # min-p
            m_ax = fig.add_subplot(gs[0, 2])
            threshold = 0.15 * y_tokens[0]
            m_ax.bar(x_tokens, y_tokens, color="white", edgecolor=TEXT, linewidth=0.4, width=0.7, zorder=2)
            m_ax.axhline(threshold, color=TEXT, linestyle=":", linewidth=0.6)
            m_ax.scatter([w_x], [w_y], s=12, color=RED, zorder=4)
            m_ax.text(w_x, w_y + 0.015, "$w$", ha="center", fontsize=5.5, weight="bold", color=RED)
            m_ax.set_title("Min-p (m=0.15)", fontsize=6.5, weight="bold", pad=3, color=TEXT)
            m_ax.spines[["top", "right"]].set_visible(False)
            m_ax.spines[["left", "bottom"]].set_color(TEXT)
            m_ax.spines[["left", "bottom"]].set_linewidth(0.4)
            m_ax.tick_params(colors=TEXT, labelsize=5.0, length=1.5, pad=1.5)
            m_ax.set_xlim(0.3, 12.7)
            m_ax.set_ylim(0, y_tokens[0] * 1.35)
            m_ax.set_xticks([1, 12])
            m_ax.set_yticklabels([])

        elif i == 3:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[2.5, 3.5], wspace=0.25)
            math_ax = fig.add_subplot(gs[0, 0])
            math_ax.axis("off")
            math_ax.set_xlim(0, 1)
            math_ax.set_ylim(0, 1)
            math_ax.text(0.01, 0.85, "Reachability Formulation:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.70, r"$\mathcal{R}_{\theta}(w, c) \in \{0, 1\}$", fontsize=7.5, color=TEXT)
            math_ax.text(0.01, 0.38, "Corpus Word Coverage Score:", fontsize=5.5, weight="bold", color=TEXT)
            math_ax.text(0.08, 0.20, r"$\text{WCS}_{\theta} = \frac{1}{|\mathcal{D}|} \sum \mathcal{R}_{\theta}(w,c)$", fontsize=7.5, color=TEXT)
            
            mat_ax = fig.add_subplot(gs[0, 1])
            mat_ax.axis("off")
            mat_ax.set_xlim(0, 1)
            mat_ax.set_ylim(0, 1)
            matrix_data = [
                [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0]
            ]
            x_coords = np.linspace(0.35, 0.75, 5)
            y_coords = np.linspace(0.64, 0.16, 5)
            for c in range(5):
                mat_ax.text(x_coords[c], 0.76, f"$c_{c+1}$", ha="center", fontsize=4.8, color=MUTED)
            mat_ax.text(0.89, 0.76, "WCS", ha="center", fontsize=4.8, color=MUTED, weight="bold")
            for r, word in enumerate(TARGET_WORDS):
                y = y_coords[r]
                mat_ax.text(0.01, y, word["word"], ha="left", va="center", fontsize=5.0, color=TEXT)
                for c in range(5):
                    val = matrix_data[r][c]
                    sym = "1" if val == 1 else "0"
                    color = GREEN if val == 1 else RED
                    mat_ax.text(x_coords[c], y, sym, ha="center", va="center", fontsize=5.5, color=color, weight="bold")
                successes = sum(matrix_data[r])
                mat_ax.text(0.89, y, f"{int((successes/5)*100)}%", ha="center", va="center", fontsize=5.0, color=TEXT, weight="bold")
            mat_ax.plot([0.30, 0.96], [0.08, 0.08], color=TEXT, linewidth=0.4)
            mat_ax.text(0.01, 0.02, "Corpus average score", ha="left", va="center", fontsize=5.0, color=TEXT, weight="bold", style="italic")
            mat_ax.text(0.89, 0.02, "52%", ha="center", va="center", fontsize=5.0, color=ACCENT, weight="bold")

    for j in range(3):
        pos_above = grid[j, 0].get_position(fig)
        pos_below = grid[j+1, 0].get_position(fig)
        y = (pos_above.y0 + pos_below.y1) / 2.0
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=BORDER, linewidth=0.4))

    fig.savefig(out_png, facecolor=BG, edgecolor="none", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, default=ROOT / "data/raw/norvig_count_1w.txt")
    args = parser.parse_args()

    ranks, counts = load_norvig_frequency(args.frequency)

    # Create subdirectory if not exists
    out_dir = ROOT / "figs/alternatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dark Mode
    dm_path = out_dir / "wcs_alt_dark_mode.png"
    build_dark_mode(ranks, counts, dm_path)
    print(f"Generated Variant 1: {dm_path}")

    # 2. Swiss Minimalist
    sm_path = out_dir / "wcs_alt_minimalist.png"
    build_minimalist(ranks, counts, sm_path)
    print(f"Generated Variant 2: {sm_path}")

    # 3. XKCD Hand-Drawn
    xk_path = out_dir / "wcs_alt_xkcd.png"
    build_xkcd(ranks, counts, xk_path)
    print(f"Generated Variant 3: {xk_path}")

    # 4. Flowchart Pipeline
    pl_path = out_dir / "wcs_alt_pipeline.png"
    build_pipeline(ranks, counts, pl_path)
    print(f"Generated Variant 4: {pl_path}")

    # 5. Warm Editorial
    re_path = out_dir / "wcs_alt_retro_editorial.png"
    build_retro_editorial(ranks, counts, re_path)
    print(f"Generated Variant 5: {re_path}")

    print("Successfully generated all 5 alternatives!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
