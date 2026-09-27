"""
Shared plotting style, used by both scripts/make_figures.py and the
EDA notebook so figures are visually consistent throughout the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIGURE_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"

PALETTE = "viridis"
FIGSIZE_SINGLE = (6, 4.5)
FIGSIZE_WIDE = (10, 4.5)
DPI = 300


def set_paper_style() -> None:
    """Apply a consistent, print-friendly style to all matplotlib /
    seaborn figures produced by this project.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save `fig` to results/figures/<name>.png, creating the
    directory if needed, and return the path.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(out_path)
    return out_path
