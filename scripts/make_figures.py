#!/usr/bin/env python3
"""
Regenerate all paper figures from already-saved results (ANOVA table,
LOLO-CV metrics, fitted surrogate). Does NOT fit any models itself --
run scripts/run_anova.py, run_lolo_cv.py, and run_surrogate_fit.py
first.

Figures produced (into results/figures/):
    01_pairwise_trends.png       - input vs. Voc/Jsc/FF/PCE trends
    02_eta_squared_heatmap.png   - ANOVA variance decomposition
    03_lolo_cv_kernel_comparison.png
    04_pce_extrapolation_curve.png

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.data import DEFAULT_RAW_PATH, INPUT_COLS, load_dataset  # noqa: E402
from solar_gpr.plotting import FIGSIZE_WIDE, savefig, set_paper_style  # noqa: E402


def fig_pairwise_trends(df: pd.DataFrame) -> None:
    outputs = ["Voc (V)", "Jsc (mA/cm2)", "FF (%)", "PCE (%)"]
    fig, axes = plt.subplots(len(INPUT_COLS), len(outputs), figsize=(14, 14))
    for i, inp in enumerate(INPUT_COLS):
        for j, out in enumerate(outputs):
            ax = axes[i, j]
            sns.scatterplot(data=df, x=inp, y=out, s=4, alpha=0.3, ax=ax, legend=False)
            if inp in ("N_A (1/cm3)", "N_D (1/cm3)"):
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(out, fontsize=9)
            if j != 0:
                ax.set_ylabel("")
    fig.tight_layout()
    savefig(fig, "01_pairwise_trends")
    plt.close(fig)


def fig_eta_squared_heatmap(eta_path: Path) -> None:
    if not eta_path.exists():
        print(f"  skip: {eta_path} not found (run scripts/run_anova.py first)")
        return
    eta = pd.read_csv(eta_path, index_col=0, encoding="utf-8")
    eta = eta.drop(index=[i for i in eta.index if i == "Residual"], errors="ignore")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(eta, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": r"$\eta^2$"})
    ax.set_ylabel("")
    fig.tight_layout()
    savefig(fig, "02_eta_squared_heatmap")
    plt.close(fig)


def fig_lolo_cv_kernel_comparison(summary_path: Path) -> None:
    if not summary_path.exists():
        print(f"  skip: {summary_path} not found (run scripts/run_lolo_cv.py first)")
        return
    summary = pd.read_csv(summary_path, encoding="utf-8")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.barplot(data=summary, x="target", y="r2", hue="kernel", ax=ax)
    ax.set_ylabel(r"Mean LOLO-CV $R^2$")
    ax.set_xlabel("")
    fig.tight_layout()
    savefig(fig, "03_lolo_cv_kernel_comparison")
    plt.close(fig)


def fig_pce_extrapolation_curve(candidates_path: Path) -> None:
    if not candidates_path.exists():
        print(f"  skip: {candidates_path} not found (run scripts/run_bo_extrapolation.py first)")
        return
    cand = pd.read_csv(candidates_path, encoding="utf-8")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.errorbar(
        cand["N_A (1/cm3)"], cand["mean_pce"], yerr=cand["std_pce"],
        fmt="o-", capsize=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_A$ (cm$^{-3}$)")
    ax.set_ylabel("Predicted PCE (%)")
    fig.tight_layout()
    savefig(fig, "04_pce_extrapolation_curve")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_PATH)
    args = parser.parse_args()

    set_paper_style()
    df = load_dataset(args.data_path)

    tables_dir = REPO_ROOT / "results" / "tables"
    metrics_dir = REPO_ROOT / "results" / "metrics"

    print("Figure 1: pairwise trends (from raw data, always available)")
    fig_pairwise_trends(df)

    print("Figure 2: eta-squared heatmap")
    fig_eta_squared_heatmap(tables_dir / "anova_eta_squared.csv")

    print("Figure 3: LOLO-CV kernel comparison")
    fig_lolo_cv_kernel_comparison(metrics_dir / "lolo_cv_kernel_summary.csv")

    print("Figure 4: PCE extrapolation curve")
    fig_pce_extrapolation_curve(tables_dir / "targeted_extrapolation_candidates.csv")

    print(f"\nFigures written to {REPO_ROOT / 'results' / 'figures'}")


if __name__ == "__main__":
    main()
