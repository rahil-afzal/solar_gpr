#!/usr/bin/env python3
"""
Run the Type-II ANOVA / eta-squared variance decomposition across all
four output metrics and write results to results/tables/ and
results/metrics/.

Usage:
    python scripts/run_anova.py
    python scripts/run_anova.py --data-path data/raw/SOLAR_DATASET.xlsx

This is a genuinely cheap step for this dataset (one OLS fit per
target, 5000 rows, <=5 levels/factor) but is still gated behind the
CLI (not run on import) so the pipeline stage boundaries stay
explicit and consistent with the other (expensive) scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.anova import run_anova_all_targets  # noqa: E402
from solar_gpr.data import (  # noqa: E402
    DEFAULT_RAW_PATH,
    INPUT_COLS,
    OUTPUT_COLS,
    load_dataset,
    verify_dataset_integrity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_RAW_PATH,
        help="Path to the raw SCAPS-1D dataset (.xlsx).",
    )
    parser.add_argument(
        "--targets", nargs="+", default=["PCE (%)", "Voc (V)", "Jsc (mA/cm2)", "FF (%)"],
        help="Output columns to run ANOVA on.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results" / "tables",
        help="Directory to write the eta-squared table (CSV + LaTeX).",
    )
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    report = verify_dataset_integrity(df)
    if not report.ok:
        print(report.summary(), file=sys.stderr)
        raise SystemExit("Dataset integrity check failed; aborting ANOVA run.")

    unknown_targets = [t for t in args.targets if t not in OUTPUT_COLS]
    if unknown_targets:
        raise SystemExit(f"Unknown target column(s): {unknown_targets}")

    print(f"Running ANOVA for targets: {args.targets}")
    eta_table = run_anova_all_targets(df, target_cols=args.targets, factor_cols=INPUT_COLS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "anova_eta_squared.csv"
    tex_path = args.out_dir / "anova_eta_squared.tex"

    eta_table.to_csv(csv_path, encoding="utf-8")
    eta_table.to_latex(tex_path, float_format="%.4f", encoding="utf-8")

    print(eta_table)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
