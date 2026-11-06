#!/usr/bin/env python3
"""
LOLO-CV of the derived PCE for one kernel (default Matern 3/2).

Fits Voc, Jsc and FF GPs in each of the 28 folds (84 GP fits), derives PCE
from the out-of-fold sub-metric predictions, and writes:

  results/metrics/lolo_cv_pce_oof_<kernel>.csv      out-of-fold predictions
  results/metrics/lolo_cv_pce_by_fold_<kernel>.csv  per-fold PCE metrics
  results/tables/pce_lolo_by_factor_<kernel>.csv    table for the paper
  results/tables/pce_lolo_by_factor_<kernel>.tex    LaTeX (Table tab:pce-lolo)
  results/tables/residual_corr_<kernel>.csv         Voc/Jsc/FF residual correlation

EXPENSIVE: each fold fits three exact GPs on ~4000-4375 rows. Memory per
worker is roughly 0.6 GB for the ARD gradient tensor, so use --n-jobs 4
only with >= 4 GB free.

Usage:
    python scripts/run_lolo_cv_pce.py --kernel matern32 --n-jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.data import DEFAULT_RAW_PATH, load_dataset, verify_dataset_integrity  # noqa: E402
from solar_gpr.pce_validation import (  # noqa: E402
    residual_correlation,
    run_pce_lolo_cv,
    summarize_pce_by_fold,
    summarize_pce_by_factor,
)
from solar_gpr.surrogate import KERNEL_NAMES  # noqa: E402

FACTOR_LATEX = {
    "HTL thickness (μm)": r"$t_{HTL}$",
    "Absorber Thickness (μm)": r"$t_{abs}$",
    "ETL thickness (μm)": r"$t_{ETL}$",
    "N_A (1/cm3)": r"$N_A$",
    "N_D (1/cm3)": r"$N_D$",
}


def to_latex(table) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{LOLO-CV performance of the derived PCE by held-out factor.}",
        r"\label{tab:pce-lolo}", r"\begin{tabular}{lcccc}", r"\toprule",
        r"Held-out factor & Mean RMSE (pp) & Max.\ fold RMSE (pp) & Pooled $R^2$ & Max.\ absolute error (pp) \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{FACTOR_LATEX.get(r.factor_col, r.factor_col)} & "
            f"{r.mean_fold_rmse_pp:.4f} & {r.max_fold_rmse_pp:.4f} & "
            f"{r.pooled_r2:.4f} & {r.max_abs_error_pp:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=DEFAULT_RAW_PATH)
    p.add_argument("--kernel", default="matern32", choices=list(KERNEL_NAMES))
    p.add_argument("--n-restarts-optimizer", type=int, default=3)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "results" / "metrics")
    p.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = p.parse_args()

    df = load_dataset(args.data_path)
    report = verify_dataset_integrity(df)
    if not report.ok:
        print(report.summary(), file=sys.stderr)
        raise SystemExit("Dataset integrity check failed; aborting.")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PCE LOLO-CV] kernel={args.kernel} restarts={args.n_restarts_optimizer} "
          f"n_jobs={args.n_jobs} ...")
    t0 = time.time()
    oof = run_pce_lolo_cv(
        df, kernel_name=args.kernel,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state, n_jobs=args.n_jobs,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    k = args.kernel
    oof.to_csv(args.metrics_dir / f"lolo_cv_pce_oof_{k}.csv", index=False, encoding="utf-8")
    by_fold = summarize_pce_by_fold(oof)
    by_fold.to_csv(args.metrics_dir / f"lolo_cv_pce_by_fold_{k}.csv", index=False, encoding="utf-8")
    by_factor = summarize_pce_by_factor(oof)
    by_factor.to_csv(args.tables_dir / f"pce_lolo_by_factor_{k}.csv", index=False, encoding="utf-8")
    (args.tables_dir / f"pce_lolo_by_factor_{k}.tex").write_text(
        to_latex(by_factor), encoding="utf-8"
    )
    corr = residual_correlation(oof)
    corr.to_csv(args.tables_dir / f"residual_corr_{k}.csv", encoding="utf-8")

    print(by_factor.to_string(index=False))
    print("\nOut-of-fold residual correlation (Voc, Jsc, FF):")
    print(corr.round(3).to_string())
    print("\nLargest-error folds:")
    print(by_fold.sort_values("rmse_pp", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
