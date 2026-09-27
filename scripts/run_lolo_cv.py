#!/usr/bin/env python3
"""
Run Leave-One-Level-Out Cross-Validation (LOLO-CV) across kernel
candidates and sub-metric targets, and write results to
results/metrics/.

This is the EXPENSIVE step: for each kernel candidate (3) x each
sub-metric target (3) x each held-out level across all 5 factors
(5+8+5+5+5 = 28 folds), a fresh GaussianProcessRegressor is fit from
scratch. That is 3 * 3 * 28 = 252 GP fits. Folds are embarrassingly
parallel (each fold fits an independent model), so --n-jobs is passed
straight to joblib.Parallel; use --n-jobs 4 on a 4-core machine.

Usage:
    python scripts/run_lolo_cv.py --n-jobs 4
    python scripts/run_lolo_cv.py --kernels matern52 --n-jobs 4  # single kernel
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.data import DEFAULT_RAW_PATH, load_dataset, verify_dataset_integrity  # noqa: E402
from solar_gpr.surrogate import KERNEL_NAMES, make_fit_predict_fn  # noqa: E402
from solar_gpr.validation import run_lolo_cv, summarize_lolo_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "gpr_kernels.yaml"
    )
    parser.add_argument(
        "--kernels", nargs="+", default=None, choices=list(KERNEL_NAMES),
        help="Subset of kernels to run (default: all candidates in config).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Parallel workers for joblib (overrides config; use 4 locally).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results" / "metrics"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_dataset(args.data_path)
    report = verify_dataset_integrity(df)
    if not report.ok:
        print(report.summary(), file=sys.stderr)
        raise SystemExit("Dataset integrity check failed; aborting LOLO-CV run.")

    kernels = args.kernels or cfg["candidates"]
    n_jobs = args.n_jobs or cfg["lolo_cv"]["n_jobs"]
    factor_cols = cfg["lolo_cv"]["factor_cols"]
    targets = cfg["lolo_cv"]["targets"]
    n_restarts = cfg["gp"]["n_restarts_optimizer"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for kernel_name in kernels:
        fit_predict_fn = make_fit_predict_fn(kernel_name, n_restarts_optimizer=n_restarts)
        for target in targets:
            print(f"[LOLO-CV] kernel={kernel_name} target={target} n_jobs={n_jobs} ...")
            t0 = time.time()
            results = run_lolo_cv(
                df, target_col=target, factor_cols=factor_cols,
                fit_predict_fn=fit_predict_fn, n_jobs=n_jobs,
            )
            results["kernel"] = kernel_name
            results["target"] = target
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s, mean R2={results['r2'].mean():.4f}")
            all_results.append(results)

            fold_path = args.out_dir / f"lolo_cv_folds_{kernel_name}_{target.split(' ')[0]}.csv"
            results.to_csv(fold_path, index=False, encoding="utf-8")

    import pandas as pd
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = args.out_dir / "lolo_cv_all_folds.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    summary = (
        combined.groupby(["kernel", "target"])[["r2", "rmse"]]
        .mean()
        .sort_values("r2", ascending=False)
    )
    summary_path = args.out_dir / "lolo_cv_kernel_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8")

    print("\n=== Kernel comparison summary (mean across folds) ===")
    print(summary)
    print(f"\nWrote {combined_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
