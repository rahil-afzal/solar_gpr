#!/usr/bin/env python3
"""
Fit the final Voc/Jsc/FF GP surrogate models (using the kernel selected
by scripts/run_lolo_cv.py) on the FULL dataset, and persist them to
results/metrics/surrogate_model.joblib for reuse by
scripts/run_bo_extrapolation.py and scripts/make_figures.py.

This is an EXPENSIVE step (3 GP hyperparameter optimizations on 5000
points each). Run only after LOLO-CV has identified the winning
kernel.

Usage:
    python scripts/run_surrogate_fit.py --kernel matern52
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.data import DEFAULT_RAW_PATH, load_dataset, verify_dataset_integrity  # noqa: E402
from solar_gpr.surrogate import KERNEL_NAMES, SolarSurrogate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--kernel", required=True, choices=list(KERNEL_NAMES),
        help="Kernel selected by LOLO-CV (see results/metrics/lolo_cv_kernel_summary.csv).",
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--out-path", type=Path,
        default=REPO_ROOT / "results" / "metrics" / "surrogate_model.joblib",
    )
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    report = verify_dataset_integrity(df)
    if not report.ok:
        print(report.summary(), file=sys.stderr)
        raise SystemExit("Dataset integrity check failed; aborting surrogate fit.")

    print(f"Fitting SolarSurrogate (kernel={args.kernel}) on {len(df)} rows ...")
    t0 = time.time()
    surrogate = SolarSurrogate.fit(
        df,
        kernel_name=args.kernel,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(surrogate, args.out_path)
    print(f"Wrote {args.out_path}")

    for name, gp in [
        ("Voc", surrogate.voc_gp), ("Jsc", surrogate.jsc_gp), ("FF", surrogate.ff_gp),
    ]:
        print(f"{name}: kernel_ = {gp.model.kernel_}")


if __name__ == "__main__":
    main()
