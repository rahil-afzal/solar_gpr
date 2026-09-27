#!/usr/bin/env python3
"""
Load the fitted surrogate (scripts/run_surrogate_fit.py output),
propagate Monte Carlo uncertainty over a small candidate grid around
the hypothesized N_A extrapolation target, rank candidates by UCB, and
write the proposed next simulation point to results/tables/.

This is an EXPENSIVE-ish step (10k MC samples x ~25 candidates through
3 GPs) but far cheaper than LOLO-CV or surrogate fitting; still gated
behind the CLI for pipeline consistency.

Usage:
    python scripts/run_bo_extrapolation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.data import DEFAULT_RAW_PATH, INPUT_COLS, load_dataset  # noqa: E402
from solar_gpr.optimization import (  # noqa: E402
    make_extrapolation_candidate_grid,
    rank_candidates_by_ucb,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--surrogate-path", type=Path,
        default=REPO_ROOT / "results" / "metrics" / "surrogate_model.joblib",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "bo_settings.yaml"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "results" / "tables"
    )
    args = parser.parse_args()

    if not args.surrogate_path.exists():
        raise SystemExit(
            f"No fitted surrogate at {args.surrogate_path}. "
            f"Run scripts/run_surrogate_fit.py first."
        )

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    surrogate = joblib.load(args.surrogate_path)
    df = load_dataset(args.data_path)

    ex_cfg = cfg["extrapolation_target"]
    if ex_cfg["hold_other_inputs_at"] == "best_observed_pce":
        base_row = df.loc[df["PCE (%)"].idxmax(), INPUT_COLS].to_dict()
    else:
        raise NotImplementedError(
            f"Unsupported hold_other_inputs_at: {ex_cfg['hold_other_inputs_at']}"
        )

    if ex_cfg["log_spaced"]:
        values = np.logspace(
            np.log10(ex_cfg["search_min"]), np.log10(ex_cfg["search_max"]),
            ex_cfg["n_points"],
        )
    else:
        values = np.linspace(ex_cfg["search_min"], ex_cfg["search_max"], ex_cfg["n_points"])

    candidates = make_extrapolation_candidate_grid(
        base_row=base_row, varying_col=ex_cfg["varying_col"], values=values
    )

    result = rank_candidates_by_ucb(
        surrogate, candidates,
        beta=cfg["ucb"]["beta"],
        n_mc_samples=cfg["monte_carlo"]["n_samples"],
        random_state=cfg["monte_carlo"]["random_state"],
    )

    top = result.top(5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "targeted_extrapolation_candidates.csv"
    top.to_csv(out_path, index=False, encoding="utf-8")

    print("Top 5 candidates by UCB score:")
    print(top)
    print(f"\nWrote {out_path}")
    print(
        "\nNOTE: this is a hypothesis-generation step. The top candidate is a "
        "proposed NEXT SCAPS-1D simulation point, not a validated PCE prediction "
        "(see Methodology: Uncertainty Propagation and Targeted Extrapolation)."
    )


if __name__ == "__main__":
    main()
