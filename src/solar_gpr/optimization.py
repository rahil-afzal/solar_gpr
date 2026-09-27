"""
Upper Confidence Bound (UCB) acquisition and targeted extrapolation
search.

This is deliberately NOT a generic closed-loop Bayesian Optimization
campaign. It is used narrowly to rank a small, physically motivated
set of candidate points (e.g. a grid around the polynomial-predicted
PCE turnover near N_A ~ 9.2e20 cm^-3) by UCB score, so the highest-
scoring candidate can be proposed as the single next SCAPS-1D
simulation to run.

UCB(x) = mean_pce(x) + beta * std_pce(x)

using the Monte Carlo mean/std from uncertainty.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .surrogate import SolarSurrogate
from .uncertainty import propagate_pce_uncertainty


@dataclass
class UcbRankingResult:
    candidates: pd.DataFrame  # original candidate columns
    mean_pce: np.ndarray
    std_pce: np.ndarray
    ucb_score: np.ndarray
    ranked_index: np.ndarray  # candidates.index sorted by ucb_score desc

    def top(self, n: int = 1) -> pd.DataFrame:
        out = self.candidates.loc[self.ranked_index[:n]].copy()
        out["mean_pce"] = pd.Series(self.mean_pce, index=self.candidates.index).loc[
            self.ranked_index[:n]
        ].to_numpy()
        out["std_pce"] = pd.Series(self.std_pce, index=self.candidates.index).loc[
            self.ranked_index[:n]
        ].to_numpy()
        out["ucb_score"] = pd.Series(
            self.ucb_score, index=self.candidates.index
        ).loc[self.ranked_index[:n]].to_numpy()
        return out


def rank_candidates_by_ucb(
    surrogate: SolarSurrogate,
    candidates: pd.DataFrame,
    beta: float = 2.0,
    n_mc_samples: int = 10_000,
    random_state: int | None = 0,
) -> UcbRankingResult:
    """Score and rank a fixed set of candidate design points by UCB.

    Parameters
    ----------
    surrogate:
        A fitted SolarSurrogate (see surrogate.SolarSurrogate.fit).
    candidates:
        DataFrame of candidate points in the original input-column
        space (solar_gpr.data.INPUT_COLS), e.g. a small grid around a
        hypothesized extrapolation target.
    beta:
        Exploration weight in UCB = mean + beta * std. Larger values
        favor high-uncertainty (more exploratory) candidates.
    n_mc_samples:
        Monte Carlo sample count used for uncertainty propagation.
    """
    result = propagate_pce_uncertainty(
        surrogate, candidates, n_samples=n_mc_samples, random_state=random_state
    )
    ucb = result.mean_pce + beta * result.std_pce
    order = np.argsort(ucb)[::-1]
    ranked_index = candidates.index.to_numpy()[order]

    return UcbRankingResult(
        candidates=candidates,
        mean_pce=result.mean_pce,
        std_pce=result.std_pce,
        ucb_score=ucb,
        ranked_index=ranked_index,
    )


def make_extrapolation_candidate_grid(
    base_row: dict[str, float],
    varying_col: str,
    values: np.ndarray,
) -> pd.DataFrame:
    """Build a small candidate grid that varies one input column
    (typically N_A) across `values` while holding all other inputs
    fixed at `base_row`. Used to probe a specific, physically
    motivated hypothesis (e.g. the polynomial-predicted PCE turnover)
    rather than searching the full 5D space.
    """
    from .data import INPUT_COLS

    missing = [c for c in INPUT_COLS if c not in base_row and c != varying_col]
    if missing:
        raise ValueError(f"base_row is missing required columns: {missing}")

    rows = []
    for v in values:
        row = dict(base_row)
        row[varying_col] = v
        rows.append(row)
    return pd.DataFrame(rows, columns=INPUT_COLS)
