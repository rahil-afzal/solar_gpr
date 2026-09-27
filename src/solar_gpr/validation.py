"""
Leave-One-Level-Out Cross-Validation (LOLO-CV).

Standard random train/test splits are misleadingly optimistic on a
full-factorial grid, because a randomly held-out point is always
surrounded by its grid neighbors in every dimension -- trivial to
interpolate. LOLO-CV instead holds out an *entire discrete level* of
one factor at a time (e.g. all rows with Absorber Thickness at its
8th tested level) and asks the model to predict that whole slice from
the remaining levels. This is the regime the surrogate will actually
be used in (interpolating/extrapolating across gaps), so it is the
metric used for kernel selection and reported validation performance.

This module is deliberately decoupled from any specific model: the
caller supplies a `fit_predict_fn(train_df, test_df, target_col) ->
np.ndarray`. That keeps this file cheap to unit-test (see
tests/test_data.py-style trivial predictors) without importing
scikit-learn's GaussianProcessRegressor here, and lets
scripts/run_lolo_cv.py plug in the real GPR fitting logic.

Fold generation and execution are separated so the (potentially
expensive) execution step can be parallelized with joblib across
multiple folds independently -- this is the natural place to use all
available CPU cores locally (vs. Colab's 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score

FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame, str], np.ndarray]


@dataclass(frozen=True)
class LoloFold:
    factor_col: str
    held_out_level: float
    train_index: np.ndarray
    test_index: np.ndarray


@dataclass
class LoloFoldResult:
    factor_col: str
    held_out_level: float
    n_train: int
    n_test: int
    r2: float
    rmse: float


def generate_lolo_folds(df: pd.DataFrame, factor_col: str) -> Iterator[LoloFold]:
    """Yield one LoloFold per unique level of `factor_col`."""
    levels = np.sort(df[factor_col].unique())
    for level in levels:
        test_mask = df[factor_col] == level
        yield LoloFold(
            factor_col=factor_col,
            held_out_level=float(level),
            train_index=df.index[~test_mask].to_numpy(),
            test_index=df.index[test_mask].to_numpy(),
        )


def generate_all_lolo_folds(
    df: pd.DataFrame, factor_cols: list[str]
) -> list[LoloFold]:
    """Yield LoloFolds across every level of every factor in
    `factor_cols` (i.e. the full LOLO-CV scheme described in the
    methodology, repeated independently per dimension).
    """
    folds: list[LoloFold] = []
    for factor_col in factor_cols:
        folds.extend(generate_lolo_folds(df, factor_col))
    return folds


def _run_single_fold(
    df: pd.DataFrame,
    fold: LoloFold,
    target_col: str,
    fit_predict_fn: FitPredictFn,
) -> LoloFoldResult:
    train_df = df.loc[fold.train_index]
    test_df = df.loc[fold.test_index]

    y_true = test_df[target_col].to_numpy()
    y_pred = fit_predict_fn(train_df, test_df, target_col)
    y_pred = np.asarray(y_pred)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"fit_predict_fn returned shape {y_pred.shape}, "
            f"expected {y_true.shape}"
        )

    return LoloFoldResult(
        factor_col=fold.factor_col,
        held_out_level=fold.held_out_level,
        n_train=len(train_df),
        n_test=len(test_df),
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def run_lolo_cv(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: list[str],
    fit_predict_fn: FitPredictFn,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run LOLO-CV for `target_col` across every level of every factor
    in `factor_cols`, in parallel across folds.

    Parameters
    ----------
    df:
        Full dataset (design-space columns + target_col).
    target_col:
        Output column to predict, e.g. "Voc (V)".
    factor_cols:
        Which input columns to hold out levels of (usually
        solar_gpr.data.INPUT_COLS).
    fit_predict_fn:
        Callable(train_df, test_df, target_col) -> np.ndarray of
        predictions aligned with test_df.index order. Supplying the
        actual GPR fitting logic here (see scripts/run_lolo_cv.py) is
        the expensive step this module does not perform itself.
    n_jobs:
        Number of parallel workers (joblib). Use -1 for all cores, or
        e.g. 4 to match a 4-core local machine. Folds are independent
        (each fold fits its own model), so this parallelizes cleanly.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns: factor_col, held_out_level,
        n_train, n_test, r2, rmse.
    """
    folds = generate_all_lolo_folds(df, factor_cols)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_fold)(df, fold, target_col, fit_predict_fn)
        for fold in folds
    )

    return pd.DataFrame([r.__dict__ for r in results])


def summarize_lolo_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold LOLO-CV results into mean/std R2 and RMSE per
    held-out factor (i.e. "how well do we generalize across gaps in
    absorber thickness" vs. "...in N_A", etc).
    """
    return (
        results.groupby("factor_col")[["r2", "rmse"]]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("r2", "mean"), ascending=False)
    )
