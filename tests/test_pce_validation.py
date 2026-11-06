"""Lightweight tests for solar_gpr.pce_validation (tiny synthetic grid)."""

import itertools
import warnings

import numpy as np
import pandas as pd

from solar_gpr.data import INPUT_COLS
from solar_gpr.pce_validation import (
    extract_hyperparameters,
    flag_bound_hits,
    residual_correlation,
    run_pce_lolo_cv,
    summarize_pce_by_factor,
)
from solar_gpr.features import FEATURE_COLS
from solar_gpr.surrogate import fit_sub_metric_gp


def _toy_grid() -> pd.DataFrame:
    levels = [
        [0.12, 0.14], [1.0, 1.5, 2.0], [0.02, 0.04], [5e19, 1e20, 5e20], [6e20, 6e21],
    ]
    rows = list(itertools.product(*levels))
    df = pd.DataFrame(rows, columns=INPUT_COLS)
    t, na = df[INPUT_COLS[1]], np.log10(df[INPUT_COLS[3]])
    df["Voc (V)"] = 1.0 + 0.05 * na + 0.02 * t
    df["Jsc (mA/cm2)"] = 20 + 5 * t
    df["FF (%)"] = 80 + 2 * na - t
    df["PCE (%)"] = df["Voc (V)"] * df["Jsc (mA/cm2)"] * df["FF (%)"] / 100
    return df


def test_pce_lolo_shapes_and_identity():
    df = _toy_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = run_pce_lolo_cv(df, "matern32", n_restarts_optimizer=0, n_jobs=1)
    assert len(oof) == len(df) * len(INPUT_COLS)  # each row held out once per factor
    expected = oof["Voc (V)_pred"] * oof["Jsc (mA/cm2)_pred"] * oof["FF (%)_pred"] / 100
    np.testing.assert_allclose(oof["PCE_pred"], expected, rtol=1e-10)
    table = summarize_pce_by_factor(oof)
    assert set(table.factor_col) == set(INPUT_COLS)
    assert residual_correlation(oof).shape == (3, 3)


def test_hyperparameter_extraction():
    df = _toy_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = fit_sub_metric_gp(df, "Voc (V)", "matern32", n_restarts_optimizer=0)
    hp = extract_hyperparameters(gp.model, list(FEATURE_COLS))
    assert all(f in hp for f in FEATURE_COLS)
    assert hp["signal_variance"] > 0 and hp["noise_variance"] > 0
    assert isinstance(flag_bound_hits(gp.model), list)
