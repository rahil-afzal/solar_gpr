"""
LOLO-CV of the *derived* PCE, and extraction of fitted GP hyperparameters.

The sub-metric LOLO-CV in validation.py scores Voc, Jsc and FF separately.
The surrogate's headline quantity is the derived PCE, so this module
refits the three sub-metric GPs in every fold, derives PCE from the
out-of-fold predictions, and keeps those predictions so that pooled
metrics and residual correlations can be computed afterwards.

Cost: n_folds (28) x 3 GP fits per kernel. Folds are independent and are
parallelised with joblib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score

from solar_gpr.data import INPUT_COLS
from solar_gpr.features import to_feature_space
from solar_gpr.surrogate import SUB_METRIC_TARGETS, derive_pce, fit_sub_metric_gp
from solar_gpr.validation import LoloFold, generate_all_lolo_folds

PCE_COL = "PCE (%)"


def _run_fold(
    df: pd.DataFrame,
    fold: LoloFold,
    kernel_name: str,
    n_restarts_optimizer: int,
    random_state: int,
) -> pd.DataFrame:
    """Fit Voc/Jsc/FF GPs on the training rows of one fold and return
    out-of-fold predictions (sub-metrics and derived PCE) for the test rows.
    """
    train_df = df.loc[fold.train_index]
    test_df = df.loc[fold.test_index]
    X_test = to_feature_space(test_df).to_numpy(dtype=np.float64)

    preds = {}
    for target in SUB_METRIC_TARGETS:
        gp = fit_sub_metric_gp(
            train_df,
            target,
            kernel_name=kernel_name,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        preds[target] = gp.predict(X_test)

    out = test_df[INPUT_COLS + [PCE_COL] + SUB_METRIC_TARGETS].copy()
    out = out.rename(columns={t: f"{t}_true" for t in SUB_METRIC_TARGETS})
    out = out.rename(columns={PCE_COL: "PCE_true"})
    for target in SUB_METRIC_TARGETS:
        out[f"{target}_pred"] = preds[target]
    out["PCE_pred"] = derive_pce(
        preds["Voc (V)"], preds["Jsc (mA/cm2)"], preds["FF (%)"]
    )
    out["factor_col"] = fold.factor_col
    out["held_out_level"] = fold.held_out_level
    return out


def run_pce_lolo_cv(
    df: pd.DataFrame,
    kernel_name: str = "matern32",
    factor_cols: list[str] | None = None,
    n_restarts_optimizer: int = 3,
    random_state: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Out-of-fold predictions for every row, once per held-out factor.

    Returns a long DataFrame (n_rows x n_factors rows) with the inputs,
    true and predicted Voc/Jsc/FF/PCE, `factor_col` and `held_out_level`.
    """
    factor_cols = factor_cols or INPUT_COLS
    folds = generate_all_lolo_folds(df, factor_cols)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(df, f, kernel_name, n_restarts_optimizer, random_state)
        for f in folds
    )
    return pd.concat(parts, ignore_index=True)


def summarize_pce_by_factor(oof: pd.DataFrame) -> pd.DataFrame:
    """Table of PCE LOLO-CV performance by held-out factor.

    Columns: mean_fold_rmse, max_fold_rmse (over the folds of that factor),
    pooled_r2 (over all out-of-fold rows of that factor), max_abs_error.
    Units are percentage points for PCE.
    """
    rows = []
    for factor, g in oof.groupby("factor_col", sort=False):
        fold_rmse = g.groupby("held_out_level").apply(
            lambda x: float(np.sqrt(mean_squared_error(x["PCE_true"], x["PCE_pred"]))),
            include_groups=False,
        )
        err = (g["PCE_pred"] - g["PCE_true"]).abs()
        rows.append(
            {
                "factor_col": factor,
                "n_folds": int(fold_rmse.size),
                "mean_fold_rmse_pp": float(fold_rmse.mean()),
                "max_fold_rmse_pp": float(fold_rmse.max()),
                "pooled_r2": float(r2_score(g["PCE_true"], g["PCE_pred"])),
                "max_abs_error_pp": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_pce_by_fold(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-fold PCE RMSE, R2 and max absolute error."""
    rows = []
    for (factor, level), g in oof.groupby(["factor_col", "held_out_level"], sort=False):
        rows.append(
            {
                "factor_col": factor,
                "held_out_level": level,
                "n_test": len(g),
                "r2": float(r2_score(g["PCE_true"], g["PCE_pred"])),
                "rmse_pp": float(np.sqrt(mean_squared_error(g["PCE_true"], g["PCE_pred"]))),
                "max_abs_error_pp": float((g["PCE_pred"] - g["PCE_true"]).abs().max()),
            }
        )
    return pd.DataFrame(rows)


def residual_correlation(oof: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of out-of-fold residuals (pred - true) of Voc,
    Jsc and FF. Non-zero off-diagonals mean the independence assumption
    used in the Monte Carlo propagation ignores correlated errors.
    """
    res = pd.DataFrame(
        {t: oof[f"{t}_pred"] - oof[f"{t}_true"] for t in SUB_METRIC_TARGETS}
    )
    return res.corr()


def extract_hyperparameters(gp_model, feature_names: list[str]) -> dict:
    """Read fitted hyperparameters from a scikit-learn GP whose kernel is
    ConstantKernel * (RBF | Matern) + WhiteKernel, as built by
    solar_gpr.surrogate.build_kernel. All values are in normalized units
    (standardized inputs, normalized target).
    """
    p = gp_model.kernel_.get_params()
    ls = np.atleast_1d(p["k1__k2__length_scale"]).astype(float)
    out = {name: float(v) for name, v in zip(feature_names, ls)}
    out["signal_variance"] = float(p["k1__k1__constant_value"])
    out["noise_variance"] = float(p["k2__noise_level"])
    out["log_marginal_likelihood"] = float(gp_model.log_marginal_likelihood_value_)
    return out


def flag_bound_hits(gp_model, rtol: float = 1e-2) -> list[str]:
    """Names of hyperparameters sitting at one of their optimizer bounds.
    A fit with hits is a sign that the bounds, not the data, set the value.
    """
    flags = []
    params = gp_model.kernel_.get_params()
    for hp in gp_model.kernel_.hyperparameters:
        bounds = np.atleast_2d(hp.bounds)
        value = np.atleast_1d(params[hp.name]).astype(float)
        for i, v in enumerate(value):
            lo, hi = bounds[min(i, len(bounds) - 1)]
            if np.isclose(v, lo, rtol=rtol) or np.isclose(v, hi, rtol=rtol):
                flags.append(f"{hp.name}[{i}]")
    return flags
