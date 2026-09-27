"""
Gaussian Process surrogate models.

Trains independent GPs for Voc, Jsc, FF (never PCE directly) and
derives PCE algebraically:

    PCE_hat = (Voc_hat * Jsc_hat * FF_hat) / P_in * 100

so the surrogate can never violate the device efficiency identity,
regardless of how the three underlying GPs individually extrapolate.
This module intentionally does NOT bound the individual sub-metrics
(e.g. FF <= 100%, Voc <= Eg/e) -- see the "Physically Consistent
Surrogate Modeling" methodology subsection for why that is flagged as
a limitation rather than silently assumed away.

Kernel fitting (hyperparameter optimization via GaussianProcessRegressor
.fit) is the expensive step here; this module defines how to build and
fit models, it does not fit anything at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from .data import P_IN_MW_CM2
from .features import FEATURE_COLS, to_feature_space

KERNEL_NAMES = ("rbf", "matern32", "matern52")

SUB_METRIC_TARGETS = ["Voc (V)", "Jsc (mA/cm2)", "FF (%)"]


def build_kernel(name: str, n_features: int, length_scale: float = 1.0):
    """Build a (Constant * base_kernel + White) kernel with per-feature
    (ARD) length scales, so heterogeneous feature units (um vs.
    log10-cm^-3) each get their own learned length scale.

    A small WhiteKernel is included for numerical stability even
    though the underlying simulator is deterministic -- GPR without
    any noise term can become ill-conditioned on near-duplicate
    inputs; the optimizer is left free to shrink it towards zero.
    """
    if name not in KERNEL_NAMES:
        raise ValueError(f"Unknown kernel '{name}', expected one of {KERNEL_NAMES}")

    length_scale_vec = np.full(n_features, length_scale)
    base: RBF | Matern
    if name == "rbf":
        base = RBF(length_scale=length_scale_vec)
    elif name == "matern32":
        base = Matern(length_scale=length_scale_vec, nu=1.5)
    else:  # matern52
        base = Matern(length_scale=length_scale_vec, nu=2.5)

    return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2)
    )


@dataclass
class SubMetricGP:
    """A single fitted GP for one sub-metric (Voc, Jsc, or FF), plus
    the feature-space mean/std used to standardize inputs.
    """

    target_col: str
    kernel_name: str
    model: GaussianProcessRegressor
    x_mean: np.ndarray
    x_std: np.ndarray

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def predict(self, X: np.ndarray, return_std: bool = False):
        Xs = self._standardize(X)
        return self.model.predict(Xs, return_std=return_std)


def fit_sub_metric_gp(
    df: pd.DataFrame,
    target_col: str,
    kernel_name: str = "matern52",
    n_restarts_optimizer: int = 3,
    random_state: int = 0,
) -> SubMetricGP:
    """Fit a single GP for one sub-metric. This is the expensive step
    (hyperparameter optimization via marginal likelihood) -- not run
    at import time, called explicitly from scripts/run_surrogate_fit.py
    or scripts/run_lolo_cv.py.
    """
    X = to_feature_space(df).to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    Xs = (X - x_mean) / x_std

    kernel = build_kernel(kernel_name, n_features=X.shape[1])
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True,
        random_state=random_state,
    )
    gp.fit(Xs, y)

    return SubMetricGP(
        target_col=target_col,
        kernel_name=kernel_name,
        model=gp,
        x_mean=x_mean,
        x_std=x_std,
    )


@dataclass
class SolarSurrogate:
    """Bundle of the three sub-metric GPs plus PCE derivation."""

    voc_gp: SubMetricGP
    jsc_gp: SubMetricGP
    ff_gp: SubMetricGP

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        kernel_name: str = "matern52",
        n_restarts_optimizer: int = 3,
        random_state: int = 0,
    ) -> "SolarSurrogate":
        gps = {
            target: fit_sub_metric_gp(
                df,
                target,
                kernel_name=kernel_name,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state,
            )
            for target in SUB_METRIC_TARGETS
        }
        return cls(
            voc_gp=gps["Voc (V)"],
            jsc_gp=gps["Jsc (mA/cm2)"],
            ff_gp=gps["FF (%)"],
        )

    def predict_sub_metrics(
        self, df_query: pd.DataFrame, return_std: bool = False
    ):
        X = to_feature_space(df_query).to_numpy(dtype=np.float64)
        if return_std:
            voc, voc_std = self.voc_gp.predict(X, return_std=True)
            jsc, jsc_std = self.jsc_gp.predict(X, return_std=True)
            ff, ff_std = self.ff_gp.predict(X, return_std=True)
            return (voc, jsc, ff), (voc_std, jsc_std, ff_std)
        voc = self.voc_gp.predict(X)
        jsc = self.jsc_gp.predict(X)
        ff = self.ff_gp.predict(X)
        return voc, jsc, ff

    def predict_pce(self, df_query: pd.DataFrame) -> np.ndarray:
        """Point estimate of PCE, derived algebraically from the point
        estimates of Voc, Jsc, FF (mean-of-product, not
        product-of-means-adjusted -- see uncertainty.py for the
        distribution-aware version used by the acquisition function).
        """
        voc, jsc, ff = self.predict_sub_metrics(df_query)
        return derive_pce(voc, jsc, ff)


def derive_pce(
    voc: np.ndarray, jsc: np.ndarray, ff_percent: np.ndarray
) -> np.ndarray:
    """PCE (%) = Voc (V) * Jsc (mA/cm2) * FF (fraction) / P_in * 100,
    matching the identity verified in data.verify_dataset_integrity.
    """
    return (voc * jsc * (ff_percent / 100.0)) / P_IN_MW_CM2 * 100.0


def make_fit_predict_fn(kernel_name: str, n_restarts_optimizer: int = 2):
    """Build a `fit_predict_fn(train_df, test_df, target_col)` closure
    compatible with validation.run_lolo_cv, for a single sub-metric
    target (Voc, Jsc, or FF -- not PCE; PCE LOLO-CV should compose the
    three sub-metric predictions via derive_pce if needed).
    """

    def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        gp = fit_sub_metric_gp(
            train_df,
            target_col,
            kernel_name=kernel_name,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        X_test = to_feature_space(test_df).to_numpy(dtype=np.float64)
        return gp.predict(X_test)

    return _fit_predict
