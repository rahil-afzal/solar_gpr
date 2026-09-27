"""
Monte Carlo uncertainty propagation for the derived PCE surrogate.

PCE_hat = Voc_hat * Jsc_hat * FF_hat / P_in * 100 is a product of three
Gaussian predictive distributions, so its own distribution is
non-Gaussian. Rather than approximate it analytically, we draw samples
from each sub-metric GP's predictive Normal(mean, std) independently
per query point, compute the empirical PCE distribution, and summarize
it with a sample mean and standard deviation.

As documented in the methodology, treating the three GPs as
independent during sampling is a known simplification (their errors
are likely correlated since all three outputs share the same five
inputs); this tends to inflate sigma_PCE, which biases any downstream
acquisition function toward more exploration rather than less -- a
conservative direction, not treated here as a bug to silently fix.

Sampling itself is cheap (just numpy). The expense in the full
pipeline comes from calling this once per candidate point in an
optimization loop (see optimization.py) -- this module only performs
the propagation math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .surrogate import SolarSurrogate, derive_pce


@dataclass
class PceUncertaintyResult:
    mean_pce: np.ndarray
    std_pce: np.ndarray
    n_samples: int


def propagate_pce_uncertainty(
    surrogate: SolarSurrogate,
    df_query: pd.DataFrame,
    n_samples: int = 10_000,
    random_state: int | None = 0,
) -> PceUncertaintyResult:
    """Monte Carlo propagate Voc/Jsc/FF predictive uncertainty into a
    mean and standard deviation for PCE, for each row of `df_query`.

    Sampling is independent across the three sub-metrics (see module
    docstring for the caveat this implies) but each sub-metric's own
    (mean, std) is per-query-point, so points far from training data
    correctly get wider PCE distributions.
    """
    rng = np.random.default_rng(random_state)

    (voc_mean, jsc_mean, ff_mean), (voc_std, jsc_std, ff_std) = (
        surrogate.predict_sub_metrics(df_query, return_std=True)
    )

    n_points = len(df_query)
    voc_samples = rng.normal(
        loc=voc_mean, scale=voc_std, size=(n_samples, n_points)
    )
    jsc_samples = rng.normal(
        loc=jsc_mean, scale=jsc_std, size=(n_samples, n_points)
    )
    ff_samples = rng.normal(loc=ff_mean, scale=ff_std, size=(n_samples, n_points))

    pce_samples = derive_pce(voc_samples, jsc_samples, ff_samples)

    return PceUncertaintyResult(
        mean_pce=pce_samples.mean(axis=0),
        std_pce=pce_samples.std(axis=0),
        n_samples=n_samples,
    )
