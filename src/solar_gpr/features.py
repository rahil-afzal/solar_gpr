"""
Physics-informed feature transforms.

Doping concentrations (N_A, N_D) span multiple orders of magnitude, so
they are log10-transformed before being handed to any distance/kernel
based model (see Methodology: Physics-Informed Feature Engineering).
All other inputs are used in their native (linear, micron) units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INPUT_COLS, LOG_SCALE_COLS

# Feature names after transform, in the order the surrogate model
# expects them. Log-scale columns get a trailing "_log10".
FEATURE_COLS: list[str] = [
    c if c not in LOG_SCALE_COLS else f"{c}_log10" for c in INPUT_COLS
]


def to_feature_space(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` restricted to INPUT_COLS with doping
    concentrations replaced by their log10 values.

    Column order matches FEATURE_COLS.
    """
    out = pd.DataFrame(index=df.index)
    for col in INPUT_COLS:
        if col in LOG_SCALE_COLS:
            values = df[col].to_numpy(dtype=np.float64)
            if np.any(values <= 0):
                raise ValueError(
                    f"Cannot log10-transform non-positive values in '{col}'"
                )
            out[f"{col}_log10"] = np.log10(values)
        else:
            out[col] = df[col]
    return out[FEATURE_COLS]


def from_feature_space(df_features: pd.DataFrame) -> pd.DataFrame:
    """Inverse of `to_feature_space`: map log10-doping features back to
    physical (linear, cm^-3) units. Useful when reporting a GP's
    proposed extrapolation point in physical units.
    """
    out = pd.DataFrame(index=df_features.index)
    for col in INPUT_COLS:
        if col in LOG_SCALE_COLS:
            out[col] = 10.0 ** df_features[f"{col}_log10"].to_numpy(dtype=np.float64)
        else:
            out[col] = df_features[col]
    return out[INPUT_COLS]
