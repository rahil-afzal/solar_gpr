"""
Tests for solar_gpr.data.

These are intentionally cheap: real-data checks here are the same
integrity checks the CLI scripts gate on (no model fitting), plus
synthetic-frame tests that construct tiny deliberately-broken inputs
to confirm the checks actually catch what they claim to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_gpr.data import (
    DEFAULT_RAW_PATH,
    INPUT_COLS,
    OUTPUT_COLS,
    P_IN_MW_CM2,
    get_level_grid,
    load_dataset,
    verify_dataset_integrity,
)


# ---------------------------------------------------------------------
# Real-data tests (skipped gracefully if the raw file isn't present)
# ---------------------------------------------------------------------

requires_raw_data = pytest.mark.skipif(
    not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present"
)


@requires_raw_data
def test_load_dataset_schema():
    df = load_dataset()
    assert list(df.columns) == INPUT_COLS + OUTPUT_COLS
    assert len(df) == 5000
    for col in ["N_A (1/cm3)", "N_D (1/cm3)"]:
        assert pd.api.types.is_float_dtype(df[col])


@requires_raw_data
def test_full_dataset_passes_integrity_checks():
    df = load_dataset()
    report = verify_dataset_integrity(df)
    assert report.ok, report.summary()
    assert report.is_full_factorial
    assert report.n_duplicate_input_combos == 0
    assert report.n_missing_values == 0
    assert report.pce_identity_max_abs_error < 1e-3


@requires_raw_data
def test_level_grid_matches_expected_counts():
    df = load_dataset()
    grid = get_level_grid(df)
    expected_counts = {
        "HTL thickness (μm)": 5,
        "Absorber Thickness (μm)": 8,
        "ETL thickness (μm)": 5,
        "N_A (1/cm3)": 5,
        "N_D (1/cm3)": 5,
    }
    for col, n in expected_counts.items():
        assert len(grid[col]) == n, f"{col}: expected {n} levels, got {len(grid[col])}"


# ---------------------------------------------------------------------
# Synthetic tests: confirm the checks catch what they claim to
# ---------------------------------------------------------------------

def _make_synthetic_clean_frame(n_a=2, n_b=2) -> pd.DataFrame:
    """Tiny 2x2 full-factorial synthetic frame with a self-consistent
    PCE identity, used to test that the checks pass on clean data
    without needing the real 5000-row dataset.
    """
    a_vals = np.linspace(0.1, 0.2, n_a)
    b_vals = np.linspace(1.0, 2.0, n_b)
    rows = []
    for a in a_vals:
        for b in b_vals:
            voc, jsc, ff = 0.8, 20.0, 75.0
            pce = (voc * jsc * (ff / 100.0)) / P_IN_MW_CM2 * 100.0
            rows.append(
                {
                    "HTL thickness (μm)": a,
                    "Absorber Thickness (μm)": b,
                    "ETL thickness (μm)": 0.03,
                    "N_A (1/cm3)": 1e20,
                    "N_D (1/cm3)": 1e21,
                    "PCE (%)": pce,
                    "Voc (V)": voc,
                    "Jsc (mA/cm2)": jsc,
                    "FF (%)": ff,
                    "V_MPP (V)": voc * 0.9,
                    "J_MPP (mA/cm2)": jsc * 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_synthetic_clean_frame_passes():
    df = _make_synthetic_clean_frame()
    report = verify_dataset_integrity(df)
    assert report.ok
    assert report.is_full_factorial


def test_detects_duplicate_input_combo():
    df = _make_synthetic_clean_frame()
    dup_row = df.iloc[[0]].copy()
    df_with_dup = pd.concat([df, dup_row], ignore_index=True)
    report = verify_dataset_integrity(df_with_dup)
    assert not report.ok
    assert report.n_duplicate_input_combos == 1
    assert not report.is_full_factorial


def test_detects_missing_grid_combo():
    df = _make_synthetic_clean_frame()
    df_missing = df.iloc[:-1].copy()  # drop last row -> incomplete grid
    report = verify_dataset_integrity(df_missing)
    assert not report.ok
    assert not report.is_full_factorial


def test_detects_pce_identity_violation():
    df = _make_synthetic_clean_frame()
    df.loc[0, "PCE (%)"] += 5.0  # break the identity for one row
    report = verify_dataset_integrity(df, pce_error_tolerance=1e-3)
    assert not report.ok
    assert report.pce_identity_max_abs_error > 1.0


def test_detects_physicality_violation():
    df = _make_synthetic_clean_frame()
    df.loc[0, "FF (%)"] = 150.0  # unphysical
    report = verify_dataset_integrity(df)
    assert not report.ok
    assert report.n_physicality_violations["FF (%)"] >= 1


def test_detects_missing_values():
    df = _make_synthetic_clean_frame()
    df.loc[0, "Voc (V)"] = np.nan
    report = verify_dataset_integrity(df)
    assert not report.ok
    assert report.n_missing_values >= 1
