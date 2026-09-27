"""
Tests for solar_gpr.features / surrogate / validation / optimization.

These use tiny synthetic datasets (8-16 rows) and minimal GP settings
(n_restarts_optimizer=0-1) purely to confirm the code paths are
correct -- NOT to validate model quality. The real kernel comparison
and LOLO-CV run (5000 rows, 3 kernels, 28 folds/target) is the
expensive step deferred to `scripts/run_lolo_cv.py`, run separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solar_gpr.data import INPUT_COLS, P_IN_MW_CM2
from solar_gpr.features import FEATURE_COLS, from_feature_space, to_feature_space
from solar_gpr.optimization import (
    make_extrapolation_candidate_grid,
    rank_candidates_by_ucb,
)
from solar_gpr.surrogate import (
    KERNEL_NAMES,
    SolarSurrogate,
    build_kernel,
    derive_pce,
    fit_sub_metric_gp,
)
from solar_gpr.uncertainty import propagate_pce_uncertainty
from solar_gpr.validation import (
    generate_all_lolo_folds,
    generate_lolo_folds,
    run_lolo_cv,
    summarize_lolo_results,
)


# ---------------------------------------------------------------------
# Fixtures: tiny synthetic full-factorial dataset
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """A small (3 x 2 x 1 x 2 x 1 = 12 row) full-factorial synthetic
    dataset with a deterministic, smooth relationship between inputs
    and outputs, small enough to fit a GP on in milliseconds.
    """
    rng = np.random.default_rng(0)
    htl_vals = [0.1, 0.15, 0.2]
    abs_vals = [1.0, 2.0]
    etl_vals = [0.03]
    na_vals = [1e20, 3e20]
    nd_vals = [1e21]

    rows = []
    for htl in htl_vals:
        for absorber in abs_vals:
            for etl in etl_vals:
                for na in na_vals:
                    for nd in nd_vals:
                        # Smooth synthetic ground truth, no noise.
                        voc = 0.6 + 0.05 * np.log10(na / 1e19) - 0.02 * absorber
                        jsc = 15.0 + 3.0 * absorber - 2.0 * htl
                        ff = 70.0 + 2.0 * np.log10(nd / na)
                        pce = (voc * jsc * (ff / 100.0)) / P_IN_MW_CM2 * 100.0
                        rows.append(
                            {
                                "HTL thickness (μm)": htl,
                                "Absorber Thickness (μm)": absorber,
                                "ETL thickness (μm)": etl,
                                "N_A (1/cm3)": na,
                                "N_D (1/cm3)": nd,
                                "PCE (%)": pce,
                                "Voc (V)": voc,
                                "Jsc (mA/cm2)": jsc,
                                "FF (%)": ff,
                                "V_MPP (V)": voc * 0.9,
                                "J_MPP (mA/cm2)": jsc * 0.9,
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------

def test_feature_space_round_trip(tiny_df):
    features = to_feature_space(tiny_df)
    assert list(features.columns) == FEATURE_COLS
    assert "N_A (1/cm3)_log10" in features.columns

    recovered = from_feature_space(features)
    for col in INPUT_COLS:
        np.testing.assert_allclose(
            recovered[col].to_numpy(), tiny_df[col].to_numpy(), rtol=1e-10
        )


def test_log_transform_rejects_nonpositive():
    df = pd.DataFrame({c: [1.0] for c in INPUT_COLS})
    df["N_A (1/cm3)"] = -1.0
    with pytest.raises(ValueError):
        to_feature_space(df)


# ---------------------------------------------------------------------
# surrogate.py
# ---------------------------------------------------------------------

def test_derive_pce_matches_identity():
    voc = np.array([0.8, 0.7])
    jsc = np.array([20.0, 18.0])
    ff = np.array([75.0, 70.0])
    pce = derive_pce(voc, jsc, ff)
    expected = (voc * jsc * (ff / 100.0)) / P_IN_MW_CM2 * 100.0
    np.testing.assert_allclose(pce, expected)


def test_build_kernel_all_candidates():
    for name in KERNEL_NAMES:
        kernel = build_kernel(name, n_features=5)
        assert kernel is not None

    with pytest.raises(ValueError):
        build_kernel("not_a_real_kernel", n_features=5)


def test_fit_sub_metric_gp_tiny(tiny_df):
    # n_restarts_optimizer=0: fastest possible fit, just checks the
    # pipeline (standardize -> fit -> predict) works end to end.
    gp = fit_sub_metric_gp(
        tiny_df, "Voc (V)", kernel_name="rbf", n_restarts_optimizer=0
    )
    from solar_gpr.features import to_feature_space

    X = to_feature_space(tiny_df).to_numpy(dtype=np.float64)
    preds = gp.predict(X)
    assert preds.shape == (len(tiny_df),)
    assert np.all(np.isfinite(preds))


def test_solar_surrogate_fit_and_predict_pce(tiny_df):
    surrogate = SolarSurrogate.fit(tiny_df, kernel_name="rbf", n_restarts_optimizer=0)
    pce_pred = surrogate.predict_pce(tiny_df)
    assert pce_pred.shape == (len(tiny_df),)
    assert np.all(np.isfinite(pce_pred))
    # Predictions on training points for a noiseless-ish smooth
    # function with a near-zero WhiteKernel floor should be close to
    # the true values -- loose tolerance since n_restarts=0.
    np.testing.assert_allclose(pce_pred, tiny_df["PCE (%)"].to_numpy(), atol=1.0)


# ---------------------------------------------------------------------
# validation.py (LOLO-CV mechanics, using a trivial predictor)
# ---------------------------------------------------------------------

def test_generate_lolo_folds_covers_all_levels(tiny_df):
    folds = list(generate_lolo_folds(tiny_df, "HTL thickness (μm)"))
    assert len(folds) == 3  # 3 unique HTL levels
    all_test_idx = np.concatenate([f.test_index for f in folds])
    assert sorted(all_test_idx) == sorted(tiny_df.index.to_numpy())


def test_generate_all_lolo_folds(tiny_df):
    folds = generate_all_lolo_folds(tiny_df, INPUT_COLS)
    # 3 + 2 + 1 + 2 + 1 = 9 total folds across all factors
    assert len(folds) == 9


def _mean_predictor(train_df, test_df, target_col):
    """Trivial fit_predict_fn: predicts the training mean for every
    test row. Used only to test run_lolo_cv's orchestration logic
    without needing a real (expensive) GP fit per fold.
    """
    return np.full(len(test_df), train_df[target_col].mean())


def test_run_lolo_cv_with_trivial_predictor(tiny_df):
    results = run_lolo_cv(
        tiny_df, target_col="Voc (V)", factor_cols=["HTL thickness (μm)", "Absorber Thickness (μm)"],
        fit_predict_fn=_mean_predictor, n_jobs=1,
    )
    assert len(results) == 3 + 2  # 3 HTL levels + 2 Absorber levels
    assert {"factor_col", "held_out_level", "n_train", "n_test", "r2", "rmse"} <= set(results.columns)
    assert (results["n_train"] + results["n_test"] == len(tiny_df)).all()

    summary = summarize_lolo_results(results)
    assert "HTL thickness (μm)" in summary.index
    assert "Absorber Thickness (μm)" in summary.index


def test_run_lolo_cv_parallel_matches_serial(tiny_df):
    kwargs = dict(
        df=tiny_df, target_col="Voc (V)", factor_cols=["HTL thickness (μm)"],
        fit_predict_fn=_mean_predictor,
    )
    serial = run_lolo_cv(**kwargs, n_jobs=1).sort_values("held_out_level").reset_index(drop=True)
    parallel = run_lolo_cv(**kwargs, n_jobs=2).sort_values("held_out_level").reset_index(drop=True)
    pd.testing.assert_frame_equal(serial, parallel)


# ---------------------------------------------------------------------
# optimization.py / uncertainty.py
# ---------------------------------------------------------------------

def test_make_extrapolation_candidate_grid(tiny_df):
    base_row = tiny_df.loc[0, INPUT_COLS].to_dict()
    values = np.array([1e20, 2e20, 3e20])
    grid = make_extrapolation_candidate_grid(base_row, "N_A (1/cm3)", values)
    assert len(grid) == 3
    np.testing.assert_allclose(grid["N_A (1/cm3)"].to_numpy(), values)
    # other columns held fixed at base_row
    assert (grid["Absorber Thickness (μm)"] == base_row["Absorber Thickness (μm)"]).all()


def test_uncertainty_propagation_and_ucb_ranking_tiny(tiny_df):
    surrogate = SolarSurrogate.fit(tiny_df, kernel_name="rbf", n_restarts_optimizer=0)

    base_row = tiny_df.loc[0, INPUT_COLS].to_dict()
    values = np.array([1e20, 2e20, 3e20])
    candidates = make_extrapolation_candidate_grid(base_row, "N_A (1/cm3)", values)

    # Small n_samples: this is a plumbing test, not a precision test.
    uncertainty = propagate_pce_uncertainty(surrogate, candidates, n_samples=200, random_state=0)
    assert uncertainty.mean_pce.shape == (3,)
    assert np.all(uncertainty.std_pce >= 0)

    ranking = rank_candidates_by_ucb(surrogate, candidates, beta=2.0, n_mc_samples=200, random_state=0)
    assert len(ranking.ranked_index) == 3
    top1 = ranking.top(1)
    assert len(top1) == 1
    assert {"mean_pce", "std_pce", "ucb_score"} <= set(top1.columns)
