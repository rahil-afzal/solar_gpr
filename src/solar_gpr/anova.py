"""
Type-II ANOVA and eta-squared variance decomposition.

Identifies which of the five design parameters (and which pairwise
interactions) actually drive each output metric, treating every
parameter as categorical (its discrete tested levels) rather than
continuous -- appropriate here because the design is a full factorial
grid with only 5-8 levels per factor.

NOTE: fitting the full main-effects + all-pairwise-interactions OLS
model across all four outputs is the "expensive" step referenced in
the CLI script (scripts/run_anova.py); this module only defines the
logic, it does not run it on import.
"""

from __future__ import annotations

import itertools

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .data import INPUT_COLS


def _safe_factor_names(cols: list[str]) -> dict[str, str]:
    """Map arbitrary column names (with spaces, units, symbols) to
    short, patsy-formula-safe identifiers f0, f1, ...
    """
    return {c: f"f{i}" for i, c in enumerate(cols)}


def build_anova_formula(factor_cols: list[str]) -> tuple[str, dict[str, str]]:
    """Build a patsy formula with all main effects and pairwise
    interactions among `factor_cols`, treating each as categorical.

    Returns (formula_string, safe_name_map) where safe_name_map maps
    the original column name -> the formula-safe factor name used
    inside the formula (e.g. "N_A (1/cm3)" -> "f3").
    """
    safe_names = _safe_factor_names(factor_cols)
    factors = list(safe_names.values())
    main_terms = [f"C({f})" for f in factors]
    interaction_terms = [
        f"C({a}):C({b})" for a, b in itertools.combinations(factors, 2)
    ]
    formula = "target ~ " + " + ".join(main_terms + interaction_terms)
    return formula, safe_names


def run_anova_for_target(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fit the main-effects + pairwise-interaction OLS model for a
    single output column and return a Type-II ANOVA table with an
    added `eta_sq` column, indexed by human-readable term names.

    This is the expensive-ish step (one OLS fit + anova_lm call per
    target); cheap for this dataset size (5000 rows, <=5 levels per
    factor) but intentionally NOT run at import time or module load.
    """
    factor_cols = factor_cols or INPUT_COLS
    formula, safe_names = build_anova_formula(factor_cols)
    label_map = {v: k for k, v in safe_names.items()}

    d = df[factor_cols + [target_col]].rename(columns=safe_names)
    d = d.rename(columns={target_col: "target"})

    model = smf.ols(formula, data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()

    def prettify(term: str) -> str:
        for code, name in label_map.items():
            term = term.replace(f"C({code})", name)
        return term

    table.index = [prettify(i) for i in table.index]
    return table.sort_values("eta_sq", ascending=False)


def run_anova_all_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run `run_anova_for_target` for each target and return a single
    DataFrame of eta-squared values, one column per target, indexed by
    term (main effects + pairwise interactions).
    """
    results = {}
    for target in target_cols:
        table = run_anova_for_target(df, target, factor_cols=factor_cols)
        results[target] = table["eta_sq"]
    return pd.DataFrame(results).round(4)
