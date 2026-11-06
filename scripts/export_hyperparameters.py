#!/usr/bin/env python3
"""
Export the fitted hyperparameters of the final Voc/Jsc/FF GPs
(results/metrics/surrogate_model.joblib, written by run_surrogate_fit.py)
as a CSV and a LaTeX table (Table tab:hyperparameters).

Values are in normalized units: inputs are standardized (zero mean, unit
variance after the log10 transform of N_A and N_D) and the target is
normalized inside scikit-learn. Cheap: no fitting is done here.

The script also lists any hyperparameter that sits at an optimizer bound,
and prints the log marginal likelihood of each model.

Usage:
    python scripts/export_hyperparameters.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_gpr.features import FEATURE_COLS  # noqa: E402
from solar_gpr.pce_validation import extract_hyperparameters, flag_bound_hits  # noqa: E402


def sci(x: float) -> str:
    """Compact scientific notation for LaTeX."""
    if x == 0:
        return "0"
    m, e = f"{x:.2e}".split("e")
    e = int(e)
    return f"{float(m):.2f}" if e == 0 else rf"${float(m):.2f}\times10^{{{e}}}$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "results" / "metrics" / "surrogate_model.joblib")
    p.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = p.parse_args()

    surrogate = joblib.load(args.model_path)
    models = {
        "Voc": surrogate.voc_gp, "Jsc": surrogate.jsc_gp, "FF": surrogate.ff_gp,
    }

    rows, flags = [], {}
    for name, gp in models.items():
        hp = extract_hyperparameters(gp.model, list(FEATURE_COLS))
        hp["target"] = name
        hp["kernel"] = gp.kernel_name
        hp["kernel_"] = str(gp.model.kernel_)
        rows.append(hp)
        flags[name] = flag_bound_hits(gp.model)

    df = pd.DataFrame(rows)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.tables_dir / "final_gp_hyperparameters.csv", index=False, encoding="utf-8")

    feat = list(FEATURE_COLS)
    header = (r"Target & $\ell_{t_{HTL}}$ & $\ell_{t_{abs}}$ & $\ell_{t_{ETL}}$ & "
              r"$\ell_{\log N_A}$ & $\ell_{\log N_D}$ & $\sigma_f^2$ & $\sigma_n^2$ \\")
    tex = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Fitted hyperparameters of the final Mat\'ern 3/2 models (normalized units).}",
        r"\label{tab:hyperparameters}", r"\begin{tabular}{lccccccc}", r"\toprule",
        header, r"\midrule",
    ]
    label = {"Voc": r"$V_{oc}$", "Jsc": r"$J_{sc}$", "FF": r"$FF$"}
    for _, r in df.iterrows():
        cells = [label[r.target]] + [sci(r[f]) for f in feat] + [
            sci(r.signal_variance), sci(r.noise_variance)]
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.tables_dir / "final_gp_hyperparameters.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8")

    print(df.drop(columns=["kernel_"]).to_string(index=False))
    for name, f in flags.items():
        print(f"{name}: hyperparameters at an optimizer bound -> {f if f else 'none'}")
    print("\nFeature order:", feat)


if __name__ == "__main__":
    main()
