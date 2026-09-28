# Physics-Informed GPR Surrogate Modeling for a SCAPS-1D Solar Cell Dataset

Reproducible code for the ML surrogate-modeling subsection of [paper
title / venue — TBD]. Given a full-factorial SCAPS-1D dataset of 5000
simulated solar cell devices, this repository:

1. Verifies the dataset's structure and physical self-consistency.
2. Decomposes output variance (PCE, Voc, Jsc, FF) via ANOVA to
   identify the dominant design parameters.
3. Fits Gaussian Process surrogates for Voc, Jsc, FF (never PCE
   directly — PCE is derived algebraically to guarantee the device
   efficiency identity is never violated).
4. Validates the surrogate honestly via Leave-One-Level-Out
   Cross-Validation (LOLO-CV), which is appropriate for a
   full-factorial design (random splits are misleadingly easy here).
5. Uses the validated surrogate to propose a small number of
   physically motivated extrapolation points (via Monte
   Carlo-propagated uncertainty and a UCB acquisition score) as
   candidates for the next SCAPS-1D simulation to run.

See `[paper draft link — TBD]` for the full methodology writeup this
code implements.

## Repository structure

```
data/           raw dataset (+ provenance notes) and cached processed artifacts
notebooks/      01_eda.ipynb — exploratory data analysis only
src/solar_gpr/  the actual package: data, features, anova, validation,
                surrogate, uncertainty, optimization, plotting
configs/        YAML settings for kernel comparison and BO/UCB search
scripts/        CLI entry points that run the pipeline stages
results/        figures / tables / metrics written by scripts/
tests/          pytest suite (fast; see "Tests" below)
```

All reusable logic lives in `src/solar_gpr/`; `scripts/` are thin CLI
wrappers around it. Nothing expensive runs on import — every model
fit is triggered explicitly from a script or an explicit function
call.

## Setup

```bash
conda env create -f environment.yml
conda activate solar-gpr
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Place the raw dataset at `data/raw/SOLAR_DATASET.xlsx` (see
`data/README.md` for provenance notes and a decision on whether this
file should be committed directly or hosted separately).

## Running the pipeline

Each stage is a standalone script; see `make help` for the full list.
**Cost note:** stages are ordered cheap -> expensive. `lolo-cv` and
`surrogate-fit` are the expensive ones (each fits many Gaussian
Processes with hyperparameter optimization); everything else is fast.

```bash
make check             # dataset integrity verification (cheap)
make anova              # ANOVA / eta-squared table (cheap)
make lolo-cv NJOBS=4    # kernel comparison via LOLO-CV (EXPENSIVE, parallel across cores)
make surrogate-fit KERNEL=matern52   # fit final GPs with the winning kernel (EXPENSIVE)
make bo-extrapolation   # rank extrapolation candidates by UCB (moderate)
make figures            # regenerate all figures from saved results (cheap)
```

`make all` runs the full pipeline in order; prefer running stages
individually the first time through so you can inspect
`results/metrics/lolo_cv_kernel_summary.csv` before choosing `KERNEL`
for `surrogate-fit`.

Every script also runs standalone with `--help` for its full option
list, e.g.:

```bash
python scripts/run_lolo_cv.py --help
```

## Tests

```bash
pytest
```

The test suite is fast (small synthetic datasets, `n_restarts_optimizer=0`
GP fits, a handful of Monte Carlo samples) — it checks that the code
paths are correct, not that the models are well-tuned. It also
includes integrity checks run directly against the real dataset (data
schema, full-factorial completeness, PCE identity, physicality
bounds) — these are the same checks the pipeline scripts gate on
before doing any expensive work.

## Key methodological notes

- The dataset is a full factorial 5×8×5×5×5 grid with 5000 points. It is not a random sample or Latin Hypercube design. For this reason, validation uses LOLO-CV rather than a random train/test split. The GP is used to model the response across and beyond the tested grid, rather than to replace simulations that have not yet been performed.

- PCE is calculated from separately fitted GPs for Voc, Jsc, and FF rather than being modeled as a separate target. This keeps the efficiency identity consistent during interpolation and extrapolation. It does not impose physical bounds on the individual quantities, such as FF ≤ 100%; this limitation is discussed in the "Physically Consistent Surrogate Modeling" section of the methodology.

- For Monte Carlo uncertainty propagation, the three sub-metric GPs are treated as independent. This is a simplifying assumption documented in 'solar_gpr.uncertainty'. Ignoring correlations can make the propagated PCE uncertainty larger, so the resulting uncertainty estimate is generally conservative.

## License

MIT — see `LICENSE`.