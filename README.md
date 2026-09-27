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

- **Full factorial, not random sampling.** The dataset is an
  exhaustive 5×8×5×5×5 grid, not a Latin Hypercube or random sample —
  this is why LOLO-CV (not a random train/test split) is used for
  validation, and why the GP's role is interpolation/extrapolation
  between/beyond tested grid points, not "avoiding expensive
  simulations" (all 5000 grid points already exist).
- **PCE is always derived, never modeled directly**, from independently
  fit Voc/Jsc/FF GPs, to guarantee the efficiency identity holds even
  when extrapolating. This does *not* bound the individual sub-metrics
  (e.g. FF <= 100%) — see the methodology's "Physically Consistent
  Surrogate Modeling" subsection for why that is a stated limitation,
  not silently assumed away.
- **The three sub-metric GPs are treated as independent** during Monte
  Carlo uncertainty propagation, which is a known simplification (see
  `solar_gpr.uncertainty` docstring) that tends to inflate the
  propagated PCE uncertainty — a conservative bias, not a silent bug.

## License

MIT — see `LICENSE`. *(Confirm this is the intended license before
publishing.)*
