# Data

## `raw/SOLAR_DATASET.xlsx`

The original, unmodified SCAPS-1D output file. **Never edit this file
in place** — all cleaning/transforms happen in code
(`solar_gpr.data.load_dataset`) so the pipeline is reproducible from
this file alone.

Verified structure (see `solar_gpr.data.verify_dataset_integrity`,
confirmed against this file on 2026-09-26):

- 5000 rows, a **complete full-factorial grid**:
  5 (HTL thickness) × 8 (Absorber thickness) × 5 (ETL thickness) ×
  5 (N_A) × 5 (N_D) = 5000, every combination present exactly once.
- No missing values, no duplicate rows or input combinations.
- Deterministic simulator: `PCE = Voc * Jsc * FF / P_in` holds to
  within ~1e-6 (floating-point precision only, no simulator noise).
- All outputs within physical bounds (0 <= Voc <= 2V, FF <= 100%,
  etc.); N_A < N_D for every row in the tested grid (this means the
  `N_A <= N_D` p-n junction constraint referenced in the BO
  methodology is never active within these bounds — see
  `configs/bo_settings.yaml` if the search range is later widened
  past the tested N_A/N_D ranges, at which point this constraint
  would need to be enforced explicitly, e.g. in `optimization.py`).

## Simulator provenance

*(Fill in: SCAPS-1D version, which parameters were held fixed, mesh
settings, illumination spectrum/intensity used to generate this
dataset. Not yet documented here — needed for the paper's simulator
setup subsection and for anyone trying to regenerate/extend this
dataset with new SCAPS runs.)*

## Should this file be committed to the repo?

`SOLAR_DATASET.xlsx` is ~428 KB — small enough to commit directly
without issue. The current `.gitignore` excludes `data/raw/*` on the
assumption you might prefer to host the dataset separately (e.g.
Zenodo, an institutional repository) and link it here, which is often
preferred for a citable, stable data location in a published paper.

**Decision needed:** either
  (a) remove the `data/raw/*` line from `.gitignore` and commit the
      file directly (simplest, fully self-contained repo), or
  (b) upload it to a data repository, put the DOI/link here, and add
      a `scripts/download_data.py` (not yet created) that fetches it
      into `data/raw/` on first use.

## `processed/`

Empty by default. Populated by pipeline scripts with cached
intermediate artifacts (e.g. feature-space-transformed data, cross-
validation fold indices) to avoid recomputation. Nothing here is
hand-edited; delete freely and regenerate.
