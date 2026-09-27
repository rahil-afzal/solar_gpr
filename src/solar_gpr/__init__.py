"""
solar_gpr: Physics-informed Gaussian Process surrogate modeling for a
SCAPS-1D solar cell dataset.

This package accompanies a paper subsection on ML surrogate modeling for
a full-factorial SCAPS-1D solar cell simulation dataset (N=5000).

Modules
-------
data          Loading + integrity verification of the raw dataset.
features      Physics-informed feature transforms (log-doping, etc).
anova         Type-II ANOVA / eta-squared variance decomposition.
validation    Leave-One-Level-Out Cross-Validation (LOLO-CV).
surrogate     GPR surrogate models (Voc, Jsc, FF) + derived PCE.
uncertainty   Monte Carlo uncertainty propagation for sigma_PCE.
optimization  UCB acquisition + targeted extrapolation search.
plotting      Shared matplotlib styling used by scripts and notebooks.
"""

__version__ = "0.1.0"
