# Reproduces the full analysis pipeline, in dependency order.
# Each target is independently re-runnable; expensive targets are
# marked below. Run `make help` for a summary.
#
# NOTE: `make all` runs every expensive step back-to-back. Prefer
# running targets individually the first time so you can inspect
# results (and cost) at each stage.

PYTHON ?= python
NJOBS ?= 4
KERNEL ?= matern32   # set after inspecting results/metrics/lolo_cv_kernel_summary.csv

.PHONY: help check anova lolo-cv surrogate-fit bo-extrapolation figures all clean

help:
	@echo "Targets:"
	@echo "  check           - cheap: dataset integrity verification"
	@echo "  anova           - cheap: ANOVA / eta-squared table"
	@echo "  lolo-cv         - EXPENSIVE: kernel comparison via LOLO-CV (NJOBS=$(NJOBS))"
	@echo "  surrogate-fit   - EXPENSIVE: fit final GPs (KERNEL=$(KERNEL))"
	@echo "  bo-extrapolation- moderate: UCB-ranked extrapolation candidates"
	@echo "  figures         - cheap: regenerate figures from saved results"
	@echo "  all             - run the full pipeline in order (EXPENSIVE)"
	@echo "  clean           - remove generated results (NOT raw data)"

check:
	$(PYTHON) -m solar_gpr.data

anova:
	$(PYTHON) scripts/run_anova.py

lolo-cv:
	$(PYTHON) scripts/run_lolo_cv.py --n-jobs $(NJOBS)

surrogate-fit:
	$(PYTHON) scripts/run_surrogate_fit.py --kernel $(KERNEL)

bo-extrapolation:
	$(PYTHON) scripts/run_bo_extrapolation.py

figures:
	$(PYTHON) scripts/make_figures.py

all: check anova lolo-cv surrogate-fit bo-extrapolation figures

clean:
	rm -rf results/figures/* results/tables/* results/metrics/*
