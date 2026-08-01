# ==============================
# Corporate Credit Risk Survival Analysis
# ==============================

PYTHON = python

.DEFAULT_GOAL := help

.PHONY: help test clean format \
        sec pipeline \
        bankrupt controls real \
        simulation all

# ------------------------------
# Help
# ------------------------------

help:
	@echo ""
	@echo "Corporate Credit Risk Survival Analysis"
	@echo ""
	@echo "Available targets:"
	@echo "  make test          Run all tests"
	@echo "  make sec           Process SEC Company Facts"
	@echo "  make bankrupt      Build bankrupt survival dataset"
	@echo "  make controls      Build control survival dataset"
	@echo "  make real          Build combined survival dataset"
	@echo "  make simulation    Run simulation study"
	@echo "  make pipeline      Run complete real-data pipeline"
	@echo "  make clean         Remove generated outputs"
	@echo ""

# ------------------------------
# Testing
# ------------------------------

test:
	$(PYTHON) -m pytest -v

# ------------------------------
# SEC Processing
# ------------------------------

sec:
	$(PYTHON) -m scripts.build_raw_financial_facts
	$(PYTHON) -m scripts.build_quarterly_panel
	$(PYTHON) -m scripts.build_financial_features
	$(PYTHON) -m scripts.clean_financial_features

# ------------------------------
# Real Survival Datasets
# ------------------------------

bankrupt:
	$(PYTHON) -m scripts.build_survival_dataset

controls:
	$(PYTHON) -m scripts.build_control_survival_dataset

real:
	$(PYTHON) -m scripts.build_real_survival_dataset

pipeline: bankrupt controls real

# ------------------------------
# Simulation
# ------------------------------

simulation:
	$(PYTHON) -m scripts.run_time_varying_monte_carlo
	$(PYTHON) -m scripts.generate_simulation_report

# ------------------------------
# Cleanup
# ------------------------------

clean:
	@echo "Removing generated outputs..."

	@if exist data\processed del /Q data\processed\*.csv
	@if exist reports del /Q reports\*.csv
	@if exist reports del /Q reports\*.png