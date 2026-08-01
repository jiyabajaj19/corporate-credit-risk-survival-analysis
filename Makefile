PYTHON := python

.DEFAULT_GOAL := help

.PHONY: help test simulation simulation-report \
	bankrupt-survival control-survival real-survival \
	model-selection model-comparison figures \
	final-analysis clean

help:
	@echo ""
	@echo "Corporate Credit Risk Survival Analysis"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make test                 Run the complete test suite"
	@echo "  make simulation           Run time-varying Monte Carlo simulation"
	@echo "  make simulation-report    Generate the simulation study report"
	@echo "  make bankrupt-survival    Build bankrupt-firm survival intervals"
	@echo "  make control-survival     Build censored control intervals"
	@echo "  make real-survival        Build the combined real survival dataset"
	@echo "  make model-selection      Tune the ridge Cox penalizer"
	@echo "  make model-comparison     Compare all real-data Cox models"
	@echo "  make figures              Generate real-model figures"
	@echo "  make final-analysis       Run model selection, comparison and figures"
	@echo "  make clean                Remove generated analysis outputs"
	@echo ""

test:
	$(PYTHON) -m pytest -v

simulation:
	$(PYTHON) -m scripts.run_time_varying_monte_carlo

simulation-report:
	$(PYTHON) -m scripts.generate_simulation_report

bankrupt-survival:
	$(PYTHON) -m scripts.build_survival_dataset

control-survival:
	$(PYTHON) -m scripts.build_control_survival_dataset

real-survival: bankrupt-survival control-survival
	$(PYTHON) -m scripts.build_real_survival_dataset

model-selection:
	$(PYTHON) -m scripts.run_real_model_selection

model-comparison:
	$(PYTHON) -m scripts.run_real_model_comparison

figures:
	$(PYTHON) -m scripts.create_real_model_figures

final-analysis: model-selection model-comparison figures
	@echo ""
	@echo "Final empirical analysis completed."

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; paths=[Path('reports/real_models/model_selection'),Path('reports/real_models/figures')]; [shutil.rmtree(p) for p in paths if p.exists()]"
	$(PYTHON) -c "from pathlib import Path; files=[Path('reports/real_models/model_comparison.csv'),Path('reports/real_models/coefficient_results.csv'),Path('reports/real_models/feature_scaling.csv'),Path('reports/real_models/summary.txt')]; [p.unlink() for p in files if p.exists()]"
	@echo "Generated real-model outputs removed."