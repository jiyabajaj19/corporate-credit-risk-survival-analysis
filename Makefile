setup:
	pip install -r requirements.txt

simulate:
	python -m scripts.generate_baseline_data

montecarlo:
	python -m scripts.run_monte_carlo

figures:
	python -m scripts.create_monte_carlo_visualizations

test:
	pytest -v

all:
	python -m scripts.generate_baseline_data
	python -m scripts.run_monte_carlo
	python -m scripts.create_monte_carlo_visualizations