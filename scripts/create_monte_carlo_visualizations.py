from pathlib import Path

import pandas as pd

from src.evaluation.visualization import (
    create_all_visualizations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ESTIMATES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monte_carlo_estimates.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monte_carlo_summary.csv"
)

FIGURES_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


def main() -> None:
    if not ESTIMATES_PATH.exists():
        raise FileNotFoundError(
            "Monte Carlo estimates were not found. "
            "Run `python -m scripts.run_monte_carlo` "
            "first."
        )

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Monte Carlo summary was not found. "
            "Run `python -m scripts.run_monte_carlo` "
            "first."
        )

    print("Loading Monte Carlo results...")
    print(f"Estimates: {ESTIMATES_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print()

    estimates = pd.read_csv(ESTIMATES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    output_paths = create_all_visualizations(
        estimates=estimates,
        summary=summary,
        output_directory=FIGURES_DIRECTORY,
    )

    print("Visualizations created successfully:")
    print("------------------------------------")

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()