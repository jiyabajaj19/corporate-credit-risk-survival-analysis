from __future__ import annotations

from pathlib import Path

from src.evaluation.reporting import (
    create_bias_comparison_table,
    create_model_level_summary,
    load_monte_carlo_summary,
    plot_absolute_bias,
    plot_coverage_probabilities,
    plot_true_vs_estimated,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "time_varying_monte_carlo_summary.csv"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "simulation_study"
)


def main() -> None:
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = load_monte_carlo_summary(
        SUMMARY_PATH
    )

    model_summary = create_model_level_summary(
        summary
    )

    bias_comparison = (
        create_bias_comparison_table(summary)
    )

    model_summary_path = (
        REPORT_DIRECTORY
        / "model_level_summary.csv"
    )

    bias_table_path = (
        REPORT_DIRECTORY
        / "bias_comparison.csv"
    )

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    bias_comparison.to_csv(
        bias_table_path,
        index=False,
    )

    plot_absolute_bias(
        summary,
        REPORT_DIRECTORY
        / "absolute_bias_comparison.png",
    )

    plot_coverage_probabilities(
        summary,
        REPORT_DIRECTORY
        / "coverage_probabilities.png",
    )

    plot_true_vs_estimated(
        summary,
        REPORT_DIRECTORY
        / "true_vs_estimated_coefficients.png",
    )

    print("Model-level summary")
    print("-------------------")
    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Simulation report generated in:"
    )
    print(REPORT_DIRECTORY)


if __name__ == "__main__":
    main()