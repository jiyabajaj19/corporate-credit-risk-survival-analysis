from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Naive Cox",
    "Left-Truncated Cox",
]

COVARIATE_LABELS = {
    "leverage": "Leverage",
    "low_interest_coverage": (
        "Low interest coverage"
    ),
}


def _validate_estimates(
    estimates: pd.DataFrame,
) -> None:
    required_columns = {
        "model",
        "covariate",
        "true_coefficient",
        "estimate",
        "lower_95",
        "upper_95",
        "covered",
    }

    missing_columns = (
        required_columns - set(estimates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Estimates data is missing columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_summary(
    summary: pd.DataFrame,
) -> None:
    required_columns = {
        "model",
        "covariate",
        "true_coefficient",
        "bias",
        "absolute_bias",
        "rmse",
        "coverage_probability",
    }

    missing_columns = (
        required_columns - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Summary data is missing columns: "
            f"{sorted(missing_columns)}"
        )


def plot_coefficient_distributions(
    estimates: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Plot the sampling distribution of each coefficient estimate.

    A separate figure is produced for each covariate.
    """
    _validate_estimates(estimates)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths: list[Path] = []

    for covariate, display_name in (
        COVARIATE_LABELS.items()
    ):
        covariate_data = estimates[
            estimates["covariate"] == covariate
        ]

        if covariate_data.empty:
            continue

        true_value = float(
            covariate_data[
                "true_coefficient"
            ].iloc[0]
        )

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        for model in MODEL_ORDER:
            model_estimates = covariate_data.loc[
                covariate_data["model"] == model,
                "estimate",
            ]

            if model_estimates.empty:
                continue

            axis.hist(
                model_estimates,
                bins=20,
                alpha=0.55,
                density=True,
                label=model,
                edgecolor="black",
                linewidth=0.5,
            )

        axis.axvline(
            true_value,
            linestyle="--",
            linewidth=2,
            label=f"True coefficient = {true_value:.2f}",
        )

        axis.set_title(
            f"Sampling distribution: {display_name}"
        )

        axis.set_xlabel(
            "Estimated log-hazard coefficient"
        )

        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        output_path = (
            output_directory
            / f"{covariate}_estimate_distribution.png"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def plot_absolute_bias(
    summary: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Compare absolute coefficient bias across models.
    """
    _validate_summary(summary)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = summary.pivot(
        index="covariate",
        columns="model",
        values="absolute_bias",
    )

    pivot = pivot.reindex(
        index=list(COVARIATE_LABELS),
        columns=MODEL_ORDER,
    )

    x_positions = np.arange(len(pivot.index))
    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    for model_index, model in enumerate(
        MODEL_ORDER
    ):
        positions = (
            x_positions
            + (
                model_index
                - (len(MODEL_ORDER) - 1) / 2
            )
            * bar_width
        )

        values = pivot[model].to_numpy()

        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=model,
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.3f}"
                for value in values
            ],
            padding=3,
        )

    axis.set_title(
        "Absolute bias by estimator"
    )

    axis.set_xlabel("Credit-risk covariate")
    axis.set_ylabel("Absolute bias")

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            COVARIATE_LABELS[covariate]
            for covariate in pivot.index
        ]
    )

    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "absolute_bias_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_rmse_comparison(
    summary: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Compare root mean squared error across models.
    """
    _validate_summary(summary)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = summary.pivot(
        index="covariate",
        columns="model",
        values="rmse",
    )

    pivot = pivot.reindex(
        index=list(COVARIATE_LABELS),
        columns=MODEL_ORDER,
    )

    x_positions = np.arange(len(pivot.index))
    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    for model_index, model in enumerate(
        MODEL_ORDER
    ):
        positions = (
            x_positions
            + (
                model_index
                - (len(MODEL_ORDER) - 1) / 2
            )
            * bar_width
        )

        values = pivot[model].to_numpy()

        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=model,
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.3f}"
                for value in values
            ],
            padding=3,
        )

    axis.set_title(
        "Estimator RMSE comparison"
    )

    axis.set_xlabel("Credit-risk covariate")
    axis.set_ylabel("Root mean squared error")

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            COVARIATE_LABELS[covariate]
            for covariate in pivot.index
        ]
    )

    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "rmse_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_coverage_probability(
    summary: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Compare empirical 95% confidence-interval coverage.

    The dashed horizontal line represents the nominal 95% target.
    """
    _validate_summary(summary)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = summary.pivot(
        index="covariate",
        columns="model",
        values="coverage_probability",
    )

    pivot = pivot.reindex(
        index=list(COVARIATE_LABELS),
        columns=MODEL_ORDER,
    )

    x_positions = np.arange(len(pivot.index))
    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    for model_index, model in enumerate(
        MODEL_ORDER
    ):
        positions = (
            x_positions
            + (
                model_index
                - (len(MODEL_ORDER) - 1) / 2
            )
            * bar_width
        )

        values = pivot[model].to_numpy()

        bars = axis.bar(
            positions,
            values,
            width=bar_width,
            label=model,
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.1%}"
                for value in values
            ],
            padding=3,
        )

    axis.axhline(
        0.95,
        linestyle="--",
        linewidth=2,
        label="Nominal 95% coverage",
    )

    axis.set_title(
        "Confidence-interval coverage"
    )

    axis.set_xlabel("Credit-risk covariate")
    axis.set_ylabel("Coverage probability")

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            COVARIATE_LABELS[covariate]
            for covariate in pivot.index
        ]
    )

    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "coverage_probability.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_all_visualizations(
    estimates: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Generate every Monte Carlo visualization.
    """
    output_paths = (
        plot_coefficient_distributions(
            estimates=estimates,
            output_directory=output_directory,
        )
    )

    output_paths.append(
        plot_absolute_bias(
            summary=summary,
            output_directory=output_directory,
        )
    )

    output_paths.append(
        plot_rmse_comparison(
            summary=summary,
            output_directory=output_directory,
        )
    )

    output_paths.append(
        plot_coverage_probability(
            summary=summary,
            output_directory=output_directory,
        )
    )

    return output_paths