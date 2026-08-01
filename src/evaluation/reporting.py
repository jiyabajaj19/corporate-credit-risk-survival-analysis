from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Baseline-at-entry Cox",
    "Left-truncated baseline Cox",
    "Time-varying Cox",
]


def load_monte_carlo_summary(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate the Monte Carlo summary."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Monte Carlo summary not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "model",
        "covariate",
        "true_coefficient",
        "mean_estimate",
        "bias",
        "absolute_bias",
        "empirical_sd",
        "mean_standard_error",
        "rmse",
        "coverage_probability",
        "successful_repetitions",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Monte Carlo summary is missing columns: "
            f"{sorted(missing)}"
        )

    return data


def create_model_level_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate covariate-level metrics by model."""

    model_summary = (
        summary.groupby("model", as_index=False)
        .agg(
            mean_absolute_bias=(
                "absolute_bias",
                "mean",
            ),
            median_absolute_bias=(
                "absolute_bias",
                "median",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_coverage_probability=(
                "coverage_probability",
                "mean",
            ),
            minimum_coverage_probability=(
                "coverage_probability",
                "min",
            ),
            maximum_absolute_bias=(
                "absolute_bias",
                "max",
            ),
        )
    )

    model_summary["model"] = pd.Categorical(
        model_summary["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return (
        model_summary.sort_values("model")
        .reset_index(drop=True)
    )


def create_bias_comparison_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per covariate and model columns."""

    table = summary.pivot(
        index="covariate",
        columns="model",
        values="absolute_bias",
    )

    available_models = [
        model
        for model in MODEL_ORDER
        if model in table.columns
    ]

    table = table[available_models]

    if (
        "Baseline-at-entry Cox" in table.columns
        and "Time-varying Cox" in table.columns
    ):
        baseline_bias = table[
            "Baseline-at-entry Cox"
        ]

        time_varying_bias = table[
            "Time-varying Cox"
        ]

        table["bias_reduction"] = (
            baseline_bias - time_varying_bias
        )

        table[
            "relative_bias_reduction"
        ] = np.where(
            baseline_bias > 0,
            (
                baseline_bias
                - time_varying_bias
            )
            / baseline_bias,
            np.nan,
        )

    return table.reset_index()


def plot_absolute_bias(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot absolute bias by covariate and model."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = summary.copy()

    pivot = plot_data.pivot(
        index="covariate",
        columns="model",
        values="absolute_bias",
    )

    available_models = [
        model
        for model in MODEL_ORDER
        if model in pivot.columns
    ]

    pivot = pivot[available_models]
    pivot = pivot.sort_values(
        "Time-varying Cox"
        if "Time-varying Cox" in pivot.columns
        else available_models[0],
    )

    ax = pivot.plot(
        kind="barh",
        figsize=(11, 8),
    )

    ax.set_title(
        "Absolute coefficient bias by model"
    )
    ax.set_xlabel("Absolute bias")
    ax.set_ylabel("Financial covariate")
    ax.legend(title="Model")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_coverage_probabilities(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot confidence-interval coverage probabilities."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = summary.pivot(
        index="covariate",
        columns="model",
        values="coverage_probability",
    )

    available_models = [
        model
        for model in MODEL_ORDER
        if model in pivot.columns
    ]

    pivot = pivot[available_models]

    ax = pivot.plot(
        kind="barh",
        figsize=(11, 8),
    )

    ax.axvline(
        0.95,
        linestyle="--",
        linewidth=1.5,
        label="Nominal 95% coverage",
    )

    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Empirical 95% confidence-interval coverage"
    )
    ax.set_xlabel("Coverage probability")
    ax.set_ylabel("Financial covariate")
    ax.legend(title="Model")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_true_vs_estimated(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Compare true and mean estimated coefficients."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 8),
    )

    for model in MODEL_ORDER:
        model_data = summary[
            summary["model"] == model
        ]

        if model_data.empty:
            continue

        ax.scatter(
            model_data["true_coefficient"],
            model_data["mean_estimate"],
            label=model,
            alpha=0.8,
        )

    minimum = min(
        summary["true_coefficient"].min(),
        summary["mean_estimate"].min(),
    )

    maximum = max(
        summary["true_coefficient"].max(),
        summary["mean_estimate"].max(),
    )

    padding = 0.05 * (maximum - minimum)

    ax.plot(
        [
            minimum - padding,
            maximum + padding,
        ],
        [
            minimum - padding,
            maximum + padding,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Perfect recovery",
    )

    ax.set_xlim(
        minimum - padding,
        maximum + padding,
    )
    ax.set_ylim(
        minimum - padding,
        maximum + padding,
    )

    ax.set_title(
        "True versus mean estimated coefficients"
    )
    ax.set_xlabel("True coefficient")
    ax.set_ylabel("Mean estimated coefficient")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()