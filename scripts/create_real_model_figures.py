from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "real_models"
)

MODEL_SELECTION_DIRECTORY = (
    REAL_MODEL_DIRECTORY
    / "model_selection"
)

FIGURE_DIRECTORY = (
    REAL_MODEL_DIRECTORY
    / "figures"
)

PENALTY_SUMMARY_PATH = (
    MODEL_SELECTION_DIRECTORY
    / "penalty_summary.csv"
)

SELECTED_PENALTY_PATH = (
    MODEL_SELECTION_DIRECTORY
    / "selected_penalty.txt"
)

COEFFICIENT_RESULTS_PATH = (
    REAL_MODEL_DIRECTORY
    / "coefficient_results.csv"
)


def format_feature_name(
    feature: str,
) -> str:
    """
    Convert internal feature names into readable labels.
    """
    replacements = {
        "leverage": "Leverage",
        "current_ratio": "Current ratio",
        "cash_ratio": "Cash ratio",
        "return_on_assets": "Return on assets",
        "revenue_growth": "Revenue growth",
        "operating_cash_flow_ratio": (
            "Operating cash-flow ratio"
        ),
        "log_total_assets": "Log total assets",
        "operating_margin": "Operating margin",
    }

    return replacements.get(
        feature,
        feature.replace("_", " ").title(),
    )


def load_selected_penalty() -> float:
    if not SELECTED_PENALTY_PATH.exists():
        raise FileNotFoundError(
            "Selected penalty file not found: "
            f"{SELECTED_PENALTY_PATH}"
        )

    value = float(
        SELECTED_PENALTY_PATH
        .read_text(encoding="utf-8")
        .strip()
    )

    if value < 0:
        raise ValueError(
            "Selected penalty cannot be negative."
        )

    return value


def load_penalty_summary() -> pd.DataFrame:
    if not PENALTY_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Penalty summary not found: "
            f"{PENALTY_SUMMARY_PATH}"
        )

    summary = pd.read_csv(
        PENALTY_SUMMARY_PATH
    )

    required_columns = {
        "penalizer",
        "mean_validation_concordance",
        "standard_deviation_concordance",
    }

    missing = (
        required_columns
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "Penalty summary is missing columns: "
            f"{sorted(missing)}"
        )

    return (
        summary.sort_values("penalizer")
        .reset_index(drop=True)
    )


def load_coefficient_results() -> pd.DataFrame:
    if not COEFFICIENT_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Coefficient results not found: "
            f"{COEFFICIENT_RESULTS_PATH}"
        )

    coefficients = pd.read_csv(
        COEFFICIENT_RESULTS_PATH
    )

    required_columns = {
        "model",
        "feature",
        "coefficient",
        "hazard_ratio",
        "hazard_ratio_lower_95",
        "hazard_ratio_upper_95",
    }

    missing = (
        required_columns
        - set(coefficients.columns)
    )

    if missing:
        raise ValueError(
            "Coefficient results are missing columns: "
            f"{sorted(missing)}"
        )

    return coefficients


def plot_cross_validation_curve(
    penalty_summary: pd.DataFrame,
    selected_penalty: float,
) -> Path:
    """
    Plot grouped validation concordance across ridge penalties.
    """
    summary = (
        penalty_summary.sort_values(
            "penalizer"
        )
        .reset_index(drop=True)
    )

    positions = np.arange(
        len(summary)
    )

    means = summary[
        "mean_validation_concordance"
    ].to_numpy()

    deviations = summary[
        "standard_deviation_concordance"
    ].fillna(0.0).to_numpy()

    labels = [
        f"{value:g}"
        for value in summary["penalizer"]
    ]

    selected_matches = np.isclose(
        summary["penalizer"],
        selected_penalty,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2)
    )

    axis.errorbar(
        positions,
        means,
        yerr=deviations,
        marker="o",
        capsize=4,
        linewidth=1.5,
        label="Mean ± one fold SD",
    )

    if selected_matches.any():
        selected_position = int(
            np.flatnonzero(
                selected_matches
            )[0]
        )

        selected_score = float(
            means[selected_position]
        )

        axis.scatter(
            selected_position,
            selected_score,
            marker="*",
            s=180,
            zorder=5,
            label=(
                "Selected penalty "
                f"({selected_penalty:g})"
            ),
        )

        axis.annotate(
            (
                f"Selected: λ={selected_penalty:g}\n"
                f"C-index={selected_score:.3f}"
            ),
            xy=(
                selected_position,
                selected_score,
            ),
            xytext=(12, 16),
            textcoords="offset points",
        )

    axis.set_xticks(
        positions,
        labels,
    )

    axis.set_xlabel(
        "Ridge penalizer"
    )

    axis.set_ylabel(
        "Mean validation concordance"
    )

    axis.set_title(
        "Firm-level grouped cross-validation"
    )

    axis.grid(
        axis="y",
        alpha=0.35,
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "cv_penalty_curve.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_validation_model_comparison(
    penalty_summary: pd.DataFrame,
    selected_penalty: float,
) -> Path:
    """
    Compare validation performance of the unpenalized and selected
    ridge time-varying Cox models.
    """
    unpenalized = penalty_summary.loc[
        np.isclose(
            penalty_summary["penalizer"],
            0.0,
        )
    ]

    selected = penalty_summary.loc[
        np.isclose(
            penalty_summary["penalizer"],
            selected_penalty,
        )
    ]

    if unpenalized.empty:
        raise ValueError(
            "Penalty summary does not contain "
            "the unpenalized model."
        )

    if selected.empty:
        raise ValueError(
            "Penalty summary does not contain "
            "the selected ridge penalty."
        )

    model_names = [
        "Unpenalized\nTime-Varying Cox",
        (
            "Tuned Ridge\n"
            f"Time-Varying Cox (λ={selected_penalty:g})"
        ),
    ]

    means = [
        float(
            unpenalized.iloc[0][
                "mean_validation_concordance"
            ]
        ),
        float(
            selected.iloc[0][
                "mean_validation_concordance"
            ]
        ),
    ]

    deviations = [
        float(
            unpenalized.iloc[0][
                "standard_deviation_concordance"
            ]
        ),
        float(
            selected.iloc[0][
                "standard_deviation_concordance"
            ]
        ),
    ]

    improvement = (
        means[1] - means[0]
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.2)
    )

    bars = axis.bar(
        model_names,
        means,
        yerr=deviations,
        capsize=6,
    )

    for bar, value in zip(
        bars,
        means,
        strict=True,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    axis.text(
        0.5,
        min(means) - 0.035,
        (
            "Validation improvement: "
            f"+{improvement:.3f}"
        ),
        ha="center",
    )

    lower_limit = max(
        0.0,
        min(means) - max(deviations) - 0.06,
    )

    upper_limit = min(
        1.0,
        max(means) + max(deviations) + 0.06,
    )

    axis.set_ylim(
        lower_limit,
        upper_limit,
    )

    axis.set_ylabel(
        "Mean validation concordance"
    )

    axis.set_title(
        "Out-of-sample model comparison"
    )

    axis.grid(
        axis="y",
        alpha=0.35,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "validation_model_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_coefficient_comparison(
    coefficients: pd.DataFrame,
) -> Path:
    """
    Compare standardized unpenalized and ridge coefficients.
    """
    model_names = [
        "Time-Varying Cox",
        "Ridge Time-Varying Cox",
    ]

    comparison = (
        coefficients[
            coefficients["model"].isin(
                model_names
            )
        ]
        .pivot(
            index="feature",
            columns="model",
            values="coefficient",
        )
        .dropna(
            subset=model_names
        )
    )

    comparison[
        "absolute_unpenalized"
    ] = comparison[
        "Time-Varying Cox"
    ].abs()

    comparison = (
        comparison.sort_values(
            "absolute_unpenalized",
            ascending=True,
        )
        .drop(
            columns=[
                "absolute_unpenalized"
            ]
        )
    )

    labels = [
        format_feature_name(feature)
        for feature in comparison.index
    ]

    positions = np.arange(
        len(comparison)
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.scatter(
        comparison[
            "Time-Varying Cox"
        ],
        positions + 0.10,
        marker="o",
        s=65,
        label="Unpenalized",
    )

    axis.scatter(
        comparison[
            "Ridge Time-Varying Cox"
        ],
        positions - 0.10,
        marker="s",
        s=55,
        label="Tuned ridge",
    )

    for position, (_, row) in enumerate(
        comparison.iterrows()
    ):
        axis.plot(
            [
                row["Time-Varying Cox"],
                row[
                    "Ridge Time-Varying Cox"
                ],
            ],
            [
                position + 0.10,
                position - 0.10,
            ],
            linewidth=1,
            alpha=0.6,
        )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_yticks(
        positions,
        labels,
    )

    axis.set_xlabel(
        "Standardized coefficient"
    )

    axis.set_title(
        "Coefficient shrinkage from ridge regularization"
    )

    axis.grid(
        axis="x",
        alpha=0.30,
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "coefficient_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_hazard_ratio_forest(
    coefficients: pd.DataFrame,
) -> Path:
    """
    Plot hazard ratios and 95% intervals for the tuned ridge model.
    """
    ridge = coefficients.loc[
        coefficients["model"]
        == "Ridge Time-Varying Cox"
    ].copy()

    if ridge.empty:
        raise ValueError(
            "No tuned ridge coefficient rows found."
        )

    ridge = ridge.dropna(
        subset=[
            "hazard_ratio",
            "hazard_ratio_lower_95",
            "hazard_ratio_upper_95",
        ]
    )

    ridge = ridge.sort_values(
        "hazard_ratio",
        ascending=True,
    ).reset_index(drop=True)

    positions = np.arange(
        len(ridge)
    )

    lower_errors = (
        ridge["hazard_ratio"]
        - ridge[
            "hazard_ratio_lower_95"
        ]
    )

    upper_errors = (
        ridge[
            "hazard_ratio_upper_95"
        ]
        - ridge["hazard_ratio"]
    )

    labels = [
        format_feature_name(feature)
        for feature in ridge["feature"]
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.errorbar(
        ridge["hazard_ratio"],
        positions,
        xerr=[
            lower_errors,
            upper_errors,
        ],
        fmt="o",
        capsize=4,
        markersize=6,
    )

    axis.axvline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="No change in hazard",
    )

    axis.set_yticks(
        positions,
        labels,
    )

    axis.set_xlabel(
        "Hazard ratio per one-standard-deviation increase"
    )

    axis.set_title(
        "Tuned ridge time-varying Cox model"
    )

    axis.grid(
        axis="x",
        alpha=0.30,
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "hazard_ratio_forest.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main() -> None:
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    penalty_summary = (
        load_penalty_summary()
    )

    selected_penalty = (
        load_selected_penalty()
    )

    coefficients = (
        load_coefficient_results()
    )

    generated_paths = [
        plot_cross_validation_curve(
            penalty_summary=penalty_summary,
            selected_penalty=(
                selected_penalty
            ),
        ),
        plot_validation_model_comparison(
            penalty_summary=penalty_summary,
            selected_penalty=(
                selected_penalty
            ),
        ),
        plot_coefficient_comparison(
            coefficients=coefficients,
        ),
        plot_hazard_ratio_forest(
            coefficients=coefficients,
        ),
    ]

    print("Real-model figures generated")
    print("----------------------------")

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()