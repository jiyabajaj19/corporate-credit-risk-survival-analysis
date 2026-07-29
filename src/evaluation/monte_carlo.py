from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.baseline_cox import (
    fit_left_truncated_cox,
    fit_naive_cox,
)
from src.simulation.generate_data import (
    simulate_credit_survival_data,
)


TRUE_COEFFICIENTS = {
    "leverage": 0.5,
    "low_interest_coverage": 1.0,
}


@dataclass
class MonteCarloResults:
    estimates: pd.DataFrame
    summary: pd.DataFrame
    failed_repetitions: int


def _extract_model_results(
    repetition: int,
    model_name: str,
    coefficients: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    """
    Convert one fitted model's coefficient table into row records.
    """
    records: list[dict[str, float | int | str]] = []

    for covariate, true_value in TRUE_COEFFICIENTS.items():
        estimate = float(
            coefficients.loc[covariate, "coef"]
        )

        standard_error = float(
            coefficients.loc[covariate, "se(coef)"]
        )

        lower_bound = float(
            coefficients.loc[
                covariate,
                "coef lower 95%",
            ]
        )

        upper_bound = float(
            coefficients.loc[
                covariate,
                "coef upper 95%",
            ]
        )

        records.append(
            {
                "repetition": repetition,
                "model": model_name,
                "covariate": covariate,
                "true_coefficient": true_value,
                "estimate": estimate,
                "standard_error": standard_error,
                "lower_95": lower_bound,
                "upper_95": upper_bound,
                "covered": int(
                    lower_bound
                    <= true_value
                    <= upper_bound
                ),
            }
        )

    return records


def summarize_monte_carlo_results(
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Monte Carlo performance metrics.

    Metrics
    -------
    mean_estimate:
        Average coefficient estimate.

    bias:
        Average estimate minus the true coefficient.

    empirical_sd:
        Standard deviation of estimates across repetitions.

    mean_standard_error:
        Average model-reported standard error.

    rmse:
        Root mean squared error relative to the true coefficient.

    coverage_probability:
        Proportion of 95% confidence intervals containing the truth.
    """
    summary_rows: list[
        dict[str, float | int | str]
    ] = []

    grouped = estimates.groupby(
        ["model", "covariate"],
        sort=False,
    )

    for (model, covariate), group in grouped:
        true_value = float(
            group["true_coefficient"].iloc[0]
        )

        estimation_errors = (
            group["estimate"] - true_value
        )

        mean_estimate = float(
            group["estimate"].mean()
        )

        bias = float(estimation_errors.mean())

        empirical_sd = float(
            group["estimate"].std(ddof=1)
        )

        mean_standard_error = float(
            group["standard_error"].mean()
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(estimation_errors)
                )
            )
        )

        coverage_probability = float(
            group["covered"].mean()
        )

        summary_rows.append(
            {
                "model": model,
                "covariate": covariate,
                "true_coefficient": true_value,
                "repetitions": len(group),
                "mean_estimate": mean_estimate,
                "bias": bias,
                "absolute_bias": abs(bias),
                "empirical_sd": empirical_sd,
                "mean_standard_error": (
                    mean_standard_error
                ),
                "rmse": rmse,
                "coverage_probability": (
                    coverage_probability
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def run_monte_carlo(
    repetitions: int = 200,
    sample_size: int = 400,
    starting_seed: int = 1000,
    progress_interval: int = 25,
) -> MonteCarloResults:
    """
    Run repeated corporate credit-risk simulations.

    For every repetition:

    1. Generate a new delayed-entry dataset.
    2. Fit the naive Cox model.
    3. Fit the left-truncated Cox model.
    4. Store coefficient estimates and confidence intervals.
    """
    if repetitions <= 0:
        raise ValueError(
            "repetitions must be positive"
        )

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    all_records: list[
        dict[str, float | int | str]
    ] = []

    failed_repetitions = 0

    for repetition in range(1, repetitions + 1):
        seed = starting_seed + repetition

        try:
            data = simulate_credit_survival_data(
                sample_size=sample_size,
                beta_leverage=0.5,
                beta_low_coverage=1.0,
                entry_scale=1.25,
                censoring_upper_bound=3.0,
                baseline_scale=2.5,
                weibull_shape=1.6,
                seed=seed,
            )

            naive_results = fit_naive_cox(data)

            truncated_results = (
                fit_left_truncated_cox(data)
            )

            all_records.extend(
                _extract_model_results(
                    repetition=repetition,
                    model_name="Naive Cox",
                    coefficients=(
                        naive_results.coefficients
                    ),
                )
            )

            all_records.extend(
                _extract_model_results(
                    repetition=repetition,
                    model_name=(
                        "Left-Truncated Cox"
                    ),
                    coefficients=(
                        truncated_results.coefficients
                    ),
                )
            )

        except Exception as error:
            failed_repetitions += 1

            print(
                f"Repetition {repetition} failed: "
                f"{type(error).__name__}: {error}"
            )

        if (
            repetition % progress_interval == 0
            or repetition == repetitions
        ):
            print(
                f"Completed {repetition}/"
                f"{repetitions} repetitions"
            )

    if not all_records:
        raise RuntimeError(
            "Every Monte Carlo repetition failed."
        )

    estimates = pd.DataFrame(all_records)

    summary = summarize_monte_carlo_results(
        estimates
    )

    return MonteCarloResults(
        estimates=estimates,
        summary=summary,
        failed_repetitions=failed_repetitions,
    )