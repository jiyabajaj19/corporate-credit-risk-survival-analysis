from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.models.time_varying_cox import (
    fit_time_varying_cox,
    standardize_time_varying_covariates,
)
from src.simulation.generate_data import (
    simulate_time_varying_credit_data,
)


COVARIATES = [
    "leverage",
    "low_interest_coverage",
    "current_ratio",
    "cash_ratio",
    "return_on_assets",
    "debt_growth",
    "revenue_growth",
    "operating_cash_flow_ratio",
    "log_total_assets",
    "ebitda_margin",
]


@dataclass
class TimeVaryingMonteCarloResults:
    estimates: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame


def collapse_to_baseline(
    long_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert long-format company-quarter data into one row per firm.

    Financial covariates are measured at the firm's first observed
    quarter. The firm's final stop time and event status define its
    observed survival outcome.
    """
    required_columns = {
        "firm_id",
        "start",
        "stop",
        "event",
        "entry_time",
        *COVARIATES,
    }

    missing = required_columns - set(long_data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    ordered = long_data.sort_values(
        ["firm_id", "start"]
    )

    first_rows = (
        ordered.groupby("firm_id", as_index=False)
        .first()
    )

    last_rows = (
        ordered.groupby("firm_id", as_index=False)
        .last()
    )

    baseline_data = first_rows[
        [
            "firm_id",
            "entry_time",
            *COVARIATES,
        ]
    ].merge(
        last_rows[
            [
                "firm_id",
                "stop",
                "event",
            ]
        ],
        on="firm_id",
        validate="one_to_one",
    )

    baseline_data = baseline_data.rename(
        columns={"stop": "exit_time"}
    )

    baseline_data["follow_up_time"] = (
        baseline_data["exit_time"]
        - baseline_data["entry_time"]
    )

    return baseline_data


def fit_baseline_at_entry_cox(
    baseline_data: pd.DataFrame,
    penalizer: float = 0.001,
) -> CoxPHFitter:
    """
    Fit an intentionally simplified Cox model.

    Time is measured from database entry, so the model ignores the
    company's unobserved pre-entry history. Covariates are frozen at
    their first observed values.
    """
    model_data = baseline_data[
        [
            "follow_up_time",
            "event",
            *COVARIATES,
        ]
    ].copy()

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        model_data,
        duration_col="follow_up_time",
        event_col="event",
        show_progress=False,
    )

    return model


def fit_left_truncated_baseline_cox(
    baseline_data: pd.DataFrame,
    penalizer: float = 0.001,
) -> CoxPHFitter:
    """
    Fit a left-truncated Cox model using first-observed covariates.
    """
    model_data = baseline_data[
        [
            "entry_time",
            "exit_time",
            "event",
            *COVARIATES,
        ]
    ].copy()

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        model_data,
        duration_col="exit_time",
        event_col="event",
        entry_col="entry_time",
        show_progress=False,
    )

    return model


def _extract_estimates(
    model_name: str,
    coefficients: pd.Series,
    standard_errors: pd.Series,
    repetition: int,
    true_coefficients: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for covariate in COVARIATES:
        estimate = float(coefficients.loc[covariate])
        standard_error = float(
            standard_errors.loc[covariate]
        )

        true_value = true_coefficients[covariate]

        rows.append(
            {
                "repetition": repetition,
                "model": model_name,
                "covariate": covariate,
                "true_coefficient": true_value,
                "estimate": estimate,
                "standard_error": standard_error,
                "error": estimate - true_value,
            }
        )

    return rows


def summarize_time_varying_monte_carlo(
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate bias, variability, RMSE and confidence-interval coverage.
    """
    if estimates.empty:
        raise ValueError(
            "Cannot summarize an empty estimates table"
        )

    working = estimates.copy()

    working["covered"] = (
        working["true_coefficient"]
        >= (
            working["estimate"]
            - 1.96 * working["standard_error"]
        )
    ) & (
        working["true_coefficient"]
        <= (
            working["estimate"]
            + 1.96 * working["standard_error"]
        )
    )

    summary = (
        working.groupby(
            ["model", "covariate"],
            as_index=False,
        )
        .agg(
            true_coefficient=(
                "true_coefficient",
                "first",
            ),
            mean_estimate=("estimate", "mean"),
            bias=("error", "mean"),
            empirical_sd=("estimate", "std"),
            mean_standard_error=(
                "standard_error",
                "mean",
            ),
            rmse=(
                "error",
                lambda values: float(
                    np.sqrt(
                        np.mean(
                            np.square(values)
                        )
                    )
                ),
            ),
            coverage_probability=(
                "covered",
                "mean",
            ),
            successful_repetitions=(
                "repetition",
                "nunique",
            ),
        )
    )

    summary["absolute_bias"] = (
        summary["bias"].abs()
    )

    return summary[
        [
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
        ]
    ]


def run_time_varying_monte_carlo(
    repetitions: int = 200,
    sample_size: int = 800,
    beta_leverage: float = 0.80,
    beta_low_coverage: float = 0.90,
    beta_current_ratio: float = -0.35,
    beta_cash_ratio: float = -0.25,
    beta_return_on_assets: float = -0.70,
    beta_debt_growth: float = 0.40,
    beta_revenue_growth: float = -0.20,
    beta_operating_cash_flow_ratio: float = -0.45,
    beta_log_total_assets: float = -0.10,
    beta_ebitda_margin: float = -0.30,
    baseline_hazard: float = 0.05,
    maximum_periods: int = 20,
    observation_interval: float = 0.25,
    maximum_entry_period: int = 8,
    penalizer: float = 0.01,
    seed: int = 2026,
) -> TimeVaryingMonteCarloResults:
    """
    Compare three Cox specifications on repeated simulated datasets.
    """
    if repetitions <= 0:
        raise ValueError(
            "repetitions must be positive"
        )

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    seed_generator = np.random.default_rng(seed)

    simulation_seeds = seed_generator.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=repetitions,
    )

    true_coefficients = {
    "leverage": beta_leverage,
    "low_interest_coverage": (
        beta_low_coverage
    ),
    "current_ratio": beta_current_ratio,
    "cash_ratio": beta_cash_ratio,
    "return_on_assets": (
        beta_return_on_assets
    ),
    "debt_growth": beta_debt_growth,
    "revenue_growth": beta_revenue_growth,
    "operating_cash_flow_ratio": (
        beta_operating_cash_flow_ratio
    ),
    "log_total_assets": (
        beta_log_total_assets
    ),
    "ebitda_margin": beta_ebitda_margin,
    }

    estimate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for repetition, simulation_seed in enumerate(
        simulation_seeds,
        start=1,
    ):
        try:
            long_data = simulate_time_varying_credit_data(
                sample_size=sample_size,
                beta_leverage=beta_leverage,
                beta_low_coverage=beta_low_coverage,
                beta_current_ratio=beta_current_ratio,
                beta_cash_ratio=beta_cash_ratio,
                beta_return_on_assets=(
                    beta_return_on_assets
                ),
                beta_debt_growth=beta_debt_growth,
                beta_revenue_growth=(
                    beta_revenue_growth
                ),
                beta_operating_cash_flow_ratio=(
                    beta_operating_cash_flow_ratio
                ),
                beta_log_total_assets=(
                    beta_log_total_assets
                ),
                    beta_ebitda_margin=beta_ebitda_margin,
                    baseline_hazard=baseline_hazard,
                    maximum_periods=maximum_periods,
                    observation_interval=(
                    observation_interval
                ),
                maximum_entry_period=(
                    maximum_entry_period
                ),
                seed=int(simulation_seed),
            )

            long_data, _ = (
                standardize_time_varying_covariates(
                    long_data
                )
            )

            
            baseline_data = collapse_to_baseline(
                long_data
            )

            baseline_model = (
                fit_baseline_at_entry_cox(
                    baseline_data,
                    penalizer=penalizer,
                )
            )

            left_truncated_model = (
                fit_left_truncated_baseline_cox(
                    baseline_data,
                    penalizer=penalizer,
                )
            )

            time_varying_results = (
                fit_time_varying_cox(
                    long_data,
                    penalizer=penalizer,
                )
            )

            estimate_rows.extend(
                _extract_estimates(
                    model_name="Baseline-at-entry Cox",
                    coefficients=(
                        baseline_model.params_
                    ),
                    standard_errors=(
                        baseline_model.standard_errors_
                    ),
                    repetition=repetition,
                    true_coefficients=(
                        true_coefficients
                    ),
                )
            )

            estimate_rows.extend(
                _extract_estimates(
                    model_name=(
                        "Left-truncated baseline Cox"
                    ),
                    coefficients=(
                        left_truncated_model.params_
                    ),
                    standard_errors=(
                        left_truncated_model
                        .standard_errors_
                    ),
                    repetition=repetition,
                    true_coefficients=(
                        true_coefficients
                    ),
                )
            )

            estimate_rows.extend(
                _extract_estimates(
                    model_name="Time-varying Cox",
                    coefficients=(
                        time_varying_results
                        .model.params_
                    ),
                    standard_errors=(
                        time_varying_results
                        .model.standard_errors_
                    ),
                    repetition=repetition,
                    true_coefficients=(
                        true_coefficients
                    ),
                )
            )

        except Exception as error:
            failure_rows.append(
                {
                    "repetition": repetition,
                    "simulation_seed": int(
                        simulation_seed
                    ),
                    "error_type": type(
                        error
                    ).__name__,
                    "error_message": str(error),
                }
            )

        if (
            repetition % 20 == 0
            or repetition == repetitions
        ):
            print(
                f"Completed {repetition}/"
                f"{repetitions} repetitions"
            )

    estimates = pd.DataFrame(estimate_rows)
    failures = pd.DataFrame(failure_rows)

    if estimates.empty:
        raise RuntimeError(
            "Every Monte Carlo repetition failed"
        )

    summary = (
        summarize_time_varying_monte_carlo(
            estimates
        )
    )

    return TimeVaryingMonteCarloResults(
        estimates=estimates,
        summary=summary,
        failures=failures,
    )