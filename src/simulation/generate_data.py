from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_credit_survival_data(
    sample_size: int = 400,
    beta_leverage: float = 0.5,
    beta_low_coverage: float = 1.0,
    entry_scale: float = 1.25,
    censoring_upper_bound: float = 3.0,
    baseline_scale: float = 2.5,
    weibull_shape: float = 1.6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate corporate time-to-default data with delayed entry.

    The data-generating process follows a proportional-hazards model
    with a Weibull baseline hazard.

    Firms enter the observed dataset only after surviving until their
    entry time, creating left truncation and survivor-selection effects.

    Parameters
    ----------
    sample_size:
        Number of observed firms in the final dataset.

    beta_leverage:
        True log-hazard coefficient for standardized leverage.

    beta_low_coverage:
        True log-hazard coefficient for low interest coverage.

    entry_scale:
        Controls how late firms enter the observed dataset.

    censoring_upper_bound:
        Maximum follow-up time before administrative censoring.

    baseline_scale:
        Weibull scale parameter for default times.

    weibull_shape:
        Weibull shape parameter. Values above one produce increasing
        default risk over time.

    seed:
        Random-number seed for reproducibility.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if entry_scale <= 0:
        raise ValueError("entry_scale must be positive")

    if censoring_upper_bound <= 0:
        raise ValueError(
            "censoring_upper_bound must be positive"
        )

    if baseline_scale <= 0:
        raise ValueError(
            "baseline_scale must be positive"
        )

    if weibull_shape <= 0:
        raise ValueError(
            "weibull_shape must be positive"
        )

    rng = np.random.default_rng(seed)

    accepted_batches: list[pd.DataFrame] = []
    accepted_count = 0

    while accepted_count < sample_size:
        remaining = sample_size - accepted_count
        batch_size = max(remaining * 20, 2000)

        # Standardized leverage:
        # larger values represent more highly leveraged firms.
        leverage = rng.normal(
            loc=0.0,
            scale=1.0,
            size=batch_size,
        )

        # Firms with higher leverage are somewhat more likely to have
        # poor interest coverage.
        low_coverage_probability = 1.0 / (
            1.0 + np.exp(-(-0.25 + 0.55 * leverage))
        )

        low_interest_coverage = rng.binomial(
            n=1,
            p=low_coverage_probability,
            size=batch_size,
        )

        linear_predictor = (
            beta_leverage * leverage
            + beta_low_coverage
            * low_interest_coverage
        )

        # Inverse-transform sampling from a Weibull proportional-
        # hazards model:
        #
        # S(t | x) = exp(
        #     -exp(linear_predictor)
        #     * (t / baseline_scale) ** weibull_shape
        # )
        uniform_draws = rng.uniform(
            low=np.finfo(float).eps,
            high=1.0,
            size=batch_size,
        )

        default_time = baseline_scale * (
            -np.log(uniform_draws)
            / np.exp(linear_predictor)
        ) ** (1.0 / weibull_shape)

        # Later database entry creates stronger delayed observation.
        #
        # Entry also depends on observed financial characteristics.
        # Conditional on these covariates, the random entry mechanism
        # is generated independently from the default-time draw.
        entry_multiplier = np.exp(
            -0.20 * leverage
            -0.30 * low_interest_coverage
        )

        entry_time = rng.exponential(
            scale=entry_scale * entry_multiplier,
            size=batch_size,
        )

        # A firm is observed only if it survives long enough to enter
        # the financial database.
        observed = default_time > entry_time

        if not np.any(observed):
            continue

        leverage_observed = leverage[observed]

        low_coverage_observed = (
            low_interest_coverage[observed]
        )

        default_time_observed = default_time[observed]
        entry_time_observed = entry_time[observed]

        # Some firms leave observation before default due to the study
        # ending or incomplete future follow-up.
        residual_censoring_time = rng.uniform(
            low=0.25,
            high=censoring_upper_bound,
            size=len(default_time_observed),
        )

        censoring_time = (
            entry_time_observed
            + residual_censoring_time
        )

        exit_time = np.minimum(
            default_time_observed,
            censoring_time,
        )

        event = (
            default_time_observed <= censoring_time
        ).astype(int)

        batch = pd.DataFrame(
            {
                "entry_time": entry_time_observed,
                "exit_time": exit_time,
                "event": event,
                "leverage": leverage_observed,
                "low_interest_coverage": (
                    low_coverage_observed
                ),
                "true_default_time": (
                    default_time_observed
                ),
            }
        )

        accepted_batches.append(batch)
        accepted_count += len(batch)

    data = pd.concat(
        accepted_batches,
        ignore_index=True,
    ).iloc[:sample_size].copy()

    data.insert(
        0,
        "firm_id",
        np.arange(1, sample_size + 1),
    )

    data["follow_up_time"] = (
        data["exit_time"] - data["entry_time"]
    )

    return data


def summarize_simulated_data(
    data: pd.DataFrame,
) -> dict[str, float]:
    """
    Return key descriptive statistics for a simulated dataset.
    """
    return {
        "number_of_firms": float(len(data)),
        "number_of_defaults": float(
            data["event"].sum()
        ),
        "default_rate": float(
            data["event"].mean()
        ),
        "censoring_rate": float(
            1.0 - data["event"].mean()
        ),
        "mean_entry_time": float(
            data["entry_time"].mean()
        ),
        "median_entry_time": float(
            data["entry_time"].median()
        ),
        "mean_follow_up_time": float(
            data["follow_up_time"].mean()
        ),
    }