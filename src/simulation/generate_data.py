from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_credit_survival_data(
    sample_size: int = 400,
    beta_leverage: float = 0.5,
    beta_low_coverage: float = 1.0,
    entry_scale: float = 0.5,
    censoring_upper_bound: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate corporate credit-risk survival data with delayed entry.

    Firms are observed only if they survive beyond their entry time.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if entry_scale <= 0:
        raise ValueError("entry_scale must be positive")

    if censoring_upper_bound <= 0:
        raise ValueError(
            "censoring_upper_bound must be positive"
        )

    rng = np.random.default_rng(seed)

    accepted_batches: list[pd.DataFrame] = []
    accepted_count = 0

    while accepted_count < sample_size:
        remaining = sample_size - accepted_count
        batch_size = max(remaining * 10, 1000)

        leverage = rng.normal(
            loc=0.0,
            scale=1.0,
            size=batch_size,
        )

        low_interest_coverage = rng.binomial(
            n=1,
            p=0.5,
            size=batch_size,
        )

        linear_predictor = (
            beta_leverage * leverage
            + beta_low_coverage * low_interest_coverage
        )

        uniform_draws = rng.uniform(
            low=np.finfo(float).eps,
            high=1.0,
            size=batch_size,
        )

        baseline_hazard = 2.0

        default_time = (
            -np.log(uniform_draws)
            / (
                baseline_hazard
                * np.exp(linear_predictor)
            )
        )

        entry_time = rng.exponential(
            scale=entry_scale,
            size=batch_size,
        )

        observed = default_time > entry_time

        if not np.any(observed):
            continue

        leverage = leverage[observed]
        low_interest_coverage = (
            low_interest_coverage[observed]
        )
        default_time = default_time[observed]
        entry_time = entry_time[observed]

        residual_censoring_time = rng.uniform(
            low=0.0,
            high=censoring_upper_bound,
            size=len(default_time),
        )

        censoring_time = (
            entry_time + residual_censoring_time
        )

        exit_time = np.minimum(
            default_time,
            censoring_time,
        )

        event = (
            default_time <= censoring_time
        ).astype(int)

        batch = pd.DataFrame(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "event": event,
                "leverage": leverage,
                "low_interest_coverage": (
                    low_interest_coverage
                ),
                "true_default_time": default_time,
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