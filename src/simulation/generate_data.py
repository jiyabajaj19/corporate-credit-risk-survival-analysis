from __future__ import annotations

import numpy as np
import pandas as pd


TIME_VARYING_COVARIATES = [
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


def simulate_credit_survival_data(
    sample_size: int = 1_000,
    beta_leverage: float = 0.50,
    beta_low_coverage: float = 1.00,
    baseline_hazard: float | None = None,
    weibull_shape: float = 1.50,
    maximum_follow_up: float = 5.0,
    maximum_entry_time: float = 2.0,
    seed: int = 42,
    *,
    # Backward-compatible arguments used by the original
    # simulation and Monte Carlo modules.
    baseline_scale: float | None = None,
    entry_scale: float | None = None,
) -> pd.DataFrame:
    """
    Simulate one-row-per-firm corporate survival data.

    Supports both the current parameterization and the legacy
    ``baseline_scale`` and ``entry_scale`` arguments used by the
    original Monte Carlo study.
    """
    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    if weibull_shape <= 0:
        raise ValueError(
            "weibull_shape must be positive"
        )

    if maximum_follow_up <= 0:
        raise ValueError(
            "maximum_follow_up must be positive"
        )

    if maximum_entry_time < 0:
        raise ValueError(
            "maximum_entry_time cannot be negative"
        )

    if baseline_scale is not None:
        if baseline_scale <= 0:
            raise ValueError(
                "baseline_scale must be positive"
            )

        effective_baseline_hazard = (
            1.0
            / baseline_scale**weibull_shape
        )
    else:
        effective_baseline_hazard = (
            0.08
            if baseline_hazard is None
            else baseline_hazard
        )

        if effective_baseline_hazard <= 0:
            raise ValueError(
                "baseline_hazard must be positive"
            )

    if entry_scale is not None and entry_scale <= 0:
        raise ValueError(
            "entry_scale must be positive"
        )

    rng = np.random.default_rng(seed)

    rows: list[dict[str, float | int]] = []

    attempts = 0
    maximum_attempts = sample_size * 500

    while len(rows) < sample_size:
        attempts += 1

        if attempts > maximum_attempts:
            raise RuntimeError(
                "Unable to generate the requested "
                "number of observed firms."
            )

        leverage = rng.normal(
            loc=0.0,
            scale=1.0,
        )

        coverage_latent = (
            0.40 * leverage
            + rng.normal(0.0, 1.0)
        )

        low_interest_coverage = int(
            coverage_latent > 0.20
        )

        linear_predictor = (
            beta_leverage * leverage
            + beta_low_coverage
            * low_interest_coverage
        )

        true_default_time = (
            _sample_weibull_event_time(
                rng=rng,
                baseline_hazard=(
                    effective_baseline_hazard
                ),
                weibull_shape=weibull_shape,
                linear_predictor=(
                    linear_predictor
                ),
            )
        )

        if entry_scale is not None:
            # Legacy delayed-entry mechanism.
            entry_time = float(
                rng.exponential(
                    scale=entry_scale
                )
            )
        else:
            risk_score = (
                0.45 * leverage
                + 0.50
                * low_interest_coverage
            )

            entry_fraction = _sigmoid(
                0.35 * risk_score
                + rng.normal(0.0, 0.80)
            )

            entry_time = float(
                maximum_entry_time
                * entry_fraction
            )

        # Left truncation: firms that fail before entry
        # are not present in the observed sample.
        if true_default_time <= entry_time:
            continue

        administrative_exit = (
            entry_time + maximum_follow_up
        )

        exit_time = min(
            true_default_time,
            administrative_exit,
        )

        event = int(
            true_default_time
            <= administrative_exit
        )

        rows.append(
            {
                "firm_id": len(rows) + 1,
                "leverage": leverage,
                "low_interest_coverage": (
                    low_interest_coverage
                ),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "follow_up_time": (
                    exit_time - entry_time
                ),
                "event": event,
                "true_default_time": (
                    true_default_time
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_simulated_data(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Return a compact summary of simulated survival data.
    """
    required_columns = {
        "firm_id",
        "event",
        "entry_time",
        "exit_time",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    number_of_firms = int(
        data["firm_id"].nunique()
    )

    number_of_events = int(
        data["event"].sum()
    )

    return pd.Series(
        {
            "number_of_firms": number_of_firms,
            "number_of_events": number_of_events,
            "default_rate": (
                number_of_events / number_of_firms
            ),
            "mean_entry_time": float(
                data["entry_time"].mean()
            ),
            "mean_exit_time": float(
                data["exit_time"].mean()
            ),
        }
    )


def simulate_time_varying_credit_data(
    sample_size: int = 800,
    beta_leverage: float = 0.60,
    beta_low_coverage: float = 0.70,
    beta_current_ratio: float = -0.25,
    beta_cash_ratio: float = -0.15,
    beta_return_on_assets: float = -0.40,
    beta_debt_growth: float = 0.30,
    beta_revenue_growth: float = -0.20,
    beta_operating_cash_flow_ratio: float = -0.35,
    beta_log_total_assets: float = -0.15,
    beta_ebitda_margin: float = -0.30,
    baseline_hazard: float = 0.02,
    weibull_shape: float = 1.60,
    maximum_periods: int = 20,
    observation_interval: float = 0.25,
    maximum_entry_period: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate delayed-entry corporate survival data with
    quarterly time-varying financial covariates.

    Continuous financial variables are generated on
    approximately standardized scales. Their coefficients
    therefore represent changes in log default hazard for
    an approximately one-standard-deviation increase.

    The result is returned in start-stop format with one
    row per observed firm-quarter.
    """
    _validate_common_arguments(
        sample_size=sample_size,
        baseline_hazard=baseline_hazard,
        weibull_shape=weibull_shape,
    )

    if maximum_periods <= 1:
        raise ValueError(
            "maximum_periods must be greater than 1"
        )

    if observation_interval <= 0:
        raise ValueError(
            "observation_interval must be positive"
        )

    if maximum_entry_period < 0:
        raise ValueError(
            "maximum_entry_period cannot be negative"
        )

    if maximum_entry_period >= maximum_periods:
        raise ValueError(
            "maximum_entry_period must be smaller than "
            "maximum_periods"
        )

    coefficients = {
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

    rng = np.random.default_rng(seed)

    rows: list[dict[str, float | int]] = []

    observed_firms = 0
    candidate_firm_id = 0
    attempts = 0
    maximum_attempts = sample_size * 150

    while observed_firms < sample_size:
        attempts += 1
        candidate_firm_id += 1

        if attempts > maximum_attempts:
            raise RuntimeError(
                "Unable to generate the requested number "
                "of observed firms. Consider reducing the "
                "baseline hazard or maximum entry period."
            )

        latent_quality = rng.normal(0.0, 1.0)
        latent_liquidity = rng.normal(0.0, 1.0)
        latent_growth = rng.normal(0.0, 1.0)
        latent_size = rng.normal(0.0, 1.0)

        leverage = (
            -0.35 * latent_quality
            + 0.15 * latent_size
            + rng.normal(0.0, 0.85)
        )

        profitability = (
            0.45 * latent_quality
            - 0.15 * leverage
            + rng.normal(0.0, 0.80)
        )

        liquidity = (
            0.45 * latent_liquidity
            + 0.25 * latent_quality
            - 0.15 * leverage
            + rng.normal(0.0, 0.75)
        )

        growth = (
            0.45 * latent_growth
            + 0.20 * latent_quality
            + rng.normal(0.0, 0.80)
        )

        financial_stress = (
            0.45 * leverage
            - 0.35 * profitability
            - 0.25 * liquidity
            + rng.normal(0.0, 0.80)
        )

        initial_risk = (
            0.45 * leverage
            + 0.45 * financial_stress
            - 0.20 * latent_quality
        )

        entry_probability = _sigmoid(
            0.50 * initial_risk
            + rng.normal(0.0, 0.85)
        )

        entry_period = int(
            np.floor(
                entry_probability
                * (maximum_entry_period + 1)
            )
        )

        entry_period = min(
            entry_period,
            maximum_entry_period,
        )

        firm_rows: list[
            dict[str, float | int]
        ] = []

        defaulted_before_entry = False

        for period in range(maximum_periods):
            start_time = (
                period * observation_interval
            )

            stop_time = (
                (period + 1)
                * observation_interval
            )

            leverage = _evolve_state(
                previous_value=leverage,
                persistence=0.82,
                long_run_mean=(
                    -0.35 * latent_quality
                    + 0.15 * latent_size
                ),
                innovation_scale=0.35,
                rng=rng,
            )

            profitability = _evolve_state(
                previous_value=profitability,
                persistence=0.75,
                long_run_mean=(
                    0.45 * latent_quality
                    - 0.15 * leverage
                ),
                innovation_scale=0.40,
                rng=rng,
            )

            liquidity = _evolve_state(
                previous_value=liquidity,
                persistence=0.78,
                long_run_mean=(
                    0.45 * latent_liquidity
                    + 0.20 * latent_quality
                    - 0.15 * leverage
                ),
                innovation_scale=0.38,
                rng=rng,
            )

            growth = _evolve_state(
                previous_value=growth,
                persistence=0.65,
                long_run_mean=(
                    0.45 * latent_growth
                    + 0.20 * latent_quality
                ),
                innovation_scale=0.50,
                rng=rng,
            )

            financial_stress = (
                0.55 * financial_stress
                + 0.40 * leverage
                - 0.30 * profitability
                - 0.20 * liquidity
                + rng.normal(0.0, 0.55)
            )

            low_interest_coverage = int(
                financial_stress > 0.35
            )

            # These variables are intentionally constructed
            # on roughly standardized scales.
            current_ratio = (
                0.65 * liquidity
                - 0.15 * leverage
                + rng.normal(0.0, 0.70)
            )

            cash_ratio = (
                0.55 * liquidity
                - 0.10 * leverage
                + rng.normal(0.0, 0.80)
            )

            return_on_assets = (
                0.70 * profitability
                - 0.10 * leverage
                + rng.normal(0.0, 0.65)
            )

            debt_growth = (
                0.45 * growth
                + 0.30 * leverage
                + rng.normal(0.0, 0.80)
            )

            revenue_growth = (
                0.65 * growth
                + 0.20 * profitability
                + rng.normal(0.0, 0.70)
            )

            operating_cash_flow_ratio = (
                0.45 * profitability
                + 0.35 * liquidity
                - 0.10 * leverage
                + rng.normal(0.0, 0.70)
            )

            log_total_assets = (
                0.85 * latent_size
                + 0.03 * period
                + rng.normal(0.0, 0.45)
            )

            ebitda_margin = (
                0.60 * profitability
                + rng.normal(0.0, 0.75)
            )

            covariate_values = {
                "leverage": leverage,
                "low_interest_coverage": (
                    low_interest_coverage
                ),
                "current_ratio": current_ratio,
                "cash_ratio": cash_ratio,
                "return_on_assets": (
                    return_on_assets
                ),
                "debt_growth": debt_growth,
                "revenue_growth": (
                    revenue_growth
                ),
                "operating_cash_flow_ratio": (
                    operating_cash_flow_ratio
                ),
                "log_total_assets": (
                    log_total_assets
                ),
                "ebitda_margin": ebitda_margin,
            }

            linear_predictor = sum(
                coefficients[covariate]
                * covariate_values[covariate]
                for covariate
                in TIME_VARYING_COVARIATES
            )

            # Avoid numerical overflow in extreme simulated
            # observations without materially affecting the
            # usual range of the linear predictor.
            linear_predictor = float(
                np.clip(
                    linear_predictor,
                    -8.0,
                    8.0,
                )
            )

            baseline_cumulative_increment = (
                baseline_hazard
                * (
                    stop_time**weibull_shape
                    - start_time**weibull_shape
                )
            )

            interval_cumulative_hazard = (
                baseline_cumulative_increment
                * np.exp(linear_predictor)
            )

            default_probability = (
                1.0
                - np.exp(
                    -interval_cumulative_hazard
                )
            )

            default_event = int(
                rng.uniform()
                < default_probability
            )

            if period < entry_period:
                if default_event == 1:
                    defaulted_before_entry = True
                    break

                continue

            firm_rows.append(
                {
                    "firm_id": (
                        observed_firms + 1
                    ),
                    "start": start_time,
                    "stop": stop_time,
                    "event": default_event,
                    "entry_time": (
                        entry_period
                        * observation_interval
                    ),
                    **covariate_values,
                }
            )

            if default_event == 1:
                break

        if defaulted_before_entry:
            continue

        if not firm_rows:
            continue

        rows.extend(firm_rows)
        observed_firms += 1

    data = pd.DataFrame(rows)

    ordered_columns = [
        "firm_id",
        "start",
        "stop",
        "event",
        "entry_time",
        *TIME_VARYING_COVARIATES,
    ]

    return data[ordered_columns]


def _validate_common_arguments(
    sample_size: int,
    baseline_hazard: float,
    weibull_shape: float,
) -> None:
    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    if baseline_hazard <= 0:
        raise ValueError(
            "baseline_hazard must be positive"
        )

    if weibull_shape <= 0:
        raise ValueError(
            "weibull_shape must be positive"
        )


def _sample_weibull_event_time(
    rng: np.random.Generator,
    baseline_hazard: float,
    weibull_shape: float,
    linear_predictor: float,
) -> float:
    uniform_value = rng.uniform(
        low=np.finfo(float).eps,
        high=1.0,
    )

    cumulative_hazard_target = (
        -np.log(uniform_value)
    )

    risk_multiplier = np.exp(
        np.clip(
            linear_predictor,
            -8.0,
            8.0,
        )
    )

    event_time = (
        cumulative_hazard_target
        / (
            baseline_hazard
            * risk_multiplier
        )
    ) ** (1.0 / weibull_shape)

    return float(event_time)


def _evolve_state(
    previous_value: float,
    persistence: float,
    long_run_mean: float,
    innovation_scale: float,
    rng: np.random.Generator,
) -> float:
    value = (
        persistence * previous_value
        + (1.0 - persistence)
        * long_run_mean
        + rng.normal(
            0.0,
            innovation_scale,
        )
    )

    return float(value)


def _sigmoid(value: float) -> float:
    clipped_value = float(
        np.clip(
            value,
            -20.0,
            20.0,
        )
    )

    return float(
        1.0
        / (
            1.0
            + np.exp(-clipped_value)
        )
    )