from dataclasses import dataclass

import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter


DEFAULT_COVARIATES = [
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

CONTINUOUS_COVARIATES = [
    "leverage",
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
class TimeVaryingCoxResults:
    model: CoxTimeVaryingFitter
    coefficients: pd.DataFrame
    number_of_firms: int
    number_of_rows: int
    number_of_events: int


def fit_time_varying_cox(
    data: pd.DataFrame,
    covariates: list[str] | None = None,
    penalizer: float = 0.0,
) -> TimeVaryingCoxResults:
    """
    Fit a Cox model with time-varying financial covariates.

    The input must use start-stop long format with one or more
    rows per firm.
    """
    selected_covariates = (
        covariates or DEFAULT_COVARIATES
    )

    required_columns = {
        "firm_id",
        "start",
        "stop",
        "event",
        *selected_covariates,
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Missing required columns: "
            f"{missing_text}"
        )

    model_data = data[
        [
            "firm_id",
            "start",
            "stop",
            "event",
            *selected_covariates,
        ]
    ].copy()

    if (
        model_data["stop"]
        <= model_data["start"]
    ).any():
        raise ValueError(
            "Every stop time must exceed start time"
        )

    model = CoxTimeVaryingFitter(
        penalizer=penalizer
    )

    model.fit(
        model_data,
        id_col="firm_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        show_progress=False,
    )

    coefficients = model.summary[
        [
            "coef",
            "exp(coef)",
            "se(coef)",
            "z",
            "p",
            "coef lower 95%",
            "coef upper 95%",
        ]
    ].copy()

    return TimeVaryingCoxResults(
        model=model,
        coefficients=coefficients,
        number_of_firms=(
            int(model_data["firm_id"].nunique())
        ),
        number_of_rows=len(model_data),
        number_of_events=int(
            model_data["event"].sum()
        ),
    )

def standardize_time_varying_covariates(
    data: pd.DataFrame,
    continuous_covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize continuous covariates using the full observed dataset.

    Binary indicators, identifiers, event columns and time columns
    are not standardized.
    """
    selected_covariates = (
        continuous_covariates
        or CONTINUOUS_COVARIATES
    )

    missing_columns = (
        set(selected_covariates)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing standardization columns: "
            + ", ".join(sorted(missing_columns))
        )

    standardized_data = data.copy()

    parameter_rows: list[dict[str, float | str]] = []

    for covariate in selected_covariates:
        mean_value = float(
            standardized_data[covariate].mean()
        )

        standard_deviation = float(
            standardized_data[covariate].std(ddof=0)
        )

        if (
            not np.isfinite(standard_deviation)
            or standard_deviation <= 0
        ):
            raise ValueError(
                f"Cannot standardize {covariate}: "
                "standard deviation must be positive"
            )

        standardized_data[covariate] = (
            standardized_data[covariate]
            - mean_value
        ) / standard_deviation

        parameter_rows.append(
            {
                "covariate": covariate,
                "mean": mean_value,
                "standard_deviation": (
                    standard_deviation
                ),
            }
        )

    parameters = pd.DataFrame(parameter_rows)

    return standardized_data, parameters