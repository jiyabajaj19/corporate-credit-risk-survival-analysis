from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter


COVARIATES = [
    "leverage",
    "low_interest_coverage",
]


@dataclass
class CoxModelResults:
    model_name: str
    coefficients: pd.DataFrame
    concordance_index: float


def fit_naive_cox(
    data: pd.DataFrame,
) -> CoxModelResults:
    """
    Fit a Cox proportional-hazards model while ignoring delayed entry.

    This model incorrectly treats follow-up time as if all firms were
    observed from the beginning of their risk histories.
    """
    required_columns = {
        "follow_up_time",
        "event",
        *COVARIATES,
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    model_data = data[
        [
            "follow_up_time",
            "event",
            *COVARIATES,
        ]
    ].copy()

    model = CoxPHFitter()

    model.fit(
        model_data,
        duration_col="follow_up_time",
        event_col="event",
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

    return CoxModelResults(
        model_name="Naive Cox",
        coefficients=coefficients,
        concordance_index=float(
            model.concordance_index_
        ),
    )


def fit_left_truncated_cox(
    data: pd.DataFrame,
) -> CoxModelResults:
    """
    Fit a Cox model accounting for delayed entry.

    Each firm enters the risk set at entry_time and leaves at exit_time.
    """
    required_columns = {
        "entry_time",
        "exit_time",
        "event",
        *COVARIATES,
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    model_data = data[
        [
            "entry_time",
            "exit_time",
            "event",
            *COVARIATES,
        ]
    ].copy()

    model = CoxPHFitter()

    model.fit(
        model_data,
        duration_col="exit_time",
        event_col="event",
        entry_col="entry_time",
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

    return CoxModelResults(
        model_name="Left-Truncated Cox",
        coefficients=coefficients,
        concordance_index=float(
            model.concordance_index_
        ),
    )