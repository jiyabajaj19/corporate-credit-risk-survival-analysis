from __future__ import annotations

import pandas as pd
from lifelines import CoxTimeVaryingFitter

from src.models.base_cox import (
    CoxModelResults,
    create_cox_results,
    prepare_time_varying_model_data,
)


def fit_real_time_varying_cox(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
    standardize: bool = True,
    model_name: str | None = None,
) -> CoxModelResults:
    """
    Fit a Cox time-varying model to the combined real survival data.

    Parameters
    ----------
    data:
        Start-stop survival data with one or more rows per firm.
    feature_columns:
        Predictors to include. Uses MODEL_FEATURES when omitted.
    penalizer:
        Overall regularization strength. Use zero for the classical
        unpenalized model and a positive value for ridge/elastic-net.
    l1_ratio:
        Fraction of the penalty allocated to L1 regularization.
        Use zero for ridge regularization.
    standardize:
        Whether predictors should be standardized before fitting.
    model_name:
        Optional display name for reports.
    """
    if penalizer < 0:
        raise ValueError(
            "penalizer cannot be negative"
        )

    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(
            "l1_ratio must be between 0 and 1"
        )

    (
        modeling_data,
        retained_features,
        feature_means,
        feature_standard_deviations,
    ) = prepare_time_varying_model_data(
        data=data,
        feature_columns=feature_columns,
        standardize=standardize,
    )

    model = CoxTimeVaryingFitter(
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )

    model.fit(
        modeling_data,
        id_col="firm_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        show_progress=False,
    )

    if model_name is None:
        if penalizer == 0:
            model_name = "Time-Varying Cox"
        elif l1_ratio == 0:
            model_name = "Ridge Time-Varying Cox"
        else:
            model_name = (
                "Elastic-Net Time-Varying Cox"
            )

    return create_cox_results(
        model_name=model_name,
        model=model,
        modeling_data=modeling_data,
        feature_columns=retained_features,
        feature_means=feature_means,
        feature_standard_deviations=(
            feature_standard_deviations
        ),
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )


def fit_unpenalized_real_time_varying_cox(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    standardize: bool = True,
) -> CoxModelResults:
    """
    Fit the classical unpenalized time-varying Cox model.
    """
    return fit_real_time_varying_cox(
        data=data,
        feature_columns=feature_columns,
        penalizer=0.0,
        l1_ratio=0.0,
        standardize=standardize,
        model_name="Time-Varying Cox",
    )


def fit_ridge_real_time_varying_cox(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    penalizer: float = 0.10,
    standardize: bool = True,
) -> CoxModelResults:
    """
    Fit a ridge-penalized time-varying Cox model.
    """
    if penalizer <= 0:
        raise ValueError(
            "Ridge penalizer must be positive"
        )

    return fit_real_time_varying_cox(
        data=data,
        feature_columns=feature_columns,
        penalizer=penalizer,
        l1_ratio=0.0,
        standardize=standardize,
        model_name="Ridge Time-Varying Cox",
    )