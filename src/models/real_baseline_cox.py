from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from datetime import timedelta

from src.features.constants import MODEL_FEATURES
from src.models.base_cox import (
    CoxModelResults,
    create_cox_results,
    remove_near_constant_features,
    standardize_features,
    validate_time_varying_data,
)


def collapse_to_baseline_data(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse start-stop histories to one row per firm.

    Predictors come from the firm's first retained quarterly
    observation. Follow-up ends at bankruptcy or censoring.
    """
    selected_features = (
        MODEL_FEATURES.copy()
        if feature_columns is None
        else list(feature_columns)
    )

    validated = validate_time_varying_data(
        data=data,
        feature_columns=selected_features,
    )

    rows: list[dict[str, object]] = []

    for firm_id, firm_data in validated.groupby(
        "firm_id",
        sort=False,
    ):
        firm_data = (
            firm_data.sort_values(
                ["start", "stop"]
            )
            .reset_index(drop=True)
        )

        first_row = firm_data.iloc[0]
        final_row = firm_data.iloc[-1]

        duration = float(
            firm_data["stop"].max()
            - firm_data["start"].min()
        )

        if duration <= 0:
            continue

        record: dict[str, object] = {
            "firm_id": str(firm_id),
            "duration": duration,
            "event": int(
                firm_data["event"].sum()
            ),
        }

        if "end_date" in firm_data.columns:
            record["entry_date"] = pd.to_datetime(
                first_row["end_date"],
                errors="coerce",
            )

        if "event_date" in firm_data.columns:
            record["event_date"] = pd.to_datetime(
                final_row["event_date"],
                errors="coerce",
            )

        for feature in selected_features:
            record[feature] = first_row[
                feature
            ]

        rows.append(record)

    baseline = pd.DataFrame(rows)

    if baseline.empty:
        raise ValueError(
            "No baseline firm records were created."
        )

    if (
        baseline.groupby("firm_id")["event"]
        .sum()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "A baseline firm cannot have more than one event."
        )

    return baseline.reset_index(drop=True)


def prepare_baseline_model_data(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    standardize: bool = True,
) -> tuple[
    pd.DataFrame,
    list[str],
    pd.Series,
    pd.Series,
]:
    """
    Build and prepare one-row-per-firm baseline data.
    """
    selected_features = (
        MODEL_FEATURES.copy()
        if feature_columns is None
        else list(feature_columns)
    )

    baseline = collapse_to_baseline_data(
        data=data,
        feature_columns=selected_features,
    )

    retained_features = (
        remove_near_constant_features(
            data=baseline,
            feature_columns=selected_features,
        )
    )

    model_data = baseline[
        [
            "firm_id",
            "duration",
            "event",
            *[
                column
                for column in [
                    "entry_date",
                    "event_date",
                ]
                if column in baseline.columns
            ],
            *retained_features,
        ]
    ].copy()

    if standardize:
        (
            model_data,
            feature_means,
            feature_standard_deviations,
        ) = standardize_features(
            data=model_data,
            feature_columns=retained_features,
        )
    else:
        feature_means = pd.Series(
            0.0,
            index=retained_features,
            dtype=float,
        )

        feature_standard_deviations = pd.Series(
            1.0,
            index=retained_features,
            dtype=float,
        )

    return (
        model_data,
        retained_features,
        feature_means,
        feature_standard_deviations,
    )


def fit_real_naive_cox(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    standardize: bool = True,
) -> CoxModelResults:
    """
    Fit a baseline Cox model using follow-up duration from entry.

    This ignores information from financial changes after the first
    retained quarter.
    """
    (
        model_data,
        features,
        means,
        standard_deviations,
    ) = prepare_baseline_model_data(
        data=data,
        feature_columns=feature_columns,
        standardize=standardize,
    )

    fit_data = model_data[
        [
            "duration",
            "event",
            *features,
        ]
    ].copy()

    model = CoxPHFitter()

    model.fit(
        fit_data,
        duration_col="duration",
        event_col="event",
    )

    result_data = model_data[
        [
            "firm_id",
            "duration",
            "event",
            *features,
        ]
    ].copy()

    return create_cox_results(
        model_name="Naive Baseline Cox",
        model=model,
        modeling_data=result_data,
        feature_columns=features,
        feature_means=means,
        feature_standard_deviations=(
            standard_deviations
        ),
    )


def fit_real_left_truncated_cox(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    standardize: bool = True,
) -> CoxModelResults:
    """
    Fit a calendar-time left-truncated baseline Cox model.

    Firms enter the risk set on their first retained financial
    statement date and exit after their observed follow-up duration.
    """
    (
        model_data,
        features,
        means,
        standard_deviations,
    ) = prepare_baseline_model_data(
        data=data,
        feature_columns=feature_columns,
        standardize=standardize,
    )

    if "entry_date" not in model_data.columns:
        raise ValueError(
            "end_date is required for the "
            "left-truncated benchmark."
        )

    if model_data["entry_date"].isna().any():
        raise ValueError(
            "Some firms have invalid entry dates."
        )

    calendar_origin = (
    model_data["entry_date"].min()
    - timedelta(days=1)
    )

    model_data["entry_time"] = (
        model_data["entry_date"]
        - calendar_origin
    ).dt.days / 365.25

    model_data["exit_time"] = (
        model_data["entry_time"]
        + model_data["duration"]
    )

    if not (
        model_data["exit_time"]
        > model_data["entry_time"]
    ).all():
        raise ValueError(
            "Every exit time must exceed entry time."
        )

    fit_data = model_data[
        [
            "entry_time",
            "exit_time",
            "event",
            *features,
        ]
    ].copy()

    model = CoxPHFitter()

    model.fit(
        fit_data,
        duration_col="exit_time",
        event_col="event",
        entry_col="entry_time",
    )

    result_data = model_data[
        [
            "firm_id",
            "entry_time",
            "exit_time",
            "event",
            *features,
        ]
    ].copy()

    return create_cox_results(
        model_name=(
            "Left-Truncated Baseline Cox"
        ),
        model=model,
        modeling_data=result_data,
        feature_columns=features,
        feature_means=means,
        feature_standard_deviations=(
            standard_deviations
        ),
    )