from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from src.features.constants import MODEL_FEATURES
from src.models.base_cox import (
    remove_near_constant_features,
    validate_time_varying_data,
)


DEFAULT_PENALIZER_GRID = [
    0.0,
    0.001,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]


@dataclass(frozen=True)
class RealModelSelectionResults:
    fold_results: pd.DataFrame
    penalty_summary: pd.DataFrame
    selected_penalizer: float
    number_of_folds: int


def build_firm_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct one row per firm for grouped fold assignment.
    """
    summary = (
        data.groupby("firm_id", as_index=False)
        .agg(
            event=("event", "max"),
            duration=("stop", "max"),
            number_of_rows=("firm_id", "size"),
        )
    )

    if summary["firm_id"].duplicated().any():
        raise ValueError(
            "Firm summary contains duplicate firm identifiers."
        )

    if not summary["event"].isin([0, 1]).all():
        raise ValueError(
            "Firm-level events must be binary."
        )

    if summary["event"].nunique() < 2:
        raise ValueError(
            "Firm summary must contain events and censored firms."
        )

    return summary


def create_grouped_folds(
    data: pd.DataFrame,
    number_of_folds: int = 5,
    random_seed: int = 2026,
) -> list[tuple[set[str], set[str]]]:
    """
    Create stratified folds at the firm level.
    """
    if number_of_folds < 2:
        raise ValueError(
            "number_of_folds must be at least 2."
        )

    firm_summary = build_firm_summary(data)

    event_count = int(
        firm_summary["event"].sum()
    )

    censored_count = int(
        len(firm_summary) - event_count
    )

    maximum_folds = min(
        event_count,
        censored_count,
    )

    if number_of_folds > maximum_folds:
        raise ValueError(
            "number_of_folds exceeds the number of firms "
            "available in the smallest outcome group."
        )

    splitter = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_seed,
    )

    firm_ids = (
        firm_summary["firm_id"]
        .astype(str)
        .to_numpy()
    )

    outcomes = (
        firm_summary["event"]
        .astype(int)
        .to_numpy()
    )

    folds: list[
        tuple[set[str], set[str]]
    ] = []

    for train_indices, validation_indices in (
        splitter.split(firm_ids, outcomes)
    ):
        train_firms = set(
            firm_ids[train_indices]
        )

        validation_firms = set(
            firm_ids[validation_indices]
        )

        if train_firms & validation_firms:
            raise RuntimeError(
                "A firm appeared in both training "
                "and validation sets."
            )

        folds.append(
            (
                train_firms,
                validation_firms,
            )
        )

    return folds


def standardize_train_validation(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Fit feature scaling on training data and apply it to both sets.
    """
    training = training_data.copy()
    validation = validation_data.copy()

    means = (
        training[feature_columns]
        .mean()
        .astype(float)
    )

    standard_deviations = (
        training[feature_columns]
        .std(ddof=0)
        .astype(float)
    )

    invalid = (
        ~np.isfinite(standard_deviations)
        | (standard_deviations <= 0)
    )

    if invalid.any():
        invalid_features = (
            standard_deviations[
                invalid
            ].index.tolist()
        )

        raise ValueError(
            "Invalid training-fold feature variation: "
            f"{invalid_features}"
        )

    training[feature_columns] = (
        training[feature_columns] - means
    ) / standard_deviations

    validation[feature_columns] = (
        validation[feature_columns] - means
    ) / standard_deviations

    return (
        training,
        validation,
        means,
        standard_deviations,
    )


def calculate_validation_concordance(
    model: CoxTimeVaryingFitter,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """
    Calculate firm-level validation concordance.

    Each firm's risk score is taken from its final observed interval.
    Higher predicted hazard corresponds to shorter survival, so the
    negative partial hazard is passed to concordance_index.
    """
    ordered = (
        validation_data.sort_values(
            ["firm_id", "start", "stop"]
        )
        .reset_index(drop=True)
    )

    final_rows = (
        ordered.groupby(
            "firm_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    partial_hazards = (
        model.predict_partial_hazard(
            final_rows[feature_columns]
        )
    )

    risk_scores = np.asarray(
        partial_hazards
    ).reshape(-1)

    durations = (
        final_rows["stop"]
        .astype(float)
        .to_numpy()
    )

    events = (
        ordered.groupby("firm_id")["event"]
        .max()
        .reindex(
            final_rows["firm_id"]
        )
        .astype(int)
        .to_numpy()
    )

    score = concordance_index(
        event_times=durations,
        predicted_scores=-risk_scores,
        event_observed=events,
    )

    return float(score)


def evaluate_penalizer_grid(
    data: pd.DataFrame,
    penalizer_grid: list[float] | None = None,
    feature_columns: list[str] | None = None,
    number_of_folds: int = 5,
    random_seed: int = 2026,
) -> RealModelSelectionResults:
    """
    Evaluate unpenalized and ridge time-varying Cox models using
    firm-level grouped cross-validation.
    """
    selected_features = (
        MODEL_FEATURES.copy()
        if feature_columns is None
        else list(feature_columns)
    )

    penalties = (
        DEFAULT_PENALIZER_GRID.copy()
        if penalizer_grid is None
        else list(penalizer_grid)
    )

    if not penalties:
        raise ValueError(
            "At least one penalizer value is required."
        )

    if any(
        penalty < 0
        for penalty in penalties
    ):
        raise ValueError(
            "Penalizer values cannot be negative."
        )

    validated = validate_time_varying_data(
        data=data,
        feature_columns=selected_features,
    )

    folds = create_grouped_folds(
        data=validated,
        number_of_folds=number_of_folds,
        random_seed=random_seed,
    )

    rows: list[dict[str, object]] = []

    for penalizer in penalties:
        for fold_number, (
            training_firms,
            validation_firms,
        ) in enumerate(
            folds,
            start=1,
        ):
            training = validated[
                validated["firm_id"]
                .astype(str)
                .isin(training_firms)
            ].copy()

            validation = validated[
                validated["firm_id"]
                .astype(str)
                .isin(validation_firms)
            ].copy()

            retained_features = (
                remove_near_constant_features(
                    data=training,
                    feature_columns=(
                        selected_features
                    ),
                )
            )

            training_columns = [
                "firm_id",
                "start",
                "stop",
                "event",
                *retained_features,
            ]

            training_model_data = (
                training[
                    training_columns
                ].copy()
            )

            validation_model_data = (
                validation[
                    training_columns
                ].copy()
            )

            try:
                (
                    training_model_data,
                    validation_model_data,
                    _,
                    _,
                ) = standardize_train_validation(
                    training_data=(
                        training_model_data
                    ),
                    validation_data=(
                        validation_model_data
                    ),
                    feature_columns=(
                        retained_features
                    ),
                )

                model = CoxTimeVaryingFitter(
                    penalizer=float(penalizer),
                    l1_ratio=0.0,
                )

                model.fit(
                    training_model_data,
                    id_col="firm_id",
                    start_col="start",
                    stop_col="stop",
                    event_col="event",
                    show_progress=False,
                )

                validation_concordance = (
                    calculate_validation_concordance(
                        model=model,
                        validation_data=(
                            validation_model_data
                        ),
                        feature_columns=(
                            retained_features
                        ),
                    )
                )

                rows.append(
                    {
                        "penalizer": float(
                            penalizer
                        ),
                        "fold": fold_number,
                        "validation_concordance": (
                            validation_concordance
                        ),
                        "training_firms": len(
                            training_firms
                        ),
                        "validation_firms": len(
                            validation_firms
                        ),
                        "training_events": int(
                            training_model_data[
                                "event"
                            ].sum()
                        ),
                        "validation_events": int(
                            validation_model_data[
                                "event"
                            ].sum()
                        ),
                        "number_of_features": len(
                            retained_features
                        ),
                        "status": "success",
                        "error": "",
                    }
                )

            except Exception as error:
                rows.append(
                    {
                        "penalizer": float(
                            penalizer
                        ),
                        "fold": fold_number,
                        "validation_concordance": (
                            np.nan
                        ),
                        "training_firms": len(
                            training_firms
                        ),
                        "validation_firms": len(
                            validation_firms
                        ),
                        "training_events": int(
                            training["event"].sum()
                        ),
                        "validation_events": int(
                            validation["event"].sum()
                        ),
                        "number_of_features": len(
                            retained_features
                        ),
                        "status": "failed",
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

    fold_results = pd.DataFrame(rows)

    successful = fold_results[
        fold_results["status"]
        == "success"
    ].copy()

    if successful.empty:
        raise RuntimeError(
            "Every model-selection fit failed."
        )

    penalty_summary = (
        successful.groupby(
            "penalizer",
            as_index=False,
        )
        .agg(
            mean_validation_concordance=(
                "validation_concordance",
                "mean",
            ),
            standard_deviation_concordance=(
                "validation_concordance",
                "std",
            ),
            minimum_validation_concordance=(
                "validation_concordance",
                "min",
            ),
            maximum_validation_concordance=(
                "validation_concordance",
                "max",
            ),
            successful_folds=(
                "fold",
                "count",
            ),
        )
    )

    penalty_summary[
        "failed_folds"
    ] = (
        number_of_folds
        - penalty_summary[
            "successful_folds"
        ]
    )

    penalty_summary = (
        penalty_summary.sort_values(
            [
                "mean_validation_concordance",
                "penalizer",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    selected_penalizer = float(
        penalty_summary.iloc[0][
            "penalizer"
        ]
    )

    return RealModelSelectionResults(
        fold_results=fold_results,
        penalty_summary=penalty_summary,
        selected_penalizer=(
            selected_penalizer
        ),
        number_of_folds=number_of_folds,
    )