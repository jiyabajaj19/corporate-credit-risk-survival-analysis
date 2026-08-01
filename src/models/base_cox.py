from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.constants import MODEL_FEATURES


TIME_VARYING_REQUIRED_COLUMNS = {
    "firm_id",
    "start",
    "stop",
    "event",
}

BASELINE_REQUIRED_COLUMNS = {
    "firm_id",
    "duration",
    "event",
}


@dataclass
class CoxModelResults:
    """
    Standard result container used by all real-data Cox models.
    """

    model_name: str
    model: Any
    coefficients: pd.DataFrame
    modeling_data: pd.DataFrame
    feature_columns: list[str]
    feature_means: pd.Series
    feature_standard_deviations: pd.Series
    number_of_firms: int
    number_of_rows: int
    number_of_events: int
    penalizer: float
    l1_ratio: float
    log_likelihood: float | None


def load_survival_dataset(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the combined real survival dataset.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Survival dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=[
            "end_date",
            "event_date",
        ],
        dtype={
            "firm_id": "string",
            "cik": "string",
            "ticker": "string",
            "entity_name": "string",
            "sample_group": "string",
        },
    )

    return data


def validate_feature_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Confirm that all requested predictors exist.
    """
    if not feature_columns:
        raise ValueError(
            "At least one model feature is required."
        )

    missing = (
        set(feature_columns)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Model features are missing: "
            f"{sorted(missing)}"
        )


def validate_time_varying_data(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Validate start-stop survival data.

    Returns a cleaned and sorted copy containing complete model rows.
    """
    selected_features = (
        MODEL_FEATURES.copy()
        if feature_columns is None
        else list(feature_columns)
    )

    required_columns = {
        *TIME_VARYING_REQUIRED_COLUMNS,
        *selected_features,
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Time-varying survival data is missing columns: "
            f"{sorted(missing)}"
        )

    result = data.copy()

    numeric_columns = [
        "start",
        "stop",
        "event",
        *selected_features,
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "firm_id",
            "start",
            "stop",
            "event",
            *selected_features,
        ]
    ).copy()

    if result.empty:
        raise ValueError(
            "No complete time-varying rows remain."
        )

    if not (
        result["stop"] > result["start"]
    ).all():
        raise ValueError(
            "Every stop time must exceed start time."
        )

    if not result["event"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Event values must be 0 or 1."
        )

    event_counts = (
        result.groupby("firm_id")["event"]
        .sum()
    )

    if (event_counts > 1).any():
        raise ValueError(
            "A firm cannot contain more than one event."
        )

    if int(result["event"].sum()) == 0:
        raise ValueError(
            "The dataset contains no events."
        )

    return (
        result.sort_values(
            [
                "firm_id",
                "start",
                "stop",
            ]
        )
        .reset_index(drop=True)
    )


def remove_near_constant_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    minimum_standard_deviation: float = 1e-8,
) -> list[str]:
    """
    Remove predictors with negligible variation.
    """
    if minimum_standard_deviation < 0:
        raise ValueError(
            "minimum_standard_deviation cannot be negative."
        )

    validate_feature_columns(
        data,
        feature_columns,
    )

    retained: list[str] = []

    for feature in feature_columns:
        values = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        standard_deviation = float(
            values.std(ddof=0)
        )

        if (
            np.isfinite(standard_deviation)
            and standard_deviation
            > minimum_standard_deviation
        ):
            retained.append(feature)

    if not retained:
        raise ValueError(
            "No nonconstant predictors remain."
        )

    return retained


def standardize_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Standardize predictors to mean zero and standard deviation one.

    The returned means and standard deviations allow coefficients to
    be interpreted as the effect of a one-standard-deviation change.
    """
    validate_feature_columns(
        data,
        feature_columns,
    )

    result = data.copy()

    means = (
        result[feature_columns]
        .mean()
        .astype(float)
    )

    standard_deviations = (
        result[feature_columns]
        .std(ddof=0)
        .astype(float)
    )

    invalid = (
        ~np.isfinite(
            standard_deviations
        )
        | (
            standard_deviations
            <= 0
        )
    )

    if invalid.any():
        invalid_features = (
            standard_deviations[
                invalid
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Cannot standardize constant or invalid predictors: "
            f"{invalid_features}"
        )

    result[feature_columns] = (
        result[feature_columns] - means
    ) / standard_deviations

    return (
        result,
        means,
        standard_deviations,
    )


def prepare_time_varying_model_data(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    standardize: bool = True,
    minimum_standard_deviation: float = 1e-8,
) -> tuple[
    pd.DataFrame,
    list[str],
    pd.Series,
    pd.Series,
]:
    """
    Validate, select, and optionally standardize time-varying data.
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

    retained_features = (
        remove_near_constant_features(
            data=validated,
            feature_columns=selected_features,
            minimum_standard_deviation=(
                minimum_standard_deviation
            ),
        )
    )

    modeling_data = validated[
        [
            "firm_id",
            "start",
            "stop",
            "event",
            *retained_features,
        ]
    ].copy()

    if standardize:
        (
            modeling_data,
            means,
            standard_deviations,
        ) = standardize_features(
            data=modeling_data,
            feature_columns=retained_features,
        )
    else:
        means = pd.Series(
            0.0,
            index=retained_features,
            dtype=float,
        )

        standard_deviations = pd.Series(
            1.0,
            index=retained_features,
            dtype=float,
        )

    return (
        modeling_data,
        retained_features,
        means,
        standard_deviations,
    )


def extract_coefficient_summary(
    model: Any,
    model_name: str,
) -> pd.DataFrame:
    """
    Convert a lifelines Cox summary into a common output format.
    """
    summary = model.summary.copy()

    required_columns = {
        "coef",
        "se(coef)",
        "z",
        "p",
    }

    missing = (
        required_columns
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "Model summary is missing columns: "
            f"{sorted(missing)}"
        )

    lower_column = None
    upper_column = None

    for candidate in [
        "coef lower 95%",
        "coef lower 95.0%",
    ]:
        if candidate in summary.columns:
            lower_column = candidate
            break

    for candidate in [
        "coef upper 95%",
        "coef upper 95.0%",
    ]:
        if candidate in summary.columns:
            upper_column = candidate
            break

    result = pd.DataFrame(
        {
            "model": model_name,
            "feature": summary.index.astype(
                str
            ),
            "coefficient": summary[
                "coef"
            ].to_numpy(),
            "standard_error": summary[
                "se(coef)"
            ].to_numpy(),
            "z_statistic": summary[
                "z"
            ].to_numpy(),
            "p_value": summary[
                "p"
            ].to_numpy(),
        }
    )

    result["hazard_ratio"] = np.exp(
        result["coefficient"]
    )

    if (
        lower_column is not None
        and upper_column is not None
    ):
        result[
            "coefficient_lower_95"
        ] = summary[
            lower_column
        ].to_numpy()

        result[
            "coefficient_upper_95"
        ] = summary[
            upper_column
        ].to_numpy()

        result[
            "hazard_ratio_lower_95"
        ] = np.exp(
            result[
                "coefficient_lower_95"
            ]
        )

        result[
            "hazard_ratio_upper_95"
        ] = np.exp(
            result[
                "coefficient_upper_95"
            ]
        )
    else:
        result[
            "coefficient_lower_95"
        ] = np.nan

        result[
            "coefficient_upper_95"
        ] = np.nan

        result[
            "hazard_ratio_lower_95"
        ] = np.nan

        result[
            "hazard_ratio_upper_95"
        ] = np.nan

    return result[
        [
            "model",
            "feature",
            "coefficient",
            "coefficient_lower_95",
            "coefficient_upper_95",
            "hazard_ratio",
            "hazard_ratio_lower_95",
            "hazard_ratio_upper_95",
            "standard_error",
            "z_statistic",
            "p_value",
        ]
    ]


def get_model_log_likelihood(
    model: Any,
) -> float | None:
    """
    Safely retrieve a fitted lifelines model log likelihood.
    """
    value = getattr(
        model,
        "log_likelihood_",
        None,
    )

    if value is None:
        return None

    numeric_value = float(value)

    if not np.isfinite(
        numeric_value
    ):
        return None

    return numeric_value


def create_cox_results(
    *,
    model_name: str,
    model: Any,
    modeling_data: pd.DataFrame,
    feature_columns: list[str],
    feature_means: pd.Series,
    feature_standard_deviations: pd.Series,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
) -> CoxModelResults:
    """
    Create a standardized result object for a fitted Cox model.
    """
    coefficients = (
        extract_coefficient_summary(
            model=model,
            model_name=model_name,
        )
    )

    return CoxModelResults(
        model_name=model_name,
        model=model,
        coefficients=coefficients,
        modeling_data=modeling_data,
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_standard_deviations=(
            feature_standard_deviations
        ),
        number_of_firms=int(
            modeling_data[
                "firm_id"
            ].nunique()
        ),
        number_of_rows=len(
            modeling_data
        ),
        number_of_events=int(
            modeling_data["event"].sum()
        ),
        penalizer=float(penalizer),
        l1_ratio=float(l1_ratio),
        log_likelihood=(
            get_model_log_likelihood(
                model
            )
        ),
    )