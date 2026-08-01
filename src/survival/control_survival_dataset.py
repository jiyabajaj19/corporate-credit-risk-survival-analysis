from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.constants import MODEL_FEATURES

from src.survival.survival_dataset import (
    normalize_cik_series,
)



def load_control_features(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load cleaned quarterly features for non-bankrupt control firms.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Control feature file not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=["end_date"],
        dtype={
            "ticker": "string",
            "cik": "string",
            "entity_name": "string",
            "fiscal_period": "string",
        },
    )

    required_columns = {
        "ticker",
        "cik",
        "entity_name",
        "fiscal_year",
        "fiscal_period",
        "quarter_number",
        "end_date",
        *MODEL_FEATURES,
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Control feature table is missing columns: "
            f"{sorted(missing)}"
        )

    data["cik"] = normalize_cik_series(
        data["cik"]
    )

    return (
        data.dropna(
            subset=[
                "cik",
                "end_date",
                "fiscal_year",
                "quarter_number",
            ]
        )
        .sort_values(
            [
                "cik",
                "end_date",
                "fiscal_year",
                "quarter_number",
            ]
        )
        .reset_index(drop=True)
    )


def select_eligible_control_quarters(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    minimum_quarters: int = 4,
    maximum_quarters: int | None = 32,
) -> pd.DataFrame:
    """
    Select control firms with enough complete quarterly observations.

    Missingness is checked only for the selected model feature set.
    """
    if minimum_quarters < 2:
        raise ValueError(
            "minimum_quarters must be at least 2"
        )

    if (
        maximum_quarters is not None
        and maximum_quarters < minimum_quarters
    ):
        raise ValueError(
            "maximum_quarters must be at least minimum_quarters"
        )

    selected_features = (
        MODEL_FEATURES
        if feature_columns is None
        else feature_columns
    )

    missing_features = (
        set(selected_features)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing requested control features: "
            f"{sorted(missing_features)}"
        )

    complete = features.dropna(
        subset=selected_features
    ).copy()

    retained_tables: list[pd.DataFrame] = []

    for _, firm_data in complete.groupby(
        "cik",
        sort=False,
    ):
        firm_data = (
            firm_data.sort_values("end_date")
            .drop_duplicates(
                subset=["end_date"],
                keep="last",
            )
        )

        if len(firm_data) < minimum_quarters:
            continue

        if maximum_quarters is not None:
            firm_data = firm_data.tail(
                maximum_quarters
            )

        retained_tables.append(
            firm_data
        )

    if not retained_tables:
        return pd.DataFrame(
            columns=features.columns
        )

    return (
        pd.concat(
            retained_tables,
            ignore_index=True,
        )
        .sort_values(
            ["cik", "end_date"]
        )
        .reset_index(drop=True)
    )


def build_control_survival_intervals(
    eligible_controls: pd.DataFrame,
    minimum_quarters: int = 4,
) -> pd.DataFrame:
    """
    Convert control-company quarterly histories into censored intervals.

    Every interval has event = 0. The final stop time is extended by
    one quarter beyond the final observed quarter.
    """
    if minimum_quarters < 2:
        raise ValueError(
            "minimum_quarters must be at least 2"
        )

    if eligible_controls.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for cik, firm_data in eligible_controls.groupby(
        "cik",
        sort=False,
    ):
        firm_data = (
            firm_data.sort_values("end_date")
            .drop_duplicates(
                subset=["end_date"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        if len(firm_data) < minimum_quarters:
            continue

        first_date = firm_data.loc[
            0,
            "end_date",
        ]

        for row_number, row in firm_data.iterrows():
            start_time = (
                row["end_date"] - first_date
            ).days / 365.25

            if row_number < len(firm_data) - 1:
                next_date = firm_data.loc[
                    row_number + 1,
                    "end_date",
                ]

                stop_time = (
                    next_date - first_date
                ).days / 365.25
            else:
                # Approximate one additional quarter of censored
                # follow-up after the last observed financial report.
                censor_date = (
                    row["end_date"]
                    + pd.Timedelta(days=91)
                )

                stop_time = (
                    censor_date - first_date
                ).days / 365.25

            if stop_time <= start_time:
                continue

            record = row.to_dict()

            record.update(
                {
                    "firm_id": str(cik),
                    "start": float(start_time),
                    "stop": float(stop_time),
                    "event": 0,
                    "event_date": pd.NaT,
                    "observation_number": (
                        row_number + 1
                    ),
                    "number_of_observations": (
                        len(firm_data)
                    ),
                    "is_bankrupt_firm": 0,
                    "chapter": pd.NA,
                }
            )

            rows.append(record)

    if not rows:
        return pd.DataFrame()

    survival = pd.DataFrame(rows)

    ordered_columns = [
        "firm_id",
        "cik",
        "ticker",
        "entity_name",
        "end_date",
        "event_date",
        "start",
        "stop",
        "event",
        "observation_number",
        "number_of_observations",
        "is_bankrupt_firm",
        "chapter",
        *[
            feature
            for feature in MODEL_FEATURES
            if feature in survival.columns
        ],
    ]

    remaining_columns = [
        column
        for column in survival.columns
        if column not in ordered_columns
    ]

    return (
        survival[
            ordered_columns
            + remaining_columns
        ]
        .sort_values(
            ["firm_id", "start"]
        )
        .reset_index(drop=True)
    )


def validate_control_survival_dataset(
    survival: pd.DataFrame,
) -> None:
    """
    Validate censored-control survival intervals.
    """
    if survival.empty:
        raise ValueError(
            "Control survival dataset is empty"
        )

    if not (
        survival["stop"] > survival["start"]
    ).all():
        raise ValueError(
            "Every control stop time must exceed start time"
        )

    if survival["event"].sum() != 0:
        raise ValueError(
            "Control firms cannot contain bankruptcy events"
        )

    if not (
        survival["is_bankrupt_firm"] == 0
    ).all():
        raise ValueError(
            "All control rows must have is_bankrupt_firm = 0"
        )


def build_control_survival_dataset(
    feature_path: str | Path,
    feature_columns: list[str] | None = None,
    minimum_quarters: int = 4,
    maximum_quarters: int | None = 32,
) -> pd.DataFrame:
    """
    Run the complete control survival-data construction.
    """
    features = load_control_features(
        feature_path
    )

    eligible = select_eligible_control_quarters(
        features=features,
        feature_columns=feature_columns,
        minimum_quarters=minimum_quarters,
        maximum_quarters=maximum_quarters,
    )

    survival = build_control_survival_intervals(
        eligible_controls=eligible,
        minimum_quarters=minimum_quarters,
    )

    validate_control_survival_dataset(
        survival
    )

    return survival