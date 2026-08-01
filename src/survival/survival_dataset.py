from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FINAL_MODEL_FEATURES = [
    "leverage",
    "current_ratio",
    "cash_ratio",
    "return_on_assets",
    "revenue_growth",
    "operating_cash_flow_ratio",
    "log_total_assets",
    "operating_margin",
]

IDENTIFIER_COLUMNS = [
    "ticker",
    "cik",
    "entity_name",
    "fiscal_year",
    "fiscal_period",
    "quarter_number",
    "end_date",
]


DEFAULT_FEATURE_COLUMNS = [
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
    "interest_coverage",
    "operating_margin",
]


def normalize_cik_series(
    values: pd.Series,
) -> pd.Series:
    """
    Normalize SEC CIK identifiers to ten-character strings.
    """

    def normalize(value: object) -> str | None:
        if pd.isna(value):
            return None

        text = str(value).strip()

        if text.endswith(".0"):
            text = text[:-2]

        digits = "".join(
            character
            for character in text
            if character.isdigit()
        )

        if not digits:
            return None

        return digits.zfill(10)

    return values.map(normalize).astype("string")


def load_clean_features(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load cleaned SEC company-quarter features.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Clean feature file not found: {file_path}"
        )

    features = pd.read_csv(
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
        *IDENTIFIER_COLUMNS,
        *DEFAULT_FEATURE_COLUMNS,
    }

    missing = required_columns - set(features.columns)

    if missing:
        raise ValueError(
            "Clean feature table is missing columns: "
            f"{sorted(missing)}"
        )

    features["cik"] = normalize_cik_series(
        features["cik"]
    )

    return (
        features.dropna(
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


def load_bankruptcy_events(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load normalized LoPucki bankruptcy events.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Bankruptcy event file not found: {file_path}"
        )

    events = pd.read_csv(
        file_path,
        parse_dates=["event_date"],
        dtype={
            "cik": "string",
            "company_name": "string",
            "common_name": "string",
            "chapter": "string",
            "source_case_id": "string",
        },
    )

    required_columns = {
        "cik",
        "event_date",
        "company_name",
        "chapter",
        "source_case_id",
    }

    missing = required_columns - set(events.columns)

    if missing:
        raise ValueError(
            "Bankruptcy event table is missing columns: "
            f"{sorted(missing)}"
        )

    events["cik"] = normalize_cik_series(
        events["cik"]
    )

    events = events.dropna(
        subset=["cik", "event_date"]
    )

    # Use the earliest bankruptcy filing per CIK.
    return (
        events.sort_values(
            ["cik", "event_date"]
        )
        .drop_duplicates(
            subset=["cik"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def merge_features_with_events(
    features: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match SEC feature histories to LoPucki event dates by CIK.
    """
    event_columns = [
        "cik",
        "event_date",
        "company_name",
        "common_name",
        "chapter",
        "disposition",
        "source_case_id",
    ]

    available_event_columns = [
        column
        for column in event_columns
        if column in events.columns
    ]

    merged = features.merge(
        events[available_event_columns],
        on="cik",
        how="inner",
        validate="many_to_one",
    )

    return merged.sort_values(
        ["cik", "end_date"]
    ).reset_index(drop=True)


def filter_pre_bankruptcy_quarters(
    merged: pd.DataFrame,
    minimum_days_before_event: int = 1,
    maximum_lookback_years: float | None = 8.0,
) -> pd.DataFrame:
    """
    Keep only observations before bankruptcy.

    A maximum lookback window prevents very old observations from
    dominating firms with long SEC histories.
    """
    if minimum_days_before_event < 0:
        raise ValueError(
            "minimum_days_before_event cannot be negative"
        )

    if (
        maximum_lookback_years is not None
        and maximum_lookback_years <= 0
    ):
        raise ValueError(
            "maximum_lookback_years must be positive"
        )

    data = merged.copy()

    days_to_event = (
        data["event_date"]
        - data["end_date"]
    ).dt.days

    keep = days_to_event >= minimum_days_before_event

    if maximum_lookback_years is not None:
        maximum_days = int(
            maximum_lookback_years * 365.25
        )

        keep &= days_to_event <= maximum_days

    data = data.loc[keep].copy()

    data["days_to_event"] = (
        data["event_date"]
        - data["end_date"]
    ).dt.days

    return (
        data.sort_values(
            ["cik", "end_date"]
        )
        .reset_index(drop=True)
    )


def build_bankrupt_survival_intervals(
    pre_event_data: pd.DataFrame,
    minimum_quarters: int = 4,
) -> pd.DataFrame:
    """
    Convert pre-bankruptcy quarterly histories into start-stop format.

    The final observed interval for every retained company receives
    event = 1.
    """
    if minimum_quarters < 2:
        raise ValueError(
            "minimum_quarters must be at least 2"
        )

    if pre_event_data.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for cik, firm_data in pre_event_data.groupby(
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
                row["end_date"]
                - first_date
            ).days / 365.25

            if row_number < len(firm_data) - 1:
                next_date = firm_data.loc[
                    row_number + 1,
                    "end_date",
                ]

                stop_time = (
                    next_date - first_date
                ).days / 365.25

                event = 0
            else:
                stop_time = (
                    row["event_date"]
                    - first_date
                ).days / 365.25

                event = 1

            if stop_time <= start_time:
                continue

            record = row.to_dict()

            record.update(
                {
                    "firm_id": str(cik),
                    "start": float(start_time),
                    "stop": float(stop_time),
                    "event": int(event),
                    "observation_number": (
                        row_number + 1
                    ),
                    "number_of_observations": (
                        len(firm_data)
                    ),
                    "is_bankrupt_firm": 1,
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
            column
            for column in DEFAULT_FEATURE_COLUMNS
            if column in survival.columns
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


def validate_survival_dataset(
    survival: pd.DataFrame,
) -> None:
    """
    Validate start-stop survival-data invariants.
    """
    if survival.empty:
        raise ValueError(
            "Survival dataset is empty"
        )

    if not (
        survival["stop"] > survival["start"]
    ).all():
        raise ValueError(
            "Every stop time must exceed start time"
        )

    event_counts = (
        survival.groupby("firm_id")["event"]
        .sum()
    )

    if not (event_counts == 1).all():
        raise ValueError(
            "Every bankrupt firm must have exactly one event"
        )

    for _, firm_data in survival.groupby(
        "firm_id"
    ):
        firm_data = firm_data.sort_values(
            "start"
        )

        if (
            firm_data.iloc[-1]["event"]
            != 1
        ):
            raise ValueError(
                "Bankruptcy event must occur on "
                "the final firm interval"
            )


def build_bankrupt_survival_dataset(
    feature_path: str | Path,
    event_path: str | Path,
    minimum_quarters: int = 4,
    maximum_lookback_years: float | None = 8.0,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the full bankrupt-firm survival dataset construction.
    """
    features = load_clean_features(
        feature_path
    )

    events = load_bankruptcy_events(
        event_path
    )

    merged = merge_features_with_events(
        features=features,
        events=events,
    )

    pre_event = filter_pre_bankruptcy_quarters(
        merged,
        maximum_lookback_years=(
            maximum_lookback_years
        ),
    )

    selected_features = (
        FINAL_MODEL_FEATURES
        if feature_columns is None
        else feature_columns
    )

    missing_features = (
        set(selected_features)
        - set(pre_event.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing requested bankruptcy features: "
            f"{sorted(missing_features)}"
        )

    # Filter before constructing intervals so the final retained
    # observation still receives the bankruptcy event.
    complete_pre_event = pre_event.dropna(
        subset=selected_features
    ).copy()

    survival = build_bankrupt_survival_intervals(
        complete_pre_event,
        minimum_quarters=minimum_quarters,
    )

    validate_survival_dataset(
        survival
    )

    return survival