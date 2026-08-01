from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.features.constants import MODEL_FEATURES


def load_survival_table(
    path: str | Path,
) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Survival dataset not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        parse_dates=[
            "end_date",
            "event_date",
        ],
        dtype={
            "firm_id": "string",
            "cik": "string",
            "ticker": "string",
        },
    )


def combine_survival_datasets(
    bankrupt: pd.DataFrame,
    controls: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Combine bankrupt and censored-control survival intervals.
    """
    selected_features = (
        MODEL_FEATURES
        if feature_columns is None
        else feature_columns
    )

    required_columns = {
        "firm_id",
        "cik",
        "ticker",
        "entity_name",
        "end_date",
        "start",
        "stop",
        "event",
        "is_bankrupt_firm",
        *selected_features,
    }

    for name, table in [
        ("bankrupt", bankrupt),
        ("controls", controls),
    ]:
        missing = (
            required_columns
            - set(table.columns)
        )

        if missing:
            raise ValueError(
                f"{name} survival table is missing: "
                f"{sorted(missing)}"
            )

    combined = pd.concat(
        [
            bankrupt,
            controls,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = combined.dropna(
        subset=selected_features
    ).copy()

    combined = combined[
        combined["stop"]
        > combined["start"]
    ].copy()

    # Prefix IDs so an unlikely CIK collision cannot merge the groups.
    combined["firm_id"] = (
        combined["is_bankrupt_firm"]
        .map(
            {
                1: "B_",
                0: "C_",
            }
        )
        + combined["cik"].astype(str)
    )

    combined["sample_group"] = (
        combined["is_bankrupt_firm"]
        .map(
            {
                1: "bankrupt",
                0: "control",
            }
        )
    )

    ordered_columns = [
        "firm_id",
        "cik",
        "ticker",
        "entity_name",
        "sample_group",
        "is_bankrupt_firm",
        "end_date",
        "event_date",
        "start",
        "stop",
        "event",
        *selected_features,
    ]

    remaining_columns = [
        column
        for column in combined.columns
        if column not in ordered_columns
    ]

    return (
        combined[
            ordered_columns
            + remaining_columns
        ]
        .sort_values(
            ["firm_id", "start"]
        )
        .reset_index(drop=True)
    )


def validate_combined_survival_dataset(
    data: pd.DataFrame,
) -> None:
    """
    Validate the combined bankrupt and control survival dataset.
    """
    if data.empty:
        raise ValueError(
            "Combined survival dataset is empty"
        )

    if not (
        data["stop"] > data["start"]
    ).all():
        raise ValueError(
            "Every stop time must exceed start time"
        )

    groups = set(
        data["sample_group"].dropna()
    )

    if groups != {
        "bankrupt",
        "control",
    }:
        raise ValueError(
            "Combined dataset must contain both "
            "bankrupt and control firms"
        )

    event_counts = (
        data.groupby("firm_id")["event"]
        .sum()
    )

    if (event_counts > 1).any():
        raise ValueError(
            "A firm cannot have more than one event"
        )

    bankrupt = data.loc[
        data["sample_group"] == "bankrupt"
    ].copy()

    controls = data.loc[
        data["sample_group"] == "control"
    ].copy()

    bankrupt_firm_count = int(
        bankrupt["firm_id"].nunique()
    )

    bankrupt_event_count = int(
        bankrupt["event"].sum()
    )

    if bankrupt_event_count != bankrupt_firm_count:
        raise ValueError(
            "Every retained bankrupt firm must have "
            "exactly one event. "
            f"Firms: {bankrupt_firm_count}, "
            f"events: {bankrupt_event_count}"
        )

    control_event_count = int(
        controls["event"].sum()
    )

    if control_event_count != 0:
        raise ValueError(
            "Control firms cannot contain "
            "bankruptcy events"
        )