import pandas as pd

from src.survival.survival_dataset import (
    build_bankrupt_survival_intervals,
    filter_pre_bankruptcy_quarters,
    validate_survival_dataset,
)


def create_example_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["TEST"] * 4,
            "cik": ["0000000001"] * 4,
            "entity_name": ["Test Corp"] * 4,
            "end_date": pd.to_datetime(
                [
                    "2019-03-31",
                    "2019-06-30",
                    "2019-09-30",
                    "2019-12-31",
                ]
            ),
            "event_date": pd.to_datetime(
                ["2020-02-15"] * 4
            ),
            "chapter": ["11"] * 4,
            "leverage": [
                0.20,
                0.25,
                0.30,
                0.40,
            ],
        }
    )


def test_filter_pre_bankruptcy_quarters() -> None:
    data = create_example_data()

    filtered = filter_pre_bankruptcy_quarters(
        data,
        maximum_lookback_years=2.0,
    )

    assert len(filtered) == 4

    assert (
        filtered["end_date"]
        < filtered["event_date"]
    ).all()


def test_build_bankrupt_survival_intervals() -> None:
    survival = build_bankrupt_survival_intervals(
        create_example_data(),
        minimum_quarters=4,
    )

    assert len(survival) == 4
    assert survival["event"].sum() == 1
    assert survival.iloc[-1]["event"] == 1
    assert (
        survival["stop"]
        > survival["start"]
    ).all()


def test_validate_survival_dataset() -> None:
    survival = build_bankrupt_survival_intervals(
        create_example_data(),
        minimum_quarters=4,
    )

    validate_survival_dataset(
        survival
    )