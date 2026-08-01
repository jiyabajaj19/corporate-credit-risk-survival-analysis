import pandas as pd

from src.data.event_universe import (
    build_eligible_bankruptcy_universe,
    create_sec_company_config,
)


def create_example_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_case_id": [
                "A",
                "B",
                "C",
                "D",
            ],
            "cik": [
                "0000000001",
                "0000000001",
                "0000000002",
                None,
            ],
            "company_name": [
                "Firm One",
                "Firm One Refile",
                "Firm Two",
                "No CIK Firm",
            ],
            "common_name": [
                "Firm One",
                "Firm One",
                "Firm Two",
                "No CIK",
            ],
            "event_date": pd.to_datetime(
                [
                    "2015-01-01",
                    "2018-01-01",
                    "2008-01-01",
                    "2019-01-01",
                ]
            ),
            "chapter": [
                "11",
                "11",
                "7",
                "11",
            ],
            "sic_description": [
                "Industry A",
                "Industry A",
                "Industry B",
                "Industry C",
            ],
        }
    )


def test_eligible_universe_filters_and_deduplicates() -> None:
    eligible = build_eligible_bankruptcy_universe(
        create_example_events(),
        minimum_event_date="2010-01-01",
        maximum_event_date="2022-12-31",
    )

    assert len(eligible) == 1

    assert (
        eligible.iloc[0]["source_case_id"]
        == "A"
    )


def test_create_sec_company_config() -> None:
    eligible = build_eligible_bankruptcy_universe(
        create_example_events()
    )

    records = create_sec_company_config(
        eligible
    )

    assert len(records) == 1
    assert records[0]["cik"] == "0000000001"
    assert records[0]["ticker"] == (
        "CIK0000000001"
    )