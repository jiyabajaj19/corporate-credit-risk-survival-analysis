import pandas as pd

from src.data.lopucki import (
    build_bankruptcy_events,
    normalize_cik_value,
)


def test_normalize_cik_value() -> None:
    assert (
        normalize_cik_value(320193)
        == "0000320193"
    )

    assert (
        normalize_cik_value("320193.0")
        == "0000320193"
    )

    assert normalize_cik_value(None) is None
    assert normalize_cik_value("") is None


def test_build_bankruptcy_events() -> None:
    cases = pd.DataFrame(
        {
            "PrimaryKey": ["CASE-1"],
            "NameCorp": ["Example Corp."],
            "CommonName": ["Example"],
            "CikBefore": [320193],
            "DateFiled": ["2020-05-01"],
            "Chapter": ["11"],
            "Disposition": ["Reorganized"],
            "SICPrimary": [1234],
            "SICDescription": [
                "Example industry"
            ],
            "AssetsBefore": [1000.0],
            "LiabBefore": [800.0],
            "SalesBefore": [900.0],
        }
    )

    events = build_bankruptcy_events(
        cases
    )

    assert len(events) == 1

    event = events.iloc[0]

    assert event["cik"] == "0000320193"

    assert (
        event["event_date"]
        == pd.Timestamp("2020-05-01")
    )

    assert (
        event["event_type"]
        == "bankruptcy_filing"
    )

    assert event["company_name"] == (
        "Example Corp."
    )