import pandas as pd
import pytest

from src.data.quarterly_panel import (
    build_duration_quarterly_table,
    classify_fact_duration,
    derive_balance_sheet_values,
)


def test_classify_fact_duration() -> None:
    facts = pd.DataFrame(
        {
            "start_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "end_date": pd.to_datetime(
                [
                    "2024-03-31",
                    "2024-06-30",
                    "2024-09-30",
                    "2024-12-31",
                ]
            ),
        }
    )

    classified = classify_fact_duration(
        facts
    )

    assert classified.loc[
        0,
        "is_single_quarter",
    ]

    assert classified.loc[
        1,
        "is_half_year_ytd",
    ]

    assert classified.loc[
        2,
        "is_nine_month_ytd",
    ]

    assert classified.loc[
        3,
        "is_annual",
    ]


def test_duration_values_are_derived() -> None:
    facts = pd.DataFrame(
        {
            "ticker": ["TEST"] * 4,
            "cik": ["0000000001"] * 4,
            "entity_name": ["Test Company"] * 4,
            "concept": ["revenue"] * 4,
            "value": [
                100.0,
                220.0,
                360.0,
                500.0,
            ],
            "start_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "end_date": pd.to_datetime(
                [
                    "2024-03-31",
                    "2024-06-30",
                    "2024-09-30",
                    "2024-12-31",
                ]
            ),
            "filed_date": pd.to_datetime(
                [
                    "2024-04-30",
                    "2024-07-30",
                    "2024-10-30",
                    "2025-02-15",
                ]
            ),
            "form": [
                "10-Q",
                "10-Q",
                "10-Q",
                "10-K",
            ],
            "fiscal_year": [2024] * 4,
            "fiscal_period": [
                "Q1",
                "Q2",
                "Q3",
                "FY",
            ],
            "tag_priority": [0] * 4,
        }
    )

    facts = classify_fact_duration(facts)

    quarterly = build_duration_quarterly_table(
        facts
    )

    values = (
        quarterly.sort_values(
            "fiscal_period"
        )
        .set_index("fiscal_period")[
            "revenue"
        ]
    )

    assert values["Q1"] == pytest.approx(
        100.0
    )

    assert values["Q2"] == pytest.approx(
        120.0
    )

    assert values["Q3"] == pytest.approx(
        140.0
    )

    assert values["FY"] == pytest.approx(
        140.0
    )


def test_derive_total_debt() -> None:
    panel = pd.DataFrame(
        {
            "short_term_debt": [20.0],
            "long_term_debt": [80.0],
            "total_debt": [float("nan")],
            "total_assets": [200.0],
            "total_liabilities": [
                float("nan")
            ],
            "stockholders_equity": [75.0],
        }
    )

    result = derive_balance_sheet_values(
        panel
    )

    assert result.loc[
        0,
        "total_debt",
    ] == pytest.approx(100.0)

    assert result.loc[
        0,
        "total_liabilities",
    ] == pytest.approx(125.0)