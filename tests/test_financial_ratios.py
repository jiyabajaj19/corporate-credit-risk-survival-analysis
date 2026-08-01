import numpy as np
import pandas as pd
import pytest

from src.features.financial_ratios import (
    build_credit_risk_features,
    calculate_quarterly_growth,
    safe_divide,
)


def create_example_panel() -> pd.DataFrame:
    million = 1_000_000.0

    return pd.DataFrame(
        {
            "ticker": ["TEST", "TEST"],
            "cik": [
                "0000000001",
                "0000000001",
            ],
            "entity_name": [
                "Test Company",
                "Test Company",
            ],
            "fiscal_year": [2024, 2024],
            "fiscal_period": ["Q1", "Q2"],
            "quarter_number": [1, 2],
            "end_date": pd.to_datetime(
                [
                    "2024-03-31",
                    "2024-06-30",
                ]
            ),
            "total_assets": [
                200 * million,
                220 * million,
            ],
            "current_assets": [
                100 * million,
                110 * million,
            ],
            "cash_and_equivalents": [
                20 * million,
                25 * million,
            ],
            "current_liabilities": [
                50 * million,
                55 * million,
            ],
            "total_debt": [
                80 * million,
                88 * million,
            ],
            "revenue": [
                100 * million,
                120 * million,
            ],
            "operating_income": [
                20 * million,
                24 * million,
            ],
            "net_income": [
                10 * million,
                12 * million,
            ],
            "operating_cash_flow": [
                15 * million,
                18 * million,
            ],
            "interest_expense": [
                5 * million,
                6 * million,
            ],
            "depreciation_and_amortization": [
                4 * million,
                5 * million,
            ],
        }
    )


def test_safe_divide() -> None:
    numerator = pd.Series(
        [10.0, 20.0, 30.0]
    )

    denominator = pd.Series(
        [2.0, 0.0, np.nan]
    )

    result = safe_divide(
        numerator,
        denominator,
    )

    assert result.iloc[0] == pytest.approx(
        5.0
    )

    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


def test_quarterly_growth() -> None:
    panel = create_example_panel()

    growth = calculate_quarterly_growth(
        panel,
        "revenue",
    )

    assert np.isnan(growth.iloc[0])

    assert growth.iloc[1] == pytest.approx(
        0.20
    )


def test_build_credit_risk_features() -> None:
    panel = create_example_panel()

    features = build_credit_risk_features(
        panel
    )

    second_row = features.iloc[1]

    assert second_row[
        "leverage"
    ] == pytest.approx(0.40)

    assert second_row[
        "current_ratio"
    ] == pytest.approx(2.0)

    assert second_row[
        "cash_ratio"
    ] == pytest.approx(
        25.0 / 55.0
    )

    assert second_row[
        "return_on_assets"
    ] == pytest.approx(
        12.0 / 220.0
    )

    assert second_row[
        "revenue_growth"
    ] == pytest.approx(0.20)

    assert second_row[
        "debt_growth"
    ] == pytest.approx(0.10)

    assert second_row[
        "interest_coverage"
    ] == pytest.approx(4.0)

    assert second_row[
        "low_interest_coverage"
    ] == pytest.approx(0.0)

    assert second_row[
        "ebitda_margin"
    ] == pytest.approx(
        29.0 / 120.0
    )


def test_missing_interest_expense_stays_missing() -> None:
    panel = create_example_panel().drop(
        columns=["interest_expense"]
    )

    features = build_credit_risk_features(
        panel
    )

    assert (
        features["interest_coverage"]
        .isna()
        .all()
    )

    assert (
        features["low_interest_coverage"]
        .isna()
        .all()
    )
