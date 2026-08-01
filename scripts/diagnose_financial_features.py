from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_features.csv"
)


def print_extremes(
    data: pd.DataFrame,
    feature: str,
    lower: float,
    upper: float,
) -> None:
    extreme = data[
        (data[feature] < lower)
        | (data[feature] > upper)
    ].copy()

    print()
    print(feature)
    print("-" * len(feature))

    if extreme.empty:
        print("No extreme values.")
        return

    columns = [
        "ticker",
        "fiscal_year",
        "fiscal_period",
        "end_date",
        feature,
    ]

    print(
        extreme[columns]
        .sort_values(feature)
        .to_string(index=False)
    )


def main() -> None:
    data = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["end_date"],
    )

    print("Financial-feature diagnostics")
    print("-----------------------------")
    print(f"Rows: {len(data)}")
    print(f"Companies: {data['ticker'].nunique()}")

    print_extremes(
        data,
        "revenue_growth",
        lower=-1.0,
        upper=2.0,
    )

    print_extremes(
        data,
        "debt_growth",
        lower=-1.0,
        upper=3.0,
    )

    print_extremes(
        data,
        "operating_margin",
        lower=-1.0,
        upper=1.0,
    )

    print_extremes(
        data,
        "interest_coverage",
        lower=-50.0,
        upper=100.0,
    )

    print_extremes(
        data,
        "ebitda_margin",
        lower=-1.0,
        upper=1.0,
    )


if __name__ == "__main__":
    main()