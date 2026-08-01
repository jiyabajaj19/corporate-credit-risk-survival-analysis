from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.financial_ratios import (
    clean_credit_risk_features,
    create_feature_coverage_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_features_clean.csv"
)

COVERAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_features_clean_coverage.csv"
)


def main() -> None:
    features = pd.read_csv(
        INPUT_PATH,
        parse_dates=["end_date"],
    )

    cleaned = clean_credit_risk_features(
        features,
        winsorize_lower=0.01,
        winsorize_upper=0.99,
    )

    coverage = create_feature_coverage_summary(
        cleaned[
            [
                column
                for column in cleaned.columns
                if not column.endswith("_raw")
            ]
        ]
    )

    cleaned.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    model_features = [
        "leverage",
        "current_ratio",
        "cash_ratio",
        "return_on_assets",
        "debt_growth",
        "revenue_growth",
        "operating_cash_flow_ratio",
        "log_total_assets",
        "interest_coverage",
        "operating_margin",
        "ebitda_margin",
    ]

    print("Cleaned financial-feature dataset")
    print("---------------------------------")
    print(f"Rows: {len(cleaned)}")
    print(
        f"Companies: "
        f"{cleaned['ticker'].nunique()}"
    )

    print()
    print("Cleaned feature ranges")
    print("----------------------")

    ranges = (
        cleaned[model_features]
        .agg(["min", "median", "max"])
        .transpose()
    )

    print(
        ranges.to_string(
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()