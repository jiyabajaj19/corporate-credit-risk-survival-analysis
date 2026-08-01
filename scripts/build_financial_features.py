from __future__ import annotations

from pathlib import Path

from src.features.financial_ratios import (
    build_credit_risk_features,
    create_feature_coverage_summary,
    load_quarterly_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_quarterly_financial_panel.csv"
)

FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_features.csv"
)

COVERAGE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_credit_risk_feature_coverage.csv"
)


def main() -> None:
    print("Building SEC credit-risk features")
    print("---------------------------------")
    print(f"Input: {INPUT_PATH}")
    print()

    panel = load_quarterly_panel(
        INPUT_PATH
    )

    features = build_credit_risk_features(
        panel
    )

    coverage = create_feature_coverage_summary(
        features
    )

    FEATURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Companies: "
        f"{features['ticker'].nunique()}"
    )

    print(
        f"Company-quarter rows: "
        f"{len(features):,}"
    )

    print()
    print("Feature coverage")
    print("----------------")

    printable_coverage = coverage.copy()

    printable_coverage["coverage"] = (
        printable_coverage["coverage"].map(
            lambda value: f"{value:.1%}"
        )
    )

    print(
        printable_coverage.to_string(
            index=False
        )
    )

    print()
    print("Feature ranges")
    print("--------------")

    feature_names = coverage["feature"].tolist()

    ranges = (
        features[feature_names]
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
    print(
        f"Features saved to: "
        f"{FEATURE_OUTPUT_PATH}"
    )

    print(
        f"Coverage saved to: "
        f"{COVERAGE_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()