from __future__ import annotations

import argparse
from pathlib import Path

from src.survival.survival_dataset import (
    build_bankrupt_survival_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bankrupt_sec_credit_risk_features_clean.csv"
)

DEFAULT_EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lopucki_bankruptcy_events.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bankrupt_survival_dataset.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pre-bankruptcy start-stop "
            "survival intervals."
        )
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENT_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--minimum-quarters",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--maximum-lookback-years",
        type=float,
        default=8.0,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    survival = build_bankrupt_survival_dataset(
        feature_path=arguments.features,
        event_path=arguments.events,
        minimum_quarters=(
            arguments.minimum_quarters
        ),
        maximum_lookback_years=(
            arguments.maximum_lookback_years
        ),
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    survival.to_csv(
        arguments.output,
        index=False,
    )

    print("Bankrupt-firm survival dataset")
    print("------------------------------")
    print(
        f"Companies: "
        f"{survival['firm_id'].nunique():,}"
    )
    print(
        f"Start-stop rows: "
        f"{len(survival):,}"
    )
    print(
        f"Bankruptcy events: "
        f"{survival['event'].sum():,}"
    )

    observations = (
        survival.groupby("firm_id")
        .size()
    )

    print(
        f"Median quarters per company: "
        f"{observations.median():.1f}"
    )
    print(
        f"Minimum quarters per company: "
        f"{observations.min():,}"
    )
    print(
        f"Maximum quarters per company: "
        f"{observations.max():,}"
    )

    print(
        f"First feature date: "
        f"{survival['end_date'].min().date()}"
    )
    print(
        f"Last bankruptcy date: "
        f"{survival['event_date'].max().date()}"
    )

    print()
    print(f"Saved to: {arguments.output}")


if __name__ == "__main__":
    main()