from __future__ import annotations

import argparse
from pathlib import Path

from src.survival.control_survival_dataset import (
    build_control_survival_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "controls_sec_credit_risk_features_clean.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "control_survival_dataset.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build censored start-stop intervals "
            "for non-bankrupt control firms."
        )
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
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
        "--maximum-quarters",
        type=int,
        default=32,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    survival = build_control_survival_dataset(
        feature_path=arguments.features,
        minimum_quarters=(
            arguments.minimum_quarters
        ),
        maximum_quarters=(
            arguments.maximum_quarters
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

    observations = (
        survival.groupby("firm_id")
        .size()
    )

    print("Control-firm survival dataset")
    print("-----------------------------")
    print(
        f"Companies: "
        f"{survival['firm_id'].nunique():,}"
    )
    print(
        f"Start-stop rows: "
        f"{len(survival):,}"
    )
    print(
        f"Events: "
        f"{survival['event'].sum():,}"
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
    print()
    print(f"Saved to: {arguments.output}")


if __name__ == "__main__":
    main()