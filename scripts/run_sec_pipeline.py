from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.sec_pipeline import (
    run_sec_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete SEC credit-risk "
            "data-processing pipeline."
        )
    )

    parser.add_argument(
        "--input-directory",
        type=Path,
        required=True,
        help=(
            "Directory containing SEC Company "
            "Facts JSON files."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "Prefix for generated files, such as "
            "'pilot', 'bankrupt', or 'controls'."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--winsorize-lower",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--winsorize-upper",
        type=float,
        default=0.99,
    )

    return parser.parse_args()


def print_coverage(
    coverage: pd.DataFrame,
) -> None:
    printable = coverage.copy()

    printable["coverage"] = (
        printable["coverage"].map(
            lambda value: f"{value:.1%}"
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )


def main() -> None:
    arguments = parse_arguments()

    print("SEC corporate-risk pipeline")
    print("---------------------------")
    print(
        f"Input: "
        f"{arguments.input_directory}"
    )

    print(
        f"Output prefix: "
        f"{arguments.output_prefix}"
    )

    print()

    print(
        "[1/4] Extracting SEC financial facts..."
    )

    results = run_sec_pipeline(
        input_directory=(
            arguments.input_directory
        ),
        output_directory=(
            arguments.output_directory
        ),
        output_prefix=(
            arguments.output_prefix
        ),
        winsorize_lower=(
            arguments.winsorize_lower
        ),
        winsorize_upper=(
            arguments.winsorize_upper
        ),
        save_outputs=True,
    )

    print(
        "[2/4] Quarterly accounting "
        "panel completed."
    )

    print(
        "[3/4] Credit-risk features "
        "completed."
    )

    print(
        "[4/4] Model-ready cleaning "
        "completed."
    )

    print()
    print("Pipeline summary")
    print("----------------")

    print(
        f"Companies: "
        f"{results.company_count:,}"
    )

    print(
        f"Financial fact observations: "
        f"{results.fact_count:,}"
    )

    print(
        f"Company-quarter rows: "
        f"{results.quarter_count:,}"
    )

    print()
    print("Clean feature coverage")
    print("----------------------")

    print_coverage(
        results.clean_feature_coverage
    )

    print()
    print("Generated files")
    print("---------------")

    for path in [
        results.paths.facts,
        results.paths.fact_coverage,
        results.paths.quarterly_panel,
        results.paths.features,
        results.paths.feature_coverage,
        results.paths.clean_features,
        results.paths.clean_feature_coverage,
    ]:
        print(path)


if __name__ == "__main__":
    main()