from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.real_model_report import (
    save_real_model_report,
)
from src.models.base_cox import (
    load_survival_dataset,
)
from src.models.real_baseline_cox import (
    fit_real_left_truncated_cox,
    fit_real_naive_cox,
)
from src.models.real_time_varying_cox import (
    fit_ridge_real_time_varying_cox,
    fit_unpenalized_real_time_varying_cox,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_survival_dataset.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "real_models"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and compare real-data Cox survival models."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--ridge-penalizer",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help=(
            "Disable predictor standardization before fitting."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    standardize = not arguments.no_standardize

    data = load_survival_dataset(
        arguments.input
    )

    print("Real corporate credit-risk model comparison")
    print("-------------------------------------------")
    print(f"Input: {arguments.input}")
    print(
        f"Firms: "
        f"{data['firm_id'].nunique():,}"
    )
    print(
        f"Start-stop rows: "
        f"{len(data):,}"
    )
    print(
        f"Bankruptcy events: "
        f"{int(data['event'].sum()):,}"
    )
    print()

    print("[1/4] Fitting naive baseline Cox...")

    naive = fit_real_naive_cox(
        data=data,
        standardize=standardize,
    )

    print(
        "[2/4] Fitting left-truncated baseline Cox..."
    )

    left_truncated = (
        fit_real_left_truncated_cox(
            data=data,
            standardize=standardize,
        )
    )

    print(
        "[3/4] Fitting unpenalized "
        "time-varying Cox..."
    )

    time_varying = (
        fit_unpenalized_real_time_varying_cox(
            data=data,
            standardize=standardize,
        )
    )

    print(
        "[4/4] Fitting ridge time-varying Cox..."
    )

    ridge = fit_ridge_real_time_varying_cox(
        data=data,
        penalizer=arguments.ridge_penalizer,
        standardize=standardize,
    )

    results = [
        naive,
        left_truncated,
        time_varying,
        ridge,
    ]

    paths = save_real_model_report(
        results=results,
        output_directory=(
            arguments.output_directory
        ),
    )

    print()
    print("Model comparison")
    print("----------------")

    for result in results:
        log_likelihood = (
            "N/A"
            if result.log_likelihood is None
            else f"{result.log_likelihood:.4f}"
        )

        print(
            f"{result.model_name}: "
            f"firms={result.number_of_firms}, "
            f"events={result.number_of_events}, "
            f"features={len(result.feature_columns)}, "
            f"penalizer={result.penalizer:.4f}, "
            f"log-likelihood={log_likelihood}"
        )

    print()
    print("Generated files")
    print("---------------")

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()