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

DEFAULT_SELECTED_PENALTY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_models"
    / "model_selection"
    / "selected_penalty.txt"
)

FALLBACK_RIDGE_PENALIZER = 0.25


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
        help=(
            "Path to the combined real survival dataset."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory where comparison reports are saved."
        ),
    )

    parser.add_argument(
        "--selected-penalty-path",
        type=Path,
        default=DEFAULT_SELECTED_PENALTY_PATH,
        help=(
            "Path to the cross-validated selected ridge "
            "penalty."
        ),
    )

    parser.add_argument(
        "--ridge-penalizer",
        type=float,
        default=None,
        help=(
            "Optional manual ridge penalty. When supplied, "
            "this overrides the value stored in "
            "selected_penalty.txt."
        ),
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help=(
            "Disable predictor standardization before fitting."
        ),
    )

    return parser.parse_args()


def load_selected_penalizer(
    selected_penalty_path: Path,
    manual_penalizer: float | None = None,
    fallback_penalizer: float = (
        FALLBACK_RIDGE_PENALIZER
    ),
) -> tuple[float, str]:
    """
    Resolve the ridge penalty used in the final model comparison.

    Priority
    --------
    1. Manual command-line override.
    2. Cross-validated value from selected_penalty.txt.
    3. Configured fallback value.
    """
    if manual_penalizer is not None:
        if manual_penalizer <= 0:
            raise ValueError(
                "--ridge-penalizer must be positive."
            )

        return (
            float(manual_penalizer),
            "command-line override",
        )

    if selected_penalty_path.exists():
        raw_value = (
            selected_penalty_path.read_text(
                encoding="utf-8"
            )
            .strip()
        )

        try:
            selected_penalizer = float(
                raw_value
            )
        except ValueError as error:
            raise ValueError(
                "Selected penalty file does not contain "
                "a valid number: "
                f"{selected_penalty_path}"
            ) from error

        if selected_penalizer <= 0:
            raise ValueError(
                "Selected ridge penalizer must be positive. "
                f"Found: {selected_penalizer}"
            )

        return (
            selected_penalizer,
            "cross-validated selection",
        )

    if fallback_penalizer <= 0:
        raise ValueError(
            "Fallback ridge penalizer must be positive."
        )

    return (
        float(fallback_penalizer),
        "fallback value",
    )


def main() -> None:
    arguments = parse_arguments()

    standardize = not arguments.no_standardize

    (
        ridge_penalizer,
        penalty_source,
    ) = load_selected_penalizer(
        selected_penalty_path=(
            arguments.selected_penalty_path
        ),
        manual_penalizer=(
            arguments.ridge_penalizer
        ),
    )

    data = load_survival_dataset(
        arguments.input
    )

    print(
        "Real corporate credit-risk model comparison"
    )
    print(
        "-------------------------------------------"
    )
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
    print(
        f"Ridge penalizer: "
        f"{ridge_penalizer:.4f}"
    )
    print(
        f"Penalty source: "
        f"{penalty_source}"
    )
    print(
        f"Standardized predictors: "
        f"{standardize}"
    )
    print()

    print(
        "[1/4] Fitting naive baseline Cox..."
    )

    naive = fit_real_naive_cox(
        data=data,
        standardize=standardize,
    )

    print(
        "[2/4] Fitting left-truncated "
        "baseline Cox..."
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
        "[4/4] Fitting tuned ridge "
        "time-varying Cox..."
    )

    ridge = fit_ridge_real_time_varying_cox(
        data=data,
        penalizer=ridge_penalizer,
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
            else (
                f"{result.log_likelihood:.4f}"
            )
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