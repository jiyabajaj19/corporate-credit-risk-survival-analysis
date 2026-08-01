from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.real_model_selection import (
    DEFAULT_PENALIZER_GRID,
    evaluate_penalizer_grid,
)
from src.models.base_cox import (
    load_survival_dataset,
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
    / "model_selection"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune ridge regularization for the real "
            "time-varying Cox model."
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
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    data = load_survival_dataset(
        arguments.input
    )

    print("Real time-varying Cox model selection")
    print("-------------------------------------")
    print(
        f"Firms: "
        f"{data['firm_id'].nunique():,}"
    )
    print(
        f"Rows: {len(data):,}"
    )
    print(
        f"Events: "
        f"{int(data['event'].sum()):,}"
    )
    print(
        f"Grouped folds: "
        f"{arguments.folds}"
    )
    print(
        "Penalizer grid: "
        f"{DEFAULT_PENALIZER_GRID}"
    )
    print()

    selection = evaluate_penalizer_grid(
        data=data,
        penalizer_grid=(
            DEFAULT_PENALIZER_GRID
        ),
        number_of_folds=(
            arguments.folds
        ),
        random_seed=arguments.seed,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_path = (
        arguments.output_directory
        / "fold_results.csv"
    )

    summary_path = (
        arguments.output_directory
        / "penalty_summary.csv"
    )

    selected_path = (
        arguments.output_directory
        / "selected_penalty.txt"
    )

    selection.fold_results.to_csv(
        fold_path,
        index=False,
    )

    selection.penalty_summary.to_csv(
        summary_path,
        index=False,
    )

    selected_path.write_text(
        (
            f"{selection.selected_penalizer}\n"
        ),
        encoding="utf-8",
    )

    print("Penalty comparison")
    print("------------------")

    print(
        selection.penalty_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Selected penalizer: "
        f"{selection.selected_penalizer:.4f}"
    )

    if selection.selected_penalizer == 0:
        final_model = (
            fit_unpenalized_real_time_varying_cox(
                data=data
            )
        )
    else:
        final_model = (
            fit_ridge_real_time_varying_cox(
                data=data,
                penalizer=(
                    selection.selected_penalizer
                ),
            )
        )

    final_coefficients_path = (
        arguments.output_directory
        / "selected_model_coefficients.csv"
    )

    final_model.coefficients.to_csv(
        final_coefficients_path,
        index=False,
    )

    print()
    print("Generated files")
    print("---------------")
    print(fold_path)
    print(summary_path)
    print(selected_path)
    print(final_coefficients_path)


if __name__ == "__main__":
    main()