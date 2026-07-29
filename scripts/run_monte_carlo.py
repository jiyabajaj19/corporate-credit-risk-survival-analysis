from pathlib import Path
from time import perf_counter

from src.evaluation.monte_carlo import (
    run_monte_carlo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ESTIMATES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monte_carlo_estimates.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monte_carlo_summary.csv"
)


def main() -> None:
    repetitions = 200
    sample_size = 400

    print("Corporate Credit Risk Monte Carlo Study")
    print("----------------------------------------")
    print(f"Repetitions: {repetitions}")
    print(f"Firms per repetition: {sample_size}")
    print()

    start_time = perf_counter()

    results = run_monte_carlo(
        repetitions=repetitions,
        sample_size=sample_size,
        starting_seed=1000,
        progress_interval=25,
    )

    elapsed_time = perf_counter() - start_time

    ESTIMATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.estimates.to_csv(
        ESTIMATES_PATH,
        index=False,
    )

    results.summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("Monte Carlo summary")
    print("-------------------")

    print(
        results.summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Failed repetitions: "
        f"{results.failed_repetitions}"
    )

    print(
        f"Elapsed time: {elapsed_time:.2f} seconds"
    )

    print()
    print(f"Estimates saved to: {ESTIMATES_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()