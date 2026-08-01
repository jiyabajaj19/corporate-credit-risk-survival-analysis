from pathlib import Path

from src.evaluation.time_varying_monte_carlo import (
    run_time_varying_monte_carlo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ESTIMATES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "time_varying_monte_carlo_estimates.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "time_varying_monte_carlo_summary.csv"
)

FAILURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "time_varying_monte_carlo_failures.csv"
)


def main() -> None:
    results = run_time_varying_monte_carlo(
        repetitions=200,
        sample_size=800,
        beta_leverage=0.80,
        beta_low_coverage=0.90,
        beta_current_ratio=-0.35,
        beta_cash_ratio=-0.25,
        beta_return_on_assets=-0.70,
        beta_debt_growth=0.40,
        beta_revenue_growth=-0.20,
        beta_operating_cash_flow_ratio=-0.45,
        beta_log_total_assets=-0.10,
        beta_ebitda_margin=-0.30,
        baseline_hazard=0.008,
        maximum_periods=20,
        observation_interval=0.25,
        maximum_entry_period=8,
        penalizer=0.0,
        seed=2026,
    )

    SUMMARY_PATH.parent.mkdir(
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

    if not results.failures.empty:
        results.failures.to_csv(
            FAILURES_PATH,
            index=False,
        )

    print("\nMonte Carlo summary")
    print("-------------------")

    print(
        results.summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nSuccessful complete repetitions: "
        f"{results.estimates['repetition'].nunique()}"
    )

    print(
        "Failed repetitions: "
        f"{len(results.failures)}"
    )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(
        f"Detailed estimates saved to: "
        f"{ESTIMATES_PATH}"
    )

    if not results.failures.empty:
        print(
            f"Failures saved to: "
            f"{FAILURES_PATH}"
        )


if __name__ == "__main__":
    main()