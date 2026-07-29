from pathlib import Path

from src.simulation.generate_data import (
    simulate_credit_survival_data,
    summarize_simulated_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "baseline_credit_survival.csv"
)


def main() -> None:
    data = simulate_credit_survival_data(
        sample_size=400,
        beta_leverage=0.5,
        beta_low_coverage=1.0,
        entry_scale=1.25,
        censoring_upper_bound=3.0,
        baseline_scale=2.5,
        weibull_shape=1.6,
        seed=42,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = summarize_simulated_data(data)

    print("Dataset created successfully.")
    print(f"Rows: {int(summary['number_of_firms'])}")
    print(
        "Defaults: "
        f"{int(summary['number_of_defaults'])}"
    )
    print(
        "Default rate: "
        f"{summary['default_rate']:.2%}"
    )
    print(
        "Censoring rate: "
        f"{summary['censoring_rate']:.2%}"
    )
    print(
        "Mean entry time: "
        f"{summary['mean_entry_time']:.4f}"
    )
    print(
        "Median entry time: "
        f"{summary['median_entry_time']:.4f}"
    )
    print(
        "Mean follow-up time: "
        f"{summary['mean_follow_up_time']:.4f}"
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()