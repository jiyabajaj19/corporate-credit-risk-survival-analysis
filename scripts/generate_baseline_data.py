from pathlib import Path

from src.simulation.generate_data import (
    simulate_credit_survival_data,
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
        entry_scale=0.5,
        censoring_upper_bound=2.0,
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

    print("Dataset created successfully.")
    print(f"Rows: {len(data)}")
    print(f"Defaults: {int(data['event'].sum())}")

    censoring_rate = (
        1.0 - data["event"].mean()
    )

    print(
        f"Censoring rate: {censoring_rate:.2%}"
    )

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()