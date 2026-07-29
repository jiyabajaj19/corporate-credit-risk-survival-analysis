from pathlib import Path

import pandas as pd

from src.models.baseline_cox import (
    fit_left_truncated_cox,
    fit_naive_cox,
)
from src.simulation.generate_data import (
    simulate_credit_survival_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "baseline_credit_survival.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline_model_results.csv"
)

TRUE_COEFFICIENTS = {
    "leverage": 0.5,
    "low_interest_coverage": 1.0,
}


def load_or_generate_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Loading data from: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("Dataset not found. Generating a new dataset.")

    data = simulate_credit_survival_data(
        sample_size=400,
        beta_leverage=0.5,
        beta_low_coverage=1.0,
        entry_scale=0.5,
        censoring_upper_bound=2.0,
        seed=42,
    )

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DATA_PATH,
        index=False,
    )

    return data


def build_comparison_table(
    naive_results,
    truncated_results,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []

    for covariate, true_coefficient in (
        TRUE_COEFFICIENTS.items()
    ):
        naive_estimate = float(
            naive_results.coefficients.loc[
                covariate,
                "coef",
            ]
        )

        truncated_estimate = float(
            truncated_results.coefficients.loc[
                covariate,
                "coef",
            ]
        )

        records.append(
            {
                "covariate": covariate,
                "true_coefficient": true_coefficient,
                "naive_estimate": naive_estimate,
                "naive_bias": (
                    naive_estimate
                    - true_coefficient
                ),
                "left_truncated_estimate": (
                    truncated_estimate
                ),
                "left_truncated_bias": (
                    truncated_estimate
                    - true_coefficient
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    data = load_or_generate_data()

    print()
    print("Dataset summary")
    print("----------------")
    print(f"Number of firms: {len(data)}")
    print(f"Observed defaults: {int(data['event'].sum())}")
    print(
        "Censoring rate: "
        f"{1.0 - data['event'].mean():.2%}"
    )
    print(
        "Mean entry time: "
        f"{data['entry_time'].mean():.4f}"
    )

    naive_results = fit_naive_cox(data)
    truncated_results = fit_left_truncated_cox(
        data
    )

    comparison = build_comparison_table(
        naive_results,
        truncated_results,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print("Coefficient comparison")
    print("----------------------")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Concordance indices")
    print("-------------------")
    print(
        f"Naive Cox: "
        f"{naive_results.concordance_index:.4f}"
    )
    print(
        f"Left-Truncated Cox: "
        f"{truncated_results.concordance_index:.4f}"
    )

    print()
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()