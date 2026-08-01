from pathlib import Path

from src.models.time_varying_cox import (
    fit_time_varying_cox,
)
from src.simulation.generate_data import (
    simulate_time_varying_credit_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "time_varying_credit_survival.csv"
)

RESULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "time_varying_cox_results.csv"
)


TRUE_COEFFICIENTS = {
    "leverage": 0.60,
    "low_interest_coverage": 0.70,
    "current_ratio": -0.25,
    "cash_ratio": -0.15,
    "return_on_assets": -0.40,
    "debt_growth": 0.30,
    "revenue_growth": -0.20,
    "operating_cash_flow_ratio": -0.35,
    "log_total_assets": -0.15,
    "ebitda_margin": -0.30,
}


def main() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=800,
        beta_leverage=(
            TRUE_COEFFICIENTS["leverage"]
        ),
        beta_low_coverage=(
            TRUE_COEFFICIENTS[
                "low_interest_coverage"
            ]
        ),
        beta_current_ratio=(
            TRUE_COEFFICIENTS["current_ratio"]
        ),
        beta_cash_ratio=(
            TRUE_COEFFICIENTS["cash_ratio"]
        ),
        beta_return_on_assets=(
            TRUE_COEFFICIENTS[
                "return_on_assets"
            ]
        ),
        beta_debt_growth=(
            TRUE_COEFFICIENTS["debt_growth"]
        ),
        beta_revenue_growth=(
            TRUE_COEFFICIENTS["revenue_growth"]
        ),
        beta_operating_cash_flow_ratio=(
            TRUE_COEFFICIENTS[
                "operating_cash_flow_ratio"
            ]
        ),
        beta_log_total_assets=(
            TRUE_COEFFICIENTS[
                "log_total_assets"
            ]
        ),
        beta_ebitda_margin=(
            TRUE_COEFFICIENTS[
                "ebitda_margin"
            ]
        ),
        baseline_hazard=0.008,
        maximum_periods=20,
        observation_interval=0.25,
        maximum_entry_period=8,
        seed=42,
    )

    results = fit_time_varying_cox(
        data,
        penalizer=0.01,
    )

    DATA_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DATA_OUTPUT_PATH,
        index=False,
    )

    results.coefficients.to_csv(
        RESULT_OUTPUT_PATH,
        index=True,
    )

    print(
        "Time-varying corporate credit model"
    )

    print(
        "-----------------------------------"
    )

    print(
        f"Companies: "
        f"{results.number_of_firms}"
    )

    print(
        f"Company-quarter rows: "
        f"{results.number_of_rows}"
    )

    print(
        f"Observed defaults: "
        f"{results.number_of_events}"
    )

    default_rate = (
        results.number_of_events
        / results.number_of_firms
    )

    print(
        f"Observed default rate: "
        f"{default_rate:.2%}"
    )

    print("\nTrue coefficients")
    print("-----------------")

    for variable, coefficient in (
        TRUE_COEFFICIENTS.items()
    ):
        print(
            f"{variable:<30} "
            f"{coefficient:>8.4f}"
        )

    comparison = (
        results.coefficients[
            [
                "coef",
                "exp(coef)",
                "se(coef)",
                "p",
            ]
        ]
        .copy()
        .rename(
            columns={
                "coef": "estimated_coefficient",
                "exp(coef)": "hazard_ratio",
                "se(coef)": "standard_error",
                "p": "p_value",
            }
        )
    )

    comparison.insert(
        0,
        "true_coefficient",
        comparison.index.map(
            TRUE_COEFFICIENTS
        ),
    )

    comparison["estimation_error"] = (
        comparison["estimated_coefficient"]
        - comparison["true_coefficient"]
    )

    comparison["absolute_error"] = (
        comparison["estimation_error"].abs()
    )

    print("\nCoefficient comparison")
    print("----------------------")

    print(
        comparison.to_string(
            float_format=lambda value: (
                f"{value:.4f}"
            )
        )
    )

    comparison.to_csv(
        RESULT_OUTPUT_PATH,
        index=True,
        index_label="covariate",
    )

    print(
        f"\nDataset saved to: "
        f"{DATA_OUTPUT_PATH}"
    )

    print(
        f"Results saved to: "
        f"{RESULT_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()