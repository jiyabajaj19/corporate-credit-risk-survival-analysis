import pandas as pd

from src.survival.control_survival_dataset import (
    build_control_survival_intervals,
    select_eligible_control_quarters,
    validate_control_survival_dataset,
)


def create_control_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["CTRL"] * 4,
            "cik": ["0000000002"] * 4,
            "entity_name": ["Control Corp"] * 4,
            "end_date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-06-30",
                    "2020-09-30",
                    "2020-12-31",
                ]
            ),
            "leverage": [0.2] * 4,
            "current_ratio": [1.5] * 4,
            "cash_ratio": [0.4] * 4,
            "return_on_assets": [0.02] * 4,
            "revenue_growth": [0.03] * 4,
            "operating_cash_flow_ratio": [0.1] * 4,
            "log_total_assets": [24.0] * 4,
            "operating_margin": [0.1] * 4,
        }
    )


def test_select_eligible_controls() -> None:
    eligible = select_eligible_control_quarters(
        create_control_data(),
        minimum_quarters=4,
    )

    assert len(eligible) == 4


def test_build_control_intervals() -> None:
    survival = build_control_survival_intervals(
        create_control_data(),
        minimum_quarters=4,
    )

    assert len(survival) == 4
    assert survival["event"].sum() == 0

    assert (
        survival["stop"]
        > survival["start"]
    ).all()


def test_validate_control_dataset() -> None:
    survival = build_control_survival_intervals(
        create_control_data(),
        minimum_quarters=4,
    )

    validate_control_survival_dataset(
        survival
    )