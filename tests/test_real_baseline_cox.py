import numpy as np
import pandas as pd
from datetime import timedelta

from src.models.real_baseline_cox import (
    collapse_to_baseline_data,
    fit_real_left_truncated_cox,
    fit_real_naive_cox,
)


FEATURES = [
    "leverage",
    "current_ratio",
]


def create_example_data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    rows: list[dict[str, object]] = []

    number_of_firms = 60
    number_of_quarters = 5
    event_firms = set(range(24))

    for firm_index in range(
        number_of_firms
    ):
        first_date = (
            pd.Timestamp("2012-01-01")
            + timedelta(
                days=30 * int(firm_index)
                )
            )

        baseline_leverage = rng.normal(
            0.35,
            0.12,
        )

        baseline_current_ratio = rng.normal(
            1.60,
            0.35,
        )

        for quarter in range(
            number_of_quarters
        ):
            end_date = (
                first_date
                + timedelta(
                    days=91 * int(quarter)
                )
            )

            rows.append(
                {
                    "firm_id": f"F{firm_index}",
                    "start": quarter * 0.25,
                    "stop": (
                        quarter + 1
                    ) * 0.25,
                    "event": int(
                        firm_index in event_firms
                        and quarter
                        == number_of_quarters - 1
                    ),
                    "end_date": end_date,
                    "event_date": (
                        end_date
                        if (
                            firm_index
                            in event_firms
                            and quarter
                            == number_of_quarters - 1
                        )
                        else pd.NaT
                    ),
                    "leverage": (
                        baseline_leverage
                        + 0.015 * quarter
                        + rng.normal(
                            0.0,
                            0.025,
                        )
                    ),
                    "current_ratio": (
                        baseline_current_ratio
                        - 0.020 * quarter
                        + rng.normal(
                            0.0,
                            0.080,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_collapse_to_baseline_data() -> None:
    baseline = collapse_to_baseline_data(
        data=create_example_data(),
        feature_columns=FEATURES,
    )

    assert len(baseline) == 60
    assert baseline["firm_id"].nunique() == 60
    assert baseline["event"].sum() == 24

    assert (
        baseline["duration"] > 0
    ).all()


def test_real_naive_cox_fits() -> None:
    result = fit_real_naive_cox(
        data=create_example_data(),
        feature_columns=FEATURES,
    )

    assert result.number_of_firms == 60
    assert result.number_of_events == 24
    assert result.model_name == (
        "Naive Baseline Cox"
    )

    assert np.isfinite(
        result.coefficients[
            "coefficient"
        ]
    ).all()


def test_real_left_truncated_cox_fits() -> None:
    result = fit_real_left_truncated_cox(
        data=create_example_data(),
        feature_columns=FEATURES,
    )

    assert result.number_of_firms == 60
    assert result.number_of_events == 24

    assert result.model_name == (
        "Left-Truncated Baseline Cox"
    )

    assert np.isfinite(
        result.coefficients[
            "coefficient"
        ]
    ).all()