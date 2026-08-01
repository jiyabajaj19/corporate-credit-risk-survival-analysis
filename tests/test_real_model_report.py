from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.evaluation.real_model_report import (
    build_feature_scaling_table,
    build_model_comparison_table,
    combine_coefficient_tables,
)


def create_example_result(
    model_name: str,
    penalizer: float,
):
    return SimpleNamespace(
        model_name=model_name,
        number_of_firms=20,
        number_of_rows=80,
        number_of_events=8,
        feature_columns=[
            "leverage",
            "current_ratio",
        ],
        penalizer=penalizer,
        l1_ratio=0.0,
        log_likelihood=-25.0,
        coefficients=pd.DataFrame(
            {
                "model": [
                    model_name,
                    model_name,
                ],
                "feature": [
                    "leverage",
                    "current_ratio",
                ],
                "coefficient": [
                    0.4,
                    -0.2,
                ],
                "coefficient_lower_95": [
                    0.1,
                    -0.5,
                ],
                "coefficient_upper_95": [
                    0.7,
                    0.1,
                ],
                "hazard_ratio": [
                    1.49,
                    0.82,
                ],
                "hazard_ratio_lower_95": [
                    1.11,
                    0.61,
                ],
                "hazard_ratio_upper_95": [
                    2.01,
                    1.11,
                ],
                "standard_error": [
                    0.15,
                    0.15,
                ],
                "z_statistic": [
                    2.67,
                    -1.33,
                ],
                "p_value": [
                    0.008,
                    0.18,
                ],
            }
        ),
        feature_means=pd.Series(
            {
                "leverage": 0.35,
                "current_ratio": 1.50,
            }
        ),
        feature_standard_deviations=pd.Series(
            {
                "leverage": 0.10,
                "current_ratio": 0.30,
            }
        ),
    )


def test_build_model_comparison_table() -> None:
    results = [
        create_example_result(
            "Time-Varying Cox",
            0.0,
        ),
        create_example_result(
            "Ridge Time-Varying Cox",
            0.1,
        ),
    ]

    comparison = build_model_comparison_table(
        results
    )

    assert len(comparison) == 2

    assert set(
        comparison["model"]
    ) == {
        "Time-Varying Cox",
        "Ridge Time-Varying Cox",
    }


def test_combine_coefficient_tables() -> None:
    results = [
        create_example_result(
            "Time-Varying Cox",
            0.0,
        ),
        create_example_result(
            "Ridge Time-Varying Cox",
            0.1,
        ),
    ]

    coefficients = (
        combine_coefficient_tables(
            results
        )
    )

    assert len(coefficients) == 4

    assert set(
        coefficients["feature"]
    ) == {
        "leverage",
        "current_ratio",
    }


def test_build_feature_scaling_table() -> None:
    results = [
        create_example_result(
            "Time-Varying Cox",
            0.0,
        )
    ]

    scaling = build_feature_scaling_table(
        results
    )

    assert len(scaling) == 2

    assert set(
        scaling["feature"]
    ) == {
        "leverage",
        "current_ratio",
    }