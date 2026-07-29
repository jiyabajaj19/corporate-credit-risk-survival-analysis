from pathlib import Path

import pandas as pd

from src.evaluation.visualization import (
    create_all_visualizations,
)


def test_create_all_visualizations(
    tmp_path: Path,
) -> None:
    estimates = pd.DataFrame(
        {
            "model": [
                "Naive Cox",
                "Naive Cox",
                "Left-Truncated Cox",
                "Left-Truncated Cox",
            ]
            * 2,
            "covariate": (
                ["leverage"] * 4
                + ["low_interest_coverage"] * 4
            ),
            "true_coefficient": (
                [0.5] * 4
                + [1.0] * 4
            ),
            "estimate": [
                0.40,
                0.42,
                0.50,
                0.52,
                0.85,
                0.90,
                1.00,
                1.04,
            ],
            "lower_95": [
                0.25,
                0.27,
                0.35,
                0.37,
                0.60,
                0.65,
                0.75,
                0.79,
            ],
            "upper_95": [
                0.55,
                0.57,
                0.65,
                0.67,
                1.10,
                1.15,
                1.25,
                1.29,
            ],
            "covered": [1] * 8,
        }
    )

    summary = pd.DataFrame(
        {
            "model": [
                "Naive Cox",
                "Left-Truncated Cox",
                "Naive Cox",
                "Left-Truncated Cox",
            ],
            "covariate": [
                "leverage",
                "leverage",
                "low_interest_coverage",
                "low_interest_coverage",
            ],
            "true_coefficient": [
                0.5,
                0.5,
                1.0,
                1.0,
            ],
            "bias": [
                -0.09,
                0.00,
                -0.13,
                0.03,
            ],
            "absolute_bias": [
                0.09,
                0.00,
                0.13,
                0.03,
            ],
            "rmse": [
                0.12,
                0.08,
                0.18,
                0.15,
            ],
            "coverage_probability": [
                0.74,
                0.95,
                0.88,
                0.94,
            ],
        }
    )

    output_paths = create_all_visualizations(
        estimates=estimates,
        summary=summary,
        output_directory=tmp_path,
    )

    assert len(output_paths) == 5

    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        