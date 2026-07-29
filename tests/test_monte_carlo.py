import numpy as np
import pandas as pd
import pytest

from src.evaluation.monte_carlo import (
    run_monte_carlo,
    summarize_monte_carlo_results,
)


def test_monte_carlo_returns_expected_models() -> None:
    results = run_monte_carlo(
        repetitions=3,
        sample_size=200,
        starting_seed=500,
        progress_interval=10,
    )

    assert set(results.estimates["model"]) == {
        "Naive Cox",
        "Left-Truncated Cox",
    }


def test_monte_carlo_returns_expected_covariates() -> None:
    results = run_monte_carlo(
        repetitions=3,
        sample_size=200,
        starting_seed=600,
        progress_interval=10,
    )

    assert set(results.estimates["covariate"]) == {
        "leverage",
        "low_interest_coverage",
    }


def test_summary_metrics_are_finite() -> None:
    results = run_monte_carlo(
        repetitions=4,
        sample_size=200,
        starting_seed=700,
        progress_interval=10,
    )

    numeric_columns = [
        "mean_estimate",
        "bias",
        "absolute_bias",
        "empirical_sd",
        "mean_standard_error",
        "rmse",
        "coverage_probability",
    ]

    assert np.all(
        np.isfinite(
            results.summary[numeric_columns]
        )
    )


def test_coverage_is_between_zero_and_one() -> None:
    results = run_monte_carlo(
        repetitions=4,
        sample_size=200,
        starting_seed=800,
        progress_interval=10,
    )

    coverage = results.summary[
        "coverage_probability"
    ]

    assert np.all(
        (coverage >= 0.0)
        & (coverage <= 1.0)
    )


def test_summary_calculation() -> None:
    estimates = pd.DataFrame(
        {
            "model": [
                "Test Model",
                "Test Model",
            ],
            "covariate": [
                "leverage",
                "leverage",
            ],
            "true_coefficient": [
                0.5,
                0.5,
            ],
            "estimate": [
                0.4,
                0.6,
            ],
            "standard_error": [
                0.1,
                0.1,
            ],
            "covered": [
                1,
                1,
            ],
        }
    )

    summary = summarize_monte_carlo_results(
        estimates
    )

    assert summary.loc[
        0,
        "mean_estimate",
    ] == pytest.approx(0.5)

    assert summary.loc[
        0,
        "bias",
    ] == pytest.approx(0.0)

    assert summary.loc[
        0,
        "rmse",
    ] == pytest.approx(0.1)

    assert summary.loc[
        0,
        "coverage_probability",
    ] == pytest.approx(1.0)


def test_invalid_repetitions_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="repetitions must be positive",
    ):
        run_monte_carlo(repetitions=0)