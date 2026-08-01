import numpy as np
import pytest

from src.evaluation.time_varying_monte_carlo import (
    collapse_to_baseline,
    run_time_varying_monte_carlo,
    summarize_time_varying_monte_carlo,
)
from src.simulation.generate_data import (
    simulate_time_varying_credit_data,
)


def test_collapse_returns_one_row_per_firm() -> None:
    long_data = (
        simulate_time_varying_credit_data(
            sample_size=80,
            baseline_hazard=0.05,
            seed=200,
        )
    )

    baseline_data = collapse_to_baseline(
        long_data
    )

    assert len(baseline_data) == 80
    assert baseline_data["firm_id"].is_unique


def test_collapsed_exit_time_matches_final_stop() -> None:
    long_data = (
        simulate_time_varying_credit_data(
            sample_size=80,
            baseline_hazard=0.05,
            seed=201,
        )
    )

    baseline_data = collapse_to_baseline(
        long_data
    )

    expected = (
        long_data.groupby("firm_id")["stop"]
        .max()
        .sort_index()
        .to_numpy()
    )

    actual = (
        baseline_data.sort_values("firm_id")[
            "exit_time"
        ].to_numpy()
    )

    assert np.allclose(actual, expected)


def test_collapsed_follow_up_is_valid() -> None:
    long_data = (
        simulate_time_varying_credit_data(
            sample_size=80,
            seed=202,
        )
    )

    baseline_data = collapse_to_baseline(
        long_data
    )

    assert (
        baseline_data["follow_up_time"] > 0
    ).all()


def test_small_monte_carlo_runs() -> None:
    results = run_time_varying_monte_carlo(
        repetitions=3,
        sample_size=150,
        baseline_hazard=0.08,
        maximum_periods=16,
        seed=203,
    )

    assert not results.estimates.empty
    assert not results.summary.empty

    assert set(
        results.estimates["model"].unique()
    ) == {
        "Baseline-at-entry Cox",
        "Left-truncated baseline Cox",
        "Time-varying Cox",
    }


def test_summary_contains_expected_metrics() -> None:
    results = run_time_varying_monte_carlo(
        repetitions=3,
        sample_size=150,
        baseline_hazard=0.08,
        maximum_periods=16,
        seed=204,
    )

    expected_columns = {
        "bias",
        "absolute_bias",
        "empirical_sd",
        "mean_standard_error",
        "rmse",
        "coverage_probability",
    }

    assert expected_columns.issubset(
        results.summary.columns
    )


def test_invalid_repetitions_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="repetitions must be positive",
    ):
        run_time_varying_monte_carlo(
            repetitions=0
        )