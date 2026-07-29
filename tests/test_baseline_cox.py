import numpy as np

from src.models.baseline_cox import (
    fit_left_truncated_cox,
    fit_naive_cox,
)
from src.simulation.generate_data import (
    simulate_credit_survival_data,
)


def create_test_data():
    return simulate_credit_survival_data(
        sample_size=300,
        seed=10,
    )


def test_naive_cox_returns_both_covariates() -> None:
    data = create_test_data()

    results = fit_naive_cox(data)

    assert set(results.coefficients.index) == {
        "leverage",
        "low_interest_coverage",
    }


def test_left_truncated_cox_returns_both_covariates() -> None:
    data = create_test_data()

    results = fit_left_truncated_cox(data)

    assert set(results.coefficients.index) == {
        "leverage",
        "low_interest_coverage",
    }


def test_naive_coefficients_are_finite() -> None:
    data = create_test_data()

    results = fit_naive_cox(data)

    assert np.all(
        np.isfinite(
            results.coefficients["coef"]
        )
    )


def test_left_truncated_coefficients_are_finite() -> None:
    data = create_test_data()

    results = fit_left_truncated_cox(data)

    assert np.all(
        np.isfinite(
            results.coefficients["coef"]
        )
    )


def test_concordance_indices_are_valid() -> None:
    data = create_test_data()

    naive = fit_naive_cox(data)
    truncated = fit_left_truncated_cox(data)

    assert 0.0 <= naive.concordance_index <= 1.0
    assert (
        0.0
        <= truncated.concordance_index
        <= 1.0
    )