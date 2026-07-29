import numpy as np

from src.simulation.generate_data import (
    simulate_credit_survival_data,
)


def test_requested_sample_size() -> None:
    data = simulate_credit_survival_data(
        sample_size=200,
        seed=1,
    )

    assert len(data) == 200


def test_entry_precedes_exit() -> None:
    data = simulate_credit_survival_data(
        sample_size=200,
        seed=2,
    )

    assert np.all(
        data["entry_time"] <= data["exit_time"]
    )


def test_event_is_binary() -> None:
    data = simulate_credit_survival_data(
        sample_size=200,
        seed=3,
    )

    assert set(data["event"].unique()).issubset(
        {0, 1}
    )


def test_firms_survive_until_entry() -> None:
    data = simulate_credit_survival_data(
        sample_size=200,
        seed=4,
    )

    assert np.all(
        data["true_default_time"]
        > data["entry_time"]
    )


def test_follow_up_is_nonnegative() -> None:
    data = simulate_credit_survival_data(
        sample_size=200,
        seed=5,
    )

    assert np.all(
        data["follow_up_time"] >= 0
    )

def test_same_seed_produces_same_data() -> None:
    first = simulate_credit_survival_data(
        sample_size=100,
        seed=25,
    )

    second = simulate_credit_survival_data(
        sample_size=100,
        seed=25,
    )

    assert first.equals(second)


def test_different_seeds_produce_different_data() -> None:
    first = simulate_credit_survival_data(
        sample_size=100,
        seed=25,
    )

    second = simulate_credit_survival_data(
        sample_size=100,
        seed=26,
    )

    assert not first.equals(second)


def test_larger_entry_scale_increases_mean_entry() -> None:
    early_entry = simulate_credit_survival_data(
        sample_size=1000,
        entry_scale=0.25,
        seed=40,
    )

    late_entry = simulate_credit_survival_data(
        sample_size=1000,
        entry_scale=1.75,
        seed=40,
    )

    assert (
        late_entry["entry_time"].mean()
        > early_entry["entry_time"].mean()
    )


def test_weibull_parameters_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError):
        simulate_credit_survival_data(
            baseline_scale=0.0
        )

    with pytest.raises(ValueError):
        simulate_credit_survival_data(
            weibull_shape=0.0
        )