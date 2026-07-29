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