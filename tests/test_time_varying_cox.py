import numpy as np
import pytest

from src.models.time_varying_cox import (
    DEFAULT_COVARIATES,
    fit_time_varying_cox,
)
from src.simulation.generate_data import (
    simulate_time_varying_credit_data,
)


EXPECTED_COVARIATES = {
    "leverage",
    "low_interest_coverage",
    "current_ratio",
    "cash_ratio",
    "return_on_assets",
    "debt_growth",
    "revenue_growth",
    "operating_cash_flow_ratio",
    "log_total_assets",
    "ebitda_margin",
}


def test_time_varying_data_has_requested_firms() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=100,
    )

    assert data["firm_id"].nunique() == 100


def test_time_intervals_are_valid() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=101,
    )

    assert (
        data["stop"] > data["start"]
    ).all()


def test_each_firm_has_at_most_one_event() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=102,
    )

    events_per_firm = (
        data.groupby("firm_id")["event"].sum()
    )

    assert (events_per_firm <= 1).all()


def test_event_is_on_final_firm_row() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=103,
    )

    for _, firm_data in data.groupby("firm_id"):
        event_rows = firm_data[
            firm_data["event"] == 1
        ]

        if not event_rows.empty:
            assert (
                event_rows.index[-1]
                == firm_data.index[-1]
            )


def test_entry_time_matches_first_interval() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=104,
    )

    first_rows = (
        data.sort_values(["firm_id", "start"])
        .groupby("firm_id")
        .first()
    )

    assert np.allclose(
        first_rows["start"],
        first_rows["entry_time"],
    )


def test_expected_financial_columns_exist() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=105,
    )

    assert EXPECTED_COVARIATES.issubset(
        data.columns
    )


def test_default_covariates_match_expected_set() -> None:
    assert set(DEFAULT_COVARIATES) == (
        EXPECTED_COVARIATES
    )


def test_financial_variables_are_finite() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        baseline_hazard=0.05,
        seed=106,
    )

    continuous_columns = [
        "leverage",
        "current_ratio",
        "cash_ratio",
        "return_on_assets",
        "debt_growth",
        "revenue_growth",
        "operating_cash_flow_ratio",
        "log_total_assets",
        "ebitda_margin",
    ]

    assert np.isfinite(
        data[continuous_columns].to_numpy()
    ).all()


def test_binary_interest_coverage_indicator() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=107,
    )

    assert set(
        data["low_interest_coverage"].unique()
    ).issubset({0, 1})


def test_standardized_ratio_variables_are_valid() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=108,
    )

    ratio_columns = [
        "current_ratio",
        "cash_ratio",
    ]

    assert np.isfinite(
        data[ratio_columns].to_numpy()
    ).all()

    for column in ratio_columns:
        assert data[column].nunique() > 1

        assert (
            data[column].std(ddof=0) > 0
        )


def test_standardized_log_total_assets_are_valid() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=109,
    )

    log_assets = data["log_total_assets"]

    assert np.isfinite(log_assets).all()

    assert log_assets.nunique() > 1

    assert log_assets.std(ddof=0) > 0

def test_growth_variables_are_not_constant() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=110,
    )

    assert data["debt_growth"].nunique() > 1
    assert data["revenue_growth"].nunique() > 1


def test_covariates_change_within_firms() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=100,
        seed=111,
    )

    leverage_counts = (
        data.groupby("firm_id")["leverage"]
        .nunique()
    )

    multi_period_firms = (
        data.groupby("firm_id")
        .size()
        .loc[lambda values: values > 1]
        .index
    )

    assert (
        leverage_counts.loc[multi_period_firms] > 1
    ).any()


def test_time_varying_model_fits() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=800,
        baseline_hazard=0.05,
        seed=112,
    )

    results = fit_time_varying_cox(
        data,
        penalizer=0.01,
    )

    assert set(
        results.coefficients.index
    ) == EXPECTED_COVARIATES

    assert np.isfinite(
        results.coefficients["coef"]
    ).all()

    assert np.isfinite(
        results.coefficients["se(coef)"]
    ).all()

    assert (
        results.coefficients["exp(coef)"] > 0
    ).all()


def test_model_result_counts_match_data() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=500,
        baseline_hazard=0.05,
        seed=113,
    )

    results = fit_time_varying_cox(
        data,
        penalizer=0.01,
    )

    assert results.number_of_firms == (
        data["firm_id"].nunique()
    )

    assert results.number_of_rows == len(data)

    assert results.number_of_events == int(
        data["event"].sum()
    )


def test_missing_columns_raise_error() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=50,
        seed=114,
    )

    data = data.drop(columns=["leverage"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        fit_time_varying_cox(data)


def test_invalid_intervals_raise_error() -> None:
    data = simulate_time_varying_credit_data(
        sample_size=50,
        seed=115,
    )

    first_index = data.index[0]

    data.loc[first_index, "stop"] = (
        data.loc[first_index, "start"]
    )

    with pytest.raises(
        ValueError,
        match="Every stop time must exceed start time",
    ):
        fit_time_varying_cox(data)