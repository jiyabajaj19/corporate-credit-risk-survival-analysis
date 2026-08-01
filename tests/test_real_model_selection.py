import numpy as np
import pandas as pd

from src.evaluation.real_model_selection import (
    build_firm_summary,
    create_grouped_folds,
    evaluate_penalizer_grid,
)


FEATURES = [
    "leverage",
    "current_ratio",
]


def create_example_data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)

    rows: list[dict[str, object]] = []

    number_of_firms = 50
    number_of_quarters = 4
    event_firms = set(range(20))

    for firm_index in range(
        number_of_firms
    ):
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
                    "leverage": (
                        baseline_leverage
                        + 0.01 * quarter
                        + rng.normal(
                            0.0,
                            0.03,
                        )
                    ),
                    "current_ratio": (
                        baseline_current_ratio
                        - 0.01 * quarter
                        + rng.normal(
                            0.0,
                            0.08,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_build_firm_summary() -> None:
    summary = build_firm_summary(
        create_example_data()
    )

    assert len(summary) == 50
    assert summary["event"].sum() == 20


def test_grouped_folds_do_not_overlap() -> None:
    data = create_example_data()

    folds = create_grouped_folds(
        data=data,
        number_of_folds=5,
    )

    assert len(folds) == 5

    for training, validation in folds:
        assert not (
            training & validation
        )


def test_evaluate_penalizer_grid() -> None:
    result = evaluate_penalizer_grid(
        data=create_example_data(),
        penalizer_grid=[
            0.01,
            0.10,
        ],
        feature_columns=FEATURES,
        number_of_folds=3,
    )

    assert len(
        result.fold_results
    ) == 6

    assert set(
        result.penalty_summary[
            "penalizer"
        ]
    ) == {
        0.01,
        0.10,
    }

    assert result.selected_penalizer in {
        0.01,
        0.10,
    }

    assert (
        result.penalty_summary[
            "mean_validation_concordance"
        ].between(0.0, 1.0)
    ).all()