import numpy as np
import pandas as pd
import pytest

from src.models.base_cox import (
    prepare_time_varying_model_data,
    remove_near_constant_features,
    standardize_features,
    validate_time_varying_data,
)


def create_example_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for firm_index in range(10):
        for quarter in range(4):
            rows.append(
                {
                    "firm_id": f"F{firm_index}",
                    "start": quarter * 0.25,
                    "stop": (
                        quarter + 1
                    ) * 0.25,
                    "event": int(
                        firm_index < 4
                        and quarter == 3
                    ),
                    "leverage": (
                        0.20
                        + 0.02 * firm_index
                        + 0.01 * quarter
                    ),
                    "current_ratio": (
                        1.80
                        - 0.03 * firm_index
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_validate_time_varying_data() -> None:
    data = create_example_data()

    validated = validate_time_varying_data(
        data=data,
        feature_columns=[
            "leverage",
            "current_ratio",
        ],
    )

    assert len(validated) == 40

    assert (
        validated["stop"]
        > validated["start"]
    ).all()

    assert validated["event"].sum() == 4


def test_invalid_intervals_raise_error() -> None:
    data = create_example_data()

    data.loc[0, "stop"] = data.loc[
        0,
        "start",
    ]

    with pytest.raises(
        ValueError,
        match="stop time",
    ):
        validate_time_varying_data(
            data=data,
            feature_columns=[
                "leverage",
                "current_ratio",
            ],
        )


def test_remove_near_constant_features() -> None:
    data = create_example_data()

    data["constant"] = 1.0

    retained = remove_near_constant_features(
        data=data,
        feature_columns=[
            "leverage",
            "constant",
        ],
    )

    assert retained == ["leverage"]


def test_standardize_features() -> None:
    data = create_example_data()

    standardized, means, deviations = (
        standardize_features(
            data=data,
            feature_columns=[
                "leverage",
                "current_ratio",
            ],
        )
    )

    assert np.allclose(
        standardized[
            [
                "leverage",
                "current_ratio",
            ]
        ].mean(),
        0.0,
        atol=1e-10,
    )

    assert np.allclose(
        standardized[
            [
                "leverage",
                "current_ratio",
            ]
        ].std(ddof=0),
        1.0,
        atol=1e-10,
    )

    assert (
        deviations > 0
    ).all()

    assert means.notna().all()


def test_prepare_time_varying_model_data() -> None:
    data = create_example_data()

    (
        modeling_data,
        features,
        means,
        deviations,
    ) = prepare_time_varying_model_data(
        data=data,
        feature_columns=[
            "leverage",
            "current_ratio",
        ],
        standardize=True,
    )

    assert features == [
        "leverage",
        "current_ratio",
    ]

    assert set(
        modeling_data.columns
    ) == {
        "firm_id",
        "start",
        "stop",
        "event",
        "leverage",
        "current_ratio",
    }

    assert means.index.tolist() == features
    assert deviations.index.tolist() == features