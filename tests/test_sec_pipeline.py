from pathlib import Path

import pytest

from src.pipelines.sec_pipeline import (
    create_pipeline_paths,
    normalize_output_prefix,
)


def test_normalize_output_prefix() -> None:
    assert (
        normalize_output_prefix(
            "Bankrupt Firms"
        )
        == "bankrupt_firms"
    )

    assert (
        normalize_output_prefix(
            "pilot-data"
        )
        == "pilot_data"
    )


def test_empty_prefix_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        normalize_output_prefix("   ")


def test_invalid_prefix_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="may contain only",
    ):
        normalize_output_prefix(
            "bankrupt/files"
        )


def test_create_pipeline_paths(
    tmp_path: Path,
) -> None:
    paths = create_pipeline_paths(
        output_directory=tmp_path,
        output_prefix="bankrupt",
    )

    assert paths.facts == (
        tmp_path
        / "bankrupt_sec_financial_facts_long.csv"
    )

    assert paths.quarterly_panel == (
        tmp_path
        / (
            "bankrupt_sec_quarterly_"
            "financial_panel.csv"
        )
    )

    assert paths.clean_features == (
        tmp_path
        / (
            "bankrupt_sec_credit_risk_"
            "features_clean.csv"
        )
    )