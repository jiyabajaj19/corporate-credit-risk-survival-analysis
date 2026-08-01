from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.companyfacts import (
    create_fact_coverage_summary,
    extract_directory_financial_facts,
)
from src.data.quarterly_panel import (
    build_quarterly_financial_panel,
)
from src.features.financial_ratios import (
    build_credit_risk_features,
    clean_credit_risk_features,
    create_feature_coverage_summary,
)


@dataclass(frozen=True)
class SecPipelinePaths:
    facts: Path
    fact_coverage: Path
    quarterly_panel: Path
    features: Path
    feature_coverage: Path
    clean_features: Path
    clean_feature_coverage: Path


@dataclass
class SecPipelineResults:
    paths: SecPipelinePaths
    company_count: int
    fact_count: int
    quarter_count: int
    facts: pd.DataFrame
    panel: pd.DataFrame
    features: pd.DataFrame
    clean_features: pd.DataFrame
    fact_coverage: pd.DataFrame
    feature_coverage: pd.DataFrame
    clean_feature_coverage: pd.DataFrame


def normalize_output_prefix(
    output_prefix: str,
) -> str:
    """
    Convert a user-provided prefix into a safe file prefix.
    """
    prefix = (
        output_prefix.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if not prefix:
        raise ValueError(
            "output_prefix cannot be empty"
        )

    if not all(
        character.isalnum()
        or character == "_"
        for character in prefix
    ):
        raise ValueError(
            "output_prefix may contain only letters, "
            "numbers, underscores, spaces, and hyphens"
        )

    return prefix


def create_pipeline_paths(
    output_directory: str | Path,
    output_prefix: str,
) -> SecPipelinePaths:
    """
    Construct all output paths used by one SEC pipeline run.
    """
    directory = Path(output_directory)
    prefix = normalize_output_prefix(
        output_prefix
    )

    return SecPipelinePaths(
        facts=(
            directory
            / f"{prefix}_sec_financial_facts_long.csv"
        ),
        fact_coverage=(
            directory
            / f"{prefix}_sec_fact_coverage.csv"
        ),
        quarterly_panel=(
            directory
            / (
                f"{prefix}_sec_quarterly_"
                "financial_panel.csv"
            )
        ),
        features=(
            directory
            / (
                f"{prefix}_sec_credit_"
                "risk_features.csv"
            )
        ),
        feature_coverage=(
            directory
            / (
                f"{prefix}_sec_credit_risk_"
                "feature_coverage.csv"
            )
        ),
        clean_features=(
            directory
            / (
                f"{prefix}_sec_credit_risk_"
                "features_clean.csv"
            )
        ),
        clean_feature_coverage=(
            directory
            / (
                f"{prefix}_sec_credit_risk_"
                "features_clean_coverage.csv"
            )
        ),
    )


def _save_table(
    table: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save a table after ensuring its output directory exists.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        path,
        index=False,
    )


def run_sec_pipeline(
    input_directory: str | Path,
    output_directory: str | Path,
    output_prefix: str,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    save_outputs: bool = True,
) -> SecPipelineResults:
    """
    Run the complete SEC financial-data pipeline.

    Stages
    ------
    1. Extract normalized Company Facts.
    2. Build the company-quarter accounting panel.
    3. Engineer quarterly credit-risk features.
    4. Create cleaned model-ready features.
    """
    input_path = Path(input_directory)
    output_path = Path(output_directory)

    if not input_path.exists():
        raise FileNotFoundError(
            "SEC Company Facts directory not found: "
            f"{input_path}"
        )

    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: "
            f"{input_path}"
        )

    if not (
        0.0
        <= winsorize_lower
        < winsorize_upper
        <= 1.0
    ):
        raise ValueError(
            "Winsorization probabilities must satisfy "
            "0 <= lower < upper <= 1"
        )

    paths = create_pipeline_paths(
        output_directory=output_path,
        output_prefix=output_prefix,
    )

    # Stage 1: normalized financial facts
    facts = extract_directory_financial_facts(
        input_path
    )

    if facts.empty:
        raise RuntimeError(
            "No supported SEC financial facts "
            "were extracted"
        )

    fact_coverage = (
        create_fact_coverage_summary(
            facts
        )
    )

    # Stage 2: quarterly accounting panel
    panel = build_quarterly_financial_panel(
        facts
    )

    if panel.empty:
        raise RuntimeError(
            "No quarterly financial panel "
            "rows were constructed"
        )

    # Stage 3: credit-risk features
    features = build_credit_risk_features(
        panel
    )

    if features.empty:
        raise RuntimeError(
            "No credit-risk features were created"
        )

    feature_coverage = (
        create_feature_coverage_summary(
            features
        )
    )

    # Stage 4: cleaned model-ready features
    clean_features = (
        clean_credit_risk_features(
            features=features,
            winsorize_lower=winsorize_lower,
            winsorize_upper=winsorize_upper,
        )
    )

    clean_columns = [
        column
        for column in clean_features.columns
        if not column.endswith("_raw")
    ]

    clean_feature_coverage = (
        create_feature_coverage_summary(
            clean_features[clean_columns]
        )
    )

    if save_outputs:
        _save_table(
            facts,
            paths.facts,
        )

        _save_table(
            fact_coverage,
            paths.fact_coverage,
        )

        _save_table(
            panel,
            paths.quarterly_panel,
        )

        _save_table(
            features,
            paths.features,
        )

        _save_table(
            feature_coverage,
            paths.feature_coverage,
        )

        _save_table(
            clean_features,
            paths.clean_features,
        )

        _save_table(
            clean_feature_coverage,
            paths.clean_feature_coverage,
        )

    return SecPipelineResults(
        paths=paths,
        company_count=int(
            clean_features[
                "ticker"
            ].nunique()
        ),
        fact_count=len(facts),
        quarter_count=len(panel),
        facts=facts,
        panel=panel,
        features=features,
        clean_features=clean_features,
        fact_coverage=fact_coverage,
        feature_coverage=feature_coverage,
        clean_feature_coverage=(
            clean_feature_coverage
        ),
    )