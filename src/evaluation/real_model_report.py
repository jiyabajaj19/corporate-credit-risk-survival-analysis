from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.base_cox import (
    CoxModelResults,
)


def build_model_comparison_table(
    results: Iterable[CoxModelResults],
) -> pd.DataFrame:
    """
    Create one model-level comparison table.
    """
    rows: list[dict[str, object]] = []

    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "number_of_firms": (
                    result.number_of_firms
                ),
                "number_of_rows": (
                    result.number_of_rows
                ),
                "number_of_events": (
                    result.number_of_events
                ),
                "number_of_features": len(
                    result.feature_columns
                ),
                "penalizer": result.penalizer,
                "l1_ratio": result.l1_ratio,
                "log_likelihood": (
                    result.log_likelihood
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        raise ValueError(
            "No model results were supplied"
        )

    return comparison


def combine_coefficient_tables(
    results: Iterable[CoxModelResults],
) -> pd.DataFrame:
    """
    Combine coefficient and hazard-ratio results across models.
    """
    tables: list[pd.DataFrame] = []

    for result in results:
        table = result.coefficients.copy()

        table["number_of_firms"] = (
            result.number_of_firms
        )

        table["number_of_events"] = (
            result.number_of_events
        )

        table["penalizer"] = (
            result.penalizer
        )

        table["l1_ratio"] = (
            result.l1_ratio
        )

        tables.append(table)

    if not tables:
        raise ValueError(
            "No coefficient tables were supplied"
        )

    return pd.concat(
        tables,
        ignore_index=True,
    )


def build_feature_scaling_table(
    results: Iterable[CoxModelResults],
) -> pd.DataFrame:
    """
    Record the means and standard deviations used for standardization.
    """
    rows: list[dict[str, object]] = []

    for result in results:
        for feature in result.feature_columns:
            rows.append(
                {
                    "model": result.model_name,
                    "feature": feature,
                    "mean": float(
                        result.feature_means[
                            feature
                        ]
                    ),
                    "standard_deviation": float(
                        result
                        .feature_standard_deviations[
                            feature
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def format_optional_number(
    value: object,
    digits: int = 4,
) -> str:
    """
    Format a possibly missing numeric value.
    """
    if value is None:
        return "N/A"

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    if not np.isfinite(numeric):
        return "N/A"

    return f"{numeric:.{digits}f}"


def build_text_summary(
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> str:
    """
    Create a readable plain-text model report.
    """
    lines = [
        "Real Corporate Credit-Risk Model Report",
        "=======================================",
        "",
        "Model comparison",
        "----------------",
    ]

    printable_comparison = (
        comparison.copy()
    )

    if "log_likelihood" in (
        printable_comparison.columns
    ):
        printable_comparison[
            "log_likelihood"
        ] = printable_comparison[
            "log_likelihood"
        ].map(format_optional_number)

    lines.append(
        printable_comparison.to_string(
            index=False
        )
    )

    lines.extend(
        [
            "",
            "Coefficient and hazard-ratio estimates",
            "--------------------------------------",
        ]
    )

    printable_coefficients = (
        coefficients.copy()
    )

    numeric_columns = [
        "coefficient",
        "coefficient_lower_95",
        "coefficient_upper_95",
        "hazard_ratio",
        "hazard_ratio_lower_95",
        "hazard_ratio_upper_95",
        "standard_error",
        "z_statistic",
        "p_value",
    ]

    for column in numeric_columns:
        if column in printable_coefficients:
            printable_coefficients[
                column
            ] = printable_coefficients[
                column
            ].map(format_optional_number)

    lines.append(
        printable_coefficients.to_string(
            index=False
        )
    )

    lines.extend(
        [
            "",
            "Interpretation note",
            "-------------------",
            (
                "Predictors are standardized before fitting, so each "
                "hazard ratio represents the estimated effect of a "
                "one-standard-deviation increase in that predictor."
            ),
            (
                "P-values and confidence intervals from penalized "
                "models should be treated as descriptive sensitivity "
                "measures rather than definitive classical inference."
            ),
        ]
    )

    return "\n".join(lines)


def save_real_model_report(
    results: list[CoxModelResults],
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Save model comparison, coefficients, scaling, and text summary.
    """
    if not results:
        raise ValueError(
            "At least one model result is required"
        )

    directory = Path(
        output_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = (
        build_model_comparison_table(
            results
        )
    )

    coefficients = (
        combine_coefficient_tables(
            results
        )
    )

    scaling = (
        build_feature_scaling_table(
            results
        )
    )

    summary = build_text_summary(
        comparison=comparison,
        coefficients=coefficients,
    )

    paths = {
        "comparison": (
            directory
            / "model_comparison.csv"
        ),
        "coefficients": (
            directory
            / "coefficient_results.csv"
        ),
        "scaling": (
            directory
            / "feature_scaling.csv"
        ),
        "summary": (
            directory
            / "summary.txt"
        ),
    }

    comparison.to_csv(
        paths["comparison"],
        index=False,
    )

    coefficients.to_csv(
        paths["coefficients"],
        index=False,
    )

    scaling.to_csv(
        paths["scaling"],
        index=False,
    )

    paths["summary"].write_text(
        summary,
        encoding="utf-8",
    )

    return paths