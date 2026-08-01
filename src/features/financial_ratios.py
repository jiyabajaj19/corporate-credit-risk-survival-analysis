from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = [
    "ticker",
    "cik",
    "entity_name",
    "fiscal_year",
    "fiscal_period",
    "quarter_number",
    "end_date",
]


def load_quarterly_panel(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the quarterly SEC accounting panel.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Quarterly financial panel not found: {file_path}"
        )

    panel = pd.read_csv(
        file_path,
        parse_dates=["end_date"],
        dtype={
            "ticker": "string",
            "cik": "string",
            "entity_name": "string",
            "fiscal_period": "string",
        },
    )

    required_columns = {
        *IDENTIFIER_COLUMNS,
        "total_assets",
        "current_assets",
        "cash_and_equivalents",
        "current_liabilities",
        "total_debt",
        "revenue",
        "operating_income",
        "net_income",
        "operating_cash_flow",
    }

    missing = required_columns - set(panel.columns)

    if missing:
        raise ValueError(
            "Quarterly panel is missing columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        column
        for column in panel.columns
        if column not in {
            "ticker",
            "cik",
            "entity_name",
            "fiscal_period",
            "end_date",
        }
    ]

    for column in numeric_columns:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    return (
        panel.sort_values(
            [
                "ticker",
                "fiscal_year",
                "quarter_number",
                "end_date",
            ]
        )
        .reset_index(drop=True)
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide while avoiding zero and non-finite denominators.
    """
    valid_denominator = (
        denominator.notna()
        & np.isfinite(denominator)
        & (denominator.abs() > 1e-12)
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=float,
    )

    result.loc[valid_denominator] = (
        numerator.loc[valid_denominator]
        / denominator.loc[valid_denominator]
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


def calculate_quarterly_growth(
    data: pd.DataFrame,
    value_column: str,
    minimum_previous_absolute_value: float = 1.0,
    require_positive_previous: bool = True,
) -> pd.Series:
    """
    Calculate growth only across consecutive fiscal quarters.

    Quarter sequence:
        Q1 -> Q2 -> Q3 -> FY -> next fiscal-year Q1

    Nonconsecutive observations are left missing.
    """
    if value_column not in data.columns:
        return pd.Series(
            np.nan,
            index=data.index,
            dtype=float,
        )

    working = data.copy()

    current_value = pd.to_numeric(
        working[value_column],
        errors="coerce",
    )

    previous_value = (
        working.groupby("ticker")[value_column]
        .shift(1)
    )

    previous_year = (
        working.groupby("ticker")["fiscal_year"]
        .shift(1)
    )

    previous_quarter = (
        working.groupby("ticker")["quarter_number"]
        .shift(1)
    )

    same_year_next_quarter = (
        working["fiscal_year"].eq(previous_year)
        & working["quarter_number"].eq(
            previous_quarter + 1
        )
    )

    next_year_first_quarter = (
        working["fiscal_year"].eq(
            previous_year + 1
        )
        & working["quarter_number"].eq(1)
        & previous_quarter.eq(4)
    )

    consecutive = (
        same_year_next_quarter
        | next_year_first_quarter
    )

    valid_previous = (
        consecutive
        & previous_value.notna()
        & np.isfinite(previous_value)
        & (
            previous_value.abs()
            > minimum_previous_absolute_value
        )
    )

    if require_positive_previous:
        valid_previous &= previous_value > 0
        valid_previous &= current_value >= 0

    growth = pd.Series(
        np.nan,
        index=working.index,
        dtype=float,
    )

    growth.loc[valid_previous] = (
        current_value.loc[valid_previous]
        / previous_value.loc[valid_previous]
        - 1.0
    )

    return growth.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def build_credit_risk_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct quarterly corporate credit-risk features.

    Ratios are based only on reported or previously derived SEC
    accounting values. Missing inputs remain missing rather than
    being replaced with assumed values.
    """
    data = panel.copy()

    required_columns = {
        *IDENTIFIER_COLUMNS,
        "total_assets",
        "current_assets",
        "cash_and_equivalents",
        "current_liabilities",
        "total_debt",
        "revenue",
        "operating_income",
        "net_income",
        "operating_cash_flow",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Cannot build credit-risk features. "
            f"Missing columns: {sorted(missing)}"
        )

    data = data.sort_values(
        [
            "ticker",
            "fiscal_year",
            "quarter_number",
            "end_date",
        ]
    ).reset_index(drop=True)

    data["leverage"] = safe_divide(
        data["total_debt"],
        data["total_assets"],
    )

    data["current_ratio"] = safe_divide(
        data["current_assets"],
        data["current_liabilities"],
    )

    data["cash_ratio"] = safe_divide(
        data["cash_and_equivalents"],
        data["current_liabilities"],
    )

    data["return_on_assets"] = safe_divide(
        data["net_income"],
        data["total_assets"],
    )

    data["operating_cash_flow_ratio"] = (
        safe_divide(
            data["operating_cash_flow"],
            data["current_liabilities"],
        )
    )

    data["operating_margin"] = safe_divide(
        data["operating_income"],
        data["revenue"],
    )

    data["log_total_assets"] = np.where(
        data["total_assets"] > 0,
        np.log(data["total_assets"]),
        np.nan,
    )

    data["revenue_growth"] = (
    calculate_quarterly_growth(
        data=data,
        value_column="revenue",
        minimum_previous_absolute_value=1_000_000.0,
        require_positive_previous=True,
        )
    )

    data["debt_growth"] = (
    calculate_quarterly_growth(
        data=data,
        value_column="total_debt",
        minimum_previous_absolute_value=1_000_000.0,
        require_positive_previous=True,
        )
    )

    if "interest_expense" in data.columns:
        positive_interest_expense = (
            data["interest_expense"].abs()
        )

        data["interest_coverage"] = (
            safe_divide(
                data["operating_income"],
                positive_interest_expense,
            )
        )
    else:
        data["interest_coverage"] = np.nan

    data["low_interest_coverage"] = (
        np.where(
            data["interest_coverage"].notna(),
            (
                data["interest_coverage"] < 1.5
            ).astype(int),
            np.nan,
        )
    )

    if (
        "depreciation_and_amortization"
        in data.columns
    ):
        data["ebitda"] = (
            data["operating_income"]
            + data[
                "depreciation_and_amortization"
            ]
        )

        data["ebitda_margin"] = safe_divide(
            data["ebitda"],
            data["revenue"],
        )
    else:
        data["ebitda"] = np.nan
        data["ebitda_margin"] = np.nan

    feature_columns = [
        *IDENTIFIER_COLUMNS,
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
        "interest_coverage",
        "operating_margin",
    ]

    return data[feature_columns]


def create_feature_coverage_summary(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize feature availability across the real panel.
    """
    feature_columns = [
        column
        for column in features.columns
        if column not in IDENTIFIER_COLUMNS
    ]

    rows: list[dict[str, float | int | str]] = []

    for feature in feature_columns:
        available = int(
            features[feature].notna().sum()
        )

        total = len(features)

        rows.append(
            {
                "feature": feature,
                "available_rows": available,
                "total_rows": total,
                "coverage": (
                    available / total
                    if total > 0
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "coverage",
            ascending=False,
        )
        .reset_index(drop=True)
    )

MODEL_FEATURE_BOUNDS = {
    "leverage": (0.0, 2.0),
    "current_ratio": (0.0, 10.0),
    "cash_ratio": (0.0, 5.0),
    "return_on_assets": (-1.0, 1.0),
    "debt_growth": (-1.0, 3.0),
    "revenue_growth": (-1.0, 3.0),
    "operating_cash_flow_ratio": (-2.0, 2.0),
    "interest_coverage": (-50.0, 100.0),
    "operating_margin": (-2.0, 2.0),
    "ebitda_margin": (-2.0, 2.0),
}


def clean_credit_risk_features(
    features: pd.DataFrame,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
) -> pd.DataFrame:
    """
    Create model-ready features while preserving raw columns.

    Values outside broad accounting plausibility bounds are marked
    missing. Remaining continuous values are winsorized within each
    feature to reduce the influence of extreme observations.
    """
    if not (
        0.0 <= winsorize_lower
        < winsorize_upper
        <= 1.0
    ):
        raise ValueError(
            "Winsorization probabilities are invalid."
        )

    result = features.copy()

    for feature, (
        lower_bound,
        upper_bound,
    ) in MODEL_FEATURE_BOUNDS.items():
        if feature not in result.columns:
            continue

        raw_column = f"{feature}_raw"

        result[raw_column] = result[feature]

        invalid = (
            (result[feature] < lower_bound)
            | (result[feature] > upper_bound)
            | ~np.isfinite(result[feature])
        )

        result.loc[invalid, feature] = np.nan

        non_missing = result[feature].dropna()

        if non_missing.empty:
            continue

        lower_quantile = float(
            non_missing.quantile(
                winsorize_lower
            )
        )

        upper_quantile = float(
            non_missing.quantile(
                winsorize_upper
            )
        )

        result[feature] = result[
            feature
        ].clip(
            lower=lower_quantile,
            upper=upper_quantile,
        )

    return result