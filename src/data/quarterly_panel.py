from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


INSTANT_CONCEPTS = {
    "total_assets",
    "current_assets",
    "cash_and_equivalents",
    "total_liabilities",
    "stockholders_equity",
    "current_liabilities",
    "short_term_debt",
    "long_term_debt",
    "total_debt",
}

DURATION_CONCEPTS = {
    "revenue",
    "operating_income",
    "net_income",
    "interest_expense",
    "operating_cash_flow",
    "depreciation_and_amortization",
}

FISCAL_PERIODS = {"Q1", "Q2", "Q3", "FY"}

QUARTER_NUMBER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "FY": 4,
}

# Examples:
# CY2024Q1  = duration fact for a calendar quarter
# CY2024Q1I = instant fact at the quarter end
DURATION_FRAME_PATTERN = re.compile(
    r"^CY\d{4}Q([1-4])$"
)

INSTANT_FRAME_PATTERN = re.compile(
    r"^CY\d{4}Q([1-4])I$"
)


def load_financial_facts(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate the normalized SEC facts table."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Financial facts file not found: {file_path}"
        )

    facts = pd.read_csv(
        file_path,
        parse_dates=[
            "start_date",
            "end_date",
            "filed_date",
        ],
        dtype={
            "ticker": "string",
            "cik": "string",
            "entity_name": "string",
            "concept": "string",
            "form": "string",
            "fiscal_period": "string",
            "frame": "string",
        },
    )

    required_columns = {
        "ticker",
        "cik",
        "entity_name",
        "concept",
        "value",
        "start_date",
        "end_date",
        "filed_date",
        "form",
        "fiscal_year",
        "fiscal_period",
        "frame",
        "tag_priority",
    }

    missing = required_columns - set(facts.columns)

    if missing:
        raise ValueError(
            "Financial facts table is missing columns: "
            f"{sorted(missing)}"
        )

    facts["ticker"] = (
        facts["ticker"]
        .str.strip()
        .str.upper()
    )

    facts["fiscal_period"] = (
        facts["fiscal_period"]
        .str.strip()
        .str.upper()
    )

    facts["form"] = (
        facts["form"]
        .str.strip()
        .str.upper()
    )

    facts["value"] = pd.to_numeric(
        facts["value"],
        errors="coerce",
    )

    facts["tag_priority"] = pd.to_numeric(
        facts["tag_priority"],
        errors="coerce",
    ).fillna(9999)

    facts["fiscal_year"] = pd.to_numeric(
        facts["fiscal_year"],
        errors="coerce",
    ).astype("Int64")

    return facts.dropna(
        subset=[
            "ticker",
            "concept",
            "end_date",
            "value",
            "fiscal_year",
            "fiscal_period",
        ]
    ).reset_index(drop=True)


def classify_fact_periods(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify SEC duration facts using both frame metadata
    and the number of days between start and end dates.
    """
    result = facts.copy()

    result["duration_days"] = (
        result["end_date"]
        - result["start_date"]
    ).dt.days

    if "frame" in result.columns:
        frame_text = (
            result["frame"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        frame_text = pd.Series(
            "",
            index=result.index,
            dtype="string",
        )

    result["frame_quarter"] = (
        frame_text.str.extract(
            DURATION_FRAME_PATTERN,
            expand=False,
        )
    )

    result["is_duration_frame"] = (
        frame_text.str.match(
            DURATION_FRAME_PATTERN,
            na=False,
        )
    )

    result["is_instant_frame"] = (
        frame_text.str.match(
            INSTANT_FRAME_PATTERN,
            na=False,
        )
    )

    # Standalone fiscal-quarter observations.
    result["is_quarter_duration"] = (
        result["duration_days"].between(
            70,
            120,
            inclusive="both",
        )
    )

    # Backward-compatible name used by existing tests.
    result["is_single_quarter"] = (
        result["is_quarter_duration"]
    )

    # Approximately six months year-to-date.
    result["is_half_year_ytd"] = (
        result["duration_days"].between(
            150,
            220,
            inclusive="both",
        )
    )

    # Approximately nine months year-to-date.
    result["is_nine_month_ytd"] = (
        result["duration_days"].between(
            230,
            310,
            inclusive="both",
        )
    )

    # Covers most 52- and 53-week fiscal years.
    result["is_annual_duration"] = (
        result["duration_days"].between(
            320,
            385,
            inclusive="both",
        )
    )

    # Backward-compatible annual column name.
    result["is_annual"] = (
        result["is_annual_duration"]
    )

    # A fact is considered a direct quarter when either:
    # 1. its SEC frame explicitly identifies a quarter; or
    # 2. its date duration is approximately one quarter.
    result["is_direct_quarter"] = (
        result["is_duration_frame"]
        | result["is_quarter_duration"]
    )

    return result


# Keep the old name so your existing tests remain compatible.
def classify_fact_duration(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    return classify_fact_periods(facts)


def _select_best_observation(
    observations: pd.DataFrame,
) -> pd.Series | None:
    """
    Select one preferred SEC observation.

    Preference order:
    1. preferred configured XBRL tag;
    2. non-amended filing;
    3. most recently filed observation.
    """
    if observations.empty:
        return None

    working = observations.copy()

    working["is_amendment"] = (
        working["form"]
        .fillna("")
        .str.endswith("/A")
    )

    working = working.sort_values(
        by=[
            "tag_priority",
            "is_amendment",
            "filed_date",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        na_position="last",
    )

    return working.iloc[0]


def build_instant_quarterly_table(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """Create quarter-end balance-sheet observations."""
    instant = facts[
        facts["concept"].isin(INSTANT_CONCEPTS)
        & facts["fiscal_period"].isin(
            FISCAL_PERIODS
        )
    ].copy()

    if instant.empty:
        return pd.DataFrame()

    selected_rows: list[pd.Series] = []

    group_columns = [
        "ticker",
        "cik",
        "entity_name",
        "fiscal_year",
        "fiscal_period",
        "concept",
    ]

    for _, group in instant.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        selected = _select_best_observation(
            group
        )

        if selected is not None:
            selected_rows.append(selected)

    if not selected_rows:
        return pd.DataFrame()

    selected = pd.DataFrame(selected_rows)

    panel = selected.pivot_table(
        index=[
            "ticker",
            "cik",
            "entity_name",
            "fiscal_year",
            "fiscal_period",
            "end_date",
        ],
        columns="concept",
        values="value",
        aggfunc="first",
    ).reset_index()

    panel.columns.name = None

    return panel


def _get_observation(
    group: pd.DataFrame,
    fiscal_period: str,
    period_type: str,
) -> pd.Series | None:
    """
    Select the best observation for one fiscal period and period type.

    Returns None when no suitable SEC observation exists.
    """
    candidates = group.loc[
        group["fiscal_period"] == fiscal_period
    ].copy()

    if candidates.empty:
        return None

    if period_type == "direct_quarter":
        candidates = candidates.loc[
            candidates["is_direct_quarter"]
        ].copy()

        if candidates.empty:
            return None

        candidates["_period_preference"] = np.where(
            candidates["is_duration_frame"],
            0,
            1,
        )

    elif period_type == "half_year_ytd":
        candidates = candidates.loc[
            candidates["is_half_year_ytd"]
        ].copy()

        if candidates.empty:
            return None

        candidates["_period_preference"] = 0

    elif period_type == "nine_month_ytd":
        candidates = candidates.loc[
            candidates["is_nine_month_ytd"]
        ].copy()

        if candidates.empty:
            return None

        candidates["_period_preference"] = 0

    elif period_type == "annual":
        candidates = candidates.loc[
            candidates["is_annual_duration"]
        ].copy()

        if candidates.empty:
            return None

        candidates["_period_preference"] = 0

    else:
        raise ValueError(
            f"Unsupported period type: {period_type}"
        )

    # Resetting the index avoids pathological assignment behavior
    # for some large SEC groups with repeated source indices.
    candidates = candidates.reset_index(drop=True)

    candidates["_is_amendment"] = (
        candidates["form"]
        .fillna("")
        .astype(str)
        .str.endswith("/A")
    )

    candidates = candidates.sort_values(
        by=[
            "_period_preference",
            "tag_priority",
            "_is_amendment",
            "filed_date",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
        na_position="last",
    )

    if candidates.empty:
        return None

    return candidates.iloc[0]


def _quarter_record(
    source: pd.Series,
    fiscal_period: str,
    value: float,
    derivation: str,
) -> dict[str, object]:
    return {
        "ticker": source["ticker"],
        "cik": source["cik"],
        "entity_name": source["entity_name"],
        "fiscal_year": int(
            source["fiscal_year"]
        ),
        "fiscal_period": fiscal_period,
        "end_date": source["end_date"],
        "concept": source["concept"],
        "value": float(value),
        "quarter_derivation": derivation,
        "source_form": source["form"],
        "source_filed_date": (
            source["filed_date"]
        ),
    }


def convert_duration_group_to_quarters(
    group: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Convert one firm/concept/fiscal-year group to standalone quarters.

    Direct quarterly values are preferred. Otherwise:

        Q2 = six-month YTD - Q1
        Q3 = nine-month YTD - six-month YTD
        Q4 = annual - nine-month YTD
    """
    records: list[dict[str, object]] = []

    q1 = _get_observation(
        group,
        "Q1",
        "direct_quarter",
    )

    q2_direct = _get_observation(
        group,
        "Q2",
        "direct_quarter",
    )

    q3_direct = _get_observation(
        group,
        "Q3",
        "direct_quarter",
    )

    q4_direct = _get_observation(
        group,
        "FY",
        "direct_quarter",
    )

    q2_ytd = _get_observation(
        group,
        "Q2",
        "half_year_ytd",
    )

    q3_ytd = _get_observation(
        group,
        "Q3",
        "nine_month_ytd",
    )

    annual = _get_observation(
        group,
        "FY",
        "annual",
    )

    q1_value: float | None = None
    q2_ytd_value: float | None = None
    q3_ytd_value: float | None = None

    if q1 is not None:
        q1_value = float(q1["value"])

        records.append(
            _quarter_record(
                source=q1,
                fiscal_period="Q1",
                value=q1_value,
                derivation="direct_quarter",
            )
        )

    if q2_ytd is not None:
        q2_ytd_value = float(
            q2_ytd["value"]
        )

    if q3_ytd is not None:
        q3_ytd_value = float(
            q3_ytd["value"]
        )

    if q2_direct is not None:
        records.append(
            _quarter_record(
                source=q2_direct,
                fiscal_period="Q2",
                value=float(
                    q2_direct["value"]
                ),
                derivation="direct_quarter",
            )
        )

    elif (
        q2_ytd is not None
        and q1_value is not None
    ):
        records.append(
            _quarter_record(
                source=q2_ytd,
                fiscal_period="Q2",
                value=(
                    q2_ytd_value
                    - q1_value
                ),
                derivation="derived_from_ytd",
            )
        )

    if q3_direct is not None:
        records.append(
            _quarter_record(
                source=q3_direct,
                fiscal_period="Q3",
                value=float(
                    q3_direct["value"]
                ),
                derivation="direct_quarter",
            )
        )

    elif (
        q3_ytd is not None
        and q2_ytd_value is not None
    ):
        records.append(
            _quarter_record(
                source=q3_ytd,
                fiscal_period="Q3",
                value=(
                    q3_ytd_value
                    - q2_ytd_value
                ),
                derivation="derived_from_ytd",
            )
        )

    if q4_direct is not None:
        records.append(
            _quarter_record(
                source=q4_direct,
                fiscal_period="FY",
                value=float(
                    q4_direct["value"]
                ),
                derivation="direct_quarter",
            )
        )

    elif (
        annual is not None
        and q3_ytd_value is not None
    ):
        records.append(
            _quarter_record(
                source=annual,
                fiscal_period="FY",
                value=(
                    float(annual["value"])
                    - q3_ytd_value
                ),
                derivation=(
                    "derived_from_annual"
                ),
            )
        )

    return records


def build_duration_quarterly_long(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce long-format standalone quarterly flow observations.
    """
    duration = facts[
        facts["concept"].isin(
            DURATION_CONCEPTS
        )
        & facts["fiscal_period"].isin(
            FISCAL_PERIODS
        )
    ].copy()

    if duration.empty:
        return pd.DataFrame()

    duration = classify_fact_periods(
        duration
    )

    # Preserve rows with missing start dates if their SEC frame
    # identifies a standalone quarter.
    duration = duration[
        duration["start_date"].notna()
        | duration["is_duration_frame"]
    ]

    records: list[dict[str, object]] = []

    group_columns = [
        "ticker",
        "cik",
        "entity_name",
        "concept",
        "fiscal_year",
    ]

    for _, group in duration.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        records.extend(
            convert_duration_group_to_quarters(
                group
            )
        )

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "ticker",
                "fiscal_year",
                "fiscal_period",
                "concept",
            ]
        )
        .reset_index(drop=True)
    )


def build_duration_quarterly_table(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """Create the wide quarterly income and cash-flow table."""
    quarterly_long = (
        build_duration_quarterly_long(facts)
    )

    if quarterly_long.empty:
        return pd.DataFrame()

    panel = quarterly_long.pivot_table(
        index=[
            "ticker",
            "cik",
            "entity_name",
            "fiscal_year",
            "fiscal_period",
            "end_date",
        ],
        columns="concept",
        values="value",
        aggfunc="first",
    ).reset_index()

    panel.columns.name = None

    return panel


def merge_quarterly_tables(
    instant_table: pd.DataFrame,
    duration_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge instant and duration facts using company fiscal-quarter keys.

    SEC observations can contain multiple reported end dates for the
    same fiscal quarter. Each input table is first collapsed to one row
    per company, fiscal year, and fiscal period before merging.
    """
    identity_keys = [
        "ticker",
        "cik",
        "entity_name",
        "fiscal_year",
        "fiscal_period",
    ]

    def collapse_to_unique_quarters(
        table: pd.DataFrame,
        end_date_name: str,
    ) -> pd.DataFrame:
        if table.empty:
            return table.copy()

        working = table.copy()

        working["end_date"] = pd.to_datetime(
            working["end_date"],
            errors="coerce",
        )

        working = working.dropna(
            subset=[
                "ticker",
                "fiscal_year",
                "fiscal_period",
                "end_date",
            ]
        )

        value_columns = [
            column
            for column in working.columns
            if column
            not in {
                *identity_keys,
                "end_date",
            }
        ]

        # Prefer the most recent quarter-end date. When different rows
        # contain different financial concepts, combine the first
        # available non-missing value from the ordered observations.
        working = working.sort_values(
            [
                *identity_keys,
                "end_date",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        )

        collapsed_rows: list[dict[str, object]] = []

        for keys, group in working.groupby(
            identity_keys,
            dropna=False,
            sort=False,
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)

            record = dict(
                zip(
                    identity_keys,
                    keys,
                    strict=True,
                )
            )

            record[end_date_name] = (
                group["end_date"].max()
            )

            for column in value_columns:
                non_missing = group[column].dropna()

                record[column] = (
                    non_missing.iloc[0]
                    if not non_missing.empty
                    else np.nan
                )

            collapsed_rows.append(record)

        return pd.DataFrame(collapsed_rows)

    instant = collapse_to_unique_quarters(
        instant_table,
        end_date_name="instant_end_date",
    )

    duration = collapse_to_unique_quarters(
        duration_table,
        end_date_name="duration_end_date",
    )

    if instant.empty:
        panel = duration.rename(
            columns={
                "duration_end_date": "end_date",
            }
        )

    elif duration.empty:
        panel = instant.rename(
            columns={
                "instant_end_date": "end_date",
            }
        )

    else:
        panel = instant.merge(
            duration,
            on=identity_keys,
            how="outer",
            validate="one_to_one",
            suffixes=(
                "_instant",
                "_duration",
            ),
        )

        panel["end_date"] = (
            panel["instant_end_date"]
            .combine_first(
                panel["duration_end_date"]
            )
        )

        panel = panel.drop(
            columns=[
                "instant_end_date",
                "duration_end_date",
            ]
        )

    if panel.empty:
        return panel

    panel["quarter_number"] = (
        panel["fiscal_period"]
        .map(QUARTER_NUMBER)
        .astype("Int64")
    )

    panel = panel.dropna(
        subset=[
            "fiscal_year",
            "quarter_number",
            "end_date",
        ]
    )

    panel["fiscal_year"] = (
        panel["fiscal_year"].astype(int)
    )

    panel["quarter_number"] = (
        panel["quarter_number"].astype(int)
    )

    panel = panel.sort_values(
        [
            "ticker",
            "fiscal_year",
            "quarter_number",
        ]
    )

    duplicate_quarters = panel.duplicated(
        subset=[
            "ticker",
            "fiscal_year",
            "fiscal_period",
        ],
        keep=False,
    )

    if duplicate_quarters.any():
        duplicate_rows = panel.loc[
            duplicate_quarters,
            [
                "ticker",
                "fiscal_year",
                "fiscal_period",
                "end_date",
            ],
        ]

        raise ValueError(
            "Quarterly panel still contains duplicate "
            "company-quarter records:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return panel.reset_index(drop=True)


def derive_balance_sheet_values(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Derive total debt and liabilities where necessary."""
    result = panel.copy()

    for column in [
        "short_term_debt",
        "long_term_debt",
        "total_debt",
        "total_assets",
        "stockholders_equity",
        "total_liabilities",
    ]:
        if column not in result:
            result[column] = np.nan

    component_debt = (
        result["short_term_debt"].fillna(0)
        + result["long_term_debt"].fillna(0)
    )

    has_debt_component = (
        result["short_term_debt"].notna()
        | result["long_term_debt"].notna()
    )

    result["total_debt"] = (
        result["total_debt"].where(
            result["total_debt"].notna(),
            component_debt.where(
                has_debt_component
            ),
        )
    )

    derived_liabilities = (
        result["total_assets"]
        - result["stockholders_equity"]
    )

    result["total_liabilities"] = (
        result["total_liabilities"].where(
            result["total_liabilities"].notna(),
            derived_liabilities,
        )
    )

    return result


def build_quarterly_financial_panel(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete company-quarter accounting panel."""
    instant_table = (
        build_instant_quarterly_table(facts)
    )

    duration_table = (
        build_duration_quarterly_table(facts)
    )

    panel = merge_quarterly_tables(
        instant_table=instant_table,
        duration_table=duration_table,
    )

    return derive_balance_sheet_values(
        panel
    )