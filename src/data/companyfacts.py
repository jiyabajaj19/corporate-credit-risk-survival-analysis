from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# Earlier tags have higher priority when multiple tags describe
# the same financial concept for the same reporting period.
CONCEPT_TAGS: dict[str, list[str]] = {
    "total_assets": [
        "Assets",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "stockholders_equity": [
    "StockholdersEquity",
    (
        "StockholdersEquityIncludingPortion"
        "AttributableToNoncontrollingInterest"
    ),
    "PartnersCapital",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "DebtCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "total_debt": [
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "DebtAndCapitalLeaseObligations",
    ],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "depreciation_and_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "Depreciation",
    ],
}


ALLOWED_FORMS = {
    "10-Q",
    "10-Q/A",
    "10-K",
    "10-K/A",
}


@dataclass(frozen=True)
class ParsedCompany:
    cik: str
    entity_name: str
    ticker: str
    source_file: str


def load_companyfacts_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load and minimally validate one SEC Company Facts file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Company Facts file not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {file_path}"
        )

    if "facts" not in payload:
        raise ValueError(
            f"Missing 'facts' object in {file_path}"
        )

    return payload


def infer_ticker_from_filename(
    path: str | Path,
) -> str:
    """
    Infer the ticker from names such as AAPL_0000320193.json.
    """
    stem = Path(path).stem

    ticker = stem.split("_", maxsplit=1)[0].strip().upper()

    if not ticker:
        raise ValueError(
            f"Could not infer ticker from filename: {path}"
        )

    return ticker


def get_company_metadata(
    payload: dict[str, Any],
    source_path: str | Path,
) -> ParsedCompany:
    """
    Extract identifying metadata from a Company Facts payload.

    Some SEC Company Facts responses do not include entityName.
    In that case, use the ticker inferred from the cached filename
    as a safe fallback so one malformed company does not stop the
    entire batch.
    """
    cik_value = payload.get("cik")

    if cik_value is None:
        raise ValueError(
            f"Missing CIK in {source_path}"
        )

    cik = str(cik_value).zfill(10)

    ticker = infer_ticker_from_filename(
        source_path
    )

    entity_name = str(
        payload.get("entityName", "")
    ).strip()

    if not entity_name:
        entity_name = ticker

    return ParsedCompany(
        cik=cik,
        entity_name=entity_name,
        ticker=ticker,
        source_file=Path(source_path).name,
    )


def _select_supported_unit(
    units: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Prefer USD observations for the accounting concepts used here.
    """
    if not isinstance(units, dict):
        return None

    observations = units.get("USD")

    if isinstance(observations, list):
        return "USD", observations

    return None


def extract_concept_observations(
    payload: dict[str, Any],
    source_path: str | Path,
    concept_name: str,
    candidate_tags: list[str],
) -> list[dict[str, Any]]:
    """
    Extract all supported observations for one standardized concept.
    """
    metadata = get_company_metadata(
        payload=payload,
        source_path=source_path,
    )

    us_gaap = (
        payload.get("facts", {})
        .get("us-gaap", {})
    )

    if not isinstance(us_gaap, dict):
        return []

    rows: list[dict[str, Any]] = []

    for tag_priority, tag in enumerate(
        candidate_tags
    ):
        tag_payload = us_gaap.get(tag)

        if not isinstance(tag_payload, dict):
            continue

        selected_unit = _select_supported_unit(
            tag_payload.get("units", {})
        )

        if selected_unit is None:
            continue

        unit, observations = selected_unit

        for observation in observations:
            if not isinstance(observation, dict):
                continue

            form = str(
                observation.get("form", "")
            ).strip()

            if form not in ALLOWED_FORMS:
                continue

            value = observation.get("val")
            end_date = observation.get("end")

            if value is None or end_date is None:
                continue

            rows.append(
                {
                    "ticker": metadata.ticker,
                    "cik": metadata.cik,
                    "entity_name": (
                        metadata.entity_name
                    ),
                    "concept": concept_name,
                    "xbrl_tag": tag,
                    "tag_priority": tag_priority,
                    "unit": unit,
                    "value": value,
                    "start_date": (
                        observation.get("start")
                    ),
                    "end_date": end_date,
                    "filed_date": (
                        observation.get("filed")
                    ),
                    "form": form,
                    "fiscal_year": (
                        observation.get("fy")
                    ),
                    "fiscal_period": (
                        observation.get("fp")
                    ),
                    "frame": (
                        observation.get("frame")
                    ),
                    "accession_number": (
                        observation.get("accn")
                    ),
                    "source_file": (
                        metadata.source_file
                    ),
                }
            )

    return rows


def extract_company_financial_facts(
    source_path: str | Path,
) -> pd.DataFrame:
    """
    Extract all configured financial concepts from one SEC JSON file.
    """
    payload = load_companyfacts_json(source_path)

    rows: list[dict[str, Any]] = []

    for concept_name, candidate_tags in (
        CONCEPT_TAGS.items()
    ):
        rows.extend(
            extract_concept_observations(
                payload=payload,
                source_path=source_path,
                concept_name=concept_name,
                candidate_tags=candidate_tags,
            )
        )

    if not rows:
        return _empty_fact_table()

    facts = pd.DataFrame(rows)

    facts["start_date"] = pd.to_datetime(
        facts["start_date"],
        errors="coerce",
    )

    facts["end_date"] = pd.to_datetime(
        facts["end_date"],
        errors="coerce",
    )

    facts["filed_date"] = pd.to_datetime(
        facts["filed_date"],
        errors="coerce",
    )

    facts["value"] = pd.to_numeric(
        facts["value"],
        errors="coerce",
    )

    facts = facts.dropna(
        subset=[
            "end_date",
            "value",
        ]
    )

    return deduplicate_financial_facts(facts)


def deduplicate_financial_facts(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove amended and alternative-tag duplicates.

    For otherwise equivalent observations, this prefers:
    1. the highest-priority configured XBRL tag;
    2. the most recently filed observation.
    """
    if facts.empty:
        return facts.copy()

    working = facts.copy()

    working["is_amendment"] = (
        working["form"].str.endswith(
            "/A",
            na=False,
        )
    )

    working = working.sort_values(
        by=[
            "ticker",
            "concept",
            "start_date",
            "end_date",
            "fiscal_year",
            "fiscal_period",
            "tag_priority",
            "filed_date",
            "is_amendment",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ],
        na_position="last",
    )

    duplicate_key = [
        "ticker",
        "concept",
        "start_date",
        "end_date",
        "fiscal_year",
        "fiscal_period",
    ]

    deduplicated = working.drop_duplicates(
        subset=duplicate_key,
        keep="first",
    )

    deduplicated = deduplicated.drop(
        columns=["is_amendment"]
    )

    return (
        deduplicated.sort_values(
            [
                "ticker",
                "concept",
                "end_date",
                "start_date",
            ]
        )
        .reset_index(drop=True)
    )


def extract_directory_financial_facts(
    input_directory: str | Path,
) -> pd.DataFrame:
    """
    Extract and combine supported facts from every Company Facts
    JSON file in a directory.

    Files with malformed JSON, missing Company Facts content, or no
    supported observations are skipped so one unusable SEC response
    does not stop a large batch.
    """
    directory = Path(input_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"SEC input directory not found: {directory}"
        )

    json_files = sorted(
        directory.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {directory}"
        )

    company_tables: list[pd.DataFrame] = []
    skipped_files: list[tuple[str, str]] = []

    for path in json_files:
        try:
            table = extract_company_financial_facts(
                path
            )

            if table.empty:
                skipped_files.append(
                    (
                        path.name,
                        "No supported financial facts",
                    )
                )
                continue

            company_tables.append(table)

        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            skipped_files.append(
                (
                    path.name,
                    str(error),
                )
            )

    if skipped_files:
        print()
        print("Skipped SEC files")
        print("-----------------")

        for filename, reason in skipped_files:
            print(
                f"{filename}: {reason}"
            )

        print()

    if not company_tables:
        return _empty_fact_table()

    return pd.concat(
        company_tables,
        ignore_index=True,
    )

def create_fact_coverage_summary(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize extracted concept coverage by company.
    """
    if facts.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "concept",
                "observation_count",
                "first_end_date",
                "last_end_date",
            ]
        )

    return (
        facts.groupby(
            ["ticker", "concept"],
            as_index=False,
        )
        .agg(
            observation_count=(
                "value",
                "size",
            ),
            first_end_date=(
                "end_date",
                "min",
            ),
            last_end_date=(
                "end_date",
                "max",
            ),
        )
        .sort_values(
            ["ticker", "concept"]
        )
        .reset_index(drop=True)
    )


def _empty_fact_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "cik",
            "entity_name",
            "concept",
            "xbrl_tag",
            "tag_priority",
            "unit",
            "value",
            "start_date",
            "end_date",
            "filed_date",
            "form",
            "fiscal_year",
            "fiscal_period",
            "frame",
            "accession_number",
            "source_file",
        ]
    )