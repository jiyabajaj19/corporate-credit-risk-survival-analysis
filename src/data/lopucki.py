from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "PrimaryKey",
    "NameCorp",
    "CommonName",
    "CikBefore",
    "DateFiled",
    "Chapter",
    "Disposition",
    "SICPrimary",
    "SICDescription",
    "AssetsBefore",
    "LiabBefore",
    "SalesBefore",
}


def normalize_cik_value(
    value: object,
) -> str | None:
    """
    Convert a LoPucki CIK value to a ten-digit SEC CIK.

    Missing, invalid, or nonpositive values return None.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    # Excel often loads integer identifiers as values such as
    # "320193.0".
    if text.endswith(".0"):
        text = text[:-2]

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        return None

    cik_number = int(digits)

    if cik_number <= 0:
        return None

    if len(str(cik_number)) > 10:
        return None

    return str(cik_number).zfill(10)


def clean_company_name(
    value: object,
) -> str | None:
    if pd.isna(value):
        return None

    name = " ".join(
        str(value).strip().split()
    )

    return name or None


def load_lopucki_cases(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load and validate the LoPucki Cases workbook.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"LoPucki file not found: {file_path}"
        )

    cases = pd.read_excel(
        file_path,
        engine="openpyxl",
    )

    missing = REQUIRED_COLUMNS - set(
        cases.columns
    )

    if missing:
        raise ValueError(
            "LoPucki workbook is missing columns: "
            f"{sorted(missing)}"
        )

    return cases


def build_bankruptcy_events(
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a normalized bankruptcy-event table.

    Each row represents one LoPucki bankruptcy case. The primary
    event date is DateFiled.
    """
    data = cases.copy()

    data["cik"] = data["CikBefore"].map(
        normalize_cik_value
    )

    data["company_name"] = data[
        "NameCorp"
    ].map(clean_company_name)

    data["common_name"] = data[
        "CommonName"
    ].map(clean_company_name)

    data["event_date"] = pd.to_datetime(
        data["DateFiled"],
        errors="coerce",
    )

    data["chapter"] = (
        data["Chapter"]
        .astype("string")
        .str.strip()
    )

    data["disposition"] = (
        data["Disposition"]
        .astype("string")
        .str.strip()
    )

    data["sic_primary"] = pd.to_numeric(
        data["SICPrimary"],
        errors="coerce",
    ).astype("Int64")

    data["sic_description"] = (
        data["SICDescription"]
        .astype("string")
        .str.strip()
    )

    for source, target in [
        ("AssetsBefore", "assets_before"),
        ("LiabBefore", "liabilities_before"),
        ("SalesBefore", "sales_before"),
    ]:
        data[target] = pd.to_numeric(
            data[source],
            errors="coerce",
        )

    data["event_type"] = "bankruptcy_filing"

    data["source_database"] = (
        "LoPucki Bankruptcy Research Database"
    )

    data["source_case_id"] = (
        data["PrimaryKey"]
        .astype("string")
        .str.strip()
    )

    events = data[
        [
            "source_case_id",
            "cik",
            "company_name",
            "common_name",
            "event_date",
            "event_type",
            "chapter",
            "disposition",
            "sic_primary",
            "sic_description",
            "assets_before",
            "liabilities_before",
            "sales_before",
            "source_database",
        ]
    ].copy()

    events = events.dropna(
        subset=[
            "source_case_id",
            "company_name",
            "event_date",
        ]
    )

    events = events.sort_values(
        [
            "event_date",
            "company_name",
        ]
    ).reset_index(drop=True)

    return events


def create_event_quality_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize match-key and field availability.
    """
    fields = [
        "cik",
        "event_date",
        "chapter",
        "disposition",
        "sic_primary",
        "assets_before",
        "liabilities_before",
        "sales_before",
    ]

    rows: list[dict[str, object]] = []

    total = len(events)

    for field in fields:
        available = int(
            events[field].notna().sum()
        )

        rows.append(
            {
                "field": field,
                "available_rows": available,
                "total_rows": total,
                "coverage": (
                    available / total
                    if total > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)