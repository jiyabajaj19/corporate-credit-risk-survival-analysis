from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.sec_api import normalize_cik


SEC_EXCHANGE_TICKERS_URL = (
    "https://www.sec.gov/files/"
    "company_tickers_exchange.json"
)

ALLOWED_EXCHANGES = {
    "Nasdaq",
    "NYSE",
    "NYSE Arca",
    "NYSE American",
}


def download_exchange_tickers(
    contact_email: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Download the official SEC ticker, CIK, and exchange file.
    """
    path = Path(output_path)

    if path.exists() and not overwrite:
        return path

    email = contact_email.strip()

    if "@" not in email:
        raise ValueError(
            "A valid contact email is required."
        )

    # Do not pass Host=data.sec.gov here because this file
    # is served from www.sec.gov.
    headers = {
        "User-Agent": (
            "CorporateCreditRiskSurvivalAnalysis "
            f"{email}"
        ),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }

    response = requests.get(
        SEC_EXCHANGE_TICKERS_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError(
            "SEC exchange ticker response must "
            "be a JSON object."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return path


def load_exchange_tickers(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load SEC exchange ticker data.

    Expected structure:
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [...]
        }
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"SEC ticker file not found: {file_path}"
        )

    payload: dict[str, Any] = json.loads(
        file_path.read_text(
            encoding="utf-8"
        )
    )

    fields = payload.get("fields")
    records = payload.get("data")

    if (
        not isinstance(fields, list)
        or not isinstance(records, list)
    ):
        raise ValueError(
            "Unexpected SEC exchange ticker structure."
        )

    companies = pd.DataFrame(
        records,
        columns=fields,
    )

    required_columns = {
        "cik",
        "name",
        "ticker",
        "exchange",
    }

    missing = required_columns - set(
        companies.columns
    )

    if missing:
        raise ValueError(
            "SEC ticker file is missing columns: "
            f"{sorted(missing)}"
        )

    companies["cik"] = (
        companies["cik"]
        .map(normalize_cik)
        .astype("string")
    )

    companies["ticker"] = (
        companies["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    companies["company_name"] = (
        companies["name"]
        .astype("string")
        .str.strip()
    )

    companies["exchange"] = (
        companies["exchange"]
        .astype("string")
        .str.strip()
    )

    return (
        companies[
            [
                "cik",
                "ticker",
                "company_name",
                "exchange",
            ]
        ]
        .dropna(
            subset=[
                "cik",
                "ticker",
                "company_name",
                "exchange",
            ]
        )
        .drop_duplicates(
            subset=["cik"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def load_excluded_bankruptcy_ciks(
    event_path: str | Path,
) -> set[str]:
    """
    Load every valid LoPucki bankruptcy CIK.
    """
    path = Path(event_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Bankruptcy events not found: {path}"
        )

    events = pd.read_csv(
        path,
        dtype={"cik": "string"},
    )

    if "cik" not in events.columns:
        raise ValueError(
            "Bankruptcy event table has no CIK column."
        )

    return set(
        events["cik"]
        .dropna()
        .map(normalize_cik)
    )


def select_control_candidates(
    companies: pd.DataFrame,
    excluded_ciks: set[str],
    candidate_count: int = 200,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """
    Select listed companies with no LoPucki bankruptcy record.

    This is a candidate pool. Firms without sufficient quarterly
    SEC history will be removed after running the SEC pipeline.
    """
    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    eligible = companies[
        companies["exchange"].isin(
            ALLOWED_EXCHANGES
        )
        & ~companies["cik"].isin(
            excluded_ciks
        )
    ].copy()

    # Avoid unusual ticker types commonly associated with
    # units, warrants, and preferred listings.
    eligible = eligible[
        ~eligible["ticker"].str.contains(
            r"[\^\+=/]",
            regex=True,
            na=False,
        )
    ]

    eligible = eligible.drop_duplicates(
        subset=["cik"],
        keep="first",
    )

    if len(eligible) < candidate_count:
        candidate_count = len(eligible)

    return (
        eligible.sample(
            n=candidate_count,
            random_state=random_seed,
        )
        .sort_values(
            ["exchange", "ticker"]
        )
        .reset_index(drop=True)
    )


def create_company_config(
    candidates: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Convert control candidates to the SEC downloader format.
    """
    records: list[dict[str, str]] = []

    for row in candidates.itertuples(
        index=False
    ):
        records.append(
            {
                "ticker": str(row.ticker),
                "name": str(row.company_name),
                "cik": normalize_cik(row.cik),
                "sector": (
                    f"Control candidate - "
                    f"{row.exchange}"
                ),
            }
        )

    return records


def save_company_config(
    records: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            records,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path