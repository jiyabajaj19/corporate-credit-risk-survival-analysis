from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/"
    "companyfacts/CIK{cik}.json"
)


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    cik: str
    sector: str


class SecApiError(RuntimeError):
    """Raised when an SEC API request cannot be completed."""


def normalize_cik(cik: str | int) -> str:
    """
    Return a CIK as a ten-character, zero-padded string.
    """
    cik_text = str(cik).strip()

    if not cik_text.isdigit():
        raise ValueError(
            f"CIK must contain digits only: {cik!r}"
        )

    if len(cik_text) > 10:
        raise ValueError(
            f"CIK cannot exceed 10 digits: {cik!r}"
        )

    return cik_text.zfill(10)


def load_company_universe(
    config_path: str | Path,
) -> list[Company]:
    """
    Load companies from a JSON configuration file.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Company configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Company configuration must contain a list."
        )

    companies: list[Company] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Company record {index} must be an object."
            )

        required_fields = {
            "ticker",
            "name",
            "cik",
            "sector",
        }

        missing = required_fields - set(record)

        if missing:
            raise ValueError(
                f"Company record {index} is missing: "
                f"{sorted(missing)}"
            )

        companies.append(
            Company(
                ticker=str(
                    record["ticker"]
                ).strip().upper(),
                name=str(record["name"]).strip(),
                cik=normalize_cik(record["cik"]),
                sector=str(
                    record["sector"]
                ).strip(),
            )
        )

    return companies


def build_sec_headers(
    contact_email: str,
    application_name: str = (
        "CorporateCreditRiskSurvivalAnalysis"
    ),
) -> dict[str, str]:
    """
    Build descriptive HTTP headers for SEC requests.

    Replace the placeholder email with your real contact email.
    """
    cleaned_email = contact_email.strip()

    if (
        not cleaned_email
        or "@" not in cleaned_email
    ):
        raise ValueError(
            "A valid contact email is required."
        )

    return {
        "User-Agent": (
            f"{application_name} "
            f"{cleaned_email}"
        ),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def request_company_facts(
    cik: str | int,
    headers: dict[str, str],
    timeout_seconds: float = 30.0,
    maximum_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """
    Download one Company Facts response from the SEC.

    Requests are retried for temporary server or network errors.
    """
    normalized_cik = normalize_cik(cik)

    url = SEC_COMPANYFACTS_URL.format(
        cik=normalized_cik
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout_seconds,
            )

            if response.status_code == 200:
                payload = response.json()

                if not isinstance(payload, dict):
                    raise SecApiError(
                        "SEC response was not a JSON object."
                    )

                return payload

            if response.status_code == 404:
                raise SecApiError(
                    "Company Facts not found for "
                    f"CIK {normalized_cik}."
                )

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise SecApiError(
                    "Temporary SEC response: "
                    f"HTTP {response.status_code}"
                )

            raise SecApiError(
                "SEC request failed with "
                f"HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        except (
            requests.RequestException,
            ValueError,
            SecApiError,
        ) as error:
            last_error = error

            if attempt == maximum_attempts:
                break

            time.sleep(
                retry_delay_seconds * attempt
            )

    raise SecApiError(
        "Unable to download Company Facts for "
        f"CIK {normalized_cik} after "
        f"{maximum_attempts} attempts."
    ) from last_error


def save_company_facts(
    payload: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save a Company Facts response as formatted JSON.
    """
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)

    return path


def download_company_facts(
    company: Company,
    output_directory: str | Path,
    headers: dict[str, str],
    overwrite: bool = False,
) -> Path:
    """
    Download and cache Company Facts for one company.
    """
    directory = Path(output_directory)

    output_path = directory / (
        f"{company.ticker}_{company.cik}.json"
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        return output_path

    payload = request_company_facts(
        cik=company.cik,
        headers=headers,
    )

    returned_cik = normalize_cik(
        payload.get("cik", company.cik)
    )

    if returned_cik != company.cik:
        raise SecApiError(
            "Returned CIK does not match the "
            f"requested company: {company.ticker}"
        )

    return save_company_facts(
        payload=payload,
        output_path=output_path,
    )