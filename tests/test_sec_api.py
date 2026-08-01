import json
from pathlib import Path

import pytest

from src.data.sec_api import (
    Company,
    build_sec_headers,
    load_company_universe,
    normalize_cik,
    save_company_facts,
)


def test_normalize_cik() -> None:
    assert normalize_cik(
        "320193"
    ) == "0000320193"

    assert normalize_cik(
        320193
    ) == "0000320193"


def test_invalid_cik_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="digits only",
    ):
        normalize_cik("ABC123")


def test_build_sec_headers() -> None:
    headers = build_sec_headers(
        contact_email="student@example.com"
    )

    assert "User-Agent" in headers
    assert (
        "student@example.com"
        in headers["User-Agent"]
    )


def test_invalid_email_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="valid contact email",
    ):
        build_sec_headers("invalid-email")


def test_load_company_universe(
    tmp_path: Path,
) -> None:
    config_path = (
        tmp_path / "companies.json"
    )

    config_path.write_text(
        json.dumps(
            [
                {
                    "ticker": "aapl",
                    "name": "Apple Inc.",
                    "cik": "320193",
                    "sector": "Technology",
                }
            ]
        ),
        encoding="utf-8",
    )

    companies = load_company_universe(
        config_path
    )

    assert companies == [
        Company(
            ticker="AAPL",
            name="Apple Inc.",
            cik="0000320193",
            sector="Technology",
        )
    ]


def test_save_company_facts(
    tmp_path: Path,
) -> None:
    payload = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {},
    }

    output_path = (
        tmp_path / "apple.json"
    )

    saved_path = save_company_facts(
        payload=payload,
        output_path=output_path,
    )

    assert saved_path.exists()

    loaded_payload = json.loads(
        saved_path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded_payload == payload