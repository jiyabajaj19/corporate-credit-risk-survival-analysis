import json
from pathlib import Path

import pandas as pd

from src.data.companyfacts import (
    create_fact_coverage_summary,
    extract_company_financial_facts,
    infer_ticker_from_filename,
)


def create_example_payload() -> dict:
    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "Total assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-03-30",
                                "val": 337411000000,
                                "accn": "0000320193-test",
                                "fy": 2024,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2024-05-03",
                                "frame": "CY2024Q1I",
                            }
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "Net Income",
                    "description": "Net income",
                    "units": {
                        "USD": [
                            {
                                "start": "2023-12-31",
                                "end": "2024-03-30",
                                "val": 23636000000,
                                "accn": "0000320193-test",
                                "fy": 2024,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2024-05-03",
                                "frame": "CY2024Q1",
                            }
                        ]
                    },
                },
            }
        },
    }


def test_infer_ticker_from_filename() -> None:
    assert infer_ticker_from_filename(
        "AAPL_0000320193.json"
    ) == "AAPL"


def test_extract_company_financial_facts(
    tmp_path: Path,
) -> None:
    source_path = (
        tmp_path / "AAPL_0000320193.json"
    )

    source_path.write_text(
        json.dumps(create_example_payload()),
        encoding="utf-8",
    )

    facts = extract_company_financial_facts(
        source_path
    )

    assert set(facts["concept"]) == {
        "total_assets",
        "net_income",
    }

    assert set(facts["ticker"]) == {
        "AAPL"
    }

    assert pd.api.types.is_datetime64_any_dtype(
        facts["end_date"]
    )

    assert facts["value"].notna().all()


def test_create_fact_coverage_summary(
    tmp_path: Path,
) -> None:
    source_path = (
        tmp_path / "AAPL_0000320193.json"
    )

    source_path.write_text(
        json.dumps(create_example_payload()),
        encoding="utf-8",
    )

    facts = extract_company_financial_facts(
        source_path
    )

    coverage = create_fact_coverage_summary(
        facts
    )

    assert set(coverage["concept"]) == {
        "total_assets",
        "net_income",
    }

    assert (
        coverage["observation_count"] == 1
    ).all()