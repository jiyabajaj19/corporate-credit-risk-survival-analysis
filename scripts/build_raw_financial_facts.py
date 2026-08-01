from __future__ import annotations

from pathlib import Path

from src.data.companyfacts import (
    create_fact_coverage_summary,
    extract_directory_financial_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_companyfacts"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FACTS_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "sec_financial_facts_long.csv"
)

COVERAGE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "sec_fact_coverage.csv"
)


def main() -> None:
    print("Building SEC financial-facts table")
    print("----------------------------------")
    print(f"Input: {INPUT_DIRECTORY}")
    print()

    facts = extract_directory_financial_facts(
        INPUT_DIRECTORY
    )

    if facts.empty:
        raise RuntimeError(
            "No supported SEC financial facts "
            "were extracted."
        )

    coverage = create_fact_coverage_summary(
        facts
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    facts.to_csv(
        FACTS_OUTPUT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Companies: "
        f"{facts['ticker'].nunique()}"
    )

    print(
        f"Standardized concepts: "
        f"{facts['concept'].nunique()}"
    )

    print(
        f"Extracted observations: "
        f"{len(facts):,}"
    )

    print()
    print("Observations by company")
    print("-----------------------")

    company_counts = (
        facts.groupby("ticker")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(company_counts.to_string())

    print()
    print("Concept coverage")
    print("----------------")

    concept_coverage = (
        facts.groupby("concept")["ticker"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(concept_coverage.to_string())

    print()
    print(f"Facts saved to: {FACTS_OUTPUT_PATH}")
    print(
        f"Coverage saved to: "
        f"{COVERAGE_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()