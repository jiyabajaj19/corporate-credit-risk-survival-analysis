from pathlib import Path

from src.data.quarterly_panel import (
    DURATION_CONCEPTS,
    build_duration_quarterly_long,
    classify_fact_periods,
    load_financial_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_financial_facts_long.csv"
)


def main() -> None:
    facts = load_financial_facts(
        FACTS_PATH
    )

    duration = facts[
        facts["concept"].isin(
            DURATION_CONCEPTS
        )
    ].copy()

    classified = classify_fact_periods(
        duration
    )

    quarterly = (
        build_duration_quarterly_long(
            facts
        )
    )

    print("Duration-fact diagnostics")
    print("-------------------------")
    print(
        f"Raw duration observations: "
        f"{len(duration):,}"
    )
    print(
        "Observations with start dates: "
        f"{duration['start_date'].notna().sum():,}"
    )
    print(
        "Observations with SEC frames: "
        f"{duration['frame'].notna().sum():,}"
    )
    print(
        "Direct-quarter candidates: "
        f"{classified['is_direct_quarter'].sum():,}"
    )
    print(
        "Half-year YTD candidates: "
        f"{classified['is_half_year_ytd'].sum():,}"
    )
    print(
        "Nine-month YTD candidates: "
        f"{classified['is_nine_month_ytd'].sum():,}"
    )
    print(
        "Annual candidates: "
        f"{classified['is_annual_duration'].sum():,}"
    )
    print(
        "Quarterly flow records created: "
        f"{len(quarterly):,}"
    )

    if not quarterly.empty:
        print()
        print("Quarterly records by concept")
        print("----------------------------")

        print(
            quarterly.groupby("concept")
            .size()
            .sort_values(
                ascending=False
            )
            .to_string()
        )


if __name__ == "__main__":
    main()