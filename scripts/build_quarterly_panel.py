from __future__ import annotations

from pathlib import Path

from src.data.quarterly_panel import (
    build_quarterly_financial_panel,
    load_financial_facts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACTS_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_financial_facts_long.csv"
)

PANEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_quarterly_financial_panel.csv"
)


def main() -> None:
    print("Building quarterly SEC financial panel")
    print("--------------------------------------")
    print(f"Input: {FACTS_INPUT_PATH}")
    print()

    facts = load_financial_facts(
        FACTS_INPUT_PATH
    )

    panel = build_quarterly_financial_panel(
        facts
    )

    if panel.empty:
        raise RuntimeError(
            "No quarterly financial observations "
            "were constructed."
        )

    PANEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_csv(
        PANEL_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Companies: "
        f"{panel['ticker'].nunique()}"
    )

    print(
        f"Company-quarter rows: "
        f"{len(panel):,}"
    )

    print(
        f"First quarter end: "
        f"{panel['end_date'].min().date()}"
    )

    print(
        f"Last quarter end: "
        f"{panel['end_date'].max().date()}"
    )

    print()
    print("Rows by company")
    print("---------------")

    counts = (
        panel.groupby("ticker")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(counts.to_string())

    print()
    print("Non-missing coverage")
    print("--------------------")

    identifier_columns = {
        "ticker",
        "cik",
        "entity_name",
        "fiscal_year",
        "fiscal_period",
        "quarter_number",
        "end_date",
    }

    financial_columns = [
        column
        for column in panel.columns
        if column not in identifier_columns
    ]

    coverage = (
        panel[financial_columns]
        .notna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        coverage.map(
            lambda value: f"{value:.1%}"
        ).to_string()
    )

    print()
    print(f"Panel saved to: {PANEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()