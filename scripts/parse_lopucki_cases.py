from __future__ import annotations

from pathlib import Path

from src.data.lopucki import (
    build_bankruptcy_events,
    create_event_quality_summary,
    load_lopucki_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lopucki"
    / "Cases.xlsx"
)

EVENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lopucki_bankruptcy_events.csv"
)

QUALITY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lopucki_bankruptcy_event_quality.csv"
)


def main() -> None:
    print("Parsing LoPucki bankruptcy cases")
    print("--------------------------------")
    print(f"Input: {INPUT_PATH}")
    print()

    cases = load_lopucki_cases(
        INPUT_PATH
    )

    events = build_bankruptcy_events(
        cases
    )

    quality = create_event_quality_summary(
        events
    )

    EVENT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        EVENT_OUTPUT_PATH,
        index=False,
    )

    quality.to_csv(
        QUALITY_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Raw cases: {len(cases):,}"
    )

    print(
        f"Normalized events: {len(events):,}"
    )

    print(
        "Events with valid CIK: "
        f"{events['cik'].notna().sum():,}"
    )

    print(
        "Unique valid CIKs: "
        f"{events['cik'].dropna().nunique():,}"
    )

    print(
        f"First filing date: "
        f"{events['event_date'].min().date()}"
    )

    print(
        f"Last filing date: "
        f"{events['event_date'].max().date()}"
    )

    print()
    print("Chapter counts")
    print("--------------")

    print(
        events["chapter"]
        .value_counts(
            dropna=False
        )
        .head(10)
        .to_string()
    )

    print()
    print("Field coverage")
    print("--------------")

    printable = quality.copy()

    printable["coverage"] = (
        printable["coverage"].map(
            lambda value: f"{value:.1%}"
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()
    print(
        f"Events saved to: "
        f"{EVENT_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()