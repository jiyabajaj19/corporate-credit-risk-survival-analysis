from __future__ import annotations

from pathlib import Path

from src.data.event_universe import (
    build_eligible_bankruptcy_universe,
    create_sec_company_config,
    load_bankruptcy_events,
    save_company_config,
    select_initial_bankruptcy_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lopucki_bankruptcy_events.csv"
)

ELIGIBLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eligible_bankruptcy_events.csv"
)

SAMPLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_bankruptcy_sample.csv"
)

CONFIG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "bankruptcy_companies.json"
)


def main() -> None:
    events = load_bankruptcy_events(
        EVENT_INPUT_PATH
    )

    eligible = (
        build_eligible_bankruptcy_universe(
            events,
            minimum_event_date="2010-01-01",
            maximum_event_date="2022-12-31",
        )
    )

    sample = select_initial_bankruptcy_sample(
        eligible,
        sample_size=100,
        random_seed=2026,
    )

    config_records = create_sec_company_config(
        sample
    )

    ELIGIBLE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    eligible.to_csv(
        ELIGIBLE_OUTPUT_PATH,
        index=False,
    )

    sample.to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
    )

    save_company_config(
        records=config_records,
        output_path=CONFIG_OUTPUT_PATH,
    )

    print("Bankruptcy company universe")
    print("---------------------------")
    print(
        f"All LoPucki events: "
        f"{len(events):,}"
    )
    print(
        f"Eligible 2010–2022 events: "
        f"{len(eligible):,}"
    )
    print(
        f"Initial download sample: "
        f"{len(sample):,}"
    )
    print(
        f"Unique sample CIKs: "
        f"{sample['cik'].nunique():,}"
    )
    print(
        f"First sample event: "
        f"{sample['event_date'].min().date()}"
    )
    print(
        f"Last sample event: "
        f"{sample['event_date'].max().date()}"
    )

    print()
    print("Sample events by year")
    print("---------------------")

    print(
        sample["event_date"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(f"SEC config saved to: {CONFIG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()