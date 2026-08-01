from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ELIGIBLE_CHAPTERS = {
    "7",
    "11",
}


def load_bankruptcy_events(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the normalized LoPucki bankruptcy-event table.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Bankruptcy event file not found: {file_path}"
        )

    events = pd.read_csv(
        file_path,
        parse_dates=["event_date"],
        dtype={
            "cik": "string",
            "company_name": "string",
            "common_name": "string",
            "chapter": "string",
            "sic_description": "string",
            "source_case_id": "string",
        },
    )

    required_columns = {
        "source_case_id",
        "cik",
        "company_name",
        "event_date",
        "chapter",
        "sic_description",
    }

    missing = required_columns - set(events.columns)

    if missing:
        raise ValueError(
            "Bankruptcy event table is missing columns: "
            f"{sorted(missing)}"
        )

    return events


def build_eligible_bankruptcy_universe(
    events: pd.DataFrame,
    minimum_event_date: str = "2010-01-01",
    maximum_event_date: str = "2022-12-31",
) -> pd.DataFrame:
    """
    Select bankruptcy firms suitable for SEC financial-data matching.

    Rules:
    - valid CIK;
    - Chapter 7 or Chapter 11;
    - event falls within the selected date range;
    - one event per CIK, using the earliest qualifying filing.
    """
    data = events.copy()

    minimum_date = pd.Timestamp(
        minimum_event_date
    )

    maximum_date = pd.Timestamp(
        maximum_event_date
    )

    data["chapter"] = (
        data["chapter"]
        .astype("string")
        .str.strip()
    )

    eligible = data[
        data["cik"].notna()
        & data["chapter"].isin(ELIGIBLE_CHAPTERS)
        & data["event_date"].between(
            minimum_date,
            maximum_date,
            inclusive="both",
        )
    ].copy()

    eligible = (
        eligible.sort_values(
            [
                "cik",
                "event_date",
            ]
        )
        .drop_duplicates(
            subset=["cik"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return eligible


def select_initial_bankruptcy_sample(
    eligible_events: pd.DataFrame,
    sample_size: int = 100,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """
    Select an initial reproducible bankruptcy sample.

    Events are sampled across filing years rather than taking only
    the earliest or latest cases.
    """
    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    if len(eligible_events) <= sample_size:
        return eligible_events.copy()

    working = eligible_events.copy()

    working["event_year"] = (
        working["event_date"].dt.year
    )

    sampled_parts: list[pd.DataFrame] = []

    year_counts = (
        working["event_year"]
        .value_counts()
        .sort_index()
    )

    for year, year_count in year_counts.items():
        year_data = working[
            working["event_year"] == year
        ]

        allocated = max(
            1,
            round(
                sample_size
                * year_count
                / len(working)
            ),
        )

        allocated = min(
            allocated,
            len(year_data),
        )

        sampled_parts.append(
            year_data.sample(
                n=allocated,
                random_state=(
                    random_seed + int(year)
                ),
            )
        )

    sampled = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    if len(sampled) > sample_size:
        sampled = sampled.sample(
            n=sample_size,
            random_state=random_seed,
        )

    elif len(sampled) < sample_size:
        remaining = working[
            ~working["cik"].isin(
                sampled["cik"]
            )
        ]

        additional_count = min(
            sample_size - len(sampled),
            len(remaining),
        )

        additional = remaining.sample(
            n=additional_count,
            random_state=random_seed,
        )

        sampled = pd.concat(
            [sampled, additional],
            ignore_index=True,
        )

    return (
        sampled.sort_values(
            "event_date"
        )
        .reset_index(drop=True)
    )


def create_sec_company_config(
    events: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Convert bankruptcy events into the format used by the SEC
    Company Facts downloader.

    Matching later will rely on CIK rather than ticker.
    """
    records: list[dict[str, str]] = []

    for row in events.itertuples(
        index=False
    ):
        cik = str(row.cik).zfill(10)

        company_name = (
            str(row.company_name).strip()
        )

        sector = (
            str(row.sic_description).strip()
            if pd.notna(row.sic_description)
            else "Unknown"
        )

        records.append(
            {
                "ticker": f"CIK{cik}",
                "name": company_name,
                "cik": cik,
                "sector": sector,
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