from __future__ import annotations

import argparse
from pathlib import Path

from src.data.control_universe import (
    create_company_config,
    download_exchange_tickers,
    load_exchange_tickers,
    load_excluded_bankruptcy_ciks,
    save_company_config,
    select_control_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKER_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_reference"
    / "company_tickers_exchange.json"
)

BANKRUPTCY_EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lopucki_bankruptcy_events.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "control_candidates.csv"
)

CONFIG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "control_candidates.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-bankrupt SEC-listed "
            "control candidate universe."
        )
    )

    parser.add_argument(
        "--email",
        required=True,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--overwrite-tickers",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    download_exchange_tickers(
        contact_email=arguments.email,
        output_path=TICKER_PATH,
        overwrite=arguments.overwrite_tickers,
    )

    companies = load_exchange_tickers(
        TICKER_PATH
    )

    excluded_ciks = (
        load_excluded_bankruptcy_ciks(
            BANKRUPTCY_EVENT_PATH
        )
    )

    candidates = select_control_candidates(
        companies=companies,
        excluded_ciks=excluded_ciks,
        candidate_count=(
            arguments.candidate_count
        ),
        random_seed=2026,
    )

    records = create_company_config(
        candidates
    )

    CANDIDATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
    )

    save_company_config(
        records,
        CONFIG_OUTPUT_PATH,
    )

    overlap = set(
        candidates["cik"]
    ).intersection(excluded_ciks)

    print("Control candidate universe")
    print("--------------------------")
    print(
        f"Listed SEC companies: "
        f"{len(companies):,}"
    )
    print(
        f"LoPucki CIKs excluded: "
        f"{len(excluded_ciks):,}"
    )
    print(
        f"Control candidates: "
        f"{len(candidates):,}"
    )
    print(
        f"Bankruptcy overlap: "
        f"{len(overlap)}"
    )

    print()
    print("Candidates by exchange")
    print("----------------------")
    print(
        candidates["exchange"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Config saved to: "
        f"{CONFIG_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()