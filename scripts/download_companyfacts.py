from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data.sec_api import (
    SecApiError,
    build_sec_headers,
    download_company_facts,
    load_company_universe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "companies.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec_companyfacts"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SEC Company Facts JSON "
            "for the configured company universe."
        )
    )

    parser.add_argument(
        "--email",
        required=True,
        help=(
            "Contact email included in the "
            "SEC User-Agent header."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the company JSON file.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for raw SEC JSON files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help=(
            "Delay in seconds between company "
            "requests."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.delay < 0:
        raise ValueError(
            "--delay cannot be negative."
        )

    companies = load_company_universe(
        arguments.config
    )

    headers = build_sec_headers(
        contact_email=arguments.email
    )

    successful_downloads = 0
    failures: list[tuple[str, str]] = []

    print("SEC Company Facts downloader")
    print("----------------------------")
    print(f"Companies: {len(companies)}")
    print(
        "Output directory: "
        f"{arguments.output_directory}"
    )
    print()

    for index, company in enumerate(
        companies,
        start=1,
    ):
        print(
            f"[{index}/{len(companies)}] "
            f"{company.ticker}: ",
            end="",
        )

        try:
            output_path = download_company_facts(
                company=company,
                output_directory=(
                    arguments.output_directory
                ),
                headers=headers,
                overwrite=arguments.overwrite,
            )

            successful_downloads += 1
            print(f"saved to {output_path.name}")

        except SecApiError as error:
            failures.append(
                (
                    company.ticker,
                    str(error),
                )
            )
            print(f"FAILED — {error}")

        if index < len(companies):
            time.sleep(arguments.delay)

    print()
    print("Download summary")
    print("----------------")
    print(
        f"Successful: {successful_downloads}"
    )
    print(f"Failed: {len(failures)}")

    if failures:
        print()

        for ticker, error_message in failures:
            print(
                f"{ticker}: {error_message}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()