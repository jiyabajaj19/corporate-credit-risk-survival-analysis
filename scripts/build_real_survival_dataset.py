from __future__ import annotations

from pathlib import Path

from src.survival.combined_dataset import (
    combine_survival_datasets,
    load_survival_table,
    validate_combined_survival_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANKRUPT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bankrupt_survival_dataset.csv"
)

CONTROL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "control_survival_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_survival_dataset.csv"
)


def main() -> None:
    bankrupt = load_survival_table(
        BANKRUPT_PATH
    )

    controls = load_survival_table(
        CONTROL_PATH
    )

    combined = combine_survival_datasets(
        bankrupt=bankrupt,
        controls=controls,
    )

    validate_combined_survival_dataset(
        combined
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    firm_summary = (
        combined[
            [
                "firm_id",
                "sample_group",
            ]
        ]
        .drop_duplicates()
        ["sample_group"]
        .value_counts()
    )

    print("Combined real survival dataset")
    print("------------------------------")
    print(
        f"Companies: "
        f"{combined['firm_id'].nunique():,}"
    )
    print(
        f"Start-stop rows: "
        f"{len(combined):,}"
    )
    print(
        f"Bankruptcy events: "
        f"{combined['event'].sum():,}"
    )
    print()
    print("Companies by group")
    print("------------------")
    print(firm_summary.to_string())
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()