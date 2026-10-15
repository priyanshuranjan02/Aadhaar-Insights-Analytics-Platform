from src.data_loader import load_all_datasets
from src.forecasting_data import build_daily_target_series


def main():
    datasets = load_all_datasets()

    for name, raw in datasets.items():
        daily = build_daily_target_series(raw, name)

        # Inspect the more recent period separately.
        recent = daily[
            daily["date"] >= "2025-09-01"
        ].copy()

        recent["gap_days"] = (
            recent["date"].diff().dt.days
        )

        print(f"\n{'=' * 55}")
        print(f"{name.upper()} — RECENT DATE COVERAGE")
        print(f"{'=' * 55}")
        print("Observed dates:", len(recent))
        print("First date:", recent["date"].min())
        print("Last date:", recent["date"].max())
        print("One-day gaps:", int((recent["gap_days"] == 1).sum()))
        print("Gaps of 2–7 days:", int(
            recent["gap_days"].between(2, 7).sum()
        ))
        print("Gaps over 7 days:", int(
            (recent["gap_days"] > 7).sum()
        ))
        print("Median gap (days):", recent["gap_days"].median())

        print("\nMonthly observations:")
        print(
            recent.groupby(recent["date"].dt.to_period("M"))
            .size()
            .to_string()
        )


if __name__ == "__main__":
    main()