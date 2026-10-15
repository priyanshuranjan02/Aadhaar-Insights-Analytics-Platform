from pathlib import Path
import pandas as pd

root = Path("data")
targets = {"22-12-2025", "23-12-2025"}

for path in root.rglob("*"):
    if (
        not path.is_file()
        or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}
        or "processed" in path.parts
    ):
        continue

    try:
        df = (
            pd.read_csv(path)
            if path.suffix.lower() == ".csv"
            else pd.read_excel(path)
        )
    except Exception:
        continue

    date_col = next(
        (c for c in df.columns if c.strip().lower() == "date"),
        None,
    )
    if date_col is None:
        continue

    dates = pd.to_datetime(
        df[date_col], dayfirst=True, errors="coerce"
    )
    selected = df.loc[dates.dt.strftime("%d-%m-%Y").isin(targets)]

    if selected.empty:
        continue

    activity_cols = [
        c for c in selected.columns
        if c.strip().lower() in {"bio_age_5_17", "bio_age_17_"}
    ]
    if not activity_cols:
        continue

    totals = selected[activity_cols].apply(
        pd.to_numeric, errors="coerce"
    ).sum()

    print(f"\nFile: {path}")
    print(f"Rows for target dates: {len(selected)}")
    print(
        dates.loc[selected.index]
        .dt.strftime("%Y-%m-%d")
        .value_counts()
        .sort_index()
    )
    print("Activity totals:", totals.to_dict())