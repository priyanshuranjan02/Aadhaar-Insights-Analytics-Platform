from pathlib import Path
import pandas as pd

root = Path("data/raw/biometric")

for path in sorted(root.rglob("*.csv")):
    try:
        df = pd.read_csv(path, usecols=lambda c: c.strip().lower() == "date")
    except Exception as exc:
        print(f"Could not read {path.name}: {exc}")
        continue

    if df.empty or len(df.columns) == 0:
        print(f"{path.name}: no date rows")
        continue

    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    valid = dates.dropna()

    print(
        f"{path.name}: rows={len(df):,}, "
        f"valid_dates={len(valid):,}, "
        f"date_range="
        f"{valid.min().date() if not valid.empty else 'N/A'} to "
        f"{valid.max().date() if not valid.empty else 'N/A'}, "
        f"Dec22={(dates == pd.Timestamp('2025-12-22')).sum()}, "
        f"Dec23={(dates == pd.Timestamp('2025-12-23')).sum()}"
    )