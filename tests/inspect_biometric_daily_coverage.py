from pathlib import Path
import pandas as pd

path = Path(
    "data/raw/biometric/api_data_aadhar_biometric/"
    "api_data_aadhar_biometric_1500000_1861108.csv"
)

df = pd.read_csv(path)

df["date"] = pd.to_datetime(
    df["date"], dayfirst=True, errors="coerce"
)

activity_cols = ["bio_age_5_17", "bio_age_17_"]
for col in activity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["total_activity"] = df[activity_cols].sum(axis=1)

daily = (
    df.dropna(subset=["date"])
    .groupby("date")
    .agg(
        rows=("date", "size"),
        total_activity=("total_activity", "sum"),
    )
    .sort_index()
)

window = daily.loc["2025-12-15":"2025-12-29"]

print("\nDaily biometric coverage: Dec 15–29, 2025")
print(window.to_string())

print("\nOverall source-file summary")
print("Total rows:", len(df))
print("Rows with invalid dates:", df["date"].isna().sum())
print("Observed dates:", daily.shape[0])
print("Median rows per observed date:", daily["rows"].median())
print("Median activity per observed date:",
      daily["total_activity"].median())