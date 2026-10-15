from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_v2"
    / "sensitivity"
    / "matched_origin"
    / "four_scenario_predictions_common.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_v2"
    / "sensitivity"
    / "matched_origin"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df["Target Date"] = pd.to_datetime(df["Target Date"])
df = df.sort_values("Target Date")

# One actual activity value per date.
actual = (
    df[["Target Date", "Actual"]]
    .drop_duplicates()
    .sort_values("Target Date")
)

# Pivot predictions into separate columns.
predictions = df.pivot(
    index="Target Date",
    columns="Scenario",
    values="Predicted",
).reset_index()

plot_df = actual.merge(predictions, on="Target Date", validate="one_to_one")

# Chart 1: actual versus all four predictions.
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    plot_df["Target Date"],
    plot_df["Actual"],
    marker="o",
    label="Actual activity",
)

ax.plot(
    plot_df["Target Date"],
    plot_df["A_original"],
    linestyle="--",
    label="Original Linear Regression",
)

ax.plot(
    plot_df["Target Date"],
    plot_df["B_exclusion"],
    linestyle="--",
    label="Exclusion Linear Regression",
)

ax.plot(
    plot_df["Target Date"],
    plot_df["C_persistence_original"],
    alpha=0.8,
    label="Original Persistence",
)

ax.plot(
    plot_df["Target Date"],
    plot_df["D_persistence_exclusion"],
    alpha=0.8,
    label="Exclusion Persistence",
)

ax.set_title("Biometric Activity: Actual vs One-Step-Ahead Predictions")
ax.set_xlabel("Target date")
ax.set_ylabel("Total activity")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "actual_vs_predictions.png", dpi=160)
plt.close(fig)

# Chart 2: absolute error by date.
error_df = df.pivot(
    index="Target Date",
    columns="Scenario",
    values="Absolute Error",
).sort_index()

fig, ax = plt.subplots(figsize=(15, 6))

for scenario in error_df.columns:
    ax.plot(
        error_df.index,
        error_df[scenario],
        marker=".",
        label=scenario,
    )

ax.set_title("Absolute Prediction Error by Target Date")
ax.set_xlabel("Target date")
ax.set_ylabel("Absolute error")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "absolute_error_by_date.png", dpi=160)
plt.close(fig)

print("Visualization complete.")
print(f"Saved: {OUTPUT_DIR / 'actual_vs_predictions.png'}")
print(f"Saved: {OUTPUT_DIR / 'absolute_error_by_date.png'}")