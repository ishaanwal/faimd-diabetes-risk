import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"

# expected value ranges based on the documentation
RANGES = {
    # the target variable
    "Diabetes_012": (0, 2),

    # mostly binary indicators
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),

    # Ordinal features
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),

    # Continuous, range just decided
    "BMI": (10, 80)
}

PLOT_COLS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Income"]

def ensure_dirs():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

def range_violations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        s = df[col]
        bad_mask = (s < lo) | (s > hi)
        rows.append({
            "variable": col,
            "expected_min": lo,
            "expected_max": hi,
            "n_violations": int(bad_mask.sum()),
            "pct_violations": float(bad_mask.mean() * 100),
            "min_observed": float(s.min()),
            "max_observed": float(s.max())
        })
    return pd.DataFrame(rows).sort_values("n_violations", ascending=False)

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "n_rows": [len(df)],
        "n_columns": [df.shape[1]],
        "n_missing_total": [int(df.isna().sum().sum())],
        "n_duplicates": [int(df.duplicated().sum())]
    })

def save_histograms(df: pd.DataFrame):
    for col in PLOT_COLS:
        if col not in df.columns:
            continue
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / f"dist_{col}.png", dpi=200)
        plt.close()

def main():
    ensure_dirs()

    df_raw = pd.read_csv(DATA_PATH)

    # record duplicates numbe
    dup_count = int(df_raw.duplicated().sum())

    # drop them
    df = df_raw.drop_duplicates().reset_index(drop=True)

    # tables for report
    summ = summary_table(df_raw)
    summ.loc[0, "n_duplicates"] = dup_count
    summ.to_csv(OUT_TABLES / "data_quality_summary.csv", index=False)

    viol = range_violations(df)
    viol.to_csv(OUT_TABLES / "range_violations.csv", index=False)

    print("Saved:", OUT_TABLES / "data_quality_summary.csv")
    print("Saved:", OUT_TABLES / "range_violations.csv")
    print("\nTop range violations:")
    print(viol.head(10))

    # Figures
    save_histograms(df)
    print("Saved distribution plots to:", OUT_FIGS)

if __name__ == "__main__":
    main()
