import pandas as pd
from pathlib import Path

# Resolve project root -> data folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"  # rename if needed

def main():
    df = pd.read_csv(DATA_PATH)

    print("=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== MISSING VALUES (TOP 20) ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\n=== DUPLICATES ===")
    print(df.duplicated().sum())

    # Try to identify target column
    for col in ["Diabetes_binary", "diabetes", "Diabetes", "Outcome"]:
        if col in df.columns:
            print(f"\n=== TARGET DISTRIBUTION: {col} ===")
            print(df[col].value_counts())
            print("\nProportion:")
            print(df[col].value_counts(normalize=True))

if __name__ == "__main__":
    main()
