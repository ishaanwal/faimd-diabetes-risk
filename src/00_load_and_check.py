import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"  # rename if needed

def main():
    df = pd.read_csv(DATA_PATH)

    print("First information")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\n HEAD ")
    print(df.head())

    print("\nTop 20 missing values")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nNumber of Duplicates")
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
