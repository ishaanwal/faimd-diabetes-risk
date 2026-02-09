import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # First step: Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Second step: Create binary feature
    df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)

    # Third step: Separate the features from the target feature
    X = df.drop(columns=["Diabetes_012", "Diabetes_binary"])
    y = df["Diabetes_binary"]
    # Fourth step: THE stratisfied train n test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("the training shape:", X_train.shape)
    print("the test shape:", X_test.shape)
    print(y_train.value_counts(normalize=True))

    X_train.to_csv(PROJECT_ROOT / "data" / "X_train.csv", index=False)
    X_test.to_csv(PROJECT_ROOT / "data" / "X_test.csv", index=False)
    y_train.to_csv(PROJECT_ROOT / "data" / "y_train.csv", index=False)
    y_test.to_csv(PROJECT_ROOT / "data" / "y_test.csv", index=False)

if __name__ == "__main__":
    main()
