import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"

OUT_FIGS.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    # binary target so only diabetic vs non diabetic
    df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)

    counts = df["Diabetes_binary"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Non-diabetic / Prediabetic", "Diabetic"],
        counts.values
    )
    plt.ylabel("Number of individuals")
    plt.title("Class distribution of Diabetes_binary")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "class_imbalance_diabetes_binary.png", dpi=200)
    plt.close()

    print("Saved:", OUT_FIGS / "class_imbalance_diabetes_binary.png")
    print("Class proportions:")
    print((counts / counts.sum()).round(3))

if __name__ == "__main__":
    main()
