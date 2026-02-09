import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
TARGET_ORIGINAL = "Diabetes_012"
TARGET_BINARY = "Diabetes_binary"
DEMOGRAPHICS = ["Sex", "Age", "Education", "Income"]
def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df[TARGET_BINARY] = (df[TARGET_ORIGINAL] > 0).astype(int)
    # All the features without the binary diabetes target feature
    all_features = [c for c in df.columns if c not in [TARGET_ORIGINAL, TARGET_BINARY]]
    # groups are split for experimentation
    demographics = [c for c in DEMOGRAPHICS if c in df.columns]
    health_lifestyle = [c for c in all_features if c not in demographics]
    # feature sets, by task
    feature_sets = {
        "classification_all_features": all_features,
        "clustering_all_features": all_features,
        "clustering_no_demographics": health_lifestyle,
        "demographics_only": demographics,
        "health_lifestyle_only": health_lifestyle,
        "targets": [TARGET_ORIGINAL, TARGET_BINARY],
    }
    json_path = OUT_TABLES / "feature_sets.json"
    with open(json_path, "w") as f:
        json.dump(feature_sets, f, indent=2)
    rows = []
    for set_name, cols in feature_sets.items():
        if isinstance(cols, list):
            for col in cols:
                rows.append({"set": set_name, "feature": col})
    csv_path = OUT_TABLES / "feature_sets.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("\nFeature set counts:")
    for k, v in feature_sets.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)}")

if __name__ == "__main__":
    main()
