import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

RANDOM_STATE = 42

def main():
    # Load + clean
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    # Binary target: prediabetes OR diabetes -> 1
    df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)

    X = df.drop(columns=["Diabetes_012", "Diabetes_binary"])
    y = df["Diabetes_binary"]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline model: Logistic Regression with class weights for imbalance
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("=== Logistic Regression (class_weight=balanced) ===")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    # Save probabilities for “probability of being diabetic for each case”
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = y_pred
    out["p_diabetes"] = y_prob
    out_path = PROJECT_ROOT / "outputs" / "tables" / "test_set_predictions_logreg.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
