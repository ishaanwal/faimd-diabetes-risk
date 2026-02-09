import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
RANDOM_STATE = 42
def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)
    X = df.drop(columns=["Diabetes_012", "Diabetes_binary"])
    y = df["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Base RF with class_weight to handle imbalance
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    # Light randomized search (fast, good enough for coursework)
    param_dist = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1",          # focus on minority performance
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV F1:", round(search.best_score_, 4))
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    print("\n=== Random Forest (tuned) ===")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    # Save
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = y_pred
    out["p_diabetes"] = y_prob
    out_path = PROJECT_ROOT / "outputs" / "tables" / "test_set_predictions_rf.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    # Save feature importances (helpful for report)
    importances = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp_path = PROJECT_ROOT / "outputs" / "tables" / "rf_feature_importances.csv"
    importances.to_csv(imp_path, header=["importance"])
    print("Saved:", imp_path)

if __name__ == "__main__":
    main()
