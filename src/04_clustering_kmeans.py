import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
FEATURE_SETS_PATH = PROJECT_ROOT / "outputs" / "tables" / "feature_sets.json"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"

RANDOM_STATE = 42
SUBSAMPLE_SIZE = 30000          # speed fix
K_RANGE = range(2, 9)           # 2–8

def load_feature_sets():
    with open(FEATURE_SETS_PATH, "r") as f:
        return json.load(f)

def choose_k_with_subsample(X_scaled, tag):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=SUBSAMPLE_SIZE, replace=False)
    X_sub = X_scaled[idx]

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_sub)
        sil = silhouette_score(X_sub, labels)
        rows.append({"k": k, "silhouette": float(sil)})

    scores = pd.DataFrame(rows)

    # silhouette plot (subsample)
    plt.figure()
    plt.plot(scores["k"], scores["silhouette"], marker="o")
    plt.title(f"Silhouette (subsample) – {tag}")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / f"kmeans_silhouette_{tag}.png", dpi=200)
    plt.close()

    return scores

def cluster_profiles(X, labels):
    dfp = X.copy()
    dfp["cluster"] = labels
    profile = dfp.groupby("cluster").mean().round(3)
    sizes = dfp["cluster"].value_counts().sort_index()
    profile.insert(0, "cluster_size", sizes.values)
    return profile

def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    # Load + clean
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    # Targets (not used for clustering, but saved for later prevalence analysis)
    df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)

    feature_sets = load_feature_sets()

    configs = {
        "all_features": feature_sets["clustering_all_features"],
        "no_demographics": feature_sets["clustering_no_demographics"]
    }

    all_scores = []
    assignments = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "Diabetes_012": df["Diabetes_012"].values,
        "Diabetes_binary": df["Diabetes_binary"].values
    })

    for tag, cols in configs.items():
        print(f"\nRunning clustering for: {tag}")

        X = df[cols].copy()

        # Scale for K-Means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # STEP 1: choose k on subsample
        scores = choose_k_with_subsample(X_scaled, tag)
        scores["feature_set"] = tag
        all_scores.append(scores)

        best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
        print(f"[{tag}] selected k = {best_k}")

        # STEP 2: fit final model on FULL data
        km_final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
        labels_full = km_final.fit_predict(X_scaled)

        # Save cluster profiles (means per feature per cluster)
        profile = cluster_profiles(X, labels_full)
        profile.to_csv(OUT_TABLES / f"cluster_profile_{tag}_k{best_k}.csv", index=True)

        # Save row-level assignments so later scripts don't need to refit K-Means
        assignments[f"cluster_{tag}_k{best_k}"] = labels_full

    # Save silhouette scores (k selection evidence)
    df_scores = pd.concat(all_scores, ignore_index=True)
    df_scores.to_csv(OUT_TABLES / "kmeans_silhouette_scores.csv", index=False)

    # Save row-level cluster assignments
    assignments_path = OUT_TABLES / "cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    print("\nSaved:")
    print("- Silhouette plots:", OUT_FIGS)
    print("- Cluster profiles:", OUT_TABLES)
    print("- k selection table:", OUT_TABLES / "kmeans_silhouette_scores.csv")
    print("- Row-level assignments:", assignments_path)

if __name__ == "__main__":
    main()
