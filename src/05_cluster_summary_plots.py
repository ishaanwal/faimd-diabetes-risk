import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
ASSIGN_PATH = PROJECT_ROOT / "outputs" / "tables" / "cluster_assignments.csv"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Main clustering to interpret
CLUSTER_COL = "cluster_no_demographics_k4"

# Features used to describe cluster meaning (no-demographics set)
PROFILE_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk"
]

def load_merged():
    # Assignments should at least contain row_id and cluster col
    assign = pd.read_csv(ASSIGN_PATH)

    if "row_id" not in assign.columns:
        raise ValueError("cluster_assignments.csv must contain 'row_id' column.")
    if CLUSTER_COL not in assign.columns:
        raise ValueError(
            f"'{CLUSTER_COL}' not found in cluster_assignments.csv. "
            f"Available: {[c for c in assign.columns if c.startswith('cluster_')]}"
        )

    # Load dataset in the same order as clustering (dedup + reset index)
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    # Merge cluster labels onto df
    df = df.merge(assign[["row_id", CLUSTER_COL]], on="row_id", how="inner")
    df = df.rename(columns={CLUSTER_COL: "cluster"})

    # Ensure Diabetes_012 exists from the dataset (source of truth)
    if "Diabetes_012" not in df.columns:
        raise ValueError("Diabetes_012 not found in diabetes.csv after loading. Check your dataset file.")

    return df

def plot_stacked_prevalence(df):
    # A) Non-diabetic (0) vs Pre+Diab (>0)
    df["cat_non"] = (df["Diabetes_012"] == 0).astype(int)
    df["cat_pre_or_diab"] = (df["Diabetes_012"] > 0).astype(int)

    # B) Non+Pre (0 or 1) vs Diab only (2)
    df["cat_non_or_pre"] = (df["Diabetes_012"].isin([0, 1])).astype(int)
    df["cat_diab_only"] = (df["Diabetes_012"] == 2).astype(int)

    a = df.groupby("cluster")[["cat_non", "cat_pre_or_diab"]].mean() * 100
    b = df.groupby("cluster")[["cat_non_or_pre", "cat_diab_only"]].mean() * 100

    # order clusters by risk for consistent visuals
    a = a.sort_values("cat_pre_or_diab", ascending=False)
    b = b.loc[a.index]

    # Plot 1
    plt.figure(figsize=(9, 5))
    x = np.arange(len(a.index))
    plt.bar(x, a["cat_non"], label="Non-diabetic (Diabetes_012=0)")
    plt.bar(x, a["cat_pre_or_diab"], bottom=a["cat_non"], label="Prediabetes or Diabetes (Diabetes_012>0)")
    plt.xticks(x, [str(int(c)) for c in a.index])
    plt.ylabel("Proportion (%)")
    plt.xlabel("Cluster")
    plt.title("Cluster composition: Non-diabetic vs Prediabetes+Diabetes")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "cluster_comp_non_vs_preplusdiab.png", dpi=200)
    plt.close()

    # Plot 2
    plt.figure(figsize=(9, 5))
    x = np.arange(len(b.index))
    plt.bar(x, b["cat_non_or_pre"], label="Non-diabetic or Prediabetes (0 or 1)")
    plt.bar(x, b["cat_diab_only"], bottom=b["cat_non_or_pre"], label="Diabetes only (2)")
    plt.xticks(x, [str(int(c)) for c in b.index])
    plt.ylabel("Proportion (%)")
    plt.xlabel("Cluster")
    plt.title("Cluster composition: Non+Prediabetes vs Diabetes")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "cluster_comp_nonpre_vs_diab.png", dpi=200)
    plt.close()

    # Save underlying tables
    a_out = a.reset_index().rename(columns={"cat_non": "pct_non", "cat_pre_or_diab": "pct_pre_or_diab"})
    b_out = b.reset_index().rename(columns={"cat_non_or_pre": "pct_non_or_pre", "cat_diab_only": "pct_diab_only"})
    a_out.to_csv(OUT_TABLES / "cluster_comp_non_vs_preplusdiab.csv", index=False)
    b_out.to_csv(OUT_TABLES / "cluster_comp_nonpre_vs_diab.csv", index=False)

    return a.index.tolist()  # cluster order

def plot_cluster_meaning(df, cluster_order):
    feats = [c for c in PROFILE_FEATURES if c in df.columns]
    prof = df.groupby("cluster")[feats].mean()

    # standardise across clusters (z-scores per feature)
    scaler = StandardScaler()
    prof_z = pd.DataFrame(scaler.fit_transform(prof), index=prof.index, columns=prof.columns)

    # reorder to match prevalence charts
    prof = prof.loc[cluster_order]
    prof_z = prof_z.loc[cluster_order]

    # Plot 3: standardised heatmap
    plt.figure(figsize=(14, 6))
    plt.imshow(prof_z, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Standardised feature value (z-score)")
    plt.xticks(np.arange(len(prof_z.columns)), prof_z.columns, rotation=90)
    plt.yticks(np.arange(len(prof_z.index)), [str(int(c)) for c in prof_z.index])
    plt.title("Cluster meaning: Standardised feature profiles (z-scores)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "cluster_meaning_heatmap_z.png", dpi=200)
    plt.close()

    # Plot 4: top differentiating features
    spread = (prof_z.max(axis=0) - prof_z.min(axis=0)).sort_values(ascending=False)
    topN = 8
    top_feats = spread.head(topN).index.tolist()
    prof_top = prof_z[top_feats]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(prof_top.index))
    width = 0.09
    for i, feat in enumerate(top_feats):
        plt.bar(x + (i - (len(top_feats)-1)/2) * width, prof_top[feat], width=width, label=feat)

    plt.axhline(0, linewidth=1)
    plt.xticks(x, [str(int(c)) for c in prof_top.index])
    plt.ylabel("Standardised value (z-score)")
    plt.xlabel("Cluster")
    plt.title(f"Cluster meaning: Top {topN} differentiating features (standardised)")
    plt.legend(loc="upper right", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "cluster_meaning_top_features.png", dpi=200)
    plt.close()

    # Save profile tables
    prof.round(3).to_csv(OUT_TABLES / "cluster_feature_means.csv")
    prof_z.round(3).to_csv(OUT_TABLES / "cluster_feature_means_z.csv")

def main():
    df = load_merged()
    cluster_order = plot_stacked_prevalence(df)
    plot_cluster_meaning(df, cluster_order)

    print("Saved 4 PNGs to:", OUT_FIGS)
    print("- cluster_comp_non_vs_preplusdiab.png")
    print("- cluster_comp_nonpre_vs_diab.png")
    print("- cluster_meaning_heatmap_z.png")
    print("- cluster_meaning_top_features.png")
    print("\nSaved tables to:", OUT_TABLES)

if __name__ == "__main__":
    main()
