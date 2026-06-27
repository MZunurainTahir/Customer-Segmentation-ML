# Customer Segmentation — Full Pipeline
# =============================================

# %% [1] Imports
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px, warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.append(os.path.join(os.getcwd(), ".."))
from src.clustering import (scale_features, find_optimal_k, run_all_clustering,
                             evaluate_clustering, reduce_dimensions, detect_anomalies)
from src.visualize import plot_3d_clusters, plot_elbow_silhouette, plot_3d_cluster_surface

# %% [2] Load / Create Data
try:
    df = pd.read_csv("../data/raw/Mall_Customers.csv")
    df.columns = [c.replace(" ", "_") for c in df.columns]
except FileNotFoundError:
    np.random.seed(42); n = 200
    df = pd.DataFrame({
        "CustomerID": range(1, n+1),
        "Gender": np.random.choice(["Male","Female"], n),
        "Age": np.random.randint(18, 70, n),
        "Annual_Income": np.random.randint(15, 140, n),
        "Spending_Score": np.random.randint(1, 100, n),
    })
    df.to_csv("../data/raw/Mall_Customers.csv", index=False)
print(df.head()); print(df.describe())

# %% [3] EDA
df["Gender_Enc"] = (df["Gender"] == "Male").astype(int)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["Age"].hist(ax=axes[0], bins=20, color="steelblue"); axes[0].set_title("Age Distribution")
df["Annual_Income"].hist(ax=axes[1], bins=20, color="coral"); axes[1].set_title("Annual Income")
df["Spending_Score"].hist(ax=axes[2], bins=20, color="teal"); axes[2].set_title("Spending Score")
plt.tight_layout(); plt.savefig("../reports/figures/eda.png", dpi=150); plt.show()

# %% [4] Scale features
feature_cols = ["Age", "Annual_Income", "Spending_Score", "Gender_Enc"]
X, scaler = scale_features(df, feature_cols)

# %% [5] Find Optimal K
k_range, inertias, silhouettes = find_optimal_k(X)
plot_elbow_silhouette(k_range, inertias, silhouettes).show()

# %% [6] Run All Clustering
labels_dict = run_all_clustering(X, n_clusters=4)
print(evaluate_clustering(X, labels_dict))

# %% [7] PCA 3D
pca_3d = reduce_dimensions(X, "pca", 3)
plot_3d_clusters(pca_3d, labels_dict["KMeans"], "PCA 3D — Customer Segments").show()

# %% [8] t-SNE 3D
tsne_3d = reduce_dimensions(X, "tsne", 3)
plot_3d_clusters(tsne_3d, labels_dict["KMeans"], "t-SNE 3D — Customer Segments", "tSNE").show()

# %% [9] Original Feature Space 3D
df["Cluster"] = labels_dict["KMeans"]
plot_3d_cluster_surface(df, "Age", "Annual_Income", "Spending_Score", "Cluster").show()

# %% [10] Anomaly Detection
df["Anomaly"] = detect_anomalies(X)
print(f"Anomalies: {(df.Anomaly==-1).sum()}")
print("Done!")
