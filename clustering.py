"""Clustering algorithms for Customer Segmentation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler

def find_optimal_k(X, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return list(k_range), inertias, silhouettes

def run_all_clustering(X, n_clusters=4):
    results = {}
    results["KMeans"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    results["DBSCAN"] = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    results["Agglomerative"] = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    results["GaussianMixture"] = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X)
    return results

def evaluate_clustering(X, labels_dict):
    rows = []
    for name, labels in labels_dict.items():
        unique = len(set(labels)) - (1 if -1 in labels else 0)
        if unique < 2:
            continue
        valid = labels != -1
        rows.append({
            "Algorithm": name, "N_Clusters": unique,
            "Silhouette": round(silhouette_score(X[valid], labels[valid]), 4),
            "DaviesBouldin": round(davies_bouldin_score(X[valid], labels[valid]), 4),
            "CalinskiHarabasz": round(calinski_harabasz_score(X[valid], labels[valid]), 1),
        })
    return pd.DataFrame(rows).sort_values("Silhouette", ascending=False)

def reduce_dimensions(X, method="pca", n_components=3):
    if method == "pca":
        return PCA(n_components=n_components, random_state=42).fit_transform(X)
    elif method == "tsne":
        return TSNE(n_components=n_components, random_state=42, perplexity=30).fit_transform(X)
    elif method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=n_components, random_state=42).fit_transform(X)
        except ImportError:
            return PCA(n_components=n_components, random_state=42).fit_transform(X)

def detect_anomalies(X, contamination=0.05):
    return IsolationForest(contamination=contamination, random_state=42).fit_predict(X)
