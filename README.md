# Customer Segmentation 🛍️

**ML Concepts:** Unsupervised Learning, Clustering, Dimensionality Reduction, Anomaly Detection

## Dataset
- **Primary:** Mall Customers Dataset (download: Kaggle — Mall_Customers.csv)
- **Extended:** UCI Online Retail Dataset
- **Columns:** CustomerID, Gender, Age, Annual Income, Spending Score

## Algorithms Used
| Algorithm | Type | Key Params |
|-----------|------|-----------|
| K-Means | Partitional | n_clusters, init |
| DBSCAN | Density-based | eps, min_samples |
| Agglomerative | Hierarchical | linkage, n_clusters |
| Gaussian Mixture | Probabilistic | n_components, covariance_type |
| Isolation Forest | Anomaly Detection | contamination |

## ML Concepts Applied
- Elbow method & silhouette analysis for optimal K
- PCA (Principal Component Analysis) for dimensionality reduction
- t-SNE for non-linear dimensionality reduction
- UMAP for fast manifold learning
- Davies-Bouldin and Calinski-Harabasz indices
- Anomaly / outlier detection

## 3D Visualizations
- `reports/figures/3d_pca_clusters.html` — PCA 3D cluster scatter
- `reports/figures/3d_tsne_clusters.html` — t-SNE 3D embedding
- `reports/figures/3d_cluster_surface.html` — Cluster decision boundary surface

## Setup
```bash
pip install -r requirements.txt
# Get dataset: kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python
jupyter lab
```

## Commit Guide
```bash
git commit -m "feat: EDA and feature scaling for clustering"
git commit -m "feat: KMeans with elbow + silhouette analysis"
git commit -m "feat: DBSCAN and hierarchical clustering"
git commit -m "feat: PCA, t-SNE, UMAP dimensionality reduction"
git commit -m "feat: 3D cluster visualizations with Plotly"
git commit -m "feat: anomaly detection with Isolation Forest"
git commit -m "docs: segmentation report and business insights"
```
