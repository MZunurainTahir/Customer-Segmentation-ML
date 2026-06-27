# Customer Segmentation — Report

## Dataset
Mall Customers: 200 customers, 5 features (Age, Gender, Income, Spending Score).

## Results
| Algorithm | Silhouette | Davies-Bouldin |
|-----------|-----------|----------------|
| K-Means (k=4) | ~0.45 | ~0.82 |
| DBSCAN | ~0.38 | ~1.10 |
| Agglomerative | ~0.43 | ~0.89 |
| Gaussian Mixture | ~0.41 | ~0.94 |

## Segments
- Cluster 0: High Income / High Spend — VIP customers
- Cluster 1: High Income / Low Spend — Untapped potential
- Cluster 2: Low Income / High Spend — Impulse buyers
- Cluster 3: Low Income / Low Spend — Budget-conscious
