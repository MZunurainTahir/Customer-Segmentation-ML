"""3D visualizations for Customer Segmentation."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
COLORS = px.colors.qualitative.Set2

def plot_3d_clusters(coords_3d, labels, title="3D Cluster Visualization", method="PCA"):
    df = pd.DataFrame(coords_3d, columns=[f"{method}1", f"{method}2", f"{method}3"])
    df["Cluster"] = labels.astype(str)
    fig = px.scatter_3d(df, x=f"{method}1", y=f"{method}2", z=f"{method}3",
                        color="Cluster", opacity=0.8, height=700, title=title,
                        color_discrete_sequence=COLORS)
    fig.update_traces(marker=dict(size=5))
    fig.write_html(FIG_DIR / f"3d_{method.lower()}_clusters.html")
    return fig

def plot_elbow_silhouette(k_range, inertias, silhouettes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_range), y=inertias, name="Inertia",
                              mode="lines+markers", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouettes, name="Silhouette",
                              mode="lines+markers", yaxis="y2", line=dict(color="coral")))
    fig.update_layout(title="Elbow Method & Silhouette Score",
                      yaxis=dict(title="Inertia", side="left"),
                      yaxis2=dict(title="Silhouette Score", side="right", overlaying="y"),
                      xaxis_title="Number of Clusters (K)")
    fig.write_html(FIG_DIR / "elbow_silhouette.html")
    return fig

def plot_3d_cluster_surface(df, x_col, y_col, z_col, label_col):
    fig = go.Figure()
    for cluster in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cluster]
        fig.add_trace(go.Scatter3d(x=sub[x_col], y=sub[y_col], z=sub[z_col],
                                   mode="markers", name=f"Cluster {cluster}",
                                   marker=dict(size=5, opacity=0.8)))
    fig.update_layout(title=f"3D Cluster: {x_col} x {y_col} x {z_col}", height=700)
    fig.write_html(FIG_DIR / "3d_cluster_surface.html")
    return fig
