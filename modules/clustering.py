"""Clustering: K-means, Hierarchical, PCA visualization and optimal k detection."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from utils.helpers import setup_matplotlib_style


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def cluster_analysis(df, numeric_cols, n_clusters_range=(2, 10), method="kmeans", random_state=42):
    """Run clustering analysis with optimal k detection and PCA visualization."""
    setup_matplotlib_style()
    X = df[numeric_cols].dropna()
    if len(X) < 5 or len(numeric_cols) < 2:
        return {"success": False, "error": "Need at least 5 rows and 2 numeric columns."}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_min, n_max = n_clusters_range
    n_max = min(n_max, max(2, len(X) // 10))

    sil_scores = []
    inertia_scores = []
    for k in range(max(2, n_min), n_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append({"k": k, "silhouette": round(float(sil), 4)})
        inertia_scores.append({"k": k, "inertia": round(float(km.inertia_), 4)})

    best_k = max(sil_scores, key=lambda x: x["silhouette"])["k"] if sil_scores else 2

    if method == "kmeans":
        model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=best_k)
    labels = model.fit_predict(X_scaled)
    df_labeled = X.copy()
    df_labeled["Cluster"] = labels.astype(str)

    profiles = df_labeled.groupby("Cluster").agg(["mean", "std", "count"]).round(3)
    profiles.columns = ["_".join(c).strip() for c in profiles.columns]

    try:
        db_score = round(float(davies_bouldin_score(X_scaled, labels)), 4)
    except Exception:
        db_score = None

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    charts = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7, s=30)
    ax.set_xlabel("PC1 ({:.1%})".format(pca.explained_variance_ratio_[0]))
    ax.set_ylabel("PC2 ({:.1%})".format(pca.explained_variance_ratio_[1]))
    ax.set_title("PCA Projection - {} Clusters ({})".format(best_k, method.upper()))
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    charts["pca_clusters"] = _fig_to_b64(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ks = [s["k"] for s in sil_scores]
    ax1.plot(ks, [s["silhouette"] for s in sil_scores], "o-", color="#3366cc")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Method")
    ax1.axvline(best_k, color="#dc3912", linestyle="--", alpha=0.5)
    ax2.plot([s["k"] for s in inertia_scores], [s["inertia"] for s in inertia_scores], "o-", color="#dc3912")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow Method")
    ax2.axvline(best_k, color="#3366cc", linestyle="--", alpha=0.5)
    fig.tight_layout()
    charts["elbow_silhouette"] = _fig_to_b64(fig)

    return {"success": True, "best_k": best_k, "silhouette_scores": sil_scores,
            "inertia_scores": inertia_scores, "db_score": db_score,
            "cluster_sizes": [int((labels == i).sum()) for i in range(best_k)],
            "pca_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "profiles": profiles.reset_index().to_dict(orient="records"),
            "charts": charts, "method": method}
