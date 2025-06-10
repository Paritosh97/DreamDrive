import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

def dbscan_clustering(points, feats, eps=0.8, min_samples=5):
    """
    Clustering points based on their features using DBSCAN.
    Parameters:
    - points: ndarray of shape (N, 3), the 3D coordinates of the points
    - feats: ndarray of shape (N, feat_dim), the features of the points
    - eps: float, optional, default=0.5, the maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: int, optional, default=5, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    Returns:
    - labels: ndarray of shape (N,), the cluster labels for each point. Noisy samples are given the label -1.
    """
    print("DBScan Clustering...")
    # feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    # Initialize the DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # Fit the DBSCAN model using the features
    labels = dbscan.fit_predict(feats)
    print("Number of clusters:", len(np.unique(labels)))
    return labels

def kmeans_clustering(points, feats, colors, weights=[1, 0, 0.5], n_clusters=20):
    """
    Perform K-means clustering on the image data.

    Parameters:
    - image: ndarray of shape (H*W, 3), the image data
    - n_clusters: int, optional, default=5, the number of clusters to find

    Returns:
    - labels: ndarray of shape (H*W,), the cluster labels for each pixel
    """
    print("KMeans Clustering ...")
    # feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    combined = np.hstack([weights[0] * points, weights[1] * feats, weights[2] * colors])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(combined)
    print("Number of clusters:", len(np.unique(labels)))
    return labels


def agg_clustering(points, feats, n_clusters=100): # This can be super slow !!!
    """
    Clustering points based on their features using Agglomerative Clustering.

    Parameters:
    - points: ndarray of shape (N, 3), the 3D coordinates of the points
    - feats: ndarray of shape (N, feat_dim), the features of the points
    - n_clusters: int, optional, default=5, the number of clusters to find

    Returns:
    - labels: ndarray of shape (N,), the cluster labels for each point.
    """
    print("AgglomerativeClustering ...")
    # Combine points and features for clustering
    combined = np.hstack((points, feats))
    
    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(combined)
    print("Number of clusters:", len(np.unique(labels)))
    return labels