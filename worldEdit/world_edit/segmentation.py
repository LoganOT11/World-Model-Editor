"""Object segmentation for 3D point clouds.

Segmentation strategies:
    - segment_by_color:       DBSCAN in (xyz + rgb) space
    - segment_by_distance:    Euclidean clustering with distance threshold
    - segment_by_connected_components: Graph-based connected components on k-NN graph
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .scene import Scene


def segment_by_color(
    scene: Scene,
    eps: float = 0.15,
    min_points: int = 50,
    color_weight: float = 0.3,
    spatial_weight: float = 1.0,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """DBSCAN clustering in combined (xyz + rgb) feature space.

    Points close in both 3D space and color are clustered together.
    Suitable for separating distinct objects with different colors.

    Args:
        scene: Source Scene with point cloud data.
        eps: DBSCAN epsilon (max distance between neighbors).
        min_points: Minimum points to form a cluster.
        color_weight: Weight of color channel difference vs. spatial.
        spatial_weight: Weight of spatial distance vs. color.
        frame_idx: If given, only cluster points visible up to this frame.

    Returns:
        Array of cluster labels (shape N, dtype int).
        -1 = noise/unclustered.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    if frame_idx is not None:
        xyz, rgb, _ = scene.active_points(frame_idx)
    else:
        xyz, rgb = scene.sorted_xyz, scene.sorted_rgb

    if len(xyz) == 0:
        return np.array([], dtype=np.int32)

    # Normalize spatial and color separately
    scaler_xyz = StandardScaler()
    scaler_rgb = StandardScaler()
    xyz_norm = scaler_xyz.fit_transform(xyz) * spatial_weight
    rgb_norm = scaler_rgb.fit_transform(rgb) * color_weight

    features = np.hstack([xyz_norm, rgb_norm])

    clusterer = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1)
    labels = clusterer.fit_predict(features)

    return labels


def segment_by_distance(
    scene: Scene,
    distance_threshold: float = 0.1,
    min_points: int = 50,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Euclidean clustering: group spatially close points.

    Simpler and faster than DBSCAN; purely geometric.

    Args:
        scene: Source Scene.
        distance_threshold: Max point-to-point distance for merging.
        min_points: Minimum points for a valid cluster.
        frame_idx: If given, cluster points visible up to this frame.

    Returns:
        Cluster labels, -1 = noise.
    """
    if frame_idx is not None:
        xyz, _, _ = scene.active_points(frame_idx)
    else:
        xyz = scene.sorted_xyz

    if len(xyz) == 0:
        return np.array([], dtype=np.int32)

    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial import cKDTree

    # Build a sparse graph: connect points within distance_threshold
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(distance_threshold, output_type='ndarray')

    if len(pairs) == 0:
        return np.full(len(xyz), -1, dtype=np.int32)

    # Use AgglomerativeClustering with connectivity from pairs
    from sklearn.cluster import AgglomerativeClustering
    from scipy.sparse import coo_matrix

    N = len(xyz)
    row, col = pairs[:, 0], pairs[:, 1]
    data = np.ones(len(pairs))
    connectivity = coo_matrix((data, (row, col)), shape=(N, N))
    connectivity = connectivity + connectivity.T  # make symmetric

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        connectivity=connectivity,
        linkage='single',
    )
    labels = clusterer.fit_predict(xyz)

    # Filter small clusters
    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = unique[counts < min_points]
    for c in small_clusters:
        labels[labels == c] = -1

    return labels


def segment_by_connected_components(
    scene: Scene,
    k: int = 10,
    distance_threshold: float = 0.1,
    min_points: int = 50,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Graph-based segmentation using k-NN + connected components.

    Builds a k-NN graph, removes edges longer than threshold,
    then finds connected components.

    Args:
        scene: Source Scene.
        k: Number of nearest neighbors for graph construction.
        distance_threshold: Max edge length to keep.
        min_points: Minimum points per component.
        frame_idx: If given, operate on points up to this frame.

    Returns:
        Component labels, -1 = noise.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix

    if frame_idx is not None:
        xyz, _, _ = scene.active_points(frame_idx)
    else:
        xyz = scene.sorted_xyz

    if len(xyz) == 0:
        return np.array([], dtype=np.int32)

    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=min(k + 1, len(xyz)), n_jobs=-1)
    neigh.fit(xyz)
    distances, indices = neigh.kneighbors(xyz)

    N = len(xyz)
    rows, cols, data = [], [], []
    for i in range(N):
        for j_idx in range(1, len(indices[i])):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            if d < distance_threshold:
                rows.append(i)
                cols.append(j)
                data.append(d)

    if len(rows) == 0:
        return np.full(N, -1, dtype=np.int32)

    graph = coo_matrix((data, (rows, cols)), shape=(N, N))
    n_components, labels = connected_components(graph, directed=False)

    # Filter small components
    unique, counts = np.unique(labels, return_counts=True)
    small = unique[counts < min_points]
    for c in small:
        labels[labels == c] = -1

    return labels
