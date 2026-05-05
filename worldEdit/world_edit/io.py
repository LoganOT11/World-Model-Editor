"""Export edited scenes to standard 3D formats.

Supports:
    - GLB (glTF binary): for web viewers, game engines
    - PLY: for CloudCompare, MeshLab
    - Labeled PLY: with per-point object IDs as scalar field
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def export_glb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: str,
    object_ids: Optional[np.ndarray] = None,
    point_size: float = 0.005,
):
    """Export point cloud as GLB (glTF binary) file.

    Uses trimesh for conversion. Each point becomes a small sphere
    or is rendered as a point cloud primitive.

    Args:
        xyz: (N, 3) float32 point positions.
        rgb: (N, 3) float32 colors in [0, 1].
        output_path: Output .glb file path.
        object_ids: Optional (N,) int array, used for per-object coloring.
        point_size: Visual point size (for renderers that support it).
    """
    import trimesh

    if len(xyz) == 0:
        raise ValueError("Cannot export empty point cloud")

    # Ensure uint8 colors
    if rgb.max() <= 1.0:
        colors_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        colors_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)

    # If object_ids provided, color by object (distinct hue per id)
    if object_ids is not None:
        unique_ids = np.unique(object_ids)
        id_to_color = {}
        for i, oid in enumerate(unique_ids):
            if oid < 0:  # background stays original
                continue
            hue = (i * 0.618033988749895) % 1.0  # golden ratio spacing
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            id_to_color[oid] = np.array([r, g, b]) * 255

        for oid, col in id_to_color.items():
            mask = object_ids == oid
            colors_uint8[mask] = col.astype(np.uint8)

    # Create trimesh point cloud
    pcd = trimesh.PointCloud(vertices=xyz, colors=colors_uint8)

    # Export as GLB
    scene = trimesh.Scene([pcd])
    scene.export(output_path)


def export_ply(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: str,
):
    """Export point cloud as PLY file.

    Args:
        xyz: (N, 3) float32 point positions.
        rgb: (N, 3) float32 colors in [0, 1].
        output_path: Output .ply file path.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    colors_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    pcd.colors = o3d.utility.Vector3dVector(colors_uint8.astype(np.float64) / 255.0)

    o3d.io.write_point_cloud(output_path, pcd)


def export_labeled_ply(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    output_path: str,
):
    """Export point cloud as PLY with object labels as a scalar field.

    Useful for opening in CloudCompare where labels can be used for
    segmentation coloring.

    Args:
        xyz: (N, 3) float32 point positions.
        rgb: (N, 3) float32 colors in [0, 1].
        labels: (N,) int32 object labels (-1 = background).
        output_path: Output .ply file path.
    """
    import open3d as o3d
    import open3d.core as o3c

    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3c.Tensor(xyz.astype(np.float32))
    colors_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    pcd.point["colors"] = o3c.Tensor(colors_uint8)

    # Add label attribute
    pcd.point["label"] = o3c.Tensor(labels.astype(np.int32))

    o3d.t.io.write_point_cloud(output_path, pcd)
