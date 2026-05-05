"""3D selection tools: ray-cast pick, bbox frustum, sphere selection.

Works with world_edit.Camera for screen-to-world and world-to-screen
projection.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .camera import Camera
from .scene import Scene


class Selection:
    """A geometric selection in 3D space. Base class."""
    pass


class PointSelection(Selection):
    """Single point or ray-cast result."""
    def __init__(self, world_pos: np.ndarray, point_idx: int):
        self.world_pos = np.asarray(world_pos, dtype=np.float32)
        self.point_idx = point_idx


class BBoxSelection(Selection):
    """Axis-aligned bounding box in screen space, projected to frustum."""
    def __init__(self, screen_min: Tuple[float, float],
                 screen_max: Tuple[float, float],
                 point_indices: np.ndarray):
        self.screen_min = screen_min
        self.screen_max = screen_max
        self.point_indices = point_indices


class SphereSelection(Selection):
    """Sphere in world space."""
    def __init__(self, center: np.ndarray, radius: float,
                 point_indices: np.ndarray):
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = radius
        self.point_indices = point_indices


def pick_point(
    scene: Scene,
    screen_x: float,
    screen_y: float,
    camera: Camera,
    render_w: int,
    render_h: int,
    radius: float = 5.0,
    only_foreground: bool = True,
    frame_idx: Optional[int] = None,
) -> Optional[int]:
    """Ray-cast from screen coordinates to find nearest point.

    Projects all points to screen, finds those near (screen_x, screen_y),
    then picks the one closest to the camera.

    Args:
        scene: Source Scene.
        screen_x, screen_y: Pixel coordinates (origin top-left).
        camera: Camera defining the viewpoint.
        render_w, render_h: Viewport dimensions in pixels.
        radius: Screen-space pick radius in pixels.
        only_foreground: If True, only check points in front of the camera.
        frame_idx: If given, only consider points up to this frame.

    Returns:
        Index into scene.sorted_xyz of the nearest hit, or None.
    """
    if frame_idx is not None:
        xyz, _, _ = scene.active_points(frame_idx)
    else:
        xyz = scene.sorted_xyz

    if len(xyz) == 0:
        return None

    w2c = camera.w2c()
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Transform to camera space
    cam_pts = (R @ xyz.T + t[:, None]).T  # (N, 3)
    z = cam_pts[:, 2]

    # Filter foreground
    if only_foreground:
        fg = z > 0
        if not fg.any():
            return None
        cam_pts = cam_pts[fg]
        z = z[fg]
        valid_indices = np.where(fg)[0]
    else:
        valid_indices = np.arange(len(xyz))

    # Project to screen
    fx, fy, cx, cy = camera.fov_intrinsics(render_w, render_h)
    safe_z = np.abs(z)
    safe_z[safe_z < 1e-6] = 1e-6
    px = cam_pts[:, 0] * fx / safe_z + cx
    py = cam_pts[:, 1] * fy / safe_z + cy

    # Distance from click position
    dist = np.sqrt((px - screen_x) ** 2 + (py - screen_y) ** 2)
    candidates = dist <= radius
    if not candidates.any():
        return None

    # Pick nearest (smallest z) among candidates
    cand_z = z[candidates]
    nearest_local = np.argmin(cand_z)
    nearest_global = valid_indices[candidates][nearest_local]

    return int(nearest_global)


def pick_bbox(
    scene: Scene,
    screen_min: Tuple[float, float],
    screen_max: Tuple[float, float],
    camera: Camera,
    render_w: int,
    render_h: int,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Frustum-based rectangle selection.

    Projects all points to screen, returns indices of those inside the
    screen-space rectangle.

    Args:
        screen_min: (x_min, y_min) in pixels.
        screen_max: (x_max, y_max) in pixels.
        camera: Camera defining the viewpoint.
        render_w, render_h: Viewport dimensions.

    Returns:
        Array of indices into scene.sorted_xyz.
    """
    if frame_idx is not None:
        xyz, _, _ = scene.active_points(frame_idx)
    else:
        xyz = scene.sorted_xyz

    if len(xyz) == 0:
        return np.array([], dtype=np.int64)

    w2c = camera.w2c()
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    cam_pts = (R @ xyz.T + t[:, None]).T
    z = cam_pts[:, 2]

    fg = z > 0
    if not fg.any():
        return np.array([], dtype=np.int64)

    valid_indices = np.where(fg)[0]
    cam_pts = cam_pts[fg]
    z = z[fg]

    fx, fy, cx, cy = camera.fov_intrinsics(render_w, render_h)
    safe_z = np.abs(z)
    safe_z[safe_z < 1e-6] = 1e-6
    px = cam_pts[:, 0] * fx / safe_z + cx
    py = cam_pts[:, 1] * fy / safe_z + cy

    sx_min, sy_min = min(screen_min[0], screen_max[0]), min(screen_min[1], screen_max[1])
    sx_max, sy_max = max(screen_min[0], screen_max[0]), max(screen_min[1], screen_max[1])

    in_rect = (px >= sx_min) & (px <= sx_max) & (py >= sy_min) & (py <= sy_max)

    return valid_indices[in_rect]


def pick_sphere(
    scene: Scene,
    center: np.ndarray,
    radius: float,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Sphere selection in world space.

    Args:
        center: (3,) world-space sphere center.
        radius: Sphere radius in world units.
        frame_idx: If given, only consider points up to this frame.

    Returns:
        Array of indices into scene.sorted_xyz.
    """
    if frame_idx is not None:
        xyz, _, _ = scene.active_points(frame_idx)
    else:
        xyz = scene.sorted_xyz

    dist = np.linalg.norm(xyz - np.asarray(center), axis=1)
    return np.where(dist <= radius)[0]
