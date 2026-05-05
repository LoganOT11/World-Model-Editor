"""SceneEditor: high-level API for selecting, transforming, and deleting objects.

The editor wraps a Scene and provides a mutable object model on top of the
immutable point cloud. Each "object" is a mask (boolean array) over points
in sorted_xyz. Edits are applied via masking operations.

Usage:
    editor = SceneEditor(scene)
    editor.segment_auto()                    # auto-segment into objects
    obj_id = editor.select_click(x, y, cam)  # ray-cast selection
    editor.move_object(obj_id, offset)       # translate
    editor.delete_object(obj_id)             # remove
    editor.insert_object(points, colors)     # add new
    new_scene = editor.to_scene()            # export back to Scene
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scene import Scene


@dataclass
class EditableObject:
    """A segmented object within an editable scene."""
    id: int
    point_indices: np.ndarray       # indices into scene.sorted_xyz
    label: str = ""
    color: Optional[np.ndarray] = None  # (3,) override color
    transform: np.ndarray = field(   # 4x4 local transform (identity = no change)
        default_factory=lambda: np.eye(4, dtype=np.float32))
    visible: bool = True

    @property
    def xyz(self, scene: Scene) -> np.ndarray:
        """World-space points with current transform applied."""
        pts = scene.sorted_xyz[self.point_indices]
        if np.allclose(self.transform, np.eye(4)):
            return pts
        R = self.transform[:3, :3]
        t = self.transform[:3, 3]
        return pts @ R.T + t

    @property
    def rgb(self, scene: Scene) -> np.ndarray:
        """Colors (override or original)."""
        if self.color is not None:
            return np.tile(self.color, (len(self.point_indices), 1))
        return scene.sorted_rgb[self.point_indices]


class SceneEditor:
    """Mutable editing layer over an immutable Scene.

    Maintains a set of EditableObjects backed by the original point cloud.
    Operations are non-destructive to the source Scene — .to_scene() produces
    a new Scene with edits applied.
    """

    def __init__(self, scene: Scene):
        self._scene = scene
        self._objects: Dict[int, EditableObject] = {}
        self._next_id = 0
        # Background / unclassified points
        self._background_mask = np.ones(len(scene.sorted_xyz), dtype=bool)

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def objects(self) -> List[EditableObject]:
        return list(self._objects.values())

    @property
    def num_objects(self) -> int:
        return len(self._objects)

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, point_indices: np.ndarray,
                   label: str = "", color: Optional[np.ndarray] = None) -> int:
        """Register a new object from a set of point indices.

        Args:
            point_indices: 1D integer array indexing into scene.sorted_xyz.
            label: Optional human-readable label.
            color: Optional (3,) float RGB override in [0, 1].

        Returns:
            The new object's id.
        """
        obj_id = self._next_id
        self._next_id += 1
        self._objects[obj_id] = EditableObject(
            id=obj_id,
            point_indices=point_indices.astype(np.int64),
            label=label,
            color=color,
        )
        self._background_mask[point_indices] = False
        return obj_id

    def remove_object(self, obj_id: int):
        """Delete an object, returning its points to the background."""
        obj = self._objects.pop(obj_id, None)
        if obj is not None:
            self._background_mask[obj.point_indices] = True

    def clear(self):
        """Remove all objects (everything becomes background)."""
        self._background_mask[:] = True
        self._objects.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Transform operations
    # ------------------------------------------------------------------

    def move_object(self, obj_id: int, offset: np.ndarray):
        """Translate an object by `offset` in world space."""
        obj = self._objects[obj_id]
        obj.transform[:3, 3] += np.asarray(offset, dtype=np.float32)

    def set_object_position(self, obj_id: int, position: np.ndarray):
        """Set absolute world position (translation component only)."""
        obj = self._objects[obj_id]
        obj.transform[:3, 3] = np.asarray(position, dtype=np.float32)

    def rotate_object(self, obj_id: int, rotation: np.ndarray):
        """Apply rotation (3x3 or 4x4 matrix) to an object."""
        obj = self._objects[obj_id]
        R = np.asarray(rotation, dtype=np.float32)
        if R.shape == (4, 4):
            obj.transform = R @ obj.transform
        else:
            new_transform = np.eye(4, dtype=np.float32)
            new_transform[:3, :3] = R
            obj.transform = new_transform @ obj.transform

    def scale_object(self, obj_id: int, scale: float):
        """Scale an object around its centroid."""
        obj = self._objects[obj_id]
        pts = obj.xyz(self._scene)
        centroid = pts.mean(axis=0)
        T_to_origin = np.eye(4, dtype=np.float32)
        T_to_origin[:3, 3] = -centroid
        S = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
        T_back = np.eye(4, dtype=np.float32)
        T_back[:3, 3] = centroid
        obj.transform = T_back @ S @ T_to_origin @ obj.transform

    def toggle_visibility(self, obj_id: int):
        """Toggle object visibility."""
        self._objects[obj_id].visible = not self._objects[obj_id].visible

    # ------------------------------------------------------------------
    # Auto-segmentation (convenience wrappers)
    # ------------------------------------------------------------------

    def segment_auto(self, method: str = "dbscan", **kwargs) -> int:
        """Auto-segment the scene and populate objects.

        Args:
            method: "dbscan", "distance", "connected".
            **kwargs: passed to the segmentation function.

        Returns:
            Number of objects found.
        """
        from . import segmentation as seg

        if method == "dbscan":
            labels = seg.segment_by_color(self._scene, **kwargs)
        elif method == "distance":
            labels = seg.segment_by_distance(self._scene, **kwargs)
        elif method == "connected":
            labels = seg.segment_by_connected_components(self._scene, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        self.clear()
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # skip noise (-1)

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > kwargs.get("min_points", 10):
                self.add_object(indices, label=f"object_{label}")

        return self.num_objects

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_click(self, screen_x: float, screen_y: float,
                     camera, render_w: int, render_h: int) -> Optional[int]:
        """Ray-cast from screen coords, return object id of nearest hit.

        Args:
            screen_x, screen_y: pixel coordinates.
            camera: world_edit.Camera instance.
            render_w, render_h: viewport dimensions.

        Returns:
            Object id if an object was hit, None otherwise.
        """
        from . import selection as sel
        hit_idx = sel.pick_point(
            self._scene, screen_x, screen_y, camera,
            render_w, render_h,
            only_foreground=False,
        )
        if hit_idx is None:
            return None
        # Find which object contains this point
        for obj_id, obj in self._objects.items():
            if hit_idx in obj.point_indices:
                return obj_id
        return None

    def select_bbox(self, screen_min, screen_max,
                    camera, render_w: int, render_h: int) -> List[int]:
        """Frustum-based region selection. Returns list of object ids."""
        from . import selection as sel
        hit_indices = sel.pick_bbox(
            self._scene, screen_min, screen_max, camera,
            render_w, render_h,
        )
        found = set()
        for idx in hit_indices:
            for obj_id, obj in self._objects.items():
                if idx in obj.point_indices and obj.visible:
                    found.add(obj_id)
                    break
        return list(found)

    # ------------------------------------------------------------------
    # Export / conversion
    # ------------------------------------------------------------------

    def get_active_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xyz, rgb, object_ids) for all visible + background points.

        object_ids[i] = -1 for background, or the object id.
        """
        N = len(self._scene.sorted_xyz)
        out_xyz = np.empty((N, 3), dtype=np.float32)
        out_rgb = np.empty((N, 3), dtype=np.float32)
        out_ids = np.full(N, -1, dtype=np.int32)

        # Background first
        bg = self._background_mask
        out_xyz[bg] = self._scene.sorted_xyz[bg]
        out_rgb[bg] = self._scene.sorted_rgb[bg]

        # Objects (with transforms)
        for obj_id, obj in self._objects.items():
            if not obj.visible:
                continue
            mask = np.zeros(N, dtype=bool)
            mask[obj.point_indices] = True
            out_xyz[mask] = obj.xyz(self._scene)
            out_rgb[mask] = obj.rgb(self._scene)
            out_ids[mask] = obj_id

        return out_xyz, out_rgb, out_ids

    def to_scene(self) -> Scene:
        """Produce a new Scene with all edits baked in.

        Note: This creates a simplified Scene without octree/LOD support.
        For full octree support, rebuild from the edited points.
        """
        xyz, rgb, _ = self.get_active_points()
        ptrs = np.array([len(xyz)], dtype=np.int32)
        frames = np.zeros(len(xyz), dtype=np.int32)

        return Scene(
            sorted_xyz=xyz,
            sorted_rgb=rgb,
            sorted_frames=frames,
            ptrs=ptrs,
            c2w_poses=self._scene.c2w_poses.copy() if self._scene.c2w_poses is not None else None,
            scene_scale=self._scene.scene_scale,
        )
