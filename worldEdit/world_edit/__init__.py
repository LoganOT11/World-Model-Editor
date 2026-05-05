"""world_edit: Edit 3D scenes reconstructed from video.

Built on the lingbot-map reconstruction pipeline (NPZ → octree → point cloud),
adding object segmentation, selection, and manipulation capabilities.

Core data flow:
    NPZ files → Scene(octree, point cloud) → Editor(segmented objects)
        → Selection(click/drag) → Edit(transform/delete/insert) → Export(GLB/PLY)
"""

# Core data structures (adapted from lingbot-map)
from .scene import Scene
from .octree import OctreeSPC
from .camera import Camera, CameraPath, lookat, compute_scene_scale

# Data loading
from .loader import load_npz_data

# Geometry
from .unproject import unproject_depth_batch_gpu

# Rendering
from .renderer import Open3DRenderer

# Configuration
from .config import PipelineConfig, SceneConfig, load_config

# GPU utilities
from .gpu_mem import GpuMemoryManager

# World editing (new)
from .editor import SceneEditor
from .segmentation import segment_by_color, segment_by_distance, segment_by_connected_components
from .selection import (
    Selection, PointSelection, BBoxSelection, SphereSelection,
    pick_point, pick_bbox, pick_sphere,
)
from .io import export_glb, export_ply, export_labeled_ply
