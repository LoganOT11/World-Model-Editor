# worldEdit - 3D Scene Editor

Edit 3D environments reconstructed from video. Segment objects, move them,
delete them, and insert new ones.

Built on the [lingbot-map](https://github.com/lingbot-map) reconstruction
pipeline (RGBD → octree → point cloud).

## Directory Structure

```
worldEdit/
├── world_edit/               # Core package
│   ├── __init__.py           # Public API
│   ├── scene.py              # Scene (immutable point cloud container) ← lingbot-map
│   ├── octree.py             # Octree spatial structure (LOD, frustum cull) ← lingbot-map
│   ├── camera.py             # Camera model + path generation ← lingbot-map
│   ├── loader.py             # NPZ data loading ← lingbot-map
│   ├── unproject.py          # GPU depth→3D unprojection ← lingbot-map
│   ├── renderer.py           # Open3D offline renderer ← lingbot-map
│   ├── config.py             # Configuration dataclasses ← lingbot-map
│   ├── gpu_mem.py            # GPU memory management ← lingbot-map
│   ├── editor.py             # SceneEditor: object model + edit ops ★ NEW
│   ├── segmentation.py       # DBSCAN / distance / connected-component clustering ★ NEW
│   ├── selection.py          # Ray-cast, bbox, sphere selection tools ★ NEW
│   └── io.py                 # GLB / PLY / labeled-PLY export ★ NEW
├── scripts/
│   └── view_scene.py         # Interactive 3D viewer
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Key Files — What Came From Where

### Copied from lingbot-map (8 files):

| worldEdit file | lingbot-map source | Role |
|---|---|---|
| `scene.py` | `rgbd_render/scene.py` | Immutable point cloud container |
| `octree.py` | `rgbd_render/geometry/octree.py` | Kaolin SPC octree with LOD |
| `camera.py` | `rgbd_render/camera.py` | Camera model + path generation |
| `loader.py` | `rgbd_render/data/loader.py` | NPZ file loading |
| `unproject.py` | `rgbd_render/geometry/unproject.py` | GPU depth unprojection |
| `renderer.py` | `rgbd_render/renderer.py` | Open3D offline renderer |
| `config.py` | `rgbd_render/config.py` | YAML config dataclasses |
| `gpu_mem.py` | `rgbd_render/pipeline/gpu_mem.py` | GPU batch size calculation |

### NOT copied (post-processing / offline rendering):

- `pipeline/builder.py` — SceneBuilder; replaced by our own build flow
- `pipeline/offline.py` — Flythrough video pipeline; not needed for editing
- `pipeline/parallel.py` — Parallel rendering; not needed (yet)
- `overlay.py` — Camera trail / frame tag overlays; not needed for editing
- `video.py` — Video encoding; not needed
- `data/sky.py` — Sky segmentation; not needed for editing

### New (4 files):

| File | Purpose |
|---|---|
| `editor.py` | `SceneEditor` — mutable object model over Scene. Add/remove/move/rotate/scale objects. |
| `segmentation.py` | 3 strategies: DBSCAN (color+spatial), Euclidean distance clustering, k-NN connected components |
| `selection.py` | 3 pick modes: ray-cast click, frustum bbox, world-space sphere |
| `io.py` | Export to GLB, PLY, labeled PLY (with per-point object IDs) |

## Quick Start

```bash
pip install -r requirements.txt

# View a reconstructed scene
python scripts/view_scene.py --input scene.npz
python scripts/view_scene.py --glb scene.glb

# Edit in Python
from world_edit import SceneEditor, load_npz_data, Camera, Open3DRenderer

# Build scene (simplified — full pipeline needs octree build)
editor = SceneEditor(scene)

# Auto-segment into objects
editor.segment_auto(method="dbscan", eps=0.15, min_points=50)

# Select and move an object
obj_id = editor.select_click(screen_x=640, screen_y=360, camera=..., render_w=1280, render_h=720)
editor.move_object(obj_id, np.array([0.5, 0, 0]))  # move 0.5m right

# Delete an object
editor.delete_object(obj_id)

# Export
xyz, rgb, obj_ids = editor.get_active_points()
from world_edit.io import export_glb
export_glb(xyz, rgb, "edited_scene.glb", object_ids=obj_ids)
```

## Requirements

- Python 3.10+
- CUDA-capable GPU
- PyTorch with CUDA
- Open3D
- Kaolin (for octree)
- scikit-learn (for segmentation)
- trimesh (for GLB export)

See `requirements.txt` for pinned versions.

## Architecture Notes (Rapid Prototyping)

Current design decisions (expect to change):

1. **Objects are point-index masks** — not separate point clouds. Simple, fast to
   implement, but means all objects share the octree. Moving an object requires
   recomputing spatial indices.

2. **Editor is a view layer** — `SceneEditor` wraps an immutable `Scene`. Edits
   don't modify the source. `to_scene()` bakes edits into a new Scene.

3. **Segmentation is CPU-side** — uses sklearn. For real-time editing on large
   scenes, this should move to GPU (CUDA DBSCAN or learned segmentation).

4. **Insertion is TODO** — `add_object()` expects existing point indices. Full
   insertion (loading external meshes → point sampling → blending into scene)
   needs more work.

5. **Octree is not updated after edits** — `to_scene()` produces a Scene without
   octree. Rebuilding the octree after edits is possible but not yet automated.

## Future Directions

- GPU-accelerated segmentation (CUDA DBSCAN, SAM-3D, point transformer)
- Real-time editing with octree rebuild on change
- Mesh reconstruction from segmented objects
- Undo/redo stack
- GUI with Open3D or Dear ImGui
