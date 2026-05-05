"""Quick interactive 3D viewer for a reconstructed scene.

Usage:
    python scripts/view_scene.py --input scene.npz           # view a scene
    python scripts/view_scene.py --glb scene.glb             # view a GLB
    python scripts/view_scene.py --ply scene.ply             # view a PLY
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_edit import (
    Scene, OctreeSPC,
    Camera,
    load_npz_data,
    Open3DRenderer,
    SceneConfig,
    compute_scene_scale,
    unproject_depth_batch_gpu,
    GpuMemoryManager,
)


def view_npz(npz_path: str):
    """Load NPZ, build scene, open interactive viewer."""
    import torch
    from tqdm import tqdm

    print(f"Loading {npz_path}...")
    data = load_npz_data(npz_path)
    images = data['images']
    depths = data['depth']
    c2w = data['c2w']
    Ks = data['K']

    S = len(images)
    print(f"Loaded {S} frames, image size {images[0].shape}")

    # Build octree
    cfg = SceneConfig(octree_level=10, max_depth=100.0, downsample=2)
    mgr = GpuMemoryManager()
    H, W = depths[0].shape
    batch_size = mgr.build_batch_size(H, W, cfg.downsample)

    print(f"Unprojecting depth (batch_size={batch_size})...")

    xyz_parts, rgb_parts, frame_parts = [], [], []
    with tqdm(total=S, desc='Unprojecting') as pbar:
        for bs in range(0, S, batch_size):
            be = min(bs + batch_size, S)
            B = be - bs

            d_batch = torch.from_numpy(depths[bs:be].astype(np.float32)).cuda()
            img_batch = torch.from_numpy(images[bs:be].copy()).cuda()
            Ks_batch = torch.from_numpy(Ks[bs:be].astype(np.float32)).cuda()
            c2w_batch = torch.from_numpy(c2w[bs:be].astype(np.float32)).cuda()

            pts_xyz, pts_rgb, vcounts = unproject_depth_batch_gpu(
                d_batch, img_batch, Ks_batch, c2w_batch,
                cfg.max_depth, cfg.downsample, cfg.jitter)
            del d_batch, img_batch, Ks_batch, c2w_batch

            offset = 0
            for j in range(B):
                cnt = vcounts[j].item()
                if cnt > 0:
                    xyz_parts.append(pts_xyz[offset:offset + cnt].cpu())
                    rgb_parts.append(pts_rgb[offset:offset + cnt].cpu())
                    frame_parts.append(torch.full((cnt,), bs + j, dtype=torch.int32))
                offset += cnt
            del pts_xyz, pts_rgb
            pbar.update(B)

    all_xyz = torch.cat(xyz_parts)
    all_rgb = torch.cat(rgb_parts)
    all_frames = torch.cat(frame_parts)

    print(f"Building octree ({len(all_xyz):,} points)...")
    octree = OctreeSPC(max_level=cfg.octree_level)
    octree.build(all_xyz, all_rgb, all_frames)

    ptrs = octree.compute_ptrs(S)
    scene = Scene(
        sorted_xyz=octree.sorted_xyz,
        sorted_rgb=octree.sorted_rgb,
        sorted_frames=octree.sorted_frames,
        ptrs=ptrs,
        c2w_poses=c2w,
        scene_scale=compute_scene_scale(c2w),
        images=images,
        intrinsics=Ks,
        octree=octree,
    )

    _interactive_view(scene)


def view_glb(glb_path: str):
    """Open a GLB point cloud in Open3D viewer."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(glb_path)
    if len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(glb_path)
    else:
        pcd = mesh.sample_points_uniformly(len(mesh.vertices))
    o3d.visualization.draw_geometries([pcd], window_name="worldEdit Viewer")


def view_ply(ply_path: str):
    """Open a PLY point cloud in Open3D viewer."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd], window_name="worldEdit Viewer")


def _interactive_view(scene: Scene):
    """Launch interactive Open3D viewer with controls."""
    import open3d as o3d

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="worldEdit Viewer", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.sorted_xyz.astype(np.float64))
    colors = np.clip(scene.sorted_rgb, 0, 1).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    # Camera trajectory as lines
    if scene.c2w_poses is not None:
        positions = scene.c2w_poses[:, :3, 3]
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
        n = len(positions)
        edges = np.array([[i, i + 1] for i in range(n - 1)])
        lines.lines = o3d.utility.Vector2iVector(edges)
        lines.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (len(edges), 1)))
        vis.add_geometry(lines)

    print("\nControls:")
    print("  Mouse drag:  Rotate / Pan")
    print("  Scroll:      Zoom")
    print("  R:           Reset view")
    print("  Q / Esc:     Quit")
    print()

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="worldEdit Scene Viewer")
    parser.add_argument('--input', '-i', help='Input NPZ file')
    parser.add_argument('--glb', help='Input GLB file')
    parser.add_argument('--ply', help='Input PLY file')
    args = parser.parse_args()

    if args.input:
        view_npz(args.input)
    elif args.glb:
        view_glb(args.glb)
    elif args.ply:
        view_ply(args.ply)
    else:
        parser.print_help()
