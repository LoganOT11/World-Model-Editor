# Point Cloud Segmentation — Library Survey & Integration Plan

Survey of state-of-the-art point cloud segmentation models (2025–2026) for potential integration into **worldEdit**, the 3D scene editor built on [LingBot-Map](https://github.com/lingbot-map).

---

## 1. Segmentation Libraries — Full Inventory

### 1.1 Unified, Interactive 3D Models

These accept user prompts (clicks, text) and segment anything in a 3D scene — the closest 3D analog to Meta's SAM philosophy.

| # | Model | GitHub | Venue | Stars | Relevance to worldEdit |
|---|-------|--------|-------|-------|----------------------|
| 1 | **SNAP** | [neu-vi/SNAP](https://github.com/neu-vi/SNAP) | ICLR 2026 | ~54 | ★★★★★ Top candidate. "Segment Anything in Any Point cloud" — spatial + text prompts, zero-shot, 8/9 SOTA benchmarks. Indoor/outdoor/aerial. Direct fit for interactive editing. |
| 2 | **S2AM3D** | [sumuru789/S2AM3D](https://github.com/sumuru789/S2AM3D) | CVPR 2026 Oral | ~30 | ★★★★☆ Part-level segmentation. Controllable granularity — segment a "door handle" as precisely as a "car door." Useful for fine-grained object editing. |
| 3 | **Point-SAM**† | [zyc00/Point-SAM](https://github.com/zyc00/point-sam) | ICLR 2025 | ~425 | ★★★★☆ Promptable 3D SAM. Strong community adoption. Prompt-based (click + text). Good fallback if SNAP doesn't pan out. |

† *Not in the original 11 but surfaced during research — relevant and well-supported.*

### 1.2 2D-to-3D Bridge (Open-Vocabulary)

These lift powerful 2D foundation models (SAM, CLIP, LLMs) into 3D — enabling text-driven segmentation without 3D training.

| # | Model | GitHub | Venue | Stars | Relevance to worldEdit |
|---|-------|--------|-------|-------|----------------------|
| 4 | **PointSeg** ⚠️ | *No public code* | ICCV 2025 Workshop | — | ★★★☆☆ Training-free. Renders point cloud → 2D views → SAM → lifts masks back to 3D. Tencent Youtu Lab. Paper only; no repo yet. |
| 5 | **VDG-Uni3DSeg** | [Hanzy1996/VDG-Uni3DSeg](https://github.com/Hanzy1996/VDG-Uni3DSeg) | ICCV 2025 | ~15 | ★★★☆☆ Visual-description-guided. Uses multimodal cues to distinguish fine-grained classes. Good for "segment all chairs" type queries. |
| 6 | **CitySeg** ⚠️ | *No public code* | arXiv Aug 2025 | — | ★★☆☆☆ City-scale foundation model. Huawei. 3D open-vocabulary for UAV point clouds. arXiv only; no repo. May be too domain-specific. |
| 7 | **OpenCity3D**† | [opencity3d/opencity3d](https://github.com/opencity3d/opencity3d) | WACV 2025 | ~50 | ★★☆☆☆ Zero-shot urban scene understanding with VLMs. Related approach, ETH Zürich. |

### 1.3 High-Performance Specialized Backbones

Best-in-class accuracy for supervised 3D semantic segmentation. Strong baselines if we need to fine-tune on custom data.

| # | Model | GitHub | Venue | Stars | Relevance to worldEdit |
|---|-------|--------|-------|-------|----------------------|
| 8 | **PTv3** (Point Transformer V3) | [pointcept/pointtransformerv3](https://github.com/pointcept/pointtransformerv3) | CVPR 2024 Oral | ~800 | ★★★★☆ Industry-standard backbone. Simpler, faster, stronger. Part of the Pointcept codebase. Excellent base for fine-tuning. |
| 9 | **PointNeXt** | [guochengqian/pointnext](https://github.com/guochengqian/pointnext) | NeurIPS 2022 | ~600 | ★★★☆☆ Improved PointNet++ with modern training recipes. Strong supervised baseline. |
| 10 | **Point-MoE** | [UVA-Computer-Vision-Lab/point_moe](https://github.com/UVA-Computer-Vision-Lab/point_moe) | ICLR 2026 | ~5 | ★★★☆☆ Mixture-of-Experts for multi-dataset training. Good for generalization across domains. |
| 11 | **Sonata** | [facebookresearch/sonata](https://github.com/facebookresearch/sonata) | CVPR 2025 Highlight | ~150 | ★★★★☆ Meta's self-supervised framework built on PTv3. Pre-trained representations transfer well to downstream tasks. Strong candidate for our pipeline. |

### 1.4 Real-Time & Efficient Models

Optimized for speed and resource-constrained hardware.

| # | Model | GitHub | Venue | Stars | Relevance to worldEdit |
|---|-------|--------|-------|-------|----------------------|
| 12 | **ALPINE** | [valeoai/Alpine](https://github.com/valeoai/Alpine) | — | ~200 | ★★★★★ Training-free LiDAR instance segmentation. #1 on SemanticKITTI. Runs real-time on single CPU thread. Could serve as a fast pre-segmentation step. |
| 13 | **FARVNet** ⚠️ | *No public code* | Sensors 2025 | — | ★★☆☆☆ Range-view method. Projects point cloud → 2D range image → CNN → projects back. Chengdu Univ. No repo. |
| 14 | **FRNet**† | [Xiangxu-0103/FRNet](https://github.com/Xiangxu-0103/FRNet) | TIP 2025 | ~101 | ★★★☆☆ Frustum-Range networks. Addresses range-view corruption issues. Solid real-time alternative. |

⚠️ = Paper claims "fully open-source" but no public code exists as of 2026-05-05.
† = Bonus find during research, not in original 11.

---

## 2. Repository Status

### Cloned (8/11) — `C:/Users/logan/VibeCode/`

```
Alpine/               # Training-free LiDAR (#1 SemanticKITTI)
point_moe/            # Mixture-of-Experts multi-dataset
pointnext/            # PointNet++ successor
pointtransformerv3/   # PTv3 backbone
S2AM3D/               # Part-level controllable
SNAP/                 # Segment Anything in Any Point Cloud ★
sonata/               # Self-supervised (Meta)
VDG-Uni3DSeg/         # Visual-description-guided
```

### Not Available (3/11)

| Model | Lab | Paper | Status |
|-------|-----|-------|--------|
| PointSeg | Tencent Youtu Lab | [arXiv 2403.06403](https://arxiv.org/abs/2403.06403) | No code. Check back. |
| CitySeg | Huawei | [arXiv 2508.09470](https://arxiv.org/abs/2508.09470) | No code. May remain closed. |
| FARVNet | Chengdu Univ | [Sensors 2025, 25(9)](https://doi.org/10.3390/s25092697) | No code. Low priority. |

---

## 3. Next Steps — Project Roadmap

### Phase 1: LingBot-Map Test Suite (`tests/`)

**Goal:** Establish test coverage on the LingBot-Map reconstruction pipeline before integrating any segmentation models.

| Task | Description | Priority |
|------|-------------|----------|
| **1.1** | Set up pytest + conftest.py with test fixtures (sample NPZ scene, mock camera) | Critical |
| **1.2** | Unit tests for `scene.py` — point cloud creation, active_points, frame queries | Critical |
| **1.3** | Unit tests for `octree.py` — Kaolin SPC construction, LOD queries, frustum culling | Critical |
| **1.4** | Unit tests for `camera.py` — projection, unprojection, path generation | Critical |
| **1.5** | Unit tests for `loader.py` — NPZ loading, validation, error cases | High |
| **1.6** | Unit tests for `unproject.py` — depth→3D correctness, edge cases | High |
| **1.7** | Unit tests for `renderer.py` — output shapes, color ranges, camera matrix consistency | High |
| **1.8** | Unit tests for `config.py` — YAML parsing, dataclass validation | Medium |
| **1.9** | Unit tests for `gpu_mem.py` — batch size calculation, VRAM estimation | Medium |
| **1.10** | Integration test — full pipeline: NPZ → scene → octree → render → export | High |

**Target:** ≥80% line coverage on all modules copied from LingBot-Map.

### Phase 2: Segmentation Library Evaluation

**Goal:** Evaluate cloned segmentation repos against worldEdit's specific needs and identify reusable components or models.

| Task | Description | Priority |
|------|-------------|----------|
| **2.1** | **SNAP** — Install & run inference on a sample point cloud. Test: (a) click-based segmentation (b) text-prompt segmentation. Benchmark speed + quality. | Critical |
| **2.2** | **ALPINE** — Test on LingBot-Map output. Evaluate as a fast pre-segmentation pass before interactive editing. | Critical |
| **2.3** | **Point-SAM** — Install & test. Compare with SNAP on same scenes. Evaluate prompt flexibility. | High |
| **2.4** | **S2AM3D** — Test part-level granularity on indoor scenes. Can it segment furniture sub-parts usefully? | High |
| **2.5** | **Sonata** — Extract pre-trained features. Evaluate as a feature backbone for custom segmentation head. | Medium |
| **2.6** | **PTv3** — Benchmark inference speed. Evaluate as a fine-tuning target if we collect labeled data. | Medium |
| **2.7** | **VDG-Uni3DSeg** — Test open-vocabulary queries. "Find all chairs" → segmentation mask. | Medium |
| **2.8** | **Point-MoE** — Evaluate multi-domain generalization. Does it handle our mixed indoor/outdoor scenes? | Low |
| **2.9** | **PointNeXt** — Benchmark as supervised baseline vs. PTv3. | Low |
| **2.10** | **FRNet** — (Not cloned) Clone & test as real-time alternative to FARVNet for range-view segmentation. | Low |

### Phase 3: Integration into worldEdit

**Goal:** Wire the best 1-2 segmentation models into `world_edit/segmentation.py` as additional strategies.

| Task | Description |
|------|-------------|
| **3.1** | Add `segment_by_snap()` — wraps SNAP inference, returns cluster labels |
| **3.2** | Add `segment_by_alpine()` — fast CPU pre-segmentation for large outdoor scenes |
| **3.3** | Add `segment_by_text()` — open-vocabulary query path (SNAP text mode or VDG) |
| **3.4** | Implement GPU-accelerated path (CUDA DBSCAN) for real-time interactive use |
| **3.5** | Benchmark all strategies on same scene: accuracy, speed, memory |

---

## 4. Decision Criteria

When evaluating Phase 2 results, rank models by:

1. **Zero-shot quality** — Does it work without fine-tuning on our data?
2. **Prompt flexibility** — Click + text + bbox prompts supported?
3. **Inference speed** — Usable interactively (<2s per segmentation)?
4. **Memory footprint** — Fits in ≤8GB VRAM alongside the scene?
5. **Integration complexity** — Clean API? ONNX exportable? Dependency weight?
6. **Community maintenance** — Active repo? Recent commits? Documentation?

**Working hypothesis:** SNAP (interactive) + ALPINE (fast pre-pass) will be the winning combination for worldEdit.

---

## 5. Project Structure After Phase 2

```
worldEdit/
├── world_edit/
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── classical.py      # Current DBSCAN/distance/CC (keep)
│   │   ├── snap.py           # SNAP wrapper ★
│   │   ├── alpine.py         # ALPINE wrapper ★
│   │   └── bridge.py         # 2D→3D bridge (VDG, PointSeg if available)
│   └── ...
├── tests/
│   ├── test_lingbot_map/     # Phase 1 tests
│   │   ├── test_scene.py
│   │   ├── test_octree.py
│   │   ├── test_camera.py
│   │   ├── test_loader.py
│   │   ├── test_unproject.py
│   │   ├── test_renderer.py
│   │   ├── test_config.py
│   │   ├── test_gpu_mem.py
│   │   └── test_integration.py
│   ├── test_segmentation/    # Phase 3 tests
│   │   ├── test_classical.py
│   │   ├── test_snap.py
│   │   └── test_alpine.py
│   └── conftest.py
├── SEGMENTATION_RESEARCH.md  # This file
└── README.md
```

---

*Last updated: 2026-05-05*
